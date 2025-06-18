import datetime
import json
import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path

from openfold3.core.data.io.dataset_cache import (
    format_nested_dict_for_json,
    write_datacache_to_json,
)
from openfold3.core.data.io.sequence.fasta import (
    consolidate_preprocessed_fastas,
)
from openfold3.core.data.primitives.caches.clustering import (
    add_cluster_data,
)
from openfold3.core.data.primitives.caches.filtering import (
    add_and_filter_alignment_representatives,
    build_provisional_clustered_dataset_cache,
    filter_by_max_polymer_chains,
    filter_by_release_date,
    filter_by_resolution,
    filter_by_skipped_structures,
    filter_by_token_count,
    func_with_n_filtered_chain_log,
    set_nan_fallback_conformer_flag,
)
from openfold3.core.data.primitives.caches.format import (
    PreprocessingDataCache,
    PreprocessingStructureDataCache,
)

logger = logging.getLogger(__name__)


# TODO: reorganize metadata cache creation pipelines into a caches module
# TODO: Make docstring more complete for new args
def filter_structure_metadata_of3(
    structure_cache: PreprocessingStructureDataCache,
    max_release_date: datetime.date | str,
    min_release_date: datetime.date | str = None,
    max_resolution: float = 9.0,
    ignore_nmr: bool = True,
    max_polymer_chains: int = 300,
    max_tokens: int | None = None,
) -> PreprocessingStructureDataCache:
    """Filter the structure metadata cache for training or validation.

    TODO: Update docstring

    Applies the following filters from the AF3 SI 2.5.4 that have not yet been applied
    in preprocessing:
    - release date <= max_release_date
    - number of polymer chains <= 300
    - resolution <= 9.0

    Args:
        structure_cache:
            Structure metadata cache to filter.

    Returns:
        Filtered structure metadata cache.
    """
    if not isinstance(max_release_date, datetime.date):
        max_release_date = datetime.datetime.strptime(
            max_release_date, "%Y-%m-%d"
        ).date()
    if min_release_date and not isinstance(min_release_date, datetime.date):
        min_release_date = datetime.datetime.strptime(
            min_release_date, "%Y-%m-%d"
        ).date()

    # Convenience wrapper that logs the number of structures filtered out
    with_log = partial(func_with_n_filtered_chain_log, logger=logger)

    # Removes structures that were skipped in preprocessing (skip logging here because
    # it does not work with skipped/failed structures)
    filtered_cache = filter_by_skipped_structures(structure_cache)

    filtered_cache = with_log(filter_by_resolution)(
        filtered_cache, max_resolution, ignore_nmr=ignore_nmr
    )

    filtered_cache = with_log(filter_by_release_date)(
        filtered_cache,
        min_date=min_release_date,
        max_date=max_release_date,
    )

    filtered_cache = with_log(filter_by_max_polymer_chains)(
        filtered_cache, max_polymer_chains
    )
    if max_tokens:
        filtered_cache = with_log(filter_by_token_count)(filtered_cache, max_tokens)

    return filtered_cache


# TODO: Add docstring!
def create_pdb_training_dataset_cache_of3(
    metadata_cache_path: Path,
    preprocessed_dir: Path,
    alignment_representatives_fasta: Path,
    output_path: Path,
    dataset_name: str,
    max_release_date: datetime.date | str,
    max_resolution: float = 9.0,
    max_polymer_chains: int = 300,
    filter_missing_alignment: bool = True,
    missing_alignment_log: Path = None,
) -> None:
    """Create a training cache from a metadata cache.

    Args:
        metadata_cache_path:
            Path to the preprocessed metadata cache.
        preprocessed_dir:
            Preprocessing output directory with preprocessed structure and fasta files.
        alignment_representatives_fasta:
            A FASTA file containing the identifier of each alignment as a header and the
            query sequence as the sequence. Used to map every sequence in the
            preprocessed directory to a corresponding MSA. Every protein or RNA sequence
            without an alignment will be filtered out.
        output_path:
            Path to write the training dataset cache to.
        dataset_name:
            Name of the dataset, e.g. 'PDB-weighted'.
        max_release_date:
            Maximum release date for included structures, formatted as 'YYYY-MM-DD'.
        max_resolution:
            Maximum resolution for structures in the dataset in Å.
        max_polymer_chains:
            Maximum number of polymer chains for included structures.
        filter_missing_alignment:
            Whether to filter out chains (and their corresponding structures) whose
            sequences do not match to a representative in the
            alignment_representatives_fasta.
        missing_alignment_log:
            Path to write a JSON file containing all chains that were filtered out
            because they do not have a corresponding alignment.
    """
    metadata_cache = PreprocessingDataCache.from_json(metadata_cache_path)

    # Read in FASTAs of all sequences in the training set
    logger.info("Scanning FASTA directories...")
    id_to_sequence = consolidate_preprocessed_fastas(preprocessed_dir)

    # Get a mapping of PDB IDs to release dates before any filtering is done
    pdb_id_to_release_date = {}
    for pdb_id, metadata in metadata_cache.structure_data.items():
        pdb_id_to_release_date[pdb_id] = metadata.release_date

    # Subset the structures in the preprocessed metadata to only the desired ones
    metadata_cache.structure_data = filter_structure_metadata_of3(
        metadata_cache.structure_data,
        max_release_date=max_release_date,
        max_resolution=max_resolution,
        max_polymer_chains=max_polymer_chains,
    )

    # Create a provisional dataset training cache with extra fields for cluster and NaN
    # conformer information that will be filled in later
    dataset_cache = build_provisional_clustered_dataset_cache(
        preprocessing_cache=metadata_cache,
        dataset_name=dataset_name,
    )

    # Convenience wrapper that logs the number of structures filtered out
    with_log = partial(func_with_n_filtered_chain_log, logger=logger)

    # Map each target chain to an alignment representative, then filter all structures
    # without alignment representatives
    if filter_missing_alignment:
        if missing_alignment_log:
            structure_data, unmatched_entries = with_log(
                add_and_filter_alignment_representatives
            )(
                structure_cache=dataset_cache.structure_data,
                query_chain_to_seq=id_to_sequence,
                alignment_representatives_fasta=alignment_representatives_fasta,
                return_no_repr=True,
            )

            # Write all chains without alignment representatives to a JSON file. These
            # are excluded from training.
            with open(missing_alignment_log, "w") as f:
                # Convert the internal dataclasses to dict
                unmatched_entries = {
                    pdb_id: {chain_id: asdict(chain_data)}
                    for pdb_id, chain_data in unmatched_entries.items()
                    for chain_id, chain_data in chain_data.items()
                }

                # Format datacache-types appropriately
                unmatched_entries = format_nested_dict_for_json(unmatched_entries)

                json.dump(unmatched_entries, f, indent=4)
        else:
            structure_data = with_log(add_and_filter_alignment_representatives)(
                dataset_cache.structure_data,
                query_chain_to_seq=id_to_sequence,
                alignment_representatives_fasta=alignment_representatives_fasta,
                preprocessed_dir=preprocessed_dir,
            )

        dataset_cache.structure_data = structure_data

    # Add cluster IDs and cluster sizes for all chains
    logger.info("Adding cluster information...")
    add_cluster_data(
        dataset_cache=dataset_cache,
        id_to_sequence=id_to_sequence,
        add_sizes=True,
    )
    logger.info("Done clustering.")

    # Block usage of reference conformer coordinates from PDB-IDs that are outside the
    # training split. Needs to be run before the filtering to use the full release date
    # information in structure_data.
    set_nan_fallback_conformer_flag(
        pdb_id_to_release_date=pdb_id_to_release_date,
        reference_mol_cache=dataset_cache.reference_molecule_data,
        max_model_pdb_release_date=max_release_date,
    )

    # Write the final dataset cache to disk
    write_datacache_to_json(dataset_cache, output_path)

    logger.info("DONE.")
