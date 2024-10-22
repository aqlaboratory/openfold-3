import datetime
import json
import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path

from openfold3.core.data.io.dataset_cache import (
    read_metadata_cache,
    write_datacache_to_json,
)
from openfold3.core.data.io.sequence.fasta import (
    consolidate_preprocessed_fastas,
)
from openfold3.core.data.primitives.structure.dataset_cache import (
    PreprocessingStructureDataCache,
    add_and_filter_alignment_representatives,
    add_cluster_ids_and_sizes,
    build_provisional_clustered_dataset_cache,
    filter_by_max_polymer_chains,
    filter_by_release_date,
    filter_by_resolution,
    filter_by_skipped_structures,
    func_with_n_filtered_chain_log,
    set_nan_fallback_conformer_flag,
    subset_reference_molecule_data,
)

logger = logging.getLogger(__name__)


def filter_structure_metadata_training_af3(
    structure_cache: PreprocessingStructureDataCache,
    max_release_date: datetime.date | str,
    max_resolution: float = 9.0,
    max_polymer_chains: int = 300,
) -> PreprocessingStructureDataCache:
    """Filter the structure metadata cache to structures suitable for training.

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

    # Convenience wrapper that logs the number of structures filtered out
    with_log = partial(func_with_n_filtered_chain_log, logger=logger)

    # Removes structures that were skipped in preprocessing (skip logging here because
    # it does not work with skipped/failed structures)
    filtered_cache = filter_by_skipped_structures(structure_cache)

    filtered_cache = with_log(filter_by_resolution)(filtered_cache, max_resolution)
    filtered_cache = with_log(filter_by_release_date)(filtered_cache, max_release_date)
    filtered_cache = with_log(filter_by_max_polymer_chains)(
        filtered_cache, max_polymer_chains
    )

    return filtered_cache


def create_pdb_training_dataset_cache_af3(
    metadata_cache_path: Path,
    preprocessed_dir: Path,
    alignment_representatives_fasta: Path,
    output_path: Path,
    dataset_name: str,
    max_release_date: datetime.date | str,
    max_resolution: float = 9.0,
    max_polymer_chains: int = 300,
    write_no_alignment_repr_entries: bool = True,
) -> None:
    """Create a training cache from a metadata cache.

    Args:
        metadata_cache_path:
            Path to the preprocessed metadata cache.
        output_path:
            Path to write the training cache to.
    """
    metadata_cache = read_metadata_cache(metadata_cache_path)

    # Read in FASTAs of all sequences in the training set
    logger.info("Scanning FASTA directories...")
    id_to_sequence = consolidate_preprocessed_fastas(preprocessed_dir)

    # Get a mapping of PDB IDs to release dates before any filtering is done
    pdb_id_to_release_date = {}
    for pdb_id, metadata in metadata_cache.structure_data.items():
        pdb_id_to_release_date[pdb_id] = metadata.release_date

    # Subset the structures in the preprocessed metadata to only the desired ones
    metadata_cache.structure_data = filter_structure_metadata_training_af3(
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
    if write_no_alignment_repr_entries:
        structure_data, unmatched_entries = with_log(
            add_and_filter_alignment_representatives
        )(
            structure_cache=dataset_cache.structure_data,
            query_chain_to_seq=id_to_sequence,
            alignment_representatives_fasta=alignment_representatives_fasta,
            return_no_repr=True,
        )

        # Write all chains without alignment representatives to a JSON file. These are
        # excluded from training.
        with open(
            output_path.parent / "no_alignment_representative_entries.json", "w"
        ) as f:
            # Convert the internal dataclasses to dict
            unmatched_entries = {
                pdb_id: {chain_id: asdict(chain_data)}
                for pdb_id, chain_data in unmatched_entries.items()
                for chain_id, chain_data in chain_data.items()
            }

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
    add_cluster_ids_and_sizes(
        dataset_cache=dataset_cache,
        id_to_sequence=id_to_sequence,
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

    # Subset reference molecule data to only the structures in the training set
    subset_reference_molecule_data(dataset_cache)

    # Write the final dataset cache to disk
    write_datacache_to_json(dataset_cache, output_path)

    logger.info("DONE.")
