import datetime
import json
import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path

from tqdm import tqdm

from openfold3.core.data.io.dataset_cache import (
    format_nested_dict_for_json,
    write_datacache_to_json,
)
from openfold3.core.data.io.s3 import list_bucket_entries
from openfold3.core.data.io.sequence.fasta import (
    consolidate_preprocessed_fastas,
)
from openfold3.core.data.primitives.caches.clustering import (
    add_cluster_ids_and_sizes,
)
from openfold3.core.data.primitives.caches.format import (
    DatasetReferenceMoleculeData,
    PreprocessingDataCache,
    PreprocessingStructureDataCache,
    ProteinMonomerChainData,
    ProteinMonomerDatasetCache,
    ProteinMonomerStructureData,
)
from openfold3.core.data.primitives.caches.metadata import (
    add_and_filter_alignment_representatives,
    build_provisional_clustered_dataset_cache,
    filter_by_max_polymer_chains,
    filter_by_release_date,
    filter_by_resolution,
    filter_by_skipped_structures,
    func_with_n_filtered_chain_log,
    set_nan_fallback_conformer_flag,
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

    # TODO: we need to catch NMR etc. here which don't have a defined resolution and
    # probably set resolution for them to zero
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
    metadata_cache = PreprocessingDataCache.from_json(metadata_cache_path)

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

    # Write the final dataset cache to disk
    write_datacache_to_json(dataset_cache, output_path)

    logger.info("DONE.")


# TODO: implement a more general way to interact with both local and S3 data
def create_protein_monomer_dataset_cache_af3(
    data_directory: Path,
    protein_reference_molecule_data_file: Path,
    dataset_name: str,
    output_path: Path,
    s3_client_config: dict | None = None,
    check_filename_exists: str | None = None,
    num_workers: int = 1,
) -> None:
    """Creates a protein monomer dataset cache.

    Args:
        data_directory (Path):
            Directory containing subdirectories for each protein monomer. Names of the
            per-chain directories will be used as chain IDs and representative ids for
            the corresponding chain in the dataset cache. If the directory lives in an
            S3 bucket, the path should be "s3:/<bucket>/<prefix>".
        protein_reference_molecule_data_file (Path):
            Path to a JSON file containing reference molecule data for each canonical
            protein monomer.
        dataset_name (str):
            Name of the dataset.
        output_path (Path):
            Path to write the dataset cache to.
        s3_client_config (dict, optional):
            Configuration for the S3 client. If None, the client is started without a
            profile. Supports profile and max_keys keys. Defaults to None.,
        check_filename_exists (str, optional):
            If provided, only adds proteins to the dataset cache if the given filename
            exists within the chain directory. Defaults to None, and if None all 
            directories are added.
        num_workers (int, optional):
            Number of workers to use for parallel processing. Defaults to 1. Only used 
            if check_filename_exists is specified.
    """
    # Get all chain directories
    # S3
    if str(data_directory).startswith("s3:/"):
        if s3_client_config is None:
            s3_client_config = {}
        logger.info("1/4: Fetching chain directories from S3 bucket.")
        chain_directories = list_bucket_entries(
            bucket_name=data_directory.parts[1],
            prefix="/".join(data_directory.parts[2:]) + "/",
            profile=s3_client_config.get("profile", None),
            max_keys=s3_client_config.get("max_keys", 1000),
            check_filename_exists=check_filename_exists,
            num_workers=num_workers,
        )
    # Local
    else:
        print("1/4: Fetching chain directories locally.")
        chain_directories = [
            entry for entry in data_directory.iterdir() if entry.is_dir()
        ]

    # Load reference molecule data
    with open(protein_reference_molecule_data_file) as f:
        reference_molecule_data_dict = json.load(f)

    # Populate structure data field
    structure_data = {}
    for chain_directory in tqdm(
        chain_directories,
        total=len(chain_directories),
        desc="2/4: Populating structure data",
    ):
        chain_id = chain_directory.stem
        structure_data[chain_id] = ProteinMonomerStructureData(
            {
                "1": ProteinMonomerChainData(
                    alignment_representative_id=chain_id, template_ids=[]
                )
            }
        )

    # Reference molecule data
    print("3/4: Populating reference molecule data.")
    reference_molecule_data = {}
    for ref_mol_id, ref_mol_data in reference_molecule_data_dict.items():
        reference_molecule_data[ref_mol_id] = DatasetReferenceMoleculeData(
            conformer_gen_strategy=ref_mol_data["conformer_gen_strategy"],
            fallback_conformer_pdb_id=ref_mol_data["fallback_conformer_pdb_id"],
            canonical_smiles=ref_mol_data["canonical_smiles"],
            set_fallback_to_nan=ref_mol_data["set_fallback_to_nan"],
        )

    # Create dataset cache
    dataset_cache = ProteinMonomerDatasetCache(
        name=dataset_name,
        structure_data=structure_data,
        reference_molecule_data=reference_molecule_data,
    )

    # Write the final dataset cache to disk
    print("4/4: Writing dataset cache to disk.")
    write_datacache_to_json(dataset_cache, output_path)
