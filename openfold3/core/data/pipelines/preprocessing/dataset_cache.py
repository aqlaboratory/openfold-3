import datetime
import json
import logging
from pathlib import Path

from openfold3.core.data.io.sequence.fasta import (
    consolidate_preprocessed_fastas,
    read_multichain_fasta,
)
from openfold3.core.data.primitives.structure.dataset_cache import (
    StructureMetadataCache,
    add_chain_representatives,
    filter_by_max_polymer_chains,
    filter_by_release_date,
    filter_by_resolution,
    filter_by_skipped_structures,
    filter_no_alignment_representative,
)

logger = logging.getLogger(__name__)


def filter_structure_metadata_af3(
    cache: StructureMetadataCache,
    max_release_date: datetime.date | str,
    max_resolution: float = 9.0,
    max_polymer_chains: int = 300,
) -> StructureMetadataCache:
    """Filter the structure metadata cache to structures suitable for training.

    Applies the following filters from the AF3 SI that have not yet been applied in
    preprocessing:
    - release date <= max_release_date
    - number of polymer chains <= 300
    - resolution <= 9.0

    Args:
        cache:
            Structure metadata cache to filter.

    Returns:
        Filtered structure metadata cache.
    """
    if not isinstance(max_release_date, datetime.date):
        max_release_date = datetime.datetime.strptime(
            max_release_date, "%Y-%m-%d"
        ).date()

    # Removes structures that were skipped in preprocessing
    filtered_cache = filter_by_skipped_structures(cache)

    filtered_cache = filter_by_resolution(filtered_cache, max_resolution)
    filtered_cache = filter_by_release_date(filtered_cache, max_release_date)
    filtered_cache = filter_by_max_polymer_chains(filtered_cache, max_polymer_chains)

    return filtered_cache


def create_training_cache_af3(
    metadata_cache_path: Path,
    preprocessed_dir: Path,
    alignment_representatives_fasta: Path,
    output_path: Path,
    max_release_date: datetime.date | str,
    max_resolution: float = 9.0,
    max_polymer_chains: int = 300,
    write_unmatched_representatives: bool = False,
) -> None:
    """Create a training cache from a metadata cache.

    Args:
        metadata_cache_path:
            Path to the preprocessed metadata cache.
        output_path:
            Path to write the training cache to.
    """
    metadata_cache = json.loads(metadata_cache_path.read_text())

    training_cache = {}

    structure_data = metadata_cache["structure_data"]

    # Add dataset name TODO: handle this more cleanly
    training_cache["name"] = "PDB-weighted"

    # TEMPORARY FIX FOR PREPROCESSING BUG
    del structure_data["pdb_id"]
    del structure_data["status"]

    # Subset the structures in the preprocessed metadata to only the desired ones
    structure_data = filter_structure_metadata_af3(
        structure_data,
        max_release_date=max_release_date,
        max_resolution=max_resolution,
        max_polymer_chains=max_polymer_chains,
    )

    # Delete status
    for metadata in structure_data.values():
        del metadata["status"]

    # Map each target chain to an alignment representative
    logger.info("Scanning FASTA directories...")
    repr_chain_to_seq = read_multichain_fasta(alignment_representatives_fasta)
    query_chain_to_seq = consolidate_preprocessed_fastas(preprocessed_dir)
    add_chain_representatives(structure_data, query_chain_to_seq, repr_chain_to_seq)

    # Filter the cache to only include structures with alignment representatives
    structure_data = filter_no_alignment_representative(structure_data)

    # Run dummy clustering
    for metadata in structure_data.values():
        for chain in metadata["chains"].values():
            chain["cluster_size"] = 1

        new_interface_dict = {}

        for interface_pair in metadata["interfaces"]:
            chain_1, chain_2 = interface_pair
            new_interface_dict[f"{chain_1}_{chain_2}"] = 1

        del metadata["interfaces"]
        metadata["interface_cluster_sizes"] = new_interface_dict

    training_cache["structure_data"] = structure_data
    training_cache["reference_molecule_data"] = metadata_cache[
        "reference_molecule_data"
    ]

    with open(output_path, "w") as f:
        json.dump(training_cache, f, indent=4)

    logger.info("DONE.")


# Run test
logger = logging.getLogger("openfold3")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

metadata_cache_path = Path(
    "/pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v2/metadata.json"
)
preprocessed_dir = Path(
    "/pscratch/sd/l/ljarosch/of3_pdb_processing/pdb_data_cleaned_50w_300-poly_old/cif_files"
)
alignment_representatives_fasta = Path(
    "/pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v2/val_rep.fasta"
)
output_path = Path(
    "/pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v2/training_cache.json"
)
max_release_date = "2021-09-30"
max_resolution = 9.0
max_polymer_chains = 300

create_training_cache_af3(
    metadata_cache_path,
    preprocessed_dir,
    alignment_representatives_fasta,
    output_path,
    max_release_date,
    max_resolution,
    max_polymer_chains,
    write_unmatched_representatives=False,
)
