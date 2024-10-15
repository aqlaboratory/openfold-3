import datetime
import json
import logging
from pathlib import Path

from openfold3.core.data.io.sequence.fasta import (
    consolidate_preprocessed_fastas,
)
from openfold3.core.data.primitives.structure.dataset_cache import (
    StructureMetadataCache,
    add_and_filter_alignment_representatives,
    add_cluster_ids_and_sizes,
    filter_by_max_polymer_chains,
    filter_by_release_date,
    filter_by_resolution,
    filter_by_skipped_structures,
    remove_interface_keys,
)

logger = logging.getLogger(__name__)


def filter_structure_metadata_training_af3(
    cache: StructureMetadataCache,
    max_release_date: datetime.date | str,
    max_resolution: float = 9.0,
    max_polymer_chains: int = 300,
) -> StructureMetadataCache:
    """Filter the structure metadata cache to structures suitable for training.

    Applies the following filters from the AF3 SI 2.5.4 that have not yet been applied
    in preprocessing:
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
    metadata_cache = json.loads(metadata_cache_path.read_text())

    # Read in FASTAs of all sequences in the training set
    logger.info("Scanning FASTA directories...")
    id_to_sequence = consolidate_preprocessed_fastas(preprocessed_dir)

    training_cache = {}

    structure_data = metadata_cache["structure_data"]
    reference_mol_data = metadata_cache["reference_molecule_data"]

    # Add dataset name
    training_cache["name"] = dataset_name

    # Subset the structures in the preprocessed metadata to only the desired ones
    structure_data = filter_structure_metadata_training_af3(
        structure_data,
        max_release_date=max_release_date,
        max_resolution=max_resolution,
        max_polymer_chains=max_polymer_chains,
    )

    # Delete no longer needed status field
    for metadata in structure_data.values():
        del metadata["status"]

    # Map each target chain to an alignment representative, then filter all structures
    # without alignment representatives
    if write_no_alignment_repr_entries:
        structure_data, unmatched_entries = add_and_filter_alignment_representatives(
            structure_data,
            query_chain_to_seq=id_to_sequence,
            alignment_representatives_fasta=alignment_representatives_fasta,
            return_no_repr=True,
        )

        with open(
            output_path.parent / "no_alignment_representative_entries.json", "w"
        ) as f:
            json.dump(unmatched_entries, f, indent=4)
    else:
        structure_data = add_and_filter_alignment_representatives(
            structure_data,
            query_chain_to_seq=id_to_sequence,
            alignment_representatives_fasta=alignment_representatives_fasta,
            preprocessed_dir=preprocessed_dir,
        )

    # Add cluster IDs and cluster sizes for all chains
    add_cluster_ids_and_sizes(
        structure_cache=structure_data,
        reference_mol_cache=reference_mol_data,
        id_to_sequence=id_to_sequence,
    )

    # Remove obsolete "interface" keys whose information is effectively fully contained
    # in "interface_clusters"
    remove_interface_keys(structure_data)

    training_cache["structure_data"] = structure_data
    training_cache["reference_molecule_data"] = reference_mol_data

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
