import json
import logging
from pathlib import Path

from tqdm import tqdm

from openfold3.core.data.io.dataset_cache import (
    write_datacache_to_json,
)
from openfold3.core.data.io.s3 import list_bucket_entries
from openfold3.core.data.primitives.caches.format import (
    DatasetReferenceMoleculeData,
    ProteinMonomerChainData,
    ProteinMonomerDatasetCache,
    ProteinMonomerStructureData,
)

logger = logging.getLogger(__name__)


# TODO: implement a more general way to interact with both local and S3 data
def create_protein_monomer_dataset_cache_af3(
    data_directory: Path,
    protein_reference_molecule_data_file: Path,
    dataset_name: str,
    output_dir: Path,
    s3_client_config: dict | None = None,
    check_filename_exists: str | None = None,
    num_workers: int = 1,
) -> ProteinMonomerDatasetCache:
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
        output_dir (Path):
            Directory to write the dataset cache to.
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

    Returns:
        ProteinMonomerDatasetCache: The created dataset cache.
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
            profile=s3_client_config.get("profile"),
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
    write_datacache_to_json(
        dataset_cache, output_dir / "training_cache_protein_monomer.json"
    )

    return dataset_cache
