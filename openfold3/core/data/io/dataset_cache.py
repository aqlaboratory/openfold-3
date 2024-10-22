"""IO functions to read and write metadata and dataset caches."""

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from openfold3.core.data.primitives.structure.dataset_cache import (
    DataCache,
    PreprocessingChainData,
    PreprocessingDataCache,
    PreprocessingReferenceMoleculeData,
    PreprocessingStructureData,
)
from openfold3.core.data.resources.residues import MoleculeType


def read_metadata_cache(metadata_cache_path: Path) -> PreprocessingDataCache:
    """Read the metadata cache created in preprocessing from a JSON file.

    Args:
        metadata_cache_path:
            Path to the metadata cache JSON file.

    Returns:
        PreprocessingDataCache:
            The metadata cache in a structured dataclass format.
    """
    # Load in dict format
    metadata_cache_dict = json.loads(metadata_cache_path.read_text())

    # Format the structure data
    structure_data_cache = {}

    for pdb_id, data in metadata_cache_dict["structure_data"].items():
        # TODO: Release date should never be None with new version, fix this after
        # rerunning preprocessing
        release_date = data.get("release_date")
        if release_date is not None:
            release_date = datetime.strptime(release_date, "%Y-%m-%d").date()

        status = data["status"]
        resolution = data.get("resolution")
        chains = data.get("chains")
        interfaces = data.get("interfaces")

        if chains is not None:
            chain_data = {}

            for chain_id, per_chain_data in chains.items():
                label_asym_id = per_chain_data["label_asym_id"]
                auth_asym_id = per_chain_data["auth_asym_id"]
                entity_id = per_chain_data["entity_id"]
                molecule_type = MoleculeType[per_chain_data["molecule_type"]]

                # This is only set for ligand chains
                reference_mol_id = per_chain_data.get("reference_mol_id")

                chain_data[chain_id] = PreprocessingChainData(
                    label_asym_id=label_asym_id,
                    auth_asym_id=auth_asym_id,
                    entity_id=entity_id,
                    molecule_type=molecule_type,
                    reference_mol_id=reference_mol_id,
                )
        else:
            chain_data = None

        structure_data_cache[pdb_id] = PreprocessingStructureData(
            release_date=release_date,
            status=status,
            resolution=resolution,
            chains=chain_data,
            interfaces=interfaces,
        )

    # Format the reference molecule data
    reference_molecule_data_cache = {}

    for pdb_id, data in metadata_cache_dict["reference_molecule_data"].items():
        conformer_gen_strategy = data["conformer_gen_strategy"]
        fallback_conformer_pdb_id = data["fallback_conformer_pdb_id"]
        canonical_smiles = data["canonical_smiles"]

        reference_molecule_data_cache[pdb_id] = PreprocessingReferenceMoleculeData(
            conformer_gen_strategy=conformer_gen_strategy,
            fallback_conformer_pdb_id=fallback_conformer_pdb_id,
            canonical_smiles=canonical_smiles,
        )

    return PreprocessingDataCache(
        structure_data=structure_data_cache,
        reference_molecule_data=reference_molecule_data_cache,
    )


def encode_datacache_types(obj: object) -> object:
    """JSON encoder for any non-standard types encountered in DataCache objects."""
    if isinstance(obj, date):
        return obj.isoformat()


# TODO: better type-hint for this?
def write_datacache_to_json(datacache: DataCache, output_path: Path) -> Path:
    """Writes a DataCache dataclass to a JSON file.

    Args:
        datacache:
            DataCache dataclass to be written to a JSON file.
        output_path:
            Path to the output JSON file.

    Returns:
        Full path to the output JSON file.
    """
    datacache_dict = asdict(datacache)

    with open(output_path, "w") as f:
        json.dump(datacache_dict, f, default=encode_datacache_types, indent=4)
