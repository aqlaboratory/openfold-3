from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from openfold3.core.data.resources.residues import MoleculeType


@dataclass
class PreprocessingChainData:
    """Chain-wise data from preprocessing metadata_cache."""

    label_asym_id: str
    auth_asym_id: str
    entity_id: int
    molecule_type: MoleculeType
    reference_mol_id: str | None  # only set for ligands


@dataclass
class PreprocessingStructureData:
    """Structure-wise data from preprocessing metadata_cache."""

    status: str
    release_date: datetime.date
    resolution: float | None
    chains: dict[str, PreprocessingChainData] | None
    interfaces: list[tuple[str, str]] | None


class PreprocessingStructureDataCache(TypedDict):
    """Structure data cache from preprocessing metadata_cache."""

    pdb_id: PreprocessingStructureData


@dataclass
class PreprocessingReferenceMoleculeData:
    """Reference molecule data from preprocessing metadata_cache."""

    conformer_gen_strategy: str
    fallback_conformer_pdb_id: str | None
    canonical_smiles: str


class PreprocessingReferenceMoleculeCache(TypedDict):
    """ "Reference molecule data cache from preprocessing metadata_cache."""

    ref_mol_id: PreprocessingReferenceMoleculeData


@dataclass
class PreprocessingDataCache:
    """Complete data cache from preprocessing metadata_cache."""

    structure_data: PreprocessingStructureDataCache
    reference_molecule_data: PreprocessingReferenceMoleculeCache

    @classmethod
    def from_json(cls, file: Path) -> PreprocessingDataCache:
        """Read the metadata cache created in preprocessing from a JSON file.

        Args:
            file:
                Path to the metadata cache JSON file.

        Returns:
            PreprocessingDataCache:
                The metadata cache in a structured dataclass format.
        """
        # Load in dict format
        metadata_cache_dict = json.loads(file.read_text())

        # Format the structure data
        structure_data_cache = {}

        for pdb_id, structure_data in metadata_cache_dict["structure_data"].items():
            status = structure_data["status"]

            if "skipped" in status:
                release_date = structure_data["release_date"]
                resolution = None
                chains = None
                interfaces = None
            elif status == "success":
                release_date = structure_data["release_date"]
                resolution = structure_data["resolution"]
                chains = structure_data["chains"]
                interfaces = structure_data["interfaces"]
            # TODO: Release date should never be None with new version, fix this after
            # rerunning preprocessing
            elif status == "failed":
                release_date = None
                resolution = None
                chains = None
                interfaces = None
            else:
                raise ValueError(f"Unexpected status: {status}")

            if release_date is not None:
                release_date = datetime.strptime(release_date, "%Y-%m-%d").date()

            if chains is not None:
                chain_data = {}

                for chain_id, per_chain_data in chains.items():
                    molecule_type = MoleculeType[per_chain_data.pop("molecule_type")]

                    # This is only set for ligand chains
                    # TODO: this should be explicitly None after preprocessing refactor,
                    # so if condition should be removed
                    if "reference_mol_id" in per_chain_data:
                        reference_mol_id = per_chain_data.pop("reference_mol_id")
                    else:
                        reference_mol_id = None

                    chain_data[chain_id] = PreprocessingChainData(
                        molecule_type=molecule_type,
                        reference_mol_id=reference_mol_id,
                        **per_chain_data,
                    )
            else:
                chain_data = None

            structure_data_cache[pdb_id] = PreprocessingStructureData(
                status=status,
                release_date=release_date,
                resolution=resolution,
                chains=chain_data,
                interfaces=interfaces,
            )

        # Format the reference molecule data
        reference_molecule_data_cache = {}

        for mol_id, mol_data in metadata_cache_dict["reference_molecule_data"].items():
            reference_molecule_data_cache[mol_id] = PreprocessingReferenceMoleculeData(
                **mol_data
            )

        return cls(
            structure_data=structure_data_cache,
            reference_molecule_data=reference_molecule_data_cache,
        )


# TODO: This may need to be updated in the future to support new datasets with different
# cache formats
@dataclass
class ClusteredDatasetChainData(PreprocessingChainData):
    """Chain-wise data with cluster and alignment information."""

    # Adds the following fields:
    cluster_id: str
    cluster_size: int
    alignment_representative_id: str | None


@dataclass
class ClusteredDatasetInterfaceData:
    """Interface-wise data with cluster information."""

    cluster_id: str
    cluster_size: int


@dataclass
class ClusteredDatasetStructureData:
    """Structure data with cluster information."""

    release_date: datetime.date
    resolution: float
    chains: dict[str, ClusteredDatasetChainData]
    interface_clusters: dict[str, ClusteredDatasetInterfaceData]


class ClusteredDatasetStructureDataCache(TypedDict):
    """Structure data cache with cluster information."""

    pdb_id: ClusteredDatasetStructureData


@dataclass
class DatasetReferenceMoleculeData(PreprocessingReferenceMoleculeData):
    """Reference molecule data with additional field for data leakage prevention."""

    set_fallback_to_nan: bool


class DatasetReferenceMoleculeCache(TypedDict):
    """Reference molecule data cache."""

    ref_mol_id: DatasetReferenceMoleculeData


@dataclass
class ClusteredDatasetCache:
    """Full data cache for clustered dataset with conformer leakage prevention."""

    name: str
    structure_data: ClusteredDatasetStructureDataCache
    reference_molecule_data: DatasetReferenceMoleculeCache

    @classmethod
    def from_json(cls, file: Path) -> ClusteredDatasetCache:
        """Constructor to format a json into this dataclass structure."""
        with open(file) as f:
            data = json.load(f)

        name = data["name"]

        # Format structure data
        structure_data = {}
        for pdb_id, per_structure_data in data["structure_data"].items():
            chain_data = per_structure_data.pop("chains")
            interface_data = per_structure_data.pop("interface_clusters")

            chains = {
                chain_id: ClusteredDatasetChainData(**chain_data[chain_id])
                for chain_id in chain_data
            }

            interfaces = {
                interface_id: ClusteredDatasetInterfaceData(
                    **interface_data[interface_id]
                )
                for interface_id in interface_data
            }

            structure_data[pdb_id] = ClusteredDatasetStructureData(
                chains=chains, interface_clusters=interfaces, **per_structure_data
            )

        # Format reference molecule data
        ref_mol_data = {}
        for ref_mol_id, ref_mol_data in data["reference_molecule_data"].items():
            ref_mol_data = DatasetReferenceMoleculeData(**ref_mol_data)
            ref_mol_data[ref_mol_id] = ref_mol_data

        return cls(
            name=name,
            structure_data=structure_data,
            reference_molecule_data=ref_mol_data,
        )


# Grouped type-aliases for more convenient type-hinting of general-purpose functions
ChainData = PreprocessingChainData | ClusteredDatasetChainData
StructureDataCache = (
    PreprocessingStructureDataCache | ClusteredDatasetStructureDataCache
)
ReferenceMoleculeCache = (
    PreprocessingReferenceMoleculeCache | DatasetReferenceMoleculeCache
)
DataCache = PreprocessingDataCache | ClusteredDatasetCache
