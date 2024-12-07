from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeAlias

import openfold3.core.data.io.dataset_cache as io
from openfold3.core.data.resources.residues import MoleculeType

# This holds a mapping of the string name of all dataset cache classes to their actual
# class object. This string name is additionally stored with every dataset cache as its
# "_type" attribute, which is also written out to the JSON when saving a dataset cache.
# This has the benefit that every downstream script can easily infer which class to use
# to read the JSON file into a fully instantiated datacache object of the appropriate
# type.
# The mapping is populated anytime a new DataCache class is defined and registered with
# the register_datacache decorator.
DATASET_CACHE_CLASS_REGISTRY = {}


# TODO: Could make post-init check that this is set
def register_datacache(cls):
    """Register a specific DataCache class in the DATASET_CACHE_CLASS_REGISTRY.

    Args:
        cls (Type[DataCache]): The class to register

    Returns:
        Type[DataCache]: The registered class
    """
    DATASET_CACHE_CLASS_REGISTRY[cls.__name__] = cls
    cls._registered = True
    cls._type = cls.__name__
    return cls


# TODO: Actually update the preprocessing code to use this class, currently it's only
# used in the training dataset logic
# ==============================================================================
# PREPROCESSING CACHE
# ==============================================================================
# This is the cache that gets created by the preprocessing script, and is usually used
# to create the other dataset caches for training / validation.
@register_datacache
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

        # Remove _type field (already an internal private attribute so shouldn't be
        # defined as an explicit field)
        if "_type" in metadata_cache_dict:
            # This is conditional for legacy compatibility, should be removed after
            del metadata_cache_dict["_type"]

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
                    # so if-condition should be removed
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

    def to_json(self, file: Path) -> None:
        """Write the metadata cache to a JSON file.

        Args:
            file:
                Path to the JSON file to write the metadata cache to.
        """
        io.write_datacache_to_json(self, file)


@dataclass
class PreprocessingStructureData:
    """Structure-wise data from preprocessing metadata_cache."""

    status: str
    release_date: datetime.date
    resolution: float | None
    chains: dict[str, PreprocessingChainData] | None
    interfaces: list[tuple[str, str]] | None


PreprocessingStructureDataCache: TypeAlias = dict[str, PreprocessingStructureData]
"""Structure data cache from preprocessing metadata_cache."""


@dataclass
class PreprocessingChainData:
    """Chain-wise data from preprocessing metadata_cache."""

    label_asym_id: str
    auth_asym_id: str
    entity_id: int
    molecule_type: MoleculeType
    reference_mol_id: str | None  # only set for ligands


@dataclass
class PreprocessingReferenceMoleculeData:
    """Reference molecule data from preprocessing metadata_cache."""

    conformer_gen_strategy: str
    fallback_conformer_pdb_id: str | None
    canonical_smiles: str


PreprocessingReferenceMoleculeCache: TypeAlias = dict[
    str, PreprocessingReferenceMoleculeData
]


# ==============================================================================
# GENERAL DATASET FORMAT LAYOUT
# ==============================================================================
# This is a general template format that every other dataset cache, such as
# PDB-weighted, PDB-disordered, and PDB-validation, etc., should follow.
@dataclass
class DatasetCache(ABC):
    """Format that every Dataset Cache should have."""

    name: str  # for referencing in dataset config
    structure_data: dataclass
    reference_molecule_data: dataclass

    @classmethod
    @abstractmethod
    def from_json(cls, file: Path) -> DatasetCache:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def to_json(self, file: Path) -> None:
        """Write the dataset cache to a JSON file.

        Args:
            file:
                Path to the JSON file to write the dataset cache to.
        """
        io.write_datacache_to_json(self, file)


@dataclass
class DatasetChainData:
    """Central class for chain-wise data that can be used for general type-hinting."""

    pass


# TODO: Set fallback to NaN could be removed from here in the future?
@dataclass
class DatasetReferenceMoleculeData:
    """Fields that every Dataset format's reference molecule data should have."""

    conformer_gen_strategy: str
    fallback_conformer_pdb_id: str | None
    canonical_smiles: str
    set_fallback_to_nan: bool


# Reference molecule data should be the same for all datasets so we provide it here as a
# general type.
DatasetReferenceMoleculeCache: TypeAlias = dict[str, DatasetReferenceMoleculeData]


# ==============================================================================
# SPECIALIZED DATASETS
# ==============================================================================
# This is where all specialized training dataset caches and validation set caches should
# be implemented.


# TEMPLATE for dataset caches with chain-wise, interface-wise and reference molecule
# data.
@dataclass
class ChainInterfaceReferenceMolCache(DatasetCache):
    """Specialized dataset cache format template.

    Template for any data cache with chain-wise, interface-wise, and reference molecule
    data. For example the PDB-weighted set, PDB-disordered set, and PDB-validation set.
    """

    # This defines the individual constructors for the chain, interface, reference
    # molecule, and structure data, so that all other datasets can inherit this
    # from_json method and set their own formats.
    _chain_data_format: dataclass = None
    _interface_data_format: dataclass = None
    _structure_data_format: dataclass = None
    _ref_mol_data_format: dataclass = DatasetReferenceMoleculeData

    def from_json(cls, file: Path) -> ChainInterfaceReferenceMolCache:
        """Constructor to format a json into this dataclass structure."""
        if any(
            format_ is None
            for format_ in [
                cls._chain_data_format,
                cls._interface_data_format,
                cls._ref_mol_data_format,
                cls._structure_data_format,
            ]
        ):
            raise NotImplementedError("Data formats must be defined in subclass.")

        with open(file) as f:
            data = json.load(f)

        # Remove _type field (already an internal private attribute so shouldn't be
        # defined as an explicit field)
        if "_type" in data:
            # This is conditional for legacy compatibility, should be removed after
            del data["_type"]

        name = data["name"]

        # Format structure data
        structure_data = {}
        for pdb_id, per_structure_data in data["structure_data"].items():
            chain_data = per_structure_data.pop("chains")
            interface_data = per_structure_data.pop("interfaces")

            # Extract all chain data into respective chain data format
            chains = {
                chain_id: cls._chain_data_format(**chain_data[chain_id])
                for chain_id in chain_data
            }

            # Extract all interface data into respective interface data format
            interfaces = {
                interface_id: cls._interface_data_format(**interface_data[interface_id])
                for interface_id in interface_data
            }

            # Combine chain and interface data with remaining structure data
            structure_data[pdb_id] = cls._structure_data_format(
                chains=chains, interfaces=interfaces, **per_structure_data
            )

        # Format reference molecule data into respective format
        ref_mol_data = {}
        for ref_mol_id, per_ref_mol_data in data["reference_molecule_data"].items():
            per_ref_mol_data_fmt = cls._ref_mol_data_format(**per_ref_mol_data)
            ref_mol_data[ref_mol_id] = per_ref_mol_data_fmt

        return cls(
            name=name,
            structure_data=structure_data,
            reference_molecule_data=ref_mol_data,
        )


# TEMPLATE for metadata for PDB datasets
@dataclass
class PDBChainData(DatasetChainData):
    """Chain-wise data for PDB datasets."""

    # TODO: These are not mandatory and currently kept for debugging purposes, but may
    # be removed later
    label_asym_id: str
    auth_asym_id: str
    entity_id: int

    molecule_type: MoleculeType
    reference_mol_id: str | None  # only set for ligands
    alignment_representative_id: str | None  # not set for ligands and DNA
    template_ids: list[str] | None  # only set for proteins


# CLUSTERED DATASET FORMAT (e.g. PDB-weighted)
@dataclass
class ClusteredDatasetChainData(PDBChainData):
    """Chain-wise data with cluster information."""

    cluster_id: str
    cluster_size: int


@dataclass
class ClusteredDatasetInterfaceData:
    """Interface-wise data with cluster information."""

    cluster_id: str
    cluster_size: int


@dataclass
class ClusteredDatasetStructureData:
    """Structure data with clusters and added metadata."""

    release_date: datetime.date
    resolution: float
    chains: dict[str, ClusteredDatasetChainData]
    interfaces: dict[str, ClusteredDatasetInterfaceData]


ClusteredDatasetStructureDataCache: TypeAlias = dict[str, ClusteredDatasetStructureData]
"""Structure data cache with cluster information."""


@register_datacache
@dataclass
class ClusteredDatasetCache(ChainInterfaceReferenceMolCache):
    """Full data cache for clustered dataset.

    This is the most information-rich data cache format, with full chain-wise,
    interface-wise, and reference molecule data, with cluster information for each chain
    and interface.

    Used for:
        - PDB-weighted training set
    """

    name: str
    structure_data: ClusteredDatasetStructureDataCache
    reference_molecule_data: DatasetReferenceMoleculeCache

    # Defines the constructor formats for the inherited from_json method
    _chain_data_format = ClusteredDatasetChainData
    _interface_data_format = ClusteredDatasetInterfaceData
    _ref_mol_data_format = DatasetReferenceMoleculeData
    _structure_data_format = ClusteredDatasetStructureData


# Grouped type-aliases for more convenient type-hinting of general-purpose functions
ChainData = PreprocessingChainData | PDBChainData
StructureDataCache = (
    PreprocessingStructureDataCache | ClusteredDatasetStructureDataCache
)
ReferenceMoleculeCache = (
    PreprocessingReferenceMoleculeCache | DatasetReferenceMoleculeCache
)
DataCache = PreprocessingDataCache | DatasetCache
