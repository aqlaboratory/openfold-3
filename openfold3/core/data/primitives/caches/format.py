from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

import lmdb

from openfold3.core.data.resources.residues import MoleculeType

if TYPE_CHECKING:
    from openfold3.core.data.primitives.caches.lmdb import LMDBDict

K = TypeVar("K")
V = TypeVar("V")
# TODO: Revisit in future if this registry is still needed after template script
# refactor

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
def register_datacache(cls: DataCacheType) -> DataCacheType:
    """
    Decorator to register a DataCache class in the registry and validate that
    any attribute named like '_xxx_format' is non-None.
    """
    # 1. Detect all format-like attributes that start with '_' and end with '_format'
    format_attrs = [
        attr for attr in dir(cls) if attr.startswith("_") and attr.endswith("_format")
    ]

    # 2. Check each attribute to make sure it's not None
    missing_attrs = [attr for attr in format_attrs if getattr(cls, attr, None) is None]
    if missing_attrs:
        raise ValueError(
            f"Class {cls.__name__} is missing required format attribute(s): "
            f"{missing_attrs}."
        )

    # 3. Put it in the registry
    DATASET_CACHE_CLASS_REGISTRY[cls.__name__] = cls

    # 4. Mark the class as validated and registered
    cls._format_validated = True
    cls._registered = True
    cls._type = cls.__name__

    return cls


# TODO: Actually update the preprocessing code to use this class, currently it's only
# used in the training dataset logic
# TODO: update names to reflect that these create the metadata caches
# ==============================================================================
# PREPROCESSING CACHES
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
            # TODO make categorical variables Enums
            if "skipped" in status:
                release_date = structure_data["release_date"]
                experimental_method = None
                resolution = None
                chains = None
                interfaces = None
                token_count = None
            elif status == "success":
                release_date = structure_data["release_date"]
                experimental_method = structure_data["experimental_method"]
                resolution = structure_data["resolution"]
                chains = structure_data["chains"]
                interfaces = structure_data["interfaces"]
                token_count = structure_data["token_count"]
            # TODO: Release date should never be None with new version, fix this after
            # rerunning preprocessing
            elif status == "failed":
                release_date = None
                resolution = None
                chains = None
                interfaces = None
                token_count = None
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
                experimental_method=experimental_method,
                resolution=resolution,
                chains=chain_data,
                interfaces=interfaces,
                token_count=token_count,
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
        # Avoid circular import
        from openfold3.core.data.io.dataset_cache import write_datacache_to_json

        write_datacache_to_json(self, file)


@register_datacache
@dataclass
class DisorderedPreprocessingDataCache(PreprocessingDataCache):
    """Complete data cache from preprocessing metadata_cache."""

    structure_data: DisorderedPreprocessingStructureDataCache
    reference_molecule_data: PreprocessingReferenceMoleculeCache

    # TODO: refactor the PreprocessingDataCache to be more general so that functions
    # like this can be mostly reused
    @classmethod
    def from_json(cls, file: Path) -> DisorderedPreprocessingDataCache:
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
                experimental_method = None
                chains = None
                interfaces = None
                token_count = None
                gdt = None
                chain_map = None
                transform_array = None
                best_model_filename = None
                distance_clash_map = None
            elif status == "success":
                release_date = structure_data["release_date"]
                resolution = structure_data["resolution"]
                experimental_method = structure_data["experimental_method"]
                chains = structure_data["chains"]
                interfaces = structure_data["interfaces"]
                token_count = structure_data["token_count"]
                gdt = structure_data["gdt"]
                chain_map = structure_data["chain_map"]
                transform_array = structure_data["transform"]
                best_model_filename = structure_data["best_model_filename"]
                distance_clash_map = structure_data["distance_clash_map"]
            # TODO: Release date should never be None with new version, fix this after
            # rerunning preprocessing
            elif status == "failed":
                release_date = None
                resolution = None
                experimental_method = None
                chains = None
                interfaces = None
                token_count = None
                gdt = None
                chain_map = None
                transform_array = None
                best_model_filename = None
                distance_clash_map = None
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

            structure_data_cache[pdb_id] = DisorderedPreprocessingStructureData(
                status=status,
                release_date=release_date,
                resolution=resolution,
                experimental_method=experimental_method,
                chains=chain_data,
                interfaces=interfaces,
                token_count=token_count,
                gdt=gdt,
                chain_map=chain_map,
                transform_array=transform_array,
                best_model_filename=best_model_filename,
                distance_clash_map=distance_clash_map,
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


# TODO: make categorical attributes like experimental method Enum
@dataclass
class PreprocessingStructureData:
    """Structure-wise data from preprocessing metadata_cache."""

    status: str
    release_date: datetime.date
    experimental_method: str
    resolution: float | None
    chains: dict[str, PreprocessingChainData] | None
    interfaces: list[tuple[str, str]] | None
    token_count: int


@dataclass
class DisorderedPreprocessingStructureData(PreprocessingStructureData):
    """Structure-wise data from preprocessing metadata_cache with disordered data."""

    gdt: float
    chain_map: dict[str, str]
    transform_array: list[list[float]]
    best_model_filename: str
    distance_clash_map: dict[float, bool]


PreprocessingStructureDataCache: TypeAlias = dict[str, PreprocessingStructureData]
"""Structure data cache from preprocessing metadata_cache."""

DisorderedPreprocessingStructureDataCache: TypeAlias = dict[
    str, DisorderedPreprocessingStructureData
]
"""Structure data cache from disordered preprocessing metadata_cache."""


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
    residue_count: int  # TODO: Remove this from after-preprocessing caches except Val


PreprocessingReferenceMoleculeCache: TypeAlias = dict[
    str, PreprocessingReferenceMoleculeData
]


# ==============================================================================
# GENERAL DATASET FORMAT LAYOUT
# ==============================================================================
# This is a general template format that every other dataset cache, such as
# PDB-weighted, PDB-disordered, and PDB-validation, etc., should follow.
@dataclass
class DatasetCache:
    """Base class for all dataset cache classes."""

    name: str  # for referencing in dataset config
    structure_data: dataclass
    reference_molecule_data: dataclass

    _registered = False
    _format_validated: bool = False

    # TODO: update parsers for this base class
    @classmethod
    def from_json(cls, file: Path) -> DatasetCache:
        """Costructs a datacache from a json.

        Args:
            file (Path):
                Path to the JSON file to read the datacache from.

        Returns:
            DatasetCache:
                The constructed datacache.
        """

        with open(file) as f:
            data = json.load(f)

        return cls(
            name=cls._parse_name_json(data),
            structure_data=cls._parse_structure_data_json(data),
            reference_molecule_data=cls._parse_ref_mol_data_json(data),
        )

    def _parse_type_json(data: dict) -> None:
        # Remove _type field (already an internal private attribute so shouldn't be
        # defined as an explicit field)
        if "_type" in data:
            # This is conditional for legacy compatibility, should be removed after
            del data["_type"]

    def _parse_name_json(data: dict) -> str:
        return data["name"]

    @classmethod
    def _parse_structure_data_json(cls, data: dict) -> dict:
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
        return structure_data

    @classmethod
    def _parse_ref_mol_data_json(cls, data: dict) -> dict:
        # Format reference molecule data into respective format
        ref_mol_data = {}
        for ref_mol_id, per_ref_mol_data in data["reference_molecule_data"].items():
            per_ref_mol_data_fmt = cls._ref_mol_data_format(**per_ref_mol_data)
            ref_mol_data[ref_mol_id] = per_ref_mol_data_fmt
        return ref_mol_data

    def to_json(self, file: Path) -> None:
        """Write the dataset cache to a JSON file.

        Args:
            file:
                Path to the JSON file to write the dataset cache to.
        """
        # Avoid circular import
        from openfold3.core.data.io.dataset_cache import write_datacache_to_json

        write_datacache_to_json(self, file)

    @classmethod
    def from_lmdb(
        cls,
        lmdb_directory: Path,
        str_encoding: Literal["utf-8", "pkl"] = "utf-8",
        structure_data_encoding: Literal["utf-8", "pkl"] = "pkl",
        reference_molecule_data_encoding: Literal["utf-8", "pkl"] = "pkl",
    ) -> DatasetCache:
        """Constructs a datacache from an LMDB.

        LMDB-backed datacaches are used for large datasets that do not fit in memory as
        they allow for lazy loading of structure_data and reference_molecule_data
        entries.

        Args:
            lmdb_directory (Path):
                The directory containing the LMDB database.
        str_encoding (Literal["utf-8", "pkl"]):
            The encoding to use for the cache keys and _type and name values.
        structure_data_encoding (Literal["utf-8", "pkl"]):
            The encoding to use for the structure_data values. The 'pkl' encoding saves
            the dataclasses directly, whereas 'utf-8' encoding requires re-creating the
            dataclasses.
        reference_molecule_data_encoding (Literal["utf-8", "pkl"]):
            The encoding to use for the reference_molecule_data values.The 'pkl'
            encoding saves the dataclasses directly, whereas 'utf-8' encoding requires
            re-creating the dataclasses.

        Returns:
            DatasetCache:
                The constructed datacache.
        """

        lmdb_env = lmdb.open(
            str(lmdb_directory), readonly=True, lock=False, subdir=True
        )

        with lmdb_env.begin() as transaction:
            _ = cls._parse_type_lmdb(transaction, str_encoding)
            name = cls._parse_name_lmdb(transaction, str_encoding)
            structure_data = cls._parse_structure_data_lmdb(
                lmdb_env, str_encoding, structure_data_encoding
            )
            reference_molecule_data = cls._parse_ref_mol_data_lmdb(
                lmdb_env, str_encoding, reference_molecule_data_encoding
            )

            return cls(
                name=name,
                structure_data=structure_data,
                reference_molecule_data=reference_molecule_data,
            )

    def _parse_type_lmdb(
        transaction: lmdb.Transaction, str_encoding: Literal["utf-8", "pkl"]
    ) -> str:
        _type_bytes = transaction.get(b"_type")
        if not _type_bytes:
            raise ValueError("No _type key found in the LMDB.")
        else:
            _type = json.loads(_type_bytes.decode(str_encoding))

        return _type

    def _parse_name_lmdb(
        transaction: lmdb.Transaction, str_encoding: Literal["utf-8", "pkl"]
    ) -> str:
        name_bytes = transaction.get(b"name")
        if not name_bytes:
            raise ValueError("No name key found in the LMDB.")
        else:
            name = json.loads(name_bytes.decode(str_encoding))

        return name

    def _parse_structure_data_lmdb(
        lmdb_env: lmdb.Environment,
        str_encoding: Literal["utf-8", "pkl"],
        structure_data_encoding: Literal["utf-8", "pkl"],
    ) -> LMDBDict:
        from openfold3.core.data.primitives.caches.lmdb import (
            LMDBDict,
        )

        return LMDBDict(
            lmdb_env=lmdb_env,
            prefix="structure_data",
            key_encoding=str_encoding,
            value_encoding=structure_data_encoding,
        )

    def _parse_ref_mol_data_lmdb(
        lmdb_env: lmdb.Environment,
        str_encoding: Literal["utf-8", "pkl"],
        reference_molecule_data_encoding: Literal["utf-8", "pkl"],
    ) -> LMDBDict:
        from openfold3.core.data.primitives.caches.lmdb import (
            LMDBDict,
        )

        return LMDBDict(
            lmdb_env=lmdb_env,
            prefix="reference_molecule_data",
            key_encoding=str_encoding,
            value_encoding=reference_molecule_data_encoding,
        )

    def to_lmdb(
        self,
        lmdb_directory: Path,
        map_size: int,
        mode: Literal["single-read", "iterative"] = "single-read",
        str_encoding: Literal["utf-8", "pkl"] = "utf-8",
        structure_data_encoding: Literal["utf-8", "pkl"] = "pkl",
        reference_molecule_data_encoding: Literal["utf-8", "pkl"] = "pkl",
    ) -> None:
        """Creates an LMDB database from the dataset cache.

        Use the convert_datacache_to_lmdb function directly if you want to convert a
        datacache JSON file to an LMDB database that does not fit into memory.

        Args:
            json_file (Path | DatasetCache):
                The datacache JSON file to convert or an existing DatasetCache object.
            lmdb_directory (Path):
                The LMDB dir to which the data and lock files are to be written.
            map_size (int):
                Size of the json file in bytes, for example  2 * (1024**3) for a 2GB
                file. Provide a value slightly larger than the actual size of the json
                file.
            mode (Literal["single-read", "iterative"]):
                The mode to use to parse the json file. Can be one of 'single-read' or
                'iterative'. Use 'single-read' for small files and 'iterative' for large
                files.
            str_encoding (Literal["utf-8", "pkl"]):
                The encoding to use for the cache keys and _type and name values.
            structure_data_encoding (Literal["utf-8", "pkl"]):
                The encoding to use for the structure_data values. The 'pkl' encoding
                saves the dataclasses directly, whereas 'utf-8' encoding requires
                re-creating the dataclasses.
            reference_molecule_data_encoding (Literal["utf-8", "pkl"]):
                The encoding to use for the reference_molecule_data values.The 'pkl'
                encoding saves the dataclasses directly, whereas 'utf-8' encoding
                requires re-creating the dataclasses.
        """
        from openfold3.core.data.primitives.caches.lmdb import (
            convert_datacache_to_lmdb,
        )

        convert_datacache_to_lmdb(
            dataset_cache_file_or_obj=self,
            lmdb_directory=lmdb_directory,
            map_size=map_size,
            mode=mode,
            str_encoding=str_encoding,
            structure_data_encoding=structure_data_encoding,
            reference_molecule_data_encoding=reference_molecule_data_encoding,
        )

    def __init_subclass__(cls, **kwargs):
        """Ensure subclasses are properly initialized."""
        super().__init_subclass__(**kwargs)
        cls._format_validated = False
        cls._registered = False

    def __post_init__(self):
        # Technically these two checks are redundant
        if not self.__class__._format_validated:
            raise ValueError(
                "Datacache format was not validated. Decorate your class with "
                "@register_datacache and provide a list of required format attributes."
            )
        if not self.__class__._registered:
            raise ValueError(
                "Datacache was not registered. Decorate your class with "
                "@register_datacache."
            )


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
if TYPE_CHECKING:
    DictOrLMDBDict: TypeAlias = dict[K, V] | LMDBDict[K, V]
    DatasetReferenceMoleculeCache: TypeAlias = DictOrLMDBDict[
        str, DatasetReferenceMoleculeData
    ]


# ==============================================================================
# SPECIALIZED DATASETS
# ==============================================================================
# This is where all specialized training dataset caches and validation set caches should
# be implemented.


# --- Chain data dataclasses ---
# TEMPLATE for metadata for PDB datasets
@dataclass
class PDBChainData(DatasetChainData):
    """Chain-wise data for PDB datasets."""

    # TODO: These are not mandatory and currently kept for debugging purposes, but may
    # be removed later
    label_asym_id: str
    auth_asym_id: str
    entity_id: int

    molecule_type: str  # TODO: This should be parsed as a MoleculeType
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
class ValidationDatasetChainData(ClusteredDatasetChainData):
    """Chain-wise data with additional validation fields.

    Additional attributes:
        low_homology (bool):
            Whether the chain has low-homology with the training data (see AF3 SI 5.8).
        use_metrics (bool):
            Whether validation metrics should be calculated for this chain (see AF3 SI
            5.8).
        ranking_model_fit (float | None):
            The ranking model fit of this chain. Only applies to ligand chains.
    """

    # Adds the following fields:
    low_homology: bool
    use_metrics: bool
    ranking_model_fit: float | None


@dataclass
class ProteinMonomerChainData:
    """Chain-wise data for protein monomers."""

    alignment_representative_id: str | None
    template_ids: list[str] | None
    index: int


# --- Interface data dataclasses ---
@dataclass
class ClusteredDatasetInterfaceData:
    """Interface-wise data with cluster information."""

    cluster_id: str
    cluster_size: int


@dataclass
class ValidationDatasetInterfaceData(ClusteredDatasetInterfaceData):
    """Interface-wise data with additional validation fields.

    Additional attributes:
        low_homology (bool):
            Whether the interface has low-homology with the training data (see AF3 SI
            5.8).
        metric_eligible (bool):
            Whether the interface is eligible for validation metrics (see ligand quality
            and residue critera in AF3 SI 5.8). (Only used as an intermediate field for
            the final use_metrics)
        use_metrics (bool):
            Whether validation metrics should be calculated for this interface (see AF3
            SI 5.8).
    """

    # Adds the following fields:
    low_homology: bool
    metric_eligible: bool
    use_metrics: bool


# --- Structure data dataclasses ---
@dataclass
class ClusteredDatasetStructureData:
    """Structure data with clusters and added metadata."""

    release_date: datetime.date
    resolution: float
    chains: dict[str, ClusteredDatasetChainData]
    interfaces: dict[str, ClusteredDatasetInterfaceData]


@dataclass
class ValidationDatasetStructureData:
    """Structure data wrapper for validation set."""

    release_date: datetime.date
    resolution: float
    token_count: int
    chains: dict[str, ValidationDatasetChainData]
    interfaces: dict[str, ClusteredDatasetInterfaceData]


@dataclass
class ProteinMonomerStructureData:
    """Structure data for protein monomers."""

    chains: dict[str, ProteinMonomerChainData]


if TYPE_CHECKING:
    ClusteredDatasetStructureDataCache: TypeAlias = DictOrLMDBDict[
        str, ClusteredDatasetStructureData
    ]
    ValClusteredDatasetStructureDataCache: TypeAlias = DictOrLMDBDict[
        str, ValidationDatasetStructureData
    ]
    ProteinMonomerStructureDataCache: TypeAlias = DictOrLMDBDict[
        str, ProteinMonomerStructureData
    ]


# --- Reference molecule dataclasses ---
@dataclass
class ValidationDatasetReferenceMoleculeData(DatasetReferenceMoleculeData):
    """Reference molecule data for validation set."""

    # Adds the following field:
    residue_count: int


# --- Dataset caches ---
# TEMPLATE for dataset caches with chain-wise, interface-wise and reference molecule
# data.
@register_datacache
@dataclass
class ClusteredDatasetCache(DatasetCache):
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


# PDB VALIDATION DATASET FORMAT
# TODO: Some of these fields are only required for filtering and should not be in the
# final cache
@register_datacache
@dataclass
class ValidationDatasetCache(DatasetCache):
    """Full data cache for the validation dataset."""

    name: str
    structure_data: ClusteredDatasetStructureDataCache
    reference_molecule_data: DatasetReferenceMoleculeCache

    # Defines the constructor formats for the inherited from_json method
    _chain_data_format = ValidationDatasetChainData
    _interface_data_format = ValidationDatasetInterfaceData
    _ref_mol_data_format = ValidationDatasetReferenceMoleculeData
    _structure_data_format = ValidationDatasetStructureData


@register_datacache
@dataclass
class ProteinMonomerDatasetCache(DatasetCache):
    """Full data cache for protein monomer data from AF2."""

    name: str
    structure_data: ProteinMonomerStructureDataCache
    reference_molecule_data: DatasetReferenceMoleculeCache
    _chain_data_format = ProteinMonomerChainData
    _ref_mol_data_format = DatasetReferenceMoleculeData
    _structure_data_format = ProteinMonomerStructureData

    @classmethod
    def _parse_structure_data_json(cls, data: dict) -> dict:
        # Format structure data
        structure_data = {}
        for pdb_id, per_structure_data in data["structure_data"].items():
            chain_data = per_structure_data.pop("chains")

            # Extract all chain data into respective chain data format
            chains = {
                chain_id: cls._chain_data_format(**chain_data[chain_id])
                for chain_id in chain_data
            }

            # Combine chain and interface data with remaining structure data
            structure_data[pdb_id] = cls._structure_data_format(
                chains=chains, **per_structure_data
            )
        return structure_data


# Grouped type-aliases for more convenient type-hinting of general-purpose functions
if TYPE_CHECKING:
    ChainData: TypeAlias = (
        PreprocessingChainData | PDBChainData | ProteinMonomerChainData
    )
    StructureDataCache: TypeAlias = (
        PreprocessingStructureDataCache
        | DisorderedPreprocessingStructureDataCache
        | ClusteredDatasetStructureDataCache
        | ValClusteredDatasetStructureDataCache
        | ProteinMonomerStructureDataCache
    )
    ReferenceMoleculeCache: TypeAlias = (
        PreprocessingReferenceMoleculeCache | DatasetReferenceMoleculeCache
    )
    DataCacheType: TypeAlias = PreprocessingDataCache | DatasetCache
