"""IO functions to read and write metadata and dataset caches."""

import json
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import lmdb

from openfold3.core.data.primitives.caches.format import (
    DATASET_CACHE_CLASS_REGISTRY,
)
from openfold3.core.data.resources.residues import MoleculeType

if TYPE_CHECKING:
    from openfold3.core.data.primitives.caches.format import DataCacheType


def encode_datacache_types(obj: object) -> object:
    """Encoder for any non-standard types encountered in DataCache objects."""
    if isinstance(obj, date):
        return obj.isoformat()
    elif isinstance(obj, MoleculeType):
        return obj.name
    else:
        return obj


def format_nested_dict_for_json(data: dict) -> dict:
    """Encoder for any non-standard types encountered in DataCaches.

    For this function to work the datacache must be converted to a dict first. This is
    meant to be used before writing the datacache data to a JSON output.

    Args:
        data:
            The datacache data as a dictionary.

    Returns:
        The data dictionary with custom type encoding.
    """
    for item, value in data.items():
        if isinstance(value, dict):
            format_nested_dict_for_json(value)
        else:
            converted_obj = encode_datacache_types(value)
            data[item] = converted_obj

    return data


def convert_dataclass_to_dict(dataclass: Any) -> dict:
    """Converts a dataclass into a dictionary.

    Note: this is intended to be a general function that can be called at any level
    of the DataCache dataclass hierarchy.

    Args:
        dataclass (Any):
            The dataclass to convert. If the dataclass is a DataCache, this function
            adds the "_type" attribute to the dictionary.

    Returns:
        dict:
            The datacache as a dictionary.
    """
    # TODO: reorganize cache format modules so that we can avoid circular import hacks
    from openfold3.core.data.primitives.caches.format import (
        DatasetCache,
        PreprocessingDataCache,
    )

    DataCacheType: TypeAlias = PreprocessingDataCache | DatasetCache

    datacache_dict = asdict(dataclass)

    # Remove private fields
    datacache_dict = {k: v for k, v in datacache_dict.items() if not k.startswith("_")}

    if isinstance(dataclass, DataCacheType):
        # Add type (which is not a field but an attribute) as the very first(!) key of
        # the dict
        datacache_dict = {"_type": dataclass._type, **datacache_dict}

    datacache_dict = format_nested_dict_for_json(datacache_dict)

    return datacache_dict


def write_datacache_to_json(datacache: "DataCacheType", output_path: Path) -> Path:
    """Writes a DataCache dataclass to a JSON file.

    This ignores any private fields (those starting with an underscore) in the
    dataclass, and adds the specialized "_type" attribute which is necessary for
    reading the datacache back in.

    Args:
        datacache:
            DataCache dataclass to be written to a JSON file.
        output_path:
            Path to the output JSON file.

    Returns:
        Full path to the output JSON file.
    """
    datacache_dict = convert_dataclass_to_dict(datacache)

    with open(output_path, "w") as f:
        json.dump(datacache_dict, f, indent=4)


def _read_datacache_file(datacache_path: Path) -> "DataCacheType":
    """Loads a datacache from a json file"""
    with open(datacache_path) as f:
        next(f)
        second_line = next(f)

        # formatted like "name": "value"
        match = re.search(r'"_type":\s*"([^"]+)"', second_line)

        if match:
            dataset_cache_type = match.group(1)
        else:
            raise ValueError("Could not determine the type of the dataset cache.")

        try:
            # Infer which class to build
            dataset_cache_class = DATASET_CACHE_CLASS_REGISTRY.get(dataset_cache_type)
        except KeyError as exc:
            raise ValueError(
                f"Unknown dataset cache type: {dataset_cache_type}"
            ) from exc

    return dataset_cache_class.from_json(datacache_path)


def read_datacache(
    datacache_path: Path,
    str_encoding: Literal["utf-8", "pkl"] = "utf-8",
    structure_data_encoding: Literal["utf-8", "pkl"] = "pkl",
    reference_molecule_data_encoding: Literal["utf-8", "pkl"] = "pkl",
) -> "DataCacheType":
    """Reads a DataCache dataclass from a JSON file.

    Args:
        datacache_path:
            Path to the JSON file or LMDB directory containing the DataCache data.
        str_encoding (Literal["utf-8", "pkl"]):
            The encoding to use for the cache keys and _type and name values. Only used
            for LMDB reading.
        structure_data_encoding (Literal["utf-8", "pkl"]):
            The encoding to use for the structure_data values. The 'pkl' encoding saves
            the dataclasses directly, whereas 'utf-8' encoding requires re-creating the
            dataclasses. Only used for LMDB reading.
        reference_molecule_data_encoding (Literal["utf-8", "pkl"]):
            The encoding to use for the reference_molecule_data values.The 'pkl'
            encoding saves the dataclasses directly, whereas 'utf-8' encoding requires
            re-creating the dataclasses. Only used for LMDB reading.

    Returns:
        A fully instantiated DataCache of the appropriate type.
    """

    # Determine the type of dataset cache first without reading the whole file
    if datacache_path.is_file():
        return _read_datacache_file(datacache_path)

    elif datacache_path.is_dir():
        # Assumed to be an lmdb dir
        lmdb_env = lmdb.open(
            str(datacache_path), readonly=True, lock=False, subdir=True
        )
        type_key = "_type".encode(str_encoding)
        with lmdb_env.begin() as txn:
            dataset_cache_type = json.loads(txn.get(type_key).decode(str_encoding))

        if not dataset_cache_type:
            raise ValueError("No type found for this directory.")

        try:
            # Infer which class to build
            dataset_cache_class = DATASET_CACHE_CLASS_REGISTRY.get(dataset_cache_type)
        except KeyError as exc:
            raise ValueError(
                f"Unknown dataset cache type: {dataset_cache_type}"
            ) from exc

        dataset_cache = dataset_cache_class.from_lmdb(
            datacache_path,
            str_encoding=str_encoding,
            structure_data_encoding=structure_data_encoding,
            reference_molecule_data_encoding=reference_molecule_data_encoding,
        )
        return dataset_cache
