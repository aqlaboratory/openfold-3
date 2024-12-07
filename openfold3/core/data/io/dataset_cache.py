"""IO functions to read and write metadata and dataset caches."""

import json
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path

from openfold3.core.data.primitives.caches.format import (
    DATASET_CACHE_CLASS_REGISTRY,
    DataCache,
)
from openfold3.core.data.resources.residues import MoleculeType


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


def write_datacache_to_json(datacache: DataCache, output_path: Path) -> Path:
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
    datacache_dict = asdict(datacache)

    # Remove private fields
    datacache_dict = {k: v for k, v in datacache_dict.items() if not k.startswith("_")}

    # Add type (which is not a field but an attribute) as the very first(!) key of the
    # dict
    datacache_dict = {"_type": datacache._type, **datacache_dict}

    datacache_dict = format_nested_dict_for_json(datacache_dict)

    with open(output_path, "w") as f:
        json.dump(datacache_dict, f, indent=4)


def read_datacache(datacache_path: Path) -> DataCache:
    """Reads a DataCache dataclass from a JSON file.

    Args:
        datacache_path:
            Path to the JSON file containing the DataCache data.

    Returns:
        A fully instantiated DataCache of the appropriate type.
    """

    # Determine the type of dataset cache first without reading the whole file
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
        raise ValueError(f"Unknown dataset cache type: {dataset_cache_type}") from exc

    # Read the JSON file and return
    return dataset_cache_class.from_json(datacache_path)
