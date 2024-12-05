"""IO functions to read and write metadata and dataset caches."""

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from openfold3.core.data.primitives.caches.format import (
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

    Args:
        datacache:
            DataCache dataclass to be written to a JSON file.
        output_path:
            Path to the output JSON file.

    Returns:
        Full path to the output JSON file.
    """
    datacache_dict = asdict(datacache)

    datacache_dict = format_nested_dict_for_json(datacache_dict)

    with open(output_path, "w") as f:
        json.dump(datacache_dict, f, indent=4)
