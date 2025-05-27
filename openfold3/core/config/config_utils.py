"""
Helper functions for converting between yaml, dicts, and config dicts.
"""

import json
import logging
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import yaml
from pydantic import (
    BeforeValidator,
    DirectoryPath,
    FilePath,
)

from openfold3.core.data.resources.residues import MoleculeType


def load_yaml(path: Union[Path, str]) -> dict[str, Any]:
    """Loads a yaml file as a dictionary."""
    if not isinstance(path, Path):
        path = Path(path)
    with path.open() as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def load_json(path: Union[Path, str]) -> dict[str, Any]:
    """Loads a json file as a dictionary."""
    if not isinstance(path, Path):
        path = Path(path)
    with path.open() as f:
        json_dict = json.load(f)
    return json_dict


def _ensure_list(value: Any) -> Any:
    if not isinstance(value, list):
        logging.info("Single value: {value} will be converted to a list")
        return [value]
    else:
        return value


def _convert_molecule_type(value: Any) -> Any:
    if isinstance(value, MoleculeType):
        return value
    elif isinstance(value, str):
        try:
            return MoleculeType[value.upper()]
        except KeyError:
            logging.warning(
                f"Found invalid {value=} for molecule type, skipping this example."
            )
            return None


def is_path_none(value: Optional[Union[str, Path]]) -> Optional[Path]:
    if isinstance(value, Path):
        return value
    elif value is None or value.lower() in ["none", "null"]:
        return None
    else:
        return Path(value)


FilePathOrNone = Annotated[Optional[FilePath], BeforeValidator(is_path_none)]
DirectoryPathOrNone = Annotated[Optional[DirectoryPath], BeforeValidator(is_path_none)]
