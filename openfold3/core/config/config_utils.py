# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper functions for converting between yaml, dicts, and config dicts.
"""

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import (
    BeforeValidator,
    DirectoryPath,
    FilePath,
)

from openfold3.core.data.resources.residues import MoleculeType


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Loads a yaml file as a dictionary."""
    if not isinstance(path, Path):
        path = Path(path)
    with path.open() as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def load_json(path: Path | str) -> dict[str, Any]:
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


def _cast_keys_to_int(mapping: dict) -> dict[int, Any]:
    """Casts all keys in the dictionary to integers."""
    return {int(k): v for k, v in mapping.items()}


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
    elif isinstance(value, int):
        try:
            return MoleculeType(value)
        except ValueError:
            logging.warning(
                f"Found invalid {value=} for molecule type, skipping this example."
            )
            return None
    elif isinstance(value, list):
        return [_convert_molecule_type(v) for v in value]


def is_path_none(value: str | Path | None) -> Path | None:
    if isinstance(value, Path):
        return value
    elif value is None or value.lower() in ["none", "null"]:
        return None
    else:
        return Path(value)


FilePathOrNone = Annotated[FilePath | None, BeforeValidator(is_path_none)]
DirectoryPathOrNone = Annotated[DirectoryPath | None, BeforeValidator(is_path_none)]
