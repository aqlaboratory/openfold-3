"""
Helper functions for converting between yaml, dicts, and config dicts.
"""

import json
from pathlib import Path
from typing import Any, Union

import yaml


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
