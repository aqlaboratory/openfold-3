"""
Helper functions for converting between yaml, dicts, and config dicts.
"""

from pathlib import Path
from typing import Any

import ml_collections as mlc
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def update_config_dict(config_dict: mlc.ConfigDict, update_dict: dict):
    base = config_dict.copy_and_resolve_references()
    base.update(update_dict)
    return base
