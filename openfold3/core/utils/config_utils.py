"""
Helper functions for converting between yaml, dicts, and config dicts. 
"""

import yaml
from pathlib import Path
from typing import Any
import ml_collections as mlc

def load_yaml(Path)-> dict[str, Any]:
    with open(Path) as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict 


def update_config_dict(config_dict: mlc.ConfigDict, update_dict: dict):
    base = config_dict.copy_and_resolve_references()
    base.update(update_dict)
    return base
