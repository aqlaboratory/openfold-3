"""This module retrieves the default configuration setting for models and customizes the config.

General layout of configurations should be:
- base_config (initial training setup)
- model_preset (specific model settings)
- customized configurations (e.g. experiment / runner specific settings)
"""

from pathlib import Path
from typing import NamedTuple
from ml_collections import ConfigDict
from openfold3.core.config import config_utils
from openfold3.model_implementations.af2_monomer.base_config import (
    config as af2_monomer_config,
)
from openfold3.model_implementations.af2_multimer.config import (
    config as af2_multimer_config,
)

MODEL_IMPLEMENTATION_PATH = Path(__file__).parent.resolve()

ConfigEntry = NamedTuple("ConfigEntry", [("config", ConfigDict), ("path", Path)])

CONFIG_REGISTRY = {
    "af2_monomer": ConfigEntry(
        af2_monomer_config, MODEL_IMPLEMENTATION_PATH / "af2_monomer"
    ),
    "af2_multimer": ConfigEntry(
        af2_multimer_config, MODEL_IMPLEMENTATION_PATH / "af2_multimer"
    ),
}


def model_config(model_name: str, model_preset: str):
    """Loads the model with the selected preset"""
    model = CONFIG_REGISTRY[model_name]
    if model_preset == "initial_training":
        return model.config
    else:
        reference_configs = config_utils.load_yaml(model.path / "reference_config.yml")
        model_preset_config = reference_configs[model_preset]
        return config_utils.update_config_dict(model.config, model_preset_config)
