# TODO add license
import copy
import functools
import logging
from typing import Optional

from ml_collections import ConfigDict

from openfold3.core.config import config_utils
from openfold3.core.runners.registry_base import register_model_base

# Record of ModelEntries
MODEL_REGISTRY = {}
register_model = functools.partial(register_model_base, model_registry=MODEL_REGISTRY)


def make_config_with_preset(model_name: str, preset: Optional[str] = None):
    """Retrieves config matching preset for one of the models."""
    base_config = copy.deepcopy(MODEL_REGISTRY[model_name].base_config)
    if preset is None:
        logging.info(f"Using default training configs for {model_name}")
        return base_config
    else:
        logging.info(f"Loading {preset} configs for {model_name}")
        reference_configs = config_utils.load_yaml(
            MODEL_REGISTRY[model_name].reference_config_path
        )
        preset_config = reference_configs[preset]
        return config_utils.update_config_dict(base_config, preset_config)


def make_model_config(model_name: str, runner_yaml_path: str):
    """Construct model config.

    First loads a config based on the model_preset argument if available.
    Then updates the config the contents of the the runner_yaml itself.
    """
    runner_yaml_dict = config_utils.load_yaml(runner_yaml_path)
    preset = runner_yaml_dict.get("model_preset", None)
    config = make_config_with_preset(model_name, preset)
    if "model_update" in runner_yaml_dict:
        config = config_utils.update_config_dict(
            config, runner_yaml_dict["model_update"]
        )
    return config


def get_lightning_module(config: ConfigDict, model_name: Optional[str] = None):
    if not model_name:
        try:
            model_name = config.model_name
        except KeyError as exc:
            raise ValueError(
                "Model_name must be specified either in config or" " as an argument."
            ) from exc
    return MODEL_REGISTRY[model_name].model_runner(config)
