# TODO add license
import dataclasses
import logging
from ml_collections import ConfigDict
from pathlib import Path
from typing import Optional

from openfold3.core.config import config_utils
from openfold3.core.runners.model_runner import ModelRunner

# Record of ModelEntries
MODEL_REGISTRY = {}


@dataclasses.dataclass
class ModelEntry:
    name: str
    model_factory: ModelRunner
    base_config: ConfigDict
    reference_config_path: Optional[Path]
    # List of available presets, read from reference_config_path
    presets: Optional[list[str]]


def register_model(name, base_config, reference_config_path=None):
    """Register a specific OpenFoldModelWrapper class in the MODEL_REGISTRY.

    Args:
        cls (Type[OpenFoldModelWrapper]): The class to register.

    Returns:
        Type[OpenFoldModelWrapper]: The registered class.
    """

    def _decorator(runner_cls):
        if name in MODEL_REGISTRY:
            raise ValueError(
                "{name} has been previously registered in model_implementations.registry"
            )
        if reference_config_path:
            reference_dict = config_utils.load_yaml(reference_config_path)
            presets = list(reference_dict.keys())
        else:
            presets = None

        MODEL_REGISTRY[name] = ModelEntry(
            name, runner_cls, base_config, reference_config_path, presets
        )
        return runner_cls

    # cls._registered = True  # QUESTION do we want to enforce class registration with
    # this decorator? Part A.
    return _decorator


def make_config_with_preset(model_name: str, preset: Optional[str] = None):
    base_config = MODEL_REGISTRY[model_name].base_config
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
    preset = (
        runner_yaml_dict["model_preset"] if "model_preset" in runner_yaml_dict else None
    )
    preset_config = make_config_with_preset(model_name, preset)
    if "model_update" in runner_yaml_dict:
        updated_config = config_utils.update_config_dict(
            preset_config, runner_yaml_dict
        )
    return updated_config
