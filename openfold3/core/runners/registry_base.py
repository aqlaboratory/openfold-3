import dataclasses
import logging
from pathlib import Path
from typing import Any, Optional

from ml_collections import ConfigDict

from openfold3.core.config import config_utils
from openfold3.core.runners.model_runner import ModelRunner


@dataclasses.dataclass
class ModelEntry:
    name: str
    model_runner: ModelRunner
    base_config: ConfigDict
    reference_config_path: Optional[Path]
    # List of available presets available for given model_runner
    # Populated upon ModelEntry creation
    presets: Optional[list[str]]

    def get_config_with_preset(self, preset: Optional[str] = None):
        if preset is None:
            logging.info(f"Using default training configs for {model_name}")
            return self.base_config

        if preset not in self.presets:
            raise KeyError(
                f"{preset} preset is not supported for {self.name}"
                f"Allowed presets are {self.presets}"
            )
        reference_configs = config_utils.load_yaml(self.reference_config_path)
        preset_config = reference_configs[preset]
        return self.base_config.update(preset_config)


def register_model_base(
    name: str,
    base_config: ConfigDict,
    reference_config_path: Optional[Path] = None,
    model_registry: Optional[dict[str, Any]] = None,
):
    """Register ModelEntry container with ModelRunner and configuration settings.

    Args:
        name: Name to use for model entry
        base_config: Base configuration class for model entry
        reference_config_path: Path to yaml with configuration presets.
        model_register: Map of ModelRunner and configs by name

    Returns:
        Type[OpenFoldModelWrapper]: The registered class.
    """

    def _decorator(runner_cls):
        if name in model_registry:
            raise ValueError("{name} has been previously registered in registry")
        if reference_config_path:
            reference_dict = config_utils.load_yaml(reference_config_path)
            presets = list(reference_dict.keys())
        else:
            presets = None

        # Automatically add/update model name to base config
        # Makes it easy to refer to this ModelEntry later from a config
        base_config.model_name = name
        model_registry[name] = ModelEntry(
            name, runner_cls, base_config, reference_config_path, presets
        )
        runner_cls._registered = True
        return runner_cls

    return _decorator
