import dataclasses
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
    # List of available presets for the model_factory
    presets: Optional[list[str]]

    def __call__(self, config):
        """Creates ModelRunner with provided config settings"""
        return self.model_runner(config)


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

        model_registry[name] = ModelEntry(
            name, runner_cls, base_config, reference_config_path, presets
        )
        return runner_cls

    # cls._registered = True  # QUESTION do we want to enforce class registration with
    # this decorator? Part A.
    return _decorator
