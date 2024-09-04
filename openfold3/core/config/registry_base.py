import dataclasses
import logging
from pathlib import Path
from typing import Optional

from ml_collections import ConfigDict

from openfold3.core.config import config_utils, dataset_config_builder
from openfold3.core.runners.model_runner import ModelRunner


@dataclasses.dataclass
class ProjectEntry:
    name: str
    model_runner: ModelRunner
    dataset_config_builder: dataset_config_builder.DefaultDatasetConfigBuilder
    base_config: ConfigDict
    reference_config_path: Optional[Path]
    # List of available presets available for given model_runner
    # Populated upon ProjectEntry creation
    presets: Optional[list[str]]

    def update_config_with_preset(self, config: ConfigDict, preset: str) -> None:
        """Updates a given configdict with a preset that is part of the ProjectEntry"""
        if preset not in self.presets:
            raise KeyError(
                f"{preset} preset is not supported for {self.name}"
                f"Allowed presets are {self.presets}"
            )
        reference_configs = ConfigDict(
            config_utils.load_yaml(self.reference_config_path)
        )
        preset_model_config = reference_configs[preset]
        config.update(preset_model_config)
        return

    def get_config_with_preset(self, preset: str) -> ConfigDict:
        """Retrieves a config with specified preset applied"""
        config = self.base_config
        if preset is None or preset == "initial_training":
            logging.info(f"Using default training configs for {self.name}")
        else:
            self.update_config_with_preset(config, preset)
        return config


def _register_project_base(
    cls: ProjectEntry, project_registry: dict[str, ProjectEntry]
):
    name = cls.name
    if name in project_registry:
        raise ValueError("{name} has been previously registered in registry")
    project_registry[name] = cls
    cls._registered = True
    return cls


def make_project_entry(
    name: str,
    model_runner: ModelRunner,
    dataset_config_builder: dataset_config_builder.DefaultDatasetConfigBuilder,
    base_project_config: ConfigDict,
    project_registry: dict[str, ProjectEntry],
    reference_config_path: Optional[Path] = None,
):
    """Basic ProjectEntry creation using default ProjectEntry class

    Args:
        name: Name to use for model entry
        model_runner: Lightning Module wrapper to use for running the model
        dataset_config_builder: Builder class for creating dataset configs
        base_project_config: Base configuration class for model entry
        reference_config_path: Path to yaml with configuration presets.
        project_registry: Map of ProjectEntries and configs by name

    Returns:
        Type[ProjectEntry]: The registered class.
    """

    if reference_config_path:
        reference_dict = config_utils.load_yaml(reference_config_path)
        presets = list(reference_dict.keys())
    else:
        presets = None

    # Automatically add/update model name to base config
    # Makes it easy to refer to this ModelEntry later from a config
    entry = ProjectEntry(
        name,
        model_runner,
        dataset_config_builder,
        base_project_config,
        reference_config_path,
        presets,
    )
    _register_project_base(entry, project_registry)
    return
