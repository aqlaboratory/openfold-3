# TODO add license
import ml_collections as mlc

from openfold3.core.config import registry_base
from openfold3.core.config.dataset_config_builder import DefaultDatasetConfigBuilder
from openfold3.core.data.framework import data_module

# Record of ModelEntries
PROJECT_REGISTRY = {}


def register_project(
    name: str,
    dataset_config_builder: DefaultDatasetConfigBuilder,
    base_config: mlc.ConfigDict,
    reference_config_path: str,
):
    """Creates decorator function for registering projects."""

    def _decorator(runner_cls):
        registry_base.make_project_entry(
            name=name,
            model_runner=runner_cls,
            dataset_config_builder=dataset_config_builder,
            base_project_config=base_config,
            project_registry=PROJECT_REGISTRY,
            reference_config_path=reference_config_path,
        )
        return runner_cls

    return _decorator


def get_project_entry(project_name: str):
    return PROJECT_REGISTRY[project_name]


def make_config_with_presets(
    project_entry: registry_base.ProjectEntry, presets: list[str]
):
    """Initializes project config using provided presets.

    Args:
        project_entry: ProjectEntry class for given project
        presets: List of preset settings available for the given project
    Returns:
        A new projct config where preset updates are applied in the order they are
        provided in `presets`.
    """
    initial_preset = presets[0]
    project_config = project_entry.get_config_with_preset(initial_preset)

    if len(presets) > 1:
        for preset in presets[1:]:
            project_entry.update_config_with_preset(project_config, preset)

    return project_config


def make_dataset_module_config(
    runner_args: mlc.ConfigDict, dataset_config_builder: DefaultDatasetConfigBuilder
):
    """Constructs dataset config module for all datasets in runner configuration."""
    dataset_configs = []
    input_dataset_configs = runner_args.dataset_configs

    # loop over modes
    for dataset_type, _dataset_configs in input_dataset_configs.items():
        # loop over datasets in modes
        for name, dataset_specs in _dataset_configs.items():
            dataset_paths = runner_args.dataset_paths.get(name)
            config = dataset_config_builder.get_custom_config(
                name, dataset_type, dataset_specs, dataset_paths
            )
            dataset_configs.append(config)

    datamodule_config = data_module.DataModuleConfig(
        batch_size=runner_args.batch_size,
        num_workers=runner_args.get("num_workers", 2),
        data_seed=runner_args.get("data_seed", 17),
        epoch_len=runner_args.get("epoch_len", 1),
        num_epochs=runner_args.pl_trainer.get("max_epochs"),
        datasets=dataset_configs,
    )
    return datamodule_config
