# TODO add license
import functools
from typing import Optional

from ml_collections import ConfigDict, FrozenConfigDict

from openfold3.core.config import config_utils
from openfold3.core.runners.registry_base import register_model_base

# Record of ModelEntries
MODEL_REGISTRY = {}
register_model = functools.partial(register_model_base, model_registry=MODEL_REGISTRY)

# Remove after setup is complete
_AF3_base_data_config = ConfigDict(
    {
        "templates": {
            "use_templates": True,
            "max_template_hits": 4,
            "max_templates": 4,
        },
        "msa": {
            "uniprot_msa_depth": 8_000,
            "main_msa_depth": 16_000,
        },
        "loss_weight_mode": "default",
        "cropping": {
            "crop_size": 768,
        },
        "min_resolution": 0.1,
        "max_resolution": 4.0,
    }
)


def make_config_with_preset(model_name: str, preset: Optional[str] = None):
    """Retrieves config matching preset for one of the models."""
    model_entry = MODEL_REGISTRY[model_name]
    return model_entry.get_config_with_preset(preset)


def make_model_config(model_name: str, model_update_yaml_path: str):
    """Construct model config.

    First loads a config based on the model_preset argument if available.
    Then updates the config the contents of the the runner_yaml itself.

    Args:
        model_name: Expected to be a key in MODEL_REGISTRY
        model_update_yaml_path:
            Path to a yaml file with model updates.
            Expected to have a `model_update` header
    Returns:
        Config dict with model config and updates from the yaml file
    """
    runner_yaml_dict = config_utils.load_yaml(model_update_yaml_path)
    preset = runner_yaml_dict.get("model_preset", None)
    config = make_config_with_preset(model_name, preset)
    if "model_update" in runner_yaml_dict:
        config = config_utils.update_config_dict(
            config, runner_yaml_dict["model_update"]
        )
    return config


def get_dataset_loss_config(
    loss_modes_config: ConfigDict, loss_mode: str
) -> FrozenConfigDict:
    if loss_mode == "default":
        return FrozenConfigDict(loss_modes_config.default)

    if loss_mode not in loss_modes_config.custom:
        raise KeyError(
            f"{loss_mode} is not supported,"
            " allowed loss modes are: default,"
            f" {list(loss_modes_config.custom.keys())}"
        )

    loss_config = loss_modes_config.default.copy_and_resolve_references()
    loss_config.update(loss_modes_config.custom[loss_mode])
    loss_config = FrozenConfigDict(loss_config)
    return loss_config


def make_dataset_configs(runner_args: ConfigDict) -> list[ConfigDict]:
    # base_data_config
    # base model config - get losses
    # Runner yaml:
    #    configs for each dataset
    #    dataset_paths
    model_name = runner_args.model_name
    loss_config = MODEL_REGISTRY[model_name].base_config.loss.loss_weight_modes

    # Finalize implementation after deciding where this goes
    # base_data_config = MODEL_REGISTRY[model_name].data_config
    base_data_config = _AF3_base_data_config

    input_dataset_configs = runner_args.dataset_configs
    output_dataset_configs = []
    # loop through datasets in a given mode
    for mode, _dataset_configs in input_dataset_configs.items():
        # loop through datasets in a given mode
        for name, dataset_specs in _dataset_configs.items():
            dataset_config = base_data_config.copy_and_resolve_references()
            dataset_config.name = name
            dataset_config.mode = mode
            dataset_config.update(dataset_specs)
            dataset_config.paths = runner_args.dataset_paths[name]
            dataset_config.loss = get_dataset_loss_config(
                loss_config, dataset_specs.loss_mode
            )
            output_dataset_configs.append(dataset_config)

    return output_dataset_configs


def get_lightning_module(config: ConfigDict, model_name: Optional[str] = None):
    """Makes a lightning module for a ModelRunner class given a model config dict.

    A module can be called using the config alone, assuming that the config contains
    the model_name. E.g.

    ```
    model_config = make_model_config("af2_monomer", model_update_yaml)
    lightning_module = get_lightning_module(model_config)
    ```

    If model_name is not a key in config, then the model_name needs to be specified
    separately

    Args:
        config: ConfigDict with settings for model construction
        model_name:
            If provided, creates a ModelRunner matching the key in the
            MODEL_REGISTRY
    Returns:
        `core.runners.model_runner.ModelRunner` for specified model_name
        with the given config settings.
    """
    if not model_name:
        try:
            model_name = config.model_name
        except KeyError as exc:
            raise ValueError(
                "Model_name must be specified either in config or" " as an argument."
            ) from exc
    return MODEL_REGISTRY[model_name].model_runner(config)
