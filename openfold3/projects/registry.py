# TODO add license
import functools
from typing import Optional

from ml_collections import ConfigDict

from openfold3.core.runners.registry_base import register_project_base

# Record of ModelEntries
MODEL_REGISTRY = {}
register_model = functools.partial(register_project_base, model_registry=MODEL_REGISTRY)


def make_model_config_with_preset(model_name: str, preset: str):
    """Retrieves config matching preset for one of the models."""
    model_entry = MODEL_REGISTRY[model_name]
    return model_entry.get_config_with_preset(preset)


def get_loss_config(loss_config: ConfigDict, loss_mode: str) -> ConfigDict:
    """Constructs a loss configuration based on loss mode.

    Arguments:
        loss_config:
            Loss section from main config of model. Expected to have a
            `loss_weight_modes` key with `default` and optionally `custom` modes. If a
            custom mode is selected, it will be applied as an update to the loss config.
        loss_mode:
            One of the modes specfied by the loss config.
    Returns:
        A loss configuration with the weighting scheme indicated by the loss mode.
    """

    # TODO: Change this to only copy non-nested arguments to loss config.
    loss_dict = ConfigDict(
        {
            "min_resolution": loss_config.min_resolution,
            "max_resolution": loss_config.max_resolution,
            "confidence_loss_names": loss_config.confidence_loss_names,
        }
    )
    loss_modes_config = loss_config.loss_weight_modes
    loss_weight_config = loss_modes_config.default.copy_and_resolve_references()

    allowed_modes = list(loss_modes_config.custom.keys()) + ["default"]
    if loss_mode not in allowed_modes:
        raise KeyError(
            f"{loss_mode} is not supported, allowed loss modes are: {allowed_modes}"
        )
    elif loss_mode != "default":
        loss_weight_config.update(loss_modes_config.custom[loss_mode])

    loss_dict.loss_weight = ConfigDict(loss_weight_config)
    return loss_dict


def make_dataset_configs(
    base_data_template: ConfigDict,
    loss_weight_config: ConfigDict,
    runner_args: ConfigDict,
) -> list[ConfigDict]:
    """Constructs dataset configuration based on run script arguments.

    The following sections are expected in the runscript arguments.
    See `openfold3/examples/example_runner.yml` for a full config example.

    ```
    dataset_configs:
        train:
            dataset_1_name:
                class: <dataset1_class_name>
                weight: 0.5
                config:
                    loss_weight_mode: <loss_mweight_mode>
                    ...
            dataset_2_name:
                ...
        val:
            val_dataset_1_name:
                ...
        test:
            test_dataset_1_name:
                ...

    dataset_paths:
        dataset_1_name:
            alignments: /<dataset_1_path>/alignments
            mmcif: /<dataset_1_path>/mmcif
            template_mmcif_structures: /<dataset_1_path>/template_mmcif
        dataset_2_name:
            ...
        val_dataset_1_name:
            ...
        test_dataset_1_name:
            ...
    ```
    Args:
        base_data_config:
            Default dataset settings
        loss_weight_config:
            Loss weight settings
        runner_args:
            ConfigDict with `dataset_configs` and `dataset_paths` heading. See
            full docstring for an example or `openfold3/examples/example_runner.yml`
    Returns:
        A list of config dicts, one for each dataset used for training the model.
    """
    input_dataset_configs = runner_args.dataset_configs
    output_dataset_configs = []
    # loop through datasets for a given type
    for dataset_type, _datasets_configs in input_dataset_configs.items():
        # loop through datasets in a given mode
        for name, dataset_specs in _datasets_configs.items():
            dataset_config = ConfigDict(
                {
                    "name": name,
                    "type": dataset_type,
                    "weight": dataset_specs.weight,
                    "class": dataset_specs["class"],
                }
            )
            sub_config = base_data_template.copy_and_resolve_references()
            sub_config.update(dataset_specs.config)
            for path_name, path in runner_args.dataset_paths[name].items():
                sub_config[path_name] = path
            sub_config.loss = get_loss_config(
                loss_weight_config, sub_config.loss_weight_mode
            )
            delattr(sub_config, "loss_weight_mode")
            dataset_config.config = sub_config
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
