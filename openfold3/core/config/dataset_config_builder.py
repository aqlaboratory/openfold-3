from pathlib import Path
from typing import Optional

import ml_collections as mlc


class DefaultDatasetConfigBuilder:
    """Builder class for constructing a config for a single dataset given a project
    config.

    A configuration for a SingleDataset in this repo is expected to follow the following
    format:

    ```
        class SingleDatasetConfig:
        name: str
        dataset_class: str
        mode: DatasetMode
        weight: float
        config: mlc.ConfigDict
    ```

    This builder class will construct a config for a single dataset based on dataset
    specifications from the project level config and runner configuration arguments.
    Because the components of the dataset config will exist in different sections, this
    class provides a guideline for how to make a datset config.

    Project Config:
    The project config must have a section called `dataset_config_template`. This
    section should represent a config that could be passed into datasets for the given
    project. The project config may optionally have additional configs that are used to
    construct the dataset config, these can be passed in the `extra_configs` section of
    the project config.

    Runner config settings:

    The specifications for individual datasets will be passed in the run
    configurations yaml in two separate sections, one for dataset_configs, and one for
    dataset_paths as follows:

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
    """

    def __init__(self, project_config: mlc.ConfigDict):
        config_template = project_config.dataset_config_template
        self.config = config_template.copy_and_resolve_references()

    def _get_sanitized_paths(self, dataset_paths: mlc.ConfigDict):
        input_paths = {}
        for name, value in dataset_paths.iteritems():
            if value in ["none", "None", None]:
                input_paths[name] = None
            elif name.endswith("_file_format"):
                input_paths[name] = value
            else:
                input_paths[name] = Path(value)
        return input_paths

    def _add_paths(self, dataset_paths: mlc.ConfigDict):
        paths = self._get_sanitized_paths(dataset_paths)
        self.config.config.dataset_paths.update(paths)

    def _update_config(self, config_update: Optional[mlc.ConfigDict]):
        if config_update:
            self.config.config.update(config_update)

    def get_custom_config(
        self,
        name: str,
        mode: str,
        dataset_specs: mlc.ConfigDict,
        dataset_paths: mlc.ConfigDict,
    ):
        self.config.name = name
        self.config["class"] = dataset_specs.get("class")
        self.config.mode = mode
        self.config.weight = dataset_specs.get("weight")
        self._add_paths(dataset_paths)
        self._update_config(dataset_specs.get("config"))
        return self.config
