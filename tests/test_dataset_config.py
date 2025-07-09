import textwrap
from pathlib import Path

import ml_collections as mlc
import pytest  # noqa: F401  - used for pytest tmp fixture

from openfold3.core.config import config_utils, dataset_config_builder
from openfold3.projects.of3_all_atom.config import (
    dataset_config_builder as af3_dataset_config_builder,
)

PLACEHOLDER_PATH = Path("/path/placeholder")

DUMMY_PROJECT_CONFIG = mlc.ConfigDict(
    {
        "model": "placeholder_configs",
        "dataset_config_template": {
            "name": "placeholder name",
            "class": "placeholder class",
            "mode": "placeholder mode",
            "config": {
                "token_budget": 17,
                "dataset_paths": {
                    "alignments": PLACEHOLDER_PATH,
                    "targets": PLACEHOLDER_PATH,
                },
            },
        },
    }
)

DUMMY_AF3_PROJECT_CONFIG = mlc.ConfigDict(
    {
        "model": "placeholder_configs",
        "dataset_config_template": {
            "name": "placeholder name",
            "class": "placeholder class",
            "mode": "placeholder mode",
            "config": {
                "token_budget": 17,
                "dataset_paths": {
                    "alignments": PLACEHOLDER_PATH,
                    "targets": PLACEHOLDER_PATH,
                },
                "loss_weight_mode": "default",
                "loss": {
                    "loss_weights": {
                        "distogram": 0.0,
                        "mse": 0.0,
                    },
                },
            },
        },
        "extra_configs": {
            "loss_weight_modes": {
                "default": {
                    "distogram": 3e-2,
                    "mse": 4.0,
                },
                "custom": {
                    "self_distillation": {
                        "mse": 0.5,
                    },
                },
            },
        },
    }
)


class TestDefaultDatasetConfigConstruction:
    def test_default_dataset_config_construction(self, tmp_path):
        test_yaml_str = textwrap.dedent("""\
        dataset_paths: 
            dataset1:
                alignments: /dataset1/alignments
                targets: /dataset1/mmcifs
            dataset2:
                alignments: /dataset2/alignments
                targets: /dataset2/mmcifs

        dataset_configs:
            train:
                dataset1:
                    class: TrainDataset 
                    weight: 1.0 
            validation:
                dataset2:
                    class: ValidationDataset
                    config:
                        token_budget: 13
        """)
        test_yaml_file = tmp_path / "runner.yml"
        with open(test_yaml_file, "w") as f:
            f.write(test_yaml_str)

        runner_args = mlc.ConfigDict(config_utils.load_yaml(test_yaml_file))

        dataset_configs = []
        for mode, input_dataset_configs in runner_args.dataset_configs.items():
            for name, dataset_specs in input_dataset_configs.items():
                builder = dataset_config_builder.DefaultDatasetConfigBuilder(
                    DUMMY_PROJECT_CONFIG
                )
                dataset_specific_paths = runner_args.dataset_paths[name]
                dataset_config = builder.get_custom_config(
                    name, mode, dataset_specs, dataset_specific_paths
                )
                dataset_configs.append(dataset_config)

        assert len(dataset_configs) == 2

        # TODO: Implement a full comparison of the expected configdicts
        if dataset_configs[0].name == "dataset1":
            dataset_1_cfg, dataset_2_cfg = dataset_configs[0], dataset_configs[1]
        elif dataset_configs[1].name == "dataset1":
            dataset_1_cfg, dataset_2_cfg = dataset_configs[1], dataset_configs[0]

        assert dataset_1_cfg.config.dataset_paths.alignments == Path(
            "/dataset1/alignments"
        )
        assert dataset_1_cfg.config.dataset_paths.targets == Path("/dataset1/mmcifs")
        assert dataset_1_cfg["class"] == "TrainDataset"
        assert dataset_1_cfg.mode == "train"
        assert dataset_1_cfg.weight == 1.0
        assert dataset_1_cfg.config.token_budget == 17

        assert dataset_2_cfg.config.dataset_paths.alignments == Path(
            "/dataset2/alignments"
        )
        assert dataset_2_cfg.config.dataset_paths.targets == Path("/dataset2/mmcifs")
        assert dataset_2_cfg["class"] == "ValidationDataset"
        assert dataset_2_cfg.mode == "validation"
        assert dataset_2_cfg.get("weight") is None
        assert dataset_2_cfg.config.token_budget == 13, "Fails token budget overwrite"

    def test_af3_dataset_loss_config_construction(self, tmp_path):
        test_yaml_str = textwrap.dedent("""\
        dataset_paths: 
            dataset1:
                alignments: /dataset1/alignments
                targets: /dataset1/mmcifs
            dataset2:
                alignments: /dataset2/alignments
                targets: /dataset2/mmcifs

        dataset_configs:
            train:
                dataset1:
                    class: TrainDataset 
                    weight: 1.0
                    config:
                        loss_weight_mode: default
            validation:
                dataset2:
                    class: ValidationDataset
                    config:
                        loss_weight_mode: self_distillation
        """)
        test_yaml_file = tmp_path / "runner.yml"
        with open(test_yaml_file, "w") as f:
            f.write(test_yaml_str)

        runner_args = mlc.ConfigDict(config_utils.load_yaml(test_yaml_file))

        dataset_configs = []
        for mode, input_dataset_configs in runner_args.dataset_configs.items():
            for name, dataset_specs in input_dataset_configs.items():
                builder = af3_dataset_config_builder.AF3DatasetConfigBuilder(
                    DUMMY_AF3_PROJECT_CONFIG
                )
                dataset_specific_paths = runner_args.dataset_paths[name]
                dataset_config = builder.get_custom_config(
                    name, mode, dataset_specs, dataset_specific_paths
                )
                dataset_configs.append(dataset_config)

        assert len(dataset_configs) == 2

        dataset_1_cfg = dataset_configs[0]
        dataset_2_cfg = dataset_configs[1]

        print(dataset_2_cfg)
        print(dataset_1_cfg)

        assert dataset_1_cfg.config.dataset_paths.alignments == Path(
            "/dataset1/alignments"
        )
        assert dataset_1_cfg.config.dataset_paths.targets == Path("/dataset1/mmcifs")
        assert dataset_1_cfg["class"] == "TrainDataset"
        assert dataset_1_cfg.weight == 1.0
        assert dataset_1_cfg.config.loss.loss_weights.distogram == 3e-2
        assert dataset_1_cfg.config.loss.loss_weights.mse == 4.0

        assert dataset_2_cfg.config.dataset_paths.alignments == Path(
            "/dataset2/alignments"
        )
        assert dataset_2_cfg.config.dataset_paths.targets == Path("/dataset2/mmcifs")
        assert dataset_2_cfg["class"] == "ValidationDataset"
        assert dataset_2_cfg.get("weight") is None
        assert dataset_2_cfg.config.loss.loss_weights.distogram == 3e-2
        assert dataset_2_cfg.config.loss.loss_weights.mse == 0.5
