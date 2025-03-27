import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytorch_lightning as pl
from ml_collections import ConfigDict
from pytorch_lightning.loggers import WandbLogger

import openfold3
from openfold3.core.config import config_utils
from openfold3.entry_points.trainable_experiment.builder import (
    TrainableExperimentBuilder,
    WandbHandler,
)
from openfold3.entry_points.trainable_experiment.validator import (
    TrainableExperimentConfig,
)

MODULE_PATH = "openfold3.entry_points.trainable_experiment.builder"

EXAMPLES_PATH = Path(openfold3.__file__).parent / "examples"


class DummyTrainer:
    def __init__(self):
        self.called_method = None

    def fit(self, model, datamodule, ckpt_path):
        self.called_method = "fit"

    def validate(self, model, datamodule, ckpt_path):
        self.called_method = "validate"

    def test(self, model, datamodule, ckpt_path):
        self.called_method = "test"

    def predict(self, model, datamodule, ckpt_path):
        self.called_method = "predict"


class TestTrainableExperimentBuilder(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to act as the output_dir.
        self.temp_dir = tempfile.mkdtemp()
        self.runner_args = ConfigDict(
            {
                "wandb": None,
                "log_level": "DEBUG",
                "seed": 42,
                "output_dir": self.temp_dir,
                "mode": "train",
                "num_gpus": 1,
                "pl_trainer": ConfigDict({"num_nodes": 1, "max_epochs": 1}),
                "project_type": "dummy_project",
                "presets": [],
                "config_update": None,
                "restart_checkpoint_path": None,
                "deepspeed_config_path": None,
                "checkpoint": None,
                "log_lr": None,
                "mpi_plugin": False,
                "compile": False,
            }
        )

        patcher1 = patch(MODULE_PATH + ".registry.get_project_entry")
        self.mock_get_project_entry = patcher1.start()
        self.addCleanup(patcher1.stop)

        dummy_project_entry = MagicMock()
        dummy_project_entry.model_runner.return_value = MagicMock(
            spec=pl.LightningModule
        )
        dummy_project_entry.dataset_config_builder = MagicMock(return_value={})
        self.mock_get_project_entry.return_value = dummy_project_entry

        patcher2 = patch(
            MODULE_PATH + ".registry.make_config_with_presets",
            return_value=ConfigDict(
                {
                    "model": ConfigDict(
                        {
                            "settings": {
                                "gradient_clipping": 0.5,
                                "optimizer": {"use_deepspeed_adam": True},
                            }
                        }
                    )
                }
            ),
        )
        self.mock_make_config = patcher2.start()
        self.addCleanup(patcher2.stop)

        patcher3 = patch(
            MODULE_PATH + ".registry.make_dataset_module_config",
            return_value=MagicMock(),
        )
        self.mock_make_dataset_module_config = patcher3.start()
        self.addCleanup(patcher3.stop)

        patcher4 = patch(MODULE_PATH + "._check_data_module_config", lambda x: None)
        self.mock_check_data_module_config = patcher4.start()
        self.addCleanup(patcher4.stop)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_output_dir_creation(self):
        builder = TrainableExperimentBuilder(self.runner_args)
        output_dir = builder.output_dir
        self.assertTrue(output_dir.exists())
        self.assertEqual(output_dir, Path(self.temp_dir))

    def test_invalid_log_level(self):
        self.runner_args.log_level = "invalid"
        builder = TrainableExperimentBuilder(self.runner_args)
        with self.assertRaises(ValueError):
            builder._setup_logger()

    def test_set_random_seed_distributed_without_seed(self):
        self.runner_args.seed = None
        self.runner_args.num_gpus = 2  # Distributed training.
        self.runner_args.pl_trainer.num_nodes = 2
        builder = TrainableExperimentBuilder(self.runner_args)
        with self.assertRaises(ValueError):
            builder._set_random_seed()

    def test_set_random_seed_with_non_int(self):
        del (
            self.runner_args.seed
        )  # I need to delete it otherwise i cannot set up as not a string
        self.runner_args.seed = "not_an_int"
        builder = TrainableExperimentBuilder(self.runner_args)
        with self.assertRaises(ValueError):
            builder._set_random_seed()

    @patch(MODULE_PATH + ".TrainableExperimentBuilder.trainer", new=DummyTrainer())
    @patch(
        MODULE_PATH + ".TrainableExperimentBuilder.lightning_module", new=MagicMock()
    )
    @patch(
        MODULE_PATH + ".TrainableExperimentBuilder.lightning_data_module",
        new=MagicMock(),
    )
    @patch(MODULE_PATH + ".TrainableExperimentBuilder.ckpt_path", new=None)
    def test_run_modes(self):
        builder = TrainableExperimentBuilder(self.runner_args)
        patched_trainer = builder.trainer

        for mode, expected_method in [
            ("train", "fit"),
            ("eval", "validate"),
            ("test", "test"),
            ("predict", "predict"),
        ]:
            self.runner_args.mode = mode
            patched_trainer.called_method = None
            builder.run()
            self.assertEqual(
                patched_trainer.called_method,
                expected_method,
                f"Mode {mode} should call {expected_method}",
            )

        # Test invalid mode.
        self.runner_args.mode = "invalid_mode"
        with self.assertRaises(ValueError):
            builder.run()

    def test_is_distributed(self):
        builder = TrainableExperimentBuilder(self.runner_args)
        self.assertFalse(builder.is_distributed)

        self.runner_args.num_gpus = 2
        self.runner_args.pl_trainer.num_nodes = 2
        builder = TrainableExperimentBuilder(self.runner_args)
        self.assertTrue(builder.is_distributed)

    def test_ckpt_path_property(self):
        builder = TrainableExperimentBuilder(self.runner_args)
        self.assertIsNone(builder.ckpt_path)

        # When a checkpoint path is provided.
        self.runner_args.restart_checkpoint_path = "dummy_checkpoint.ckpt"
        builder = TrainableExperimentBuilder(self.runner_args)
        self.assertEqual(builder.ckpt_path, "dummy_checkpoint.ckpt")

    def test_strategy_deepspeed(self):
        self.runner_args.deepspeed_config_path = "dummy_deepspeed_config.json"
        with patch(
            MODULE_PATH + ".DeepSpeedStrategy", return_value=MagicMock()
        ) as mock_deepspeed_strategy:
            strategy = TrainableExperimentBuilder(self.runner_args).strategy
            self.assertIs(strategy, mock_deepspeed_strategy.return_value)
            mock_deepspeed_strategy.assert_called_once()

    def test_strategy_ddp(self):
        self.runner_args.deepspeed_config_path = None
        self.runner_args.num_gpus = 2
        self.runner_args.pl_trainer.num_nodes = 1
        builder = TrainableExperimentBuilder(self.runner_args)
        strategy = builder.strategy
        from pytorch_lightning.strategies import DDPStrategy

        self.assertIsInstance(strategy, DDPStrategy)

    def test_is_mpi_and_cluster_environment(self):
        self.runner_args.mpi_plugin = True
        builder = TrainableExperimentBuilder(self.runner_args)
        self.assertTrue(builder.is_mpi)
        self.assertIsNotNone(builder.cluster_environment)


class DummyWandbExperiment:
    def __init__(self, directory):
        self.dir = directory
        self.saved_files = []

    def save(self, filepath):
        self.saved_files.append(filepath)


class DummyWandbLogger:
    def __init__(self, experiment):
        self.experiment = experiment


class TestWandbHandler(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.wandb_args = ConfigDict(
            {
                "project": "test_project",
                "entity": "test_entity",
                "group": "test_group",
                "experiment_name": "test_experiment",
                "offline": True,
                "id": "test_id",
            }
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("wandb.init")
    def test_init_logger(self, mock_wandb_init):
        # Test that the logger is initialized and wandb.init is called for rank-zero.
        _wandb_handler = WandbHandler(
            self.wandb_args, is_mpi_rank_zero=True, output_dir=Path(".")
        )
        _wandb_handler._init_logger()
        self.assertIsNotNone(_wandb_handler.logger)
        mock_wandb_init.assert_called_once()

    @patch("wandb.init")
    def test_wandb_is_called_on_logger(self, mock_wandb_init):
        # Test that the logger is initialized and wandb.init is called for rank-zero.
        _wandb_handler = WandbHandler(
            self.wandb_args, is_mpi_rank_zero=True, output_dir=Path(".")
        )
        assert isinstance(_wandb_handler.logger, WandbLogger)
        mock_wandb_init.assert_called_once()

    @patch("os.system", return_value=0)
    def test_store_configs_creates_files(self, mock_os_system):
        _wandb_handler = WandbHandler(
            self.wandb_args, is_mpi_rank_zero=True, output_dir=Path(self.temp_dir)
        )

        # Create dummy configuration objects with a to_dict() method.
        dummy_runner_args = ConfigDict({"key": "value"})
        dummy_data_module_config = ConfigDict({"data": 123})
        dummy_model_config = ConfigDict({"model": "dummy"})

        # Set up a dummy experiment with our temporary directory.
        dummy_experiment = DummyWandbExperiment(self.temp_dir)
        dummy_logger = DummyWandbLogger(dummy_experiment)
        _wandb_handler._logger = dummy_logger

        _wandb_handler.store_configs(
            dummy_runner_args, dummy_data_module_config, dummy_model_config
        )

        expected_files = [
            "package_versions.txt",
            "runner.json",
            "data_config.json",
            "model_config.json",
        ]
        expected_files = [
            os.path.join(self.temp_dir, fname) for fname in expected_files
        ]
        assert set(dummy_experiment.saved_files) == set(expected_files)

        for fpath in expected_files:
            if fpath.endswith("package_versions.txt"):
                # Ignore this file, since i am patching its generation
                continue

            with open(fpath) as f:
                data = json.load(f)
                if fpath.endswith("runner.json"):
                    self.assertEqual(data, dummy_runner_args.to_dict())
                elif fpath.endswith("data_config.json"):
                    self.assertEqual(data, dummy_data_module_config.to_dict())
                elif fpath.endswith("model_config.json"):
                    self.assertEqual(data, dummy_model_config.to_dict())


class TestTrainableExperimentValidator(unittest.TestCase):
    def setUp(self):
        self.examples_file = list(EXAMPLES_PATH.glob("*.yml"))

    def test_validate(self):
        for fname in self.examples_file:
            raw_args = config_utils.load_yaml(fname)
            try:
                TrainableExperimentConfig(**raw_args)
            except Exception:
                self.fail(f"Validation failed for {fname}")
