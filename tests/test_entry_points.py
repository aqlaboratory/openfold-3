import json
import os
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import ml_collections as mlc
import pytest
from pytorch_lightning.loggers import WandbLogger

from openfold3.core.config import config_utils
from openfold3.core.data.framework.data_module import DataModuleConfig
from openfold3.entry_points.experiment_runner import (
    InferenceExperimentRunner,
    TrainingExperimentRunner,
    WandbHandler,
)
from openfold3.entry_points.validator import (
    InferenceExperimentConfig,
    TrainingExperimentConfig,
    WandbConfig,
)
from openfold3.projects.of3_all_atom.project_entry import ModelUpdate, OF3ProjectEntry


class TestTrainingExperiment:
    @pytest.fixture
    def expt_runner(self, tmp_path):
        """Minimal runner yaml containing only dataset configs."""
        test_dummy_file = tmp_path / "test.json"
        test_dummy_file.write_text("test")

        test_yaml_str = textwrap.dedent(f"""\
            data_module_args:
                data_seed: 114
                num_workers: 0
                                        
            model_update:
                presets:
                    - train
                custom:
                    settings:
                        model_selection_weight_scheme: fine_tuning
                    architecture:
                        shared:
                            diffusion:
                                no_samples: 32
                                        
            dataset_configs:
                train:
                    weighted-pdb:
                        dataset_class: WeightedPDBDataset 
                        weight: 1 
                        config:
                            debug_mode: true
                            crop:
                                token_budget: 640 
                            loss:
                                bond: 4.0
                                smooth_lddt: 0.0

                validation:
                    val-weighted-pdb:
                        dataset_class: ValidationPDBDataset
                        config:
                            template:
                                n_templates: 4

            dataset_paths:
                weighted-pdb:
                    alignments_directory: null
                    alignment_db_directory: null
                    alignment_array_directory: {tmp_path} 
                    target_structures_directory: {tmp_path} 
                    target_structure_file_format: npz
                    dataset_cache_file: {test_dummy_file} 
                    reference_molecule_directory: {tmp_path} 
                    template_cache_directory: {tmp_path} 
                    template_structure_array_directory: {tmp_path} 
                    template_structures_directory: null
                    template_file_format: pkl
                    ccd_file: null

                val-weighted-pdb:
                    alignments_directory: null
                    alignment_db_directory: null
                    alignment_array_directory: {tmp_path} 
                    target_structures_directory: {tmp_path}
                    target_structure_file_format: npz
                    dataset_cache_file: {test_dummy_file} 
                    reference_molecule_directory: {tmp_path} 
                    template_cache_directory: {tmp_path} 
                    template_structure_array_directory: {tmp_path} 
                    template_structures_directory: null
                    template_file_format: pkl
                    ccd_file: null
                """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        expt_config = TrainingExperimentConfig.model_validate(
            config_utils.load_yaml(test_yaml_file)
        )

        expt_runner = TrainingExperimentRunner(expt_config)
        expt_runner.setup()
        return expt_runner

    def test_model_config_update(self, expt_runner):
        assert (
            expt_runner.model_config.settings.model_selection_weight_scheme
            == "fine_tuning"
        )
        assert expt_runner.model_config.architecture.shared.diffusion.no_samples == 32
        # Check that default settings are not overwritten
        # See openfold3.projects.of3_all_atom.config.model_config
        assert (
            expt_runner.model_config.settings.memory.eval.per_sample_token_cutoff == 750
        )

    def test_model(self, expt_runner):
        # Check model creation

        assert expt_runner.lightning_module.model
        assert (
            expt_runner.lightning_module.model.aux_heads.distogram.linear.in_features
            == 128
        )

    def test_data_module(self, expt_runner):
        # Check data_module creation
        assert expt_runner.data_module_config.data_seed == 114

        assert len(expt_runner.data_module_config.datasets) == 2
        assert expt_runner.data_module_config.datasets[0].name == "weighted-pdb"
        assert expt_runner.data_module_config.datasets[1].name == "val-weighted-pdb"

        weighted_pdb_spec = expt_runner.data_module_config.datasets[0]
        assert weighted_pdb_spec.weight == 1
        assert weighted_pdb_spec.config.crop.token_budget == 640


class TestModelUpdate:
    def test_bad_model_update_fails(self):
        """Verify that a model update that has an invalid field is not allowed."""
        model_update = ModelUpdate(custom={"nonexistant_field": "bad"})
        project_entry = OF3ProjectEntry()

        with pytest.raises(KeyError, match="config is locked"):
            project_entry.get_model_config_with_update(model_update)

    def test_model_update_with_diffusion_samples(self, tmp_path):
        """Test application of model update and num_diffusion_samples cli argument."""
        test_yaml_str = textwrap.dedent("""\
            model_update:
              custom:
                architecture:
                  shared:
                    num_recycles: 1 
        """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)
        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=tmp_path / "dummy.ckpt.pt",
            **config_utils.load_yaml(test_yaml_file),
        )
        expt_runner = InferenceExperimentRunner(expt_config)

        expected_num_diffusion_samples = 17
        expt_runner.set_num_diffusion_samples(expected_num_diffusion_samples)
        model_config = expt_runner.model_config
        assert (
            model_config.architecture.shared.diffusion.no_full_rollout_samples
            == expected_num_diffusion_samples
        )
        # Verify settings from model_update section are also applied
        assert model_config.architecture.shared.num_recycles == 1


class TestLowMemoryConfig:
    def test_low_mem_model_config_preset(self, tmp_path):
        test_dummy_file = tmp_path / "test.json"
        test_dummy_file.write_text("test")

        test_yaml_str = textwrap.dedent("""\
            data_module_args:
                data_seed: 114
                                        
            model_update:
                presets:
                    - predict
                    - low_mem
            """)

        test_yaml_file = tmp_path / "runner.yml"
        dummy_ckpt = tmp_path / "dummy.ckpt.pt"
        test_yaml_file.write_text(test_yaml_str)

        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=dummy_ckpt, **config_utils.load_yaml(test_yaml_file)
        )

        expt_runner = InferenceExperimentRunner(expt_config)
        model_cfg = expt_runner.model_config

        # check that inference mode set correctly
        assert not model_cfg.settings.diffusion_training_enabled

        # check low memory settings set correctly
        assert model_cfg.settings.memory.eval.chunk_size == 4
        assert model_cfg.settings.memory.eval.offload_inference.enabled

        # test existing setting in experiment runner is not overwritten
        assert not model_cfg.settings.memory.eval.use_lma


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
        self.wandb_args = WandbConfig.model_validate(
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
            self.wandb_args, is_rank_zero=True, output_dir=Path(".")
        )
        _wandb_handler._init_logger()
        self.assertIsNotNone(_wandb_handler.logger)
        mock_wandb_init.assert_called_once()

    @patch("wandb.init")
    def test_wandb_is_called_on_logger(self, mock_wandb_init):
        # Test that the logger is initialized and wandb.init is called for rank-zero.
        _wandb_handler = WandbHandler(
            self.wandb_args, is_rank_zero=True, output_dir=Path(".")
        )
        assert isinstance(_wandb_handler.logger, WandbLogger)
        mock_wandb_init.assert_called_once()

    @patch("os.system", return_value=0)
    def test_store_configs_creates_files(self, mock_os_system):
        _wandb_handler = WandbHandler(
            self.wandb_args, is_rank_zero=True, output_dir=Path(self.temp_dir)
        )

        # Create dummy configuration objects with a to_dict() method.
        dummy_runner_args = TrainingExperimentConfig(
            dataset_configs={}, dataset_paths={}
        )
        dummy_data_module_config = DataModuleConfig(datasets=[])
        dummy_model_config = mlc.ConfigDict({"model": "dummy"})

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
                    self.assertEqual(data, dummy_runner_args.model_dump(mode="json"))
                elif fpath.endswith("data_config.json"):
                    self.assertEqual(data, dummy_data_module_config.model_dump())
                elif fpath.endswith("model_config.json"):
                    self.assertEqual(data, dummy_model_config.to_dict())


class TestInferenceCommandLineSettings:
    @pytest.mark.parametrize("use_msa_cli_arg", [True, False])
    def test_use_msa_cli(self, use_msa_cli_arg, tmp_path):
        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=tmp_path / "dummy.ckpt"
        )
        expt_runner = InferenceExperimentRunner(
            expt_config, use_msa_server=use_msa_cli_arg
        )
        assert expt_runner.use_msa_server == use_msa_cli_arg
    
    @pytest.mark.parametrize("use_templates_cli_arg", [True, False])
    def test_use_templates_cli(self, use_templates_cli_arg, tmp_path):
        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=tmp_path / "dummy.ckpt"
        )
        expt_runner = InferenceExperimentRunner(
            expt_config, use_templates=use_templates_cli_arg
        )
        assert expt_runner.use_templates == use_templates_cli_arg
