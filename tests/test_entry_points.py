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
    CHECKPOINT_NAME,
    InferenceExperimentConfig,
    TrainingExperimentConfig,
    TrainingExperimentSettings,
    WandbConfig,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from openfold3.projects.of3_all_atom.project_entry import ModelUpdate, OF3ProjectEntry


@pytest.fixture
def dummy_ckpt_file(tmp_path: Path) -> Path:
    dummy_ckpt = tmp_path / "dummy.ckpt"
    dummy_ckpt.write_text("dummy content")
    return dummy_ckpt


def _create_fake_file(path: Path) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("dummy content")


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

    @pytest.mark.parametrize("pl_checkpoint_option", [None, "last", "hpc", "registry"])
    def test_pl_checkpoint_load_options(self, pl_checkpoint_option):
        expt_config = TrainingExperimentSettings.model_validate(
            {"restart_checkpoint_path": pl_checkpoint_option}
        )
        print(expt_config.restart_checkpoint_path)
        assert expt_config.restart_checkpoint_path == pl_checkpoint_option

    def test_pl_checkpoint_load_from_path(self, tmp_path):
        dummy_ckpt = tmp_path / "dummy.ckpt"
        dummy_ckpt.write_text("test")
        expt_config = TrainingExperimentSettings.model_validate(
            {"restart_checkpoint_path": str(dummy_ckpt)}
        )
        assert expt_config.restart_checkpoint_path == str(dummy_ckpt)

        # check that loading fails when given an invalid string / path
        non_existant_path = "nonexistant.ckpt"
        with pytest.raises(ValueError):
            TrainingExperimentSettings.model_validate(
                {"restart_checkpoint_path": non_existant_path}
            )

    @pytest.mark.parametrize(
        "data_seed, model_seed, expected_data_seed", [(114, 42, 114), (None, 123, 123)]
    )
    def test_synchronize_seeds_respects_data_seed(
        self, data_seed, model_seed, expected_data_seed, tmp_path
    ):
        test_yaml_str = textwrap.dedent(f"""\
            experiment_settings:
                seed: {model_seed}
            """)

        if data_seed:
            test_yaml_str += textwrap.dedent(f"""\
                    data_module_args:
                        data_seed: {data_seed}
            """)

        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        expt_config = TrainingExperimentConfig(
            dataset_paths={},
            dataset_configs={},
            **config_utils.load_yaml(test_yaml_file),
        )
        assert expt_config.experiment_settings.seed == model_seed
        assert expt_config.data_module_args.data_seed == expected_data_seed


class TestModelUpdate:
    def test_bad_model_update_fails(self):
        """Verify that a model update that has an invalid field is not allowed."""
        model_update = ModelUpdate(custom={"nonexistant_field": "bad"})
        project_entry = OF3ProjectEntry()

        with pytest.raises(KeyError, match="config is locked"):
            project_entry.get_model_config_with_update(model_update)

    def test_model_update_with_diffusion_samples(self, tmp_path, dummy_ckpt_file):
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
            inference_ckpt_path=dummy_ckpt_file,
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

    def test_pae_disabled_if_preset_not_selected(self, tmp_path, dummy_ckpt_file):
        """Test pae not set if only predict preset specified experiment runner."""
        test_yaml_str = textwrap.dedent("""\
            model_update:
              presets: 
                - predict
        """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)
        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=dummy_ckpt_file,
            **config_utils.load_yaml(test_yaml_file),
        )
        expt_runner = InferenceExperimentRunner(expt_config)
        assert not expt_runner.pae_enabled, "Expected pae_head not to be enabled."

    def test_pae_enabled(self, tmp_path, dummy_ckpt_file):
        """Test pae enabled updates experiment runner."""
        test_yaml_str = textwrap.dedent("""\
            model_update:
              presets: 
                - predict
                - pae_enabled
        """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)
        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=dummy_ckpt_file,
            **config_utils.load_yaml(test_yaml_file),
        )
        expt_runner = InferenceExperimentRunner(expt_config)
        assert expt_runner.pae_enabled

    def test_low_mem_model_config_preset(self, tmp_path, dummy_ckpt_file):
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
        test_yaml_file.write_text(test_yaml_str)

        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=dummy_ckpt_file,
            **config_utils.load_yaml(test_yaml_file),
        )

        expt_runner = InferenceExperimentRunner(expt_config)
        model_cfg = expt_runner.model_config

        # check that inference mode set correctly
        assert not model_cfg.architecture.msa.msa_module_embedder.subsample_main_msa
        assert model_cfg.architecture.msa.msa_module_embedder.subsample_all_msa

        # check low memory settings set correctly
        assert model_cfg.settings.memory.eval.chunk_size == 4
        assert model_cfg.settings.memory.eval.offload_inference.confidence_heads
        assert model_cfg.settings.memory.eval.offload_inference.token_cutoff == 0

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
    def test_use_msa_cli(self, use_msa_cli_arg, tmp_path, dummy_ckpt_file):
        expt_config = InferenceExperimentConfig(inference_ckpt_path=dummy_ckpt_file)
        expt_runner = InferenceExperimentRunner(
            expt_config, use_msa_server=use_msa_cli_arg
        )
        assert expt_runner.use_msa_server == use_msa_cli_arg

    @pytest.mark.parametrize("use_templates_cli_arg", [True, False])
    def test_use_templates_cli(self, use_templates_cli_arg, tmp_path, dummy_ckpt_file):
        expt_config = InferenceExperimentConfig(inference_ckpt_path=dummy_ckpt_file)
        expt_runner = InferenceExperimentRunner(
            expt_config, use_templates=use_templates_cli_arg
        )
        assert expt_runner.use_templates == use_templates_cli_arg

    def test_seeding_from_num_seeds(self, dummy_ckpt_file):
        expt_config = InferenceExperimentConfig(inference_ckpt_path=dummy_ckpt_file)
        num_seeds = 7
        expt_runner = InferenceExperimentRunner(expt_config, num_model_seeds=num_seeds)
        assert len(expt_runner.seeds) == num_seeds

    def test_seeding_from_list(self, tmp_path, dummy_ckpt_file):
        test_yaml_str = textwrap.dedent("""\
            experiment_settings:
                seeds:
                  - 17 
                  - 101
            """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=dummy_ckpt_file,
            **config_utils.load_yaml(test_yaml_file),
        )
        assert expt_config.experiment_settings.seeds == [17, 101]

    @pytest.mark.parametrize(
        "data_seed, model_seed, expected_data_seed", [(114, 42, 114), (None, 123, 123)]
    )
    def test_synchronize_seeds_respects_data_seed(
        self,
        data_seed,
        model_seed,
        expected_data_seed,
        tmp_path,
        dummy_ckpt_file,
    ):
        test_yaml_str = textwrap.dedent(f"""\
            experiment_settings:
                seeds:
                  - {model_seed} 
                  - 101
            """)

        if data_seed:
            test_yaml_str += textwrap.dedent(f"""\
                    data_module_args:
                        data_seed: {data_seed}
            """)

        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=dummy_ckpt_file,
            **config_utils.load_yaml(test_yaml_file),
        )
        assert expt_config.experiment_settings.seeds == [model_seed, 101]
        assert expt_config.data_module_args.data_seed == expected_data_seed


class TestInferenceCheckpointLoading:
    def test_inference_ckpt_path_user_defined(self, dummy_ckpt_file):
        expt_config = InferenceExperimentConfig.model_validate(
            {"inference_ckpt_path": dummy_ckpt_file}
        )
        assert expt_config.inference_ckpt_path == dummy_ckpt_file

    def test_inference_ckpt_path_defaults(self, tmp_path):
        with patch(
            "openfold3.entry_points.validator._maybe_download_parameters",
            side_effect=_create_fake_file,
        ):
            expt_config = InferenceExperimentConfig.model_validate(
                {"cache_path": tmp_path}
            )
        assert expt_config.inference_ckpt_path == tmp_path / CHECKPOINT_NAME


class TestTemplatePreprocessorSettings:
    def test_overwrite_output_dir(self, tmp_path, dummy_ckpt_file):
        test_yaml_str = textwrap.dedent(f"""\
        template_preprocessor_settings:
            output_directory: {tmp_path / "custom_dir"}
        """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)
        expt_config = InferenceExperimentConfig(
            inference_ckpt_path=dummy_ckpt_file,
            **config_utils.load_yaml(test_yaml_file),
        )

        assert expt_config.template_preprocessor_settings.output_directory == (
            tmp_path / "custom_dir"
        ), "Expected structure directory to match config file setting"


class TestRemoveQuerySetDuplicates:
    @pytest.fixture
    def dummy_output_path(self, tmp_path):
        # Creates the following directories:
        # <output_directory>
        #  ├── query_1
        # 	 └── seed_42
        #         ├── query_1_seed_42_sample_1_model.cif
        #         ├── query_1_seed_42_sample_2_model.cif
        # 	 └── seed_43
        #         ├── query_1_seed_43_sample_1_model.cif
        #         ├── query_1_seed_43_sample_2_model.cif
        #  ├── query_2
        # 	 └── seed_42
        #         ├── query_1_seed_42_sample_1_model.cif
        #         ├── query_1_seed_42_sample_2_model.cif
        # 	 └── seed_43
        #         ├── query_1_seed_43_sample_1_model.cif
        #         ├── <Missing sample 2>

        expected_fnames = [
            "query_1/seed_42/query_1_seed_42_sample_1_model.cif",
            "query_1/seed_42/query_1_seed_42_sample_2_model.cif",
            "query_1/seed_43/query_1_seed_43_sample_1_model.cif",
            "query_1/seed_43/query_1_seed_43_sample_2_model.cif",
            "query_2/seed_42/query_2_seed_42_sample_1_model.cif",
            "query_2/seed_42/query_2_seed_42_sample_2_model.cif",
            "query_2/seed_43/query_2_seed_43_sample_1_model.cif",
        ]

        for fname in expected_fnames:
            _create_fake_file(tmp_path / fname)

        return tmp_path

    def test_remove_duplicates(self, dummy_output_path):
        input_query_set = InferenceQuerySet.model_validate(
            {
                "queries": {
                    "query_1": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "TEST",
                            }
                        ]
                    },
                    "query_2": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "TESTING",
                            }
                        ]
                    },
                    "query_3": {
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": ["A"],
                                "sequence": "TESTTEST",
                            }
                        ]
                    },
                }
            }
        )

        experiment_config = InferenceExperimentConfig.model_validate(
            {"experiment_settings": {"seeds": [42, 43]}}
        )
        expt_runner = InferenceExperimentRunner(
            experiment_config, num_diffusion_samples=2, output_dir=dummy_output_path
        )

        deduplicated_set = expt_runner.remove_completed_queries_from_query_set(
            input_query_set
        )

        assert set(deduplicated_set.queries.keys()) == set(["query_2", "query_3"])
