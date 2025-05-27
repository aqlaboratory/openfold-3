import textwrap

import pytest  # noqa: F401  - used for pytest tmp fixture

from openfold3.core.config import config_utils
from openfold3.core.data.framework.single_datasets.pdb import WeightedPDBDataset
from openfold3.core.data.framework.single_datasets.validation import (
    ValidationPDBDataset,
)
from openfold3.entry_points.experiment_runner import TrainingExperimentRunner
from openfold3.entry_points.validator import TrainingExperimentConfig
from openfold3.projects.af3_all_atom.config.dataset_configs import (
    TrainingDatasetSpec,
)


class TestAF3DatasetConfigConstruction:
    # DO NOT SUBMIT: Test using real paths
    def test_load_pdb_weighted_real(self, tmp_path):
        test_yaml_str = textwrap.dedent("""\
            name: dataset1
            mode: train
            dataset_class: WeightedPDBDataset
            weight: 0.37
            config:
                dataset_paths:
                    alignments_directory: none # /pscratch/sd/v/vss2134/alignments/pdb_msas_completed
                    alignment_db_directory: none
                    alignment_array_directory: /global/cfs/cdirs/m4351/gnikol/data/pdb_msas_v3_preparsed/
                    target_structure_file_format: npz 
                    target_structures_directory: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v10/preprocessed_pdb/cif_files
                    dataset_cache_file: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v7/training_cache_with_templates.json
                    reference_molecule_directory: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v10/preprocessed_pdb/reference_mols
                    template_cache_directory: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v7/template_cache/
                    template_structures_directory: none # /global/cfs/cdirs/m4351/ljarosch/of3_data/pdb_data/pdb_mmcif/mmcif_files
                    template_structure_array_directory: /pscratch/sd/g/gnikol/data/template_preprocessing/template_structure_preprocessing_v4_2/template_structure_arrays
                    template_file_format: cif
                    ccd_file: /pscratch/sd/g/gnikol/data/template_preprocessing/components.cif
        """)

        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        input_dict = config_utils.load_yaml(test_yaml_file)
        actual_config = TrainingDatasetSpec.model_validate(input_dict)
        dataset = WeightedPDBDataset(actual_config.config)
        assert dataset.name == "dataset1"
        print(f"{len(dataset)=}")

    # DO NOT SUBMIT: Test using real paths
    @pytest.mark.skip(
        reason="Validation sets used for testing missing new `metric_eligible` and `source_subset` fields."
    )
    def test_load_val_dataset_real(self, tmp_path):
        test_yaml_str = textwrap.dedent("""\
            name: val-data
            mode: validation 
            dataset_class: ValidationPDBDataset 
            config:
                dataset_paths:
                    alignments_directory: none
                    alignment_array_directory: /global/cfs/cdirs/m4351/gnikol/data/pdb_msas_v3_preparsed/
                    target_structures_directory: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v8_proto/preprocessed_pdb/cif_files
                    target_structure_file_format: pkl 
                    alignment_db_directory: none
                    dataset_cache_file: /pscratch/sd/j/jnwei22/of3_datacache_files/mini_validation_cache_Jan15.json
                    reference_molecule_directory: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v8_proto/preprocessed_pdb/reference_mols
                    template_cache_directory: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v8_proto/validation_template_processing/val_template_cache
                    template_structure_array_directory: /pscratch/sd/g/gnikol/data/template_preprocessing/template_structure_preprocessing_v4/template_structure_arrays
                    template_structures_directory: none
                    template_file_format: "cif"
                    ccd_file: none
        """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        input_dict = config_utils.load_yaml(test_yaml_file)
        actual_config = TrainingDatasetSpec.model_validate(input_dict)
        dataset = ValidationPDBDataset(actual_config.config)
        assert dataset.name == "val-data"
        print(f"{len(dataset)=}")

    def test_load_runner_yaml_args(self, tmp_path):
        test_yaml_str = textwrap.dedent("""\
            dataset_paths:
                weighted-pdb:
                    alignments_directory: none # /pscratch/sd/v/vss2134/alignments/pdb_msas_completed
                    alignment_db_directory: none
                    alignment_array_directory: /global/cfs/cdirs/m4351/gnikol/data/pdb_msas_v3_preparsed/
                    target_structure_file_format: npz 
                    target_structures_directory: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v10/preprocessed_pdb/cif_files
                    dataset_cache_file: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v7/training_cache_with_templates.json
                    reference_molecule_directory: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v10/preprocessed_pdb/reference_mols
                    template_cache_directory: /pscratch/sd/l/ljarosch/af3_dataset_releases/af3_training_data_v7/template_cache/
                    template_structures_directory: none # /global/cfs/cdirs/m4351/ljarosch/of3_data/pdb_data/pdb_mmcif/mmcif_files
                    template_structure_array_directory: /pscratch/sd/g/gnikol/data/template_preprocessing/template_structure_preprocessing_v4_2/template_structure_arrays
                    template_file_format: cif
                    ccd_file: /pscratch/sd/g/gnikol/data/template_preprocessing/components.cif

            dataset_configs:
              train:
                weighted-pdb:
                  dataset_class: WeightedPDBDataset 
                  weight: 1 
                  config:
                    template:
                        n_templates: 4
                        token_budget: 384
                    crop:
                        crop_weights:
                            contiguous: 0.2
                            spatial: 0.4
                            spatial_interface: 0.4
        """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        runner_args = config_utils.load_yaml(test_yaml_file)
        expt_config = TrainingExperimentConfig.model_validate(runner_args)
        expt_runner = TrainingExperimentRunner(expt_config)
        data_module = expt_runner.lightning_data_module

        data_module.setup()
        train_dataloader = data_module.train_dataloader()
        train_features = next(iter(train_dataloader))
        print(train_features)
