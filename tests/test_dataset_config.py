import textwrap

import pytest  # noqa: F401  - used for pytest tmp fixture

from openfold3.core.config import config_utils
from openfold3.projects.af3_all_atom.config.dataset_configs import TrainingDatasetSpec


class TestAF3DatasetConfigConstruction:
    def test_load_pdb_weighted_config(self, tmp_path):
        test_dummy_file = tmp_path / "test.json"
        test_dummy_file.write_text("test")
        test_yaml_str = textwrap.dedent(f"""\
            name: dataset1
            mode: train
            dataset_class: WeightedPDBDataset
            weight: 0.37
            config:
                crop:
                    token_budget: 10
                    crop_weights:
                        contiguous: 0.33
                        spatial: 0.33
                        spatial_interface: 0.33
                dataset_paths:
                    alignments_directory: None 
                    alignment_array_directory: {tmp_path} 
                    dataset_cache_file: {test_dummy_file} 
                    target_structure_file_format: npz
                    target_structures_directory: {tmp_path} 
                    reference_molecule_directory: {tmp_path}
                    template_cache_directory: {tmp_path} 
                    template_file_format: npz
                    ccd_file: None 
        """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        input_dict = config_utils.load_yaml(test_yaml_file)
        actual_config = TrainingDatasetSpec.model_validate(input_dict)
        expected_fields = {
            "name": "dataset1",
            "mode": "train",
            "dataset_class": "WeightedPDBDataset",
            "weight": 0.37,
            "config": {
                "crop": {
                    # based on yaml specified settings
                    "token_budget": 10,
                    "crop_weights": {
                        "contiguous": 0.33,
                        "spatial": 0.33,
                        "spatial_interface": 0.33,
                    },
                    # based on default dataset settings
                    "sample_weights": {
                        "a_prot": 3.0,
                        "a_nuc": 3.0,
                        "a_ligand": 1.0,
                        "w_chain": 0.5,
                        "w_interface": 1.0,
                    },
                },
                "dataset_paths": {
                    "alignments_directory": None,
                    "dataset_cache_file": test_dummy_file,
                    "alignment_array_directory": tmp_path,
                    "target_structures_directory": tmp_path,
                    "target_structure_file_format": "npz",
                    "reference_molecule_directory": tmp_path,
                    "template_cache_directory": tmp_path,
                    "template_file_format": "npz",
                    "ccd_file": None,
                },
            },
        }
        expected_dataset_config = TrainingDatasetSpec.model_validate(expected_fields)
        assert expected_dataset_config == actual_config
        print(expected_dataset_config.config)

    def test_load_protein_monomer_dataset_config(self, tmp_path):
        test_dummy_file = tmp_path / "test.json"
        test_dummy_file.write_text("test")
        test_yaml_str = textwrap.dedent(f"""\
            name: dataset1
            mode: train
            dataset_class: ProteinMonomerDistillationDataset 
            weight: 0.5
            config:
                dataset_paths:
                    alignments_directory: None
                    alignment_array_directory: {tmp_path} 
                    dataset_cache_file: {test_dummy_file} 
                    target_structure_file_format: npz
                    target_structures_directory: {tmp_path} 
                    reference_molecule_directory: {tmp_path}
                    template_cache_directory: {tmp_path} 
                    template_file_format: npz
                    ccd_file: None 
        """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)
        input_dict = config_utils.load_yaml(test_yaml_file)
        actual_config = TrainingDatasetSpec.model_validate(input_dict)

        expected_fields = {
            "name": "dataset1",
            "mode": "train",
            "dataset_class": "ProteinMonomerDistillationDataset",
            "weight": 0.5,
            "config": {
                # Verify that custom loss weights for protein monomer are supported
                "loss": {
                    "loss_weights": {
                        "bond_loss": 0.0,
                        "mse": 4.0,
                        "plddt": 0.0,
                        "pae": 0.0,
                        "pde": 0.0,
                    },
                },
                "dataset_paths": {
                    "alignments_directory": None,
                    "dataset_cache_file": test_dummy_file,
                    "alignment_array_directory": tmp_path,
                    "target_structures_directory": tmp_path,
                    "target_structure_file_format": "npz",
                    "reference_molecule_directory": tmp_path,
                    "template_cache_directory": tmp_path,
                    "template_file_format": "npz",
                    "ccd_file": None,
                },
            },
        }
        expected_config = TrainingDatasetSpec.model_validate(expected_fields)
        assert expected_config == actual_config
