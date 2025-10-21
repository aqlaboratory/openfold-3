# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import textwrap

import pytest  # noqa: F401  - used for pytest tmp fixture

from openfold3.core.config import config_utils
from openfold3.core.data.framework.data_module import (
    DataModuleConfig,
    InferenceDataModule,
)
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceDatasetSpec,
    InferenceJobConfig,
    TrainingDatasetPaths,
    TrainingDatasetSpec,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)


class TestOF3DatasetConfigConstruction:
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
                    template_structures_directory: {tmp_path} 
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
                    "template_structures_directory": tmp_path,
                    "template_file_format": "npz",
                    "ccd_file": None,
                },
            },
        }
        expected_dataset_config = TrainingDatasetSpec.model_validate(expected_fields)
        assert expected_dataset_config == actual_config

    def test_load_protein_monomer_dataset_config(self, tmp_path):
        test_dummy_file = tmp_path / "test.json"
        test_dummy_file.write_text("test")
        test_yaml_str = textwrap.dedent(f"""\
            name: dataset1
            mode: train
            dataset_class: ProteinMonomerDataset 
            weight: 0.5
            config:
                dataset_paths:
                    alignments_directory: none
                    alignment_array_directory: {tmp_path} 
                    dataset_cache_file: {test_dummy_file} 
                    target_structure_file_format: npz
                    target_structures_directory: {tmp_path} 
                    reference_molecule_directory: {tmp_path}
                    template_cache_directory: {tmp_path} 
                    template_structures_directory: {tmp_path} 
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
            "dataset_class": "ProteinMonomerDataset",
            "weight": 0.5,
            "config": {
                # Verify that custom loss weights for protein monomer are supported
                "loss": {
                    "loss_weights": {
                        "bond": 0.0,
                        "mse": 4.0,
                        "experimentally_resolved": 0.0,
                        "plddt": 0.0,
                        "pae": 0.0,
                        "pde": 0.0,
                    },
                },
                "dataset_paths": {
                    "alignment_array_directory": tmp_path,
                    "dataset_cache_file": test_dummy_file,
                    "target_structures_directory": tmp_path,
                    "target_structure_file_format": "npz",
                    "reference_molecule_directory": tmp_path,
                    "template_cache_directory": tmp_path,
                    "template_structures_directory": tmp_path,
                    "template_file_format": "npz",
                    "ccd_file": None,
                },
            },
        }
        expected_config = TrainingDatasetSpec.model_validate(expected_fields)
        assert expected_config == actual_config

    def test_load_disordered_pdb_dataset_config(self, tmp_path):
        test_dummy_file = tmp_path / "test.json"
        test_dummy_file.write_text("test")
        test_yaml_str = textwrap.dedent(f"""\
            name: dataset1
            mode: train
            dataset_class: DisorderedPDBDataset 
            weight: 0.02
            config:
                dataset_paths:
                    alignments_directory: none
                    alignment_array_directory: {tmp_path} 
                    dataset_cache_file: {test_dummy_file} 
                    target_structure_file_format: npz
                    target_structures_directory: {tmp_path} 
                    reference_molecule_directory: {tmp_path}
                    template_cache_directory: {tmp_path} 
                    template_structures_directory: {tmp_path} 
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
            "dataset_class": "DisorderedPDBDataset",
            "weight": 0.02,
            "config": {
                "disable_non_protein_diffusion_weights": True,
                "dataset_paths": {
                    "alignment_array_directory": tmp_path,
                    "dataset_cache_file": test_dummy_file,
                    "target_structures_directory": tmp_path,
                    "target_structure_file_format": "npz",
                    "reference_molecule_directory": tmp_path,
                    "template_cache_directory": tmp_path,
                    "template_structures_directory": tmp_path,
                    "template_file_format": "npz",
                    "ccd_file": None,
                },
                "loss": {
                    "loss_weights": {
                        "bond": 0.0,
                        "mse": 4.0,
                        "experimentally_resolved": 0.0,
                        "plddt": 0.0,
                        "pae": 0.0,
                        "pde": 0.0,
                    },
                },
            },
        }
        expected_config = TrainingDatasetSpec.model_validate(expected_fields)
        assert expected_config == actual_config

    def test_error_if_no_alignment_path_set(self, tmp_path):
        test_dummy_file = tmp_path / "test.json"
        test_dummy_file.write_text("test")
        with pytest.raises(ValueError, match="Exactly one"):
            TrainingDatasetPaths.model_validate(
                {
                    "alignments_directory": None,
                    "alignment_db_directory": None,
                    "alignment_array_directory": None,
                    "dataset_cache_file": test_dummy_file,
                    "target_structures_directory": tmp_path,
                    "target_structure_file_format": "npz",
                    "reference_molecule_directory": tmp_path,
                }
            )

    def test_error_if_two_alignment_paths_set(self, tmp_path):
        test_dummy_file = tmp_path / "test.json"
        test_dummy_file.write_text("test")
        with pytest.raises(ValueError, match="Exactly one"):
            TrainingDatasetPaths.model_validate(
                {
                    "alignments_directory": tmp_path,
                    "alignment_db_directory": tmp_path,
                    "alignment_array_directory": None,
                    "dataset_cache_file": test_dummy_file,
                    "target_structures_directory": tmp_path,
                    "target_structure_file_format": "npz",
                    "reference_molecule_directory": tmp_path,
                }
            )


class TestInferenceConfigConstruction:
    def test_inference_config_loading(self, tmp_path):
        inference_set = InferenceQuerySet.model_validate(
            {
                "queries": {
                    "query_1": {
                        "use_msas": False,
                        "chains": [
                            {
                                "molecule_type": "protein",
                                "chain_ids": "A",
                                "sequence": "PVLSCGEWQCL",
                                "main_msa_file_paths": None,
                                "paired_msa_file_paths": None,
                            },
                            {
                                "molecule_type": "ligand",
                                "chain_ids": ["F", "G", "H"],
                                "ccd_codes": "ATP",
                            },
                            {
                                "molecule_type": "ligand",
                                "chain_ids": "Z",
                                "smiles": "CC(=O)OC1C[NH+]2CCC1CC2",
                            },
                        ],
                    },
                },
            }
        )

        inference_config = InferenceJobConfig(
            query_set=inference_set,
            template_preprocessor_settings=TemplatePreprocessorSettings(mode="predict"),
        )
        inference_spec = InferenceDatasetSpec(config=inference_config)
        dataset_specs = [inference_spec]

        data_config = DataModuleConfig(
            datasets=dataset_specs,
            batch_size=1,
            epoch_len=1,
            num_epochs=1,
        )

        data_module = InferenceDataModule(
            data_config,
            use_msa_server=False,
            use_templates=False,
        )

        data_module.prepare_data()
        data_module.setup()

        dataloader = data_module.predict_dataloader()

        # Option: Test feature generation once create_features is written
        # Would require a real msa file
        assert len(dataloader) == 1
        it = iter(dataloader)
        next(it)
