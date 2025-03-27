import textwrap

import pytest  # noqa: F401  - used for pytest tmp fixture

from openfold3.core.config import config_utils
from openfold3.projects.af3_all_atom.project_entry import AF3ProjectEntry, ModelUpdate


class TestAF3ProjectConfigGeneration:
    def test_load_runner_arguments(self, tmp_path):
        test_dummy_file = tmp_path / "test.json"
        test_dummy_file.write_text("test")
        test_yaml_str = textwrap.dedent(f"""\
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

        runner_args = config_utils.load_yaml(test_yaml_file)
        project_entry = AF3ProjectEntry()

        dataset_specs = project_entry.combine_dataset_paths_with_configs(
            runner_args.get("dataset_paths"), runner_args.get("dataset_configs")
        )
        assert len(dataset_specs) == 2
        assert dataset_specs[0].dataset_class == "WeightedPDBDataset"
        assert dataset_specs[1].dataset_class == "ValidationPDBDataset"

        weighted_pdb_spec = dataset_specs[0]
        assert weighted_pdb_spec.weight == 1
        assert weighted_pdb_spec.config.crop.token_budget == 640

        model_update = ModelUpdate.model_validate(runner_args.get("model_update"))
        print(f"{model_update=}")
        model_config = project_entry.get_model_config_with_update(model_update)

        assert model_config.settings.model_selection_weight_scheme == "fine_tuning"
        assert (
            model_config.architecture.msa.msa_module.transition_ckpt_chunk_size == None
        )
        assert model_config.architecture.shared.diffusion.no_samples == 32

    def test_bad_model_update_fails(self):
        model_update = ModelUpdate(custom={"nonexistant_field": "bad"})

        project_entry = AF3ProjectEntry()

        with pytest.raises(KeyError, match="config is locked"):
            project_entry.get_model_config_with_update(model_update)
