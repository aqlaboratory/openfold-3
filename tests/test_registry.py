import textwrap

import ml_collections as mlc
import pytest  # noqa: F401  - used for pytest tmp fixture

from openfold3.core.config import config_utils
from openfold3.projects import registry
from tests import compare_utils


class TestLoadPresets:
    def test_model_registry_loads_models(self):
        # TODO: Convert other models to new config format
        expected_model_entries = {"af2_monomer", "af2_multimer", "af3_all_atom"}
        assert set(registry.PROJECT_REGISTRY.keys()) == expected_model_entries

    def test_config_preset_loading(self):
        project_entry = registry.get_project_entry("af3_all_atom")
        project_config = registry.make_config_with_presets(
            project_entry, ["finetune1", "train"]
        )
        # From base preset
        assert project_config.model.settings.ema.decay == 0.999
        # From finetune1 preset
        assert project_config.extra_configs.loss_weight_modes.default.bond == 4.0
        # From train preset
        assert project_config.model.settings.use_deepspeed_evo_attention

    def test_yaml_overwrite_preset(self, tmp_path):
        test_yaml_str = textwrap.dedent("""\
        project_type: af3_all_atom
        presets: 
          - finetune1

        config_update:
            model:
                architecture:
                    shared:
                        c_s: 47

            extra_configs:
                loss_weight_modes:
                    default:
                        mse: 7.0
        """)
        test_yaml_file = tmp_path / "runner.yml"
        with open(test_yaml_file, "w") as f:
            f.write(test_yaml_str)
        runner_args = mlc.ConfigDict(config_utils.load_yaml(test_yaml_file))

        project_entry = registry.get_project_entry(runner_args.project_type)
        project_config = registry.make_config_with_presets(
            project_entry, runner_args.presets
        )
        project_config.update(runner_args.config_update)

        # Test update to shared architecture passes to layer
        assert project_config.model.architecture.input_embedder.c_s == 47
        assert project_config.model.architecture.input_embedder.c_z == 128
        assert project_config.extra_configs.loss_weight_modes.default.bond == 4.0
        assert project_config.extra_configs.loss_weight_modes.default.mse == 7.0

    @compare_utils.skip_unless_cuda_available()
    def test_registry_model_loads(self):
        project_entry = registry.get_project_entry("af3_all_atom")
        project_config = registry.make_config_with_presets(
            project_entry,
            ["initial_training"],
        )
        model_runner = project_entry.model_runner(project_config.model, _compile=False)
        assert model_runner.model.get_submodule("input_embedder")
