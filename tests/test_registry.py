import textwrap

import pytest  # noqa: F401  - used for pytest tmp fixture
import torch

from openfold3.core.config import config_utils
from openfold3.core.runners.model_runner import ModelRunner, ModelRunnerNotRegisteredError 
from openfold3.model_implementations.registry import (
    MODEL_REGISTRY,
    get_lightning_module,
    make_config_with_preset,
    make_model_config,
)


class TestLoadPresets:
    def test_model_registry_loads_models(self):
        expected_model_entries = {"af2_monomer", "af2_multimer", "af3_all_atom"}
        assert set(MODEL_REGISTRY.keys()) == expected_model_entries

    def test_model_preset_loading(self):
        model_config = make_config_with_preset("af2_monomer", "model_1_ptm")
        assert model_config.loss.tm.weight == 0.1

    def test_yaml_overwrite_preset(self, tmp_path):
        # yaml which would change weight for monomer model_1_ptm tm loss
        test_yaml_str = textwrap.dedent("""\
        model_preset: "model_1_ptm"

        model_update:
            loss:
                tm:
                    weight: 7""")
        test_yaml_file = tmp_path / "test.yml"
        with open(test_yaml_file, "w") as f:
            f.write(test_yaml_str)
        loaded_yaml_dict = config_utils.load_yaml(test_yaml_file)
        print(loaded_yaml_dict)
        assert "model_update" in loaded_yaml_dict

        overwritten_config = make_model_config("af2_monomer", test_yaml_file)
        assert overwritten_config.loss.tm.weight == 7

    def test_registry_model_loads(self):
        # TODO: Change loaded preset to load a smaller test preset
        test_multimer_config = make_config_with_preset(
            "af2_multimer", "model_1_multimer_v3"
        )
        multimer_runner = get_lightning_module(test_multimer_config)
        assert multimer_runner.model.get_submodule("input_embedder")

class TestModelRegistry:
    def test_unregistered_model_runner_raises_error(self):
        dummy_model = torch.nn.Linear(5, 7)
        config = {'model_name': 'unregistered'}
        class UnregisteredModelRunner(ModelRunner):
            def __init__(self, model_config):
                super().__init__(dummy_model, model_config)

        with pytest.raises(ModelRunnerNotRegisteredError) as exc:
            _ = UnregisteredModelRunner(config)