import pytest
from openfold3.model_implementations import registry

class TestLoadPrests:
    def test_model_registry_loads_models(self):
        assert "af2_monomer" in registry.MODEL_REGISTRY
    
    def test_model_preset_loading(self):
        model_config = registry.make_config_with_preset("af2_monomer", "model_1_ptm")
        assert model_config.model.heads.tm.enabled == True
