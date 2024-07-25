from pathlib import Path

from openfold3.core.runners.model_runner import ModelRunner
from openfold3.model_implementations.registry import register_model
from openfold3.model_implementations.af2_monomer.config.base_config import config
from openfold3.model_implementations.af2_monomer.model import AlphaFold

REFERENCE_CONFIG_PATH = Path(__file__).parent.resolve() / "config/reference_config.yml"


@register_model("af2_monomer", config, REFERENCE_CONFIG_PATH)
class AlphaFoldMonomer(ModelRunner):
    def __init__(self, config):
        self.model = AlphaFold
