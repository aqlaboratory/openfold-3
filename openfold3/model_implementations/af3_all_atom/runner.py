from pathlib import Path

from openfold3.core.runners.model_runner import ModelRunner
from openfold3.model_implementations.af3_all_atom.config.base_config import config
from openfold3.model_implementations.af3_all_atom.model import AlphaFold3
from openfold3.model_implementations.registry import register_model

REFERENCE_CONFIG_PATH = Path(__file__).parent.resolve() / "config/reference_config.yml"


@register_model("af3_all_atom", config, REFERENCE_CONFIG_PATH)
class AlphaFold3AllAtom(ModelRunner):
    def __init__(self, model_config):
        super().__init__(AlphaFold3, model_config)