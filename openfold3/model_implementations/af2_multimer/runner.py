from pathlib import Path

from openfold3.core.runners.model_runner import ModelRunner
from openfold3.model_implementations.af2_multimer.config.base_config import config
from openfold3.model_implementations.af2_multimer.model import AlphaFold as AFMultimer
from openfold3.model_implementations.registry import register_model

REFERENCE_CONFIG_PATH = Path(__file__).parent.resolve() / "config/reference_config.yml"


@register_model("af2_multimer", config, REFERENCE_CONFIG_PATH)
class AlphaFoldMultimer(ModelRunner):
    def __init__(self, model_config):
        super().__init__(AFMultimer, model_config)
