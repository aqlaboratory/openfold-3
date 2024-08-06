from pathlib import Path

from openfold3.core.loss.loss_module import AlphaFold3Loss
from openfold3.core.runners.model_runner import ModelRunner
from openfold3.model_implementations.af3_all_atom.config.base_config import config
from openfold3.model_implementations.af3_all_atom.model import AlphaFold3
from openfold3.model_implementations.registry import register_model

REFERENCE_CONFIG_PATH = Path(__file__).parent.resolve() / "config/reference_config.yml"


@register_model("af3_all_atom", config, REFERENCE_CONFIG_PATH)
class AlphaFold3AllAtom(ModelRunner):
    def __init__(self, model_config):
        super().__init__(AlphaFold3, model_config)
        self.loss = AlphaFold3Loss(config=model_config.loss)
    
    def _compute_validation_metrics(
            self, batch, outputs, superimposition_metrics=False):
        return {}
