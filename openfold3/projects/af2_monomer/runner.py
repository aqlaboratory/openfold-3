from atexit import register
from pathlib import Path

import torch

from openfold3.core.runners.model_runner import ModelRunner
from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold3.projects.af2_monomer.config.base_config import config
from openfold3.projects.af2_monomer.model import AlphaFold
from openfold3.projects.registry import register_project

REFERENCE_CONFIG_PATH = Path(__file__).parent.resolve() / "config/reference_config.yml"


@register_project("af2_monomer", config, REFERENCE_CONFIG_PATH)
class AlphaFoldMonomer(ModelRunner):
    def __init__(self, model_config):
        super().__init__(AlphaFold, model_config)

    def configure_optimizers(
        self,
        learning_rate: float = 1e-3,
        eps: float = 1e-5,
    ) -> torch.optim.Adam:
        # Ignored as long as a DeepSpeed optimizer is configured
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=eps)

        if self.last_lr_step != -1:
            for group in optimizer.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = learning_rate

        lr_scheduler = AlphaFoldLRScheduler(optimizer, last_epoch=self.last_lr_step)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "name": "AlphaFoldLRScheduler",
            },
        }
