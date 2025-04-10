
import torch

from openfold3.core.loss.loss_module import AlphaFoldLoss
from openfold3.core.metrics.confidence import (
    compute_plddt,
    compute_predicted_aligned_error,
    compute_ptm,
)
from openfold3.core.runners.model_runner import ModelRunner
from openfold3.core.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold3.legacy.af2_monomer.model import AlphaFold


class AlphaFoldMonomer(ModelRunner):
    def __init__(self, model_config, _compile=True):
        super().__init__(model_class=AlphaFold, config=model_config, _compile=_compile)

        self.loss = (
            torch.compile(AlphaFoldLoss(config=model_config.loss))
            if _compile
            else AlphaFoldLoss(config=model_config.loss)
        )

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

    def _compute_confidence_scores(self, batch: dict, outputs: dict) -> dict:
        """Compute confidence metrics. This function is called during inference.

        Args:
            batch (dict):
                Input feature dictionary
            outputs (dict):
                Output dictionary containing the predicted trunk embeddings,
                all-atom positions, and distogram head logits

        Returns:
            confidence_scores (dict):
                Dict containing the following confidence measures:
                pLDDT, PAE, pTM
        """
        confidence_scores = {}

        # Required for relaxation later on
        confidence_scores["plddt"] = compute_plddt(outputs["lddt_logits"])

        if self.config.model.heads.tm.enabled:
            confidence_scores["ptm_score"] = compute_ptm(
                outputs["tm_logits"], **self.config.confidence.ptm
            )

            confidence_scores.update(
                compute_predicted_aligned_error(
                    outputs["tm_logits"],
                    **self.config.confidence.pae,
                )
            )

        return confidence_scores
