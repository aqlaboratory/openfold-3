import torch.nn as nn

from openfold3.core.loss.confidence import ConfidenceLoss
from openfold3.core.loss.diffusion import DiffusionLoss
from openfold3.core.loss.distogram import DistogramLoss


class AlphaFold3Loss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.confidence_loss = ConfidenceLoss(config)
        self.diffusion_loss = DiffusionLoss(config)
        self.distogram_loss = DistogramLoss(config)

    def forward(self, batch, output):
        """
        Args:
            batch: dict containing input tensors
            output: dict containing output tensors
            (see openfold3/openfold3/model_implementations/af3_all_atom/model.py
            for a list items in batch and output)
        Returns:
            loss: scalar tensor representing the total loss
        """
        alpha_confidence = self.config.loss.alpha_confidence
        alpha_diffusion = self.config.loss.alpha_diffusion
        alpha_distogram = self.config.loss.alpha_distogram

        loss = (
            alpha_confidence * self.confidence_loss(batch, output)
            + alpha_diffusion * self.diffusion_loss(batch, output)
            + alpha_distogram * self.distogram_loss(batch, output)
        )
        return loss
