import torch
import torch.nn as nn

from openfold3.core.loss.confidence import ConfidenceLoss
from openfold3.core.loss.diffusion import DiffusionLoss

# import DistogramLoss from openfold3.core.loss.distogram


class AlphaFold3Loss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.confidence_loss = ConfidenceLoss(config)
        self.diffusion_loss = DiffusionLoss(config)
        self.DitogramLoss = DitogramLoss(config)

    def forward(self, batch, output):
        alpha_confidence = self.config.loss.alpha_confidence
        alpha_diffusion = self.config.loss.alpha_diffusion
        alpha_distogram = self.config.loss.alpha_distogram

        loss = (
            alpha_confidence * self.confidence_loss(batch, output)
            + alpha_diffusion * self.diffusion_loss(batch, output)
            + alpha_distogram * distogram_loss(output),
        )
        return loss
