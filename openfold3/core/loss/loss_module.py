# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main loss modules."""

import logging

import torch.nn as nn

from openfold3.core.loss.confidence import confidence_loss
from openfold3.core.loss.diffusion import diffusion_loss
from openfold3.core.loss.distogram import all_atom_distogram_loss

logger = logging.getLogger(__name__)


class OpenFold3Loss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super().__init__()

        # Loss config
        self.config = config

    def loss(self, batch, output):
        cum_loss = 0.0
        losses = {}

        l_confidence, l_confidence_breakdown = confidence_loss(
            batch=batch, output=output, **self.config.confidence
        )
        losses.update(l_confidence_breakdown)

        if l_confidence_breakdown:
            losses["confidence_loss"] = l_confidence.detach().clone()

        # Weighted in confidence_loss()
        cum_loss = cum_loss + l_confidence

        # Run diffusion loss only if diffusion training and losses are enabled
        atom_positions_diffusion = output.get("atom_positions_diffusion")
        if atom_positions_diffusion is not None:
            l_diffusion, l_diffusion_breakdown = diffusion_loss(
                batch=batch,
                x=atom_positions_diffusion,
                t=output["noise_level"],
                **self.config.diffusion,
            )
            losses.update(l_diffusion_breakdown)

            if l_diffusion_breakdown:
                losses["diffusion_loss"] = l_diffusion.detach().clone()

            # Weighted in diffusion_loss()
            cum_loss = cum_loss + l_diffusion

        l_distogram, l_distogram_breakdown = all_atom_distogram_loss(
            batch=batch, logits=output["distogram_logits"], **self.config.distogram
        )
        losses.update(l_distogram_breakdown)

        if l_distogram_breakdown:
            losses["scaled_distogram_loss"] = l_distogram.detach().clone()

        # Weighted in all_atom_distogram_loss()
        cum_loss = cum_loss + l_distogram

        losses["loss"] = cum_loss.detach().clone()

        return cum_loss, losses

    def forward(self, batch, output, _return_breakdown=False):
        """
        Args:
            batch:
                Dict containing input tensors
            output:
                Dict containing output tensors
                (see openfold3/openfold3/model_implementations/of3_all_atom/model.py
                for a list items in batch and output)
            _return_breakdown:
                If True, also return a dictionary of individual
                loss components
        Returns:
            cum_loss: Scalar tensor representing the total loss
            losses: Dict containing individual loss components
        """
        loss, loss_breakdown = self.loss(batch, output)

        if not _return_breakdown:
            return loss

        return loss, loss_breakdown
