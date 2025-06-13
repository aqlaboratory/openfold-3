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

import torch
import torch.nn as nn

from openfold3.core.loss.confidence import (
    atom37_experimentally_resolved_loss,
    ca_plddt_loss,
    confidence_loss,
    masked_msa_loss,
    tm_loss,
)
from openfold3.core.loss.diffusion import diffusion_loss
from openfold3.core.loss.distogram import all_atom_distogram_loss, cbeta_distogram_loss
from openfold3.core.loss.loss_utils import compute_renamed_ground_truth
from openfold3.core.loss.structure import (
    chain_center_of_mass_loss,
    fape_loss,
    supervised_chi_loss,
)
from openfold3.core.loss.violation import find_structural_violations, violation_loss

logger = logging.getLogger(__name__)


class AlphaFoldLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def loss(self, batch, out, _return_breakdown=False):
        """
        Rename previous forward() as loss()
        so that can be reused in the subclass
        """
        if "violation" not in out:
            out["violation"] = find_structural_violations(
                batch,
                out["sm"]["positions"][-1],
                **self.config.violation,
            )

        if "renamed_atom14_gt_positions" not in out:
            batch.update(
                compute_renamed_ground_truth(
                    batch,
                    out["sm"]["positions"][-1],
                )
            )

        loss_fns = {
            "distogram": lambda: cbeta_distogram_loss(
                logits=out["distogram_logits"],
                **{**batch, **self.config.distogram},
            ),
            "experimentally_resolved": lambda: atom37_experimentally_resolved_loss(
                logits=out["experimentally_resolved_logits"],
                **{**batch, **self.config.experimentally_resolved},
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                self.config.fape,
            ),
            "plddt_loss": lambda: ca_plddt_loss(
                logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.plddt_loss},
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                **{**batch, **self.config.masked_msa},
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                out["sm"]["angles"],
                out["sm"]["unnormalized_angles"],
                **{**batch, **self.config.supervised_chi},
            ),
            "violation": lambda: violation_loss(
                out["violation"],
                **{**batch, **self.config.violation},
            ),
        }

        if self.config.tm.enabled:
            loss_fns["tm"] = lambda: tm_loss(
                logits=out["tm_logits"],
                **{**batch, **out, **self.config.tm},
            )

        if self.config.chain_center_of_mass.enabled:
            loss_fns["chain_center_of_mass"] = lambda: chain_center_of_mass_loss(
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.chain_center_of_mass},
            )

        cum_loss = 0.0
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                # for k,v in batch.items():
                #    if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                #        logging.warning(f"{k}: is nan")
                # logging.warning(f"{loss_name}: {loss}")
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0.0, requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = batch["aatype"].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses

    def forward(self, out, batch, _return_breakdown=False):
        if not _return_breakdown:
            cum_loss = self.loss(out, batch, _return_breakdown)
            return cum_loss

        cum_loss, losses = self.loss(out, batch, _return_breakdown)
        return cum_loss, losses


class OpenFold3Loss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super().__init__()

        # Loss config
        self.config = config

    def loss(self, batch, output, _return_breakdown=False):
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

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses

    def forward(self, batch, output, _return_breakdown=False):
        """
        Args:
            batch:
                Dict containing input tensors
            output:
                Dict containing output tensors
                (see openfold3/openfold3/model_implementations/af3_all_atom/model.py
                for a list items in batch and output)
            _return_breakdown:
                If True, also return a dictionary of individual
                loss components
        Returns:
            cum_loss: Scalar tensor representing the total loss
            losses: Dict containing individual loss components
        """
        if not _return_breakdown:
            cum_loss = self.loss(batch, output, _return_breakdown)
            return cum_loss

        cum_loss, losses = self.loss(batch, output, _return_breakdown)
        return cum_loss, losses
