# Copyright 2026 AlQuraishi Laboratory
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

"""Distogram losses."""

import torch

from openfold3.core.loss.loss_utils import loss_masked_batch_mean, softmax_cross_entropy
from openfold3.core.utils.atomize_utils import get_token_representative_atoms
from openfold3.core.utils.tensor_utils import binned_one_hot


def all_atom_distogram_loss(
    batch: dict,
    logits: torch.Tensor,
    no_bins: int,
    bin_min: float,
    bin_max: float,
    eps: float,
    **kwargs,
):
    """
    Computes loss on distogram prediction (Subsection 4.4).

    Args:
        batch:
            Feature dictionary
        logits:
            [*, N_token, N_token, no_bins] Predicted logits
        no_bins:
            Number of bins
        bin_min:
            Minimum bin value
        bin_max:
            Maximum bin value
        eps:
            Small float for numerical stability
    Returns:
        mean_loss:
            Distogram loss
        loss_breakdown:
            Dict of individual component losses (unweighted)
    """
    # Extract representative atoms
    rep_x, rep_atom_mask = get_token_representative_atoms(
        batch=batch,
        x=batch["ground_truth"]["atom_positions"],
        atom_mask=batch["ground_truth"]["atom_resolved_mask"],
    )

    # Compute distogram
    d = torch.sqrt(
        torch.sum((rep_x[..., None, :] - rep_x[..., None, :, :]) ** 2, dim=-1)
    )

    # Compute binned distogram
    bin_size = (bin_max - bin_min) / no_bins
    bin_min_offset = bin_min + bin_size / 2
    v_bins = bin_min_offset + torch.arange(no_bins, device=d.device) * bin_size
    d_b = binned_one_hot(d, v_bins).to(dtype=d.dtype)

    pair_mask = (rep_atom_mask[..., None] * rep_atom_mask[..., None, :]).bool()
    errors = softmax_cross_entropy(logits, d_b)

    # Compute distogram loss
    loss = torch.sum(errors * pair_mask, dim=(-1, -2)) / (
        torch.sum(pair_mask, dim=(-1, -2)) + eps
    )

    distogram_weight = batch["loss_weights"]["distogram"]

    # Calculate unweighted batch mean
    # Mask out samples where the loss is disabled
    loss_breakdown = {}
    if distogram_weight.any():
        mean_loss_unweighted = loss_masked_batch_mean(
            loss=loss.detach().clone(),
            weight=distogram_weight,
            apply_weight=False,
            eps=eps,
        )
        loss_breakdown = {"distogram_loss": mean_loss_unweighted}

    # Apply loss weight in batch mean
    mean_loss = loss_masked_batch_mean(
        loss=loss,
        weight=distogram_weight,
        apply_weight=True,
        eps=eps,
    )

    return mean_loss, loss_breakdown
