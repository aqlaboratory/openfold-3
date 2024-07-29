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

"""Distogram losses."""

from typing import Dict

import torch

from openfold3.core.loss.loss_utils import softmax_cross_entropy
from openfold3.core.utils.atomize_utils import get_token_representative_atoms
from openfold3.core.utils.tensor_utils import binned_one_hot


def cbeta_distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs,
):
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries**2

    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # FP16-friendly sum. Equivalent to:
    # mean = (torch.sum(errors * square_mask, dim=(-1, -2)) /
    #         (eps + torch.sum(square_mask, dim=(-1, -2))))
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = errors * square_mask
    mean = torch.sum(mean, dim=-1)
    mean = mean / denom[..., None]
    mean = torch.sum(mean, dim=-1)

    # Average over the batch dimensions
    mean = torch.mean(mean)

    return mean


def all_atom_distogram_loss(
    batch: Dict,
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
            [*, N_token, no_bins] Predicted logits
        no_bins:
            Number of bins
        bin_min:
            Minimum bin value
        bin_max:
            Maximum bin value
        eps:
            Small float for numerical stability
    Returns:
        Loss on distogram prediction
    """
    # Extract representative atoms
    rep_x, rep_atom_mask = get_token_representative_atoms(
        batch=batch, x=batch["gt_atom_positions"], atom_mask=batch["gt_atom_mask"]
    )

    # Compute distogram
    d = (
        torch.sum(eps + (rep_x[..., None, :] - rep_x[..., None, :, :]) ** 2, dim=-1)
        ** 0.5
    )

    # Compute binned distogram
    bin_size = (bin_max - bin_min) / no_bins
    v_bins = bin_min + torch.arange(no_bins, device=d.device) * bin_size
    d_b = binned_one_hot(d, v_bins)

    pair_mask = rep_atom_mask[..., None] * rep_atom_mask[..., None, :]

    errors = softmax_cross_entropy(logits, d_b)

    # Compute distogram loss
    loss = torch.sum(errors * pair_mask, dim=(-1, -2)) / (
        torch.sum(pair_mask, dim=(-1, -2)) + eps
    )

    return torch.mean(loss)
