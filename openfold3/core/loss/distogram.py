from typing import Dict

import torch

from openfold3.core.utils.atomize_utils import get_token_representative_atoms
from openfold3.core.utils.tensor_utils import binned_one_hot


def distogram_loss(
    batch: Dict,
    p_b: torch.Tensor,
    no_bins: int,
    bin_min: float,
    bin_max: float,
    eps: float,
    **kwargs
):
    """
    Computes loss on distogram prediction (Subsection 4.4).

    Args:
        batch:
            Feature dictionary
        p_b:
            [*, N_token, no_bins] Predicted probabilities
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

    # Compute distogram loss
    pair_mask = rep_atom_mask[..., None] * rep_atom_mask[..., None, :]
    loss = -torch.sum(
        torch.sum(d_b * torch.log(p_b + eps), dim=-1) * pair_mask, dim=(-1, -2)
    ) / (torch.sum(pair_mask, dim=(-1, -2)) + eps)

    return torch.mean(loss)
