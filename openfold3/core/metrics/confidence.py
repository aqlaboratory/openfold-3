from typing import Optional

import torch


def get_bin_centers(
    bin_min: float, bin_max: float, no_bins: int, device, dtype
) -> torch.Tensor:
    width = (bin_max - bin_min) / float(no_bins)
    boundaries = torch.linspace(
        bin_min, bin_max, steps=no_bins + 1, device=device, dtype=dtype
    )
    return boundaries[:-1] + 0.5 * width


def probs_to_expected_error(
    probs: torch.Tensor, bin_min: float, bin_max: float, no_bins: int, **kwargs
) -> torch.Tensor:
    """
    Computing expectation of error from binned probability
    """
    bin_centers = get_bin_centers(
        bin_min, bin_max, no_bins, device=probs.device, dtype=probs.dtype
    )
    expectation = torch.sum(probs * bin_centers, dim=-1)
    return expectation


# TODO We have this function since validation_all_atom calls this without access
# to plddt bin config, But ultimately that function should get access to bin config
def compute_plddt(logits):
    return probs_to_expected_error(
        torch.softmax(logits, dim=-1), bin_min=0, bin_max=1.0, no_bins=50
    )


def compute_global_predicted_distance_error(
    pde: torch.Tensor,
    logits: torch.Tensor,
    bin_min: int = 2,
    bin_max: int = 22,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    """Computes the gPDE metric as defined in AF3 SI 5.7 (16)"""
    device = pde.device
    probs = torch.softmax(logits, dim=-1)

    # Bins range from 2 to 22 Å
    distogram_bin_ends = torch.linspace(bin_min, bin_max, no_bins + 1, device=device)[
        1:
    ]
    # boolean mask for bins <= 8 Å
    distogram_bins_8A = distogram_bin_ends <= 8.0
    # probability of contact between tokens i and j is
    # defined as sum of probability across bins <= 8 Å
    contact_probs = torch.sum(probs[..., distogram_bins_8A], dim=-1)

    gpde = torch.sum(contact_probs * pde, dim=[-2, -1]) / (
        torch.sum(contact_probs, dim=[-2, -1]) + eps
    )

    return gpde, contact_probs


def compute_ptm(
    logits: torch.Tensor,
    has_frame: torch.Tensor,
    bin_min: int,
    bin_max: int,
    no_bins: int,
    mask_i: torch.Tensor,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Predicted TM (pTM) / interface predicted TM (ipTM).
    Implements AF3 SI 5.9.1, Eqs. (17-18).
    May compute multiple samples with same mask information.

    Args:
        logits:
            Pair-distance logits with bins in
            [num_samples, num_tokens, num_tokens, no_bins]
        has_frame:
            [num_samples, num_tokens] boolean mask of tokens with valid frames
            (outer max over i).
        bin_min:
            Lower bound (Å) for the distance bins (AF3: 0).
        bin_max:
            Upper bound (Å) for the distance bins (AF3: 32).
        no_bins:
            Number of distance bins (AF3: 64).
        mask_i:
            [num_tokens] boolean mask indicating the set D of tokens that is considered
        asym_id:
            [num_tokens] chain IDs. Required when `interface=True` to exclude same-chain
            pairs for the inner average over j.
        interface:
            If True, compute ipTM (exclude same-chain pairs); otherwise pTM.
        eps:
            Numerical stability epsilon used in denominators.

    Returns:
       pTM/ipTM score
    """
    device, dtype = logits.device, logits.dtype
    mask_i = mask_i.to(device=device, dtype=torch.bool)

    if interface and asym_id is None:
        raise ValueError("asym_id is required when interface=True")

    if asym_id is not None:
        asym_id = asym_id[mask_i].to(device=device)

    # Compute bin weights
    num_tokens_considered = mask_i.sum().clamp_min(1).to(dtype)
    clipped = torch.maximum(
        num_tokens_considered, torch.tensor(19.0, device=device, dtype=dtype)
    )
    d0 = 1.24 * (clipped - 15.0).clamp_min(0).pow(1.0 / 3.0) - 1.8

    bin_centers = get_bin_centers(bin_min, bin_max, no_bins, device, dtype)
    bin_weight = 1.0 / (1.0 + (bin_centers / d0) ** 2)

    # Subset to token mask
    logits = logits[:, mask_i, ...]
    logits = logits[..., mask_i, :]
    has_frame = has_frame[:, mask_i].bool()
    probs = torch.softmax(logits, dim=-1)
    ptm_ij = torch.sum(probs * bin_weight, dim=-1)

    # Subset tokens j to different chain from i if interface=True
    if interface:
        pair_mask = asym_id.unsqueeze(-1) != asym_id.unsqueeze(-2)
        tm_i = (ptm_ij * pair_mask).sum(dim=-1) / pair_mask.sum(dim=-1).clamp_min(eps)
    else:
        tm_i = ptm_ij.sum(dim=-1) / num_tokens_considered

    tm_i = tm_i.masked_fill(~has_frame, 0.0)
    return tm_i.max(dim=-1).values
