from typing import Optional

import torch


def get_bin_centers(bin_min: float, bin_max: float, no_bins: int, device, dtype) -> torch.Tensor:
    width = (bin_max - bin_min) / float(no_bins)
    boundaries = torch.linspace(bin_min, bin_max, steps=no_bins+1, device=device, dtype=dtype)
    return boundaries[:-1] + 0.5 * width

def probs_to_expected_error(
    probs: torch.Tensor,
    bin_min: float,
    bin_max: float,
    no_bins: int,
    **kwargs
) -> torch.Tensor:
    """
    Computing expectation of error from binned probability. Used for pLDDT, pAE, and pTM.
    """
    bin_centers = get_bin_centers(bin_min, bin_max, no_bins, device=probs.device, dtype=probs.dtype)
    expectation = torch.sum(probs * bin_centers, dim=-1)
    return expectation

# TODO We have this function since validation_all_atom calls this without access to plddt bin config,
# But ultimately that function should get access to bin config
def compute_plddt(logits):
    return probs_to_expected_error(
        torch.softmax(logits, dim=-1),
        bin_min=0, bin_max=1.0, no_bins=50
    )

def compute_global_predicted_distance_error(
    pde: torch.Tensor,
    logits: torch.Tensor,
    bin_min: int = 2,
    bin_max: int = 22,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs
) -> torch.Tensor:
    """Computes the gPDE metric as defined in AF3 SI 5.7 (16)"""
    device = pde.device
    probs = torch.softmax(logits, dim=-1)

    # Bins range from 2 to 22 Å
    distogram_bin_ends = torch.linspace(bin_min, bin_max, no_bins + 1, device=device)[1:]
    # boolean mask for bins <= 8 Å
    distogram_bins_8A = distogram_bin_ends <= 8.0  
    # probability of contact between tokens i and j is defined as sum of probability across bins <= 8 Å
    contact_probs = torch.sum(probs[..., distogram_bins_8A], dim=-1)
    
    gpde = torch.sum(contact_probs * pde, dim=[-2, -1]) / (torch.sum(contact_probs, dim=[-2, -1]) + eps)

    return gpde, contact_probs

def compute_ptm(
    logits: torch.Tensor,
    bin_min: float = 0.0,
    bin_max: float = 32.0,
    no_bins: int = 64,
    has_frame: Optional[torch.Tensor] = None,
    D_mask: Optional[torch.Tensor] = None,  # [*, N] bool – membership set D
    asym_id: Optional[torch.Tensor] = None,  # [*, N] int – required if interface=True
    interface: bool = False,  # False=pTM, True=ipTM
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Predicted TM (pTM) / interface predicted TM (ipTM) per sample.

    Args:
        logits:
            Pair-distance logits with bins in [*, N, N, B]  (bins last)
        bin_max:
            Upper bound (Å) for the distance bins (AF3: 32).
        no_bins:
            Number of distance bins (AF3: 64).
        has_frame:
            [*, N] boolean mask of tokens with valid frames (outer max over i).
            If None, all tokens are considered valid.
        D_mask:
            [*, N] boolean mask selecting the set D to average over.
            If None, D = all tokens.
        asym_id:
            [*, N] chain IDs. Required when `interface=True` to exclude same-chain
            pairs for the inner average over j.
        interface:
            If True, compute ipTM (exclude same-chain pairs); otherwise pTM.
        eps:
            Numerical stability epsilon used in denominators.

    Returns:
        score: [*] tensor with the pTM/ipTM score per leading index (batch/sample).

    Notes:
        Implements AF3 Supp. §5.9.1, Eqs. (17-18).
    """
    x = logits
    device, dtype = x.device, x.dtype
    *batch_dims, num_tokens, _,  = x.shape

    if D_mask is None:
        D_mask = torch.ones(*batch_dims, num_tokens, device=device, dtype=torch.bool)
    else:
        D_mask = D_mask.expand(*batch_dims, -1)
        D_mask = D_mask.to(device=device, dtype=torch.bool)

    if has_frame is None:
        has_frame = torch.ones(*batch_dims, num_tokens, device=device, dtype=torch.bool)
    else:
        has_frame = has_frame.to(device=device, dtype=torch.bool)

    if interface and asym_id is None:
        raise ValueError("asym_id is required when interface=True")
    if asym_id is not None:
        asym_id = asym_id.to(device=device)

    D_size = D_mask.sum(dim=-1).clamp_min(1).to(dtype)
    clipped = torch.maximum(D_size, torch.tensor(19.0, device=device, dtype=dtype))
    d0 = 1.24 * (clipped - 15.0).clamp_min(0).pow(1.0 / 3.0) - 1.8 # [*, num_tokens]

    bin_centers = get_bin_centers(bin_min, bin_max, no_bins, device, dtype)
    bin_centers = bin_centers.expand(*batch_dims, -1)
    tm_per_bin = 1.0 / (1.0 + (bin_centers/d0) ** 2) # [*, no_bins]

    probs = torch.softmax(x, dim=-1)
    exp_tm_ij = torch.sum(probs * tm_per_bin, dim=-1)

    if interface:
        same_chain = asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)
        M_ij = (~same_chain) & D_mask.unsqueeze(-2)
    else:
        M_ij = D_mask.unsqueeze(-2).expand(*batch_dims, num_tokens, num_tokens)

    M_ij_f = M_ij.to(exp_tm_ij.dtype)
    exp_tm_ij = exp_tm_ij * M_ij_f

    denom_j = M_ij_f.sum(dim=-1).clamp_min(eps)
    per_i = exp_tm_ij.sum(dim=-1) / denom_j

    valid_i = has_frame & D_mask
    per_i_masked = torch.where(valid_i, per_i, torch.full_like(per_i, float("-inf")))
    return per_i_masked.max(dim=-1).values
