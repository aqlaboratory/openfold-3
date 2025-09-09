from typing import Optional

import torch


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )

    return bin_centers


def _calculate_binned_predicted_error(
    boundaries: torch.Tensor,
    distance_error_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(boundaries)
    return (
        torch.sum(distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def compute_binned_predicted_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
) -> [torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the sum of binned predicted confidence metrics from logits.

    Args:
        logits: [*, num_res, num_res, num_bins] Logits
        max_bin: Maximum bin value
        no_bins: Number of bins
    Returns:
      confidence_probs: [*, num_res, num_res, num_bins] The predicted
        error probabilities over bins for each residue/token pair
      predicted_error: [*, num_res, num_res] The expected distance
        error for each pair of residues/tokens
      max_predicted_error: [*] The maximum predicted error possible.
    """
    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    (
        predicted_error,
        max_predicted_error,
    ) = _calculate_binned_predicted_error(
        boundaries=boundaries,
        distance_error_probs=confidence_probs,
    )

    return confidence_probs, predicted_error, max_predicted_error


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from PredictedAlignedErrorHead logits"""
    confidence_probs, predicted_error, max_predicted_error = (
        compute_binned_predicted_error(logits=logits, max_bin=max_bin, no_bins=no_bins)
    )

    return {
        "aligned_confidence_probs": confidence_probs,
        "predicted_aligned_error": predicted_error,
        "max_predicted_aligned_error": max_predicted_error,
    }


def compute_predicted_distance_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from PredictedDistanceErrorHead logits"""
    confidence_probs, predicted_error, max_predicted_error = (
        compute_binned_predicted_error(logits=logits, max_bin=max_bin, no_bins=no_bins)
    )

    return {
        "distance_confidence_probs": confidence_probs,
        "predicted_distance_error": predicted_error,
        "max_predicted_distance_error": max_predicted_error,
    }


def compute_global_predicted_distance_error(
    pde: torch.Tensor,
    distogram_probs: torch.Tensor,
    min_bin: int = 2,
    max_bin: int = 22,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes the gPDE metric as defined in AF3 SI 5.7 (16)"""
    device = pde.device
    n_bins = distogram_probs.shape[-1]

    # Bins range from 2 to 22 Å
    distogram_bin_ends = torch.linspace(min_bin, max_bin, n_bins + 1, device=device)[1:]
    distogram_bins_8A = distogram_bin_ends <= 8.0  # boolean mask for bins <= 8 Å

    # Probability of contact between tokens i and j (sum over bins <= 8 Å)
    # pij shape: [bs, n_samples, n_tokens, n_tokens]
    pij = torch.sum(distogram_probs[..., distogram_bins_8A], dim=-1)

    # Global pde: weighted by contact probability pij
    # weighted_pde shape: [bs, n_samples]
    weighted_pde = torch.sum(pij * pde, dim=[-2, -1])
    sum_pij = torch.sum(pij, dim=[-2, -1]) + eps  # avoid division by zero
    # global_pde shape: [bs, n_samples]
    global_pde = weighted_pde / sum_pij

    return global_pde


def compute_ptm(
    logits: torch.Tensor,
    max_bin: int = 31,
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
            Pair-distance logits with bins in either layout:
            - [*, N, N, B]  (bins last), or
            - [*, B, N, N]  (bins first)
        max_bin:
            Upper bound (Å) for the distance bins (AF3: 31 → covers up to ~32Å).
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
    *leading, N, N2, B = x.shape

    if D_mask is None:
        D_mask = torch.ones(*leading, N, device=device, dtype=torch.bool)
    else:
        D_mask = D_mask.expand(*leading, -1)
        D_mask = D_mask.to(device=device, dtype=torch.bool)

    if has_frame is None:
        has_frame = torch.ones(*leading, N, device=device, dtype=torch.bool)
    else:
        has_frame = has_frame.to(device=device, dtype=torch.bool)

    if interface and asym_id is None:
        raise ValueError("asym_id is required when interface=True")
    if asym_id is not None:
        asym_id = asym_id.to(device=device)

    D_size = D_mask.sum(dim=-1).clamp_min(1).to(dtype)
    clipped = torch.maximum(D_size, torch.tensor(19.0, device=device, dtype=dtype))
    d0 = 1.24 * (clipped - 15.0).clamp_min(0).pow(1.0 / 3.0) - 1.8
    d0_sq = (d0**2).view(*leading, 1, 1, 1)

    boundaries = torch.linspace(
        0.0, float(max_bin), steps=no_bins - 1, device=device, dtype=dtype
    )
    bin_centers = _calculate_bin_centers(boundaries).view(*([1] * (x.dim() - 1)), B)
    tm_per_bin = 1.0 / (1.0 + (bin_centers**2) / d0_sq)

    probs = torch.softmax(x, dim=-1)
    exp_tm_ij = torch.sum(probs * tm_per_bin, dim=-1)

    if interface:
        same_chain = asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)
        M_ij = (~same_chain) & D_mask.unsqueeze(-2)
    else:
        M_ij = D_mask.unsqueeze(-2).expand(*leading, N, N)

    M_ij_f = M_ij.to(exp_tm_ij.dtype)
    exp_tm_ij = exp_tm_ij * M_ij_f

    denom_j = M_ij_f.sum(dim=-1).clamp_min(eps)
    per_i = exp_tm_ij.sum(dim=-1) / denom_j

    valid_i = has_frame & D_mask
    per_i_masked = torch.where(valid_i, per_i, torch.full_like(per_i, float("-inf")))
    return per_i_masked.max(dim=-1).values


def compute_weighted_ptm(
    logits: torch.Tensor,
    asym_id: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    ptm_weight: int = 0.2,
    iptm_weight: int = 0.8,
    mask: Optional[torch.Tensor] = None,
    residue_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    **kwargs,
) -> dict:
    ptm_score = compute_ptm(
        logits,
        max_bin=max_bin,
        no_bins=no_bins,
        mask=mask,
        residue_weights=residue_weights,
        eps=eps,
    )

    iptm_score = compute_ptm(
        logits,
        asym_id=asym_id,
        interface=True,
        max_bin=max_bin,
        no_bins=no_bins,
        mask=mask,
        residue_weights=residue_weights,
        eps=eps,
    )

    weighted_ptm_score = iptm_weight * iptm_score + ptm_weight * ptm_score

    all_ptm_scores = {
        "ptm_score": ptm_score,
        "iptm_score": iptm_score,
        "weighted_ptm_score": weighted_ptm_score,
    }

    return all_ptm_scores
