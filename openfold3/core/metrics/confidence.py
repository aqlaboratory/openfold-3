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
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
    max_bin: int = 31,
    no_bins: int = 64,
    mask: Optional[torch.Tensor] = None,
    residue_weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    n = residue_weights.shape[-1]
    pair_mask = residue_weights.new_ones((n, n), dtype=torch.int32)
    if interface and (asym_id is not None):
        if len(asym_id.shape) > 1:
            assert len(asym_id.shape) <= 2
            batch_size = asym_id.shape[0]
            pair_mask = residue_weights.new_ones((batch_size, n, n), dtype=torch.int32)
        pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(
            dtype=pair_mask.dtype
        )

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[..., None, :] * residue_weights[..., :, None]
    )
    denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
    normed_residue_mask = pair_residue_weights / denom
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    if mask is not None:
        weighted = weighted * mask

    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


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
