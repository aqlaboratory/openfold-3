import logging

import torch

from openfold3.core.metrics.confidence import compute_predicted_distance_error
from openfold3.projects.af3_all_atom.constants import METRICS_MAXIMIZE, METRICS_MINIMIZE

logger = logging.getLogger(__name__)


def compute_model_selection_metric(
    outputs: dict,
    metrics: dict,
    weights: dict,
    eps: float = 1e-8,
) -> dict:
    """
    Implements Model Selection (Section 5.7.3) LDDT metrics computation

    Args:
        outputs: Output dictionary from the model.
        weights: Dict of weights for each metric to compute a weighted average.
        eps: Small value to avoid division by zero.

    Returns:
        metrics: Dictionary containing:
            - Keys for various LDDT metrics (e.g., 'lddt_inter_protein_protein',
              'lddt_intra_ligand', etc.), each with shape [batch_size].
            - "model_selection_metric" with shape [batch_size], representing
              the final weighted model-selection metric.
    """
    device = outputs["pde_logits"].device

    # Compute pde (predicted distance error)
    pde = compute_predicted_distance_error(
        outputs["pde_logits"].detach(), max_bin=31, no_bins=64
    )["predicted_distance_error"]

    # Compute distogram-based contact probabilities (pij)
    # distogram_logits shape: [bs, n_samples, n_tokens, n_tokens, 38]
    distogram_logits = outputs["distogram_logits"].detach()
    distogram_bins = torch.linspace(2, 22, 65, device=device)
    distogram_bins_8A = distogram_bins <= 8.0  # boolean mask for bins <= 8 Å
    distogram_bins_8A = distogram_bins_8A[1:]  # exclude the first bin (2 Å)

    # Probability of contact between tokens i and j (sum over bins <= 8 Å)
    # pij shape: [bs, n_samples, n_tokens, n_tokens]
    pij = torch.sum(distogram_logits[..., distogram_bins_8A], dim=-1)

    # Global pde: weighted by contact probability pij
    # weighted_pde shape: [bs, n_samples]
    weighted_pde = torch.sum(pij * pde, dim=[-2, -1])
    sum_pij = torch.sum(pij, dim=[-2, -1]) + eps  # avoid division by zero
    # global_pde shape: [bs, n_samples]
    global_pde = weighted_pde / sum_pij

    # Find the top-1 sample per batch based on global pde
    # top1_global_pde shape: [bs]
    top1_global_pde = torch.argmax(global_pde, dim=1)

    # Select the top-1 metric values (across the sample dimension) per batch
    metrics_top_1 = {}
    for metric_name, metric_values in metrics.items():
        # metric_values shape: [bs, n_samples]
        # Index each batch by the top-1 sample
        batch_indices = torch.arange(metric_values.shape[0], device=device)
        metrics_top_1[metric_name] = metric_values[batch_indices, top1_global_pde]

    # Compute the best metric value (top) across all samples per batch
    # (referred to as "metric_top_5" in the original AF3, though it's just max/min
    # based on the metric type)
    metric_best = {}
    for metric_name, metric_values in metrics.items():
        # Take the best across the sample dimension => shape [bs]
        metric_type = metric_name.split("_")[0]
        if metric_type in METRICS_MAXIMIZE:
            best_metric_sample = torch.max(metric_values, dim=1)[0]
        elif metric_type in METRICS_MINIMIZE:
            best_metric_sample = torch.min(metric_values, dim=1)[0]
        else:
            raise ValueError(
                f"Please specify whether metric should be maximized "
                f"or minimized in the METRICS_MAX or METRICS_MIN "
                f"constants: {metric_name}"
            )
        metric_best[metric_name] = best_metric_sample

    # Combine top-1 and top-max metrics by arithmetic mean
    final_metrics = {}
    for metric_name in metrics_top_1:
        final_metrics[metric_name] = 0.5 * (
            metrics_top_1[metric_name] + metric_best[metric_name]
        )

    # Sum up the weighted metrics (shape: [bs])
    # and then divide by the sum of weights (scalar).
    valid_metrics = list(set(weights.keys()).intersection(final_metrics.keys()))
    valid_metrics = [
        metric_name
        for metric_name in valid_metrics
        if not torch.isnan(final_metrics[metric_name]).any()
    ]

    total_weighted = 0.0
    sum_weights = 0.0
    for metric_name in valid_metrics:
        total_weighted += final_metrics[metric_name] * weights[metric_name]
        sum_weights += weights[metric_name]

    final_metrics["model_selection"] = total_weighted / sum_weights

    return final_metrics
