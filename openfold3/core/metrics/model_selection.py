import logging

import torch
from ml_collections import ConfigDict

from openfold3.core.metrics.confidence import (
    compute_global_predicted_distance_error,
    compute_predicted_distance_error,
)
from openfold3.projects.af3_all_atom.constants import METRICS_MAXIMIZE, METRICS_MINIMIZE

logger = logging.getLogger(__name__)


def compute_valid_model_selection_metrics(
    confidence_config: ConfigDict,
    outputs: dict,
    metrics: dict,
    eps: float = 1e-8,
) -> dict:
    """
    Implements Model Selection (Section 5.7.3) LDDT metrics computation

    Args:
        confidence_config: Config for confidence metrics (needed for PDE)
        outputs: Output dictionary from the model
        metrics: Dict of metrics for all rollout samples
        eps: Small value to avoid division by zero

    Returns:
        final_metrics:
            Dictionary containing keys for various LDDT metrics
            (e.g., 'lddt_inter_protein_protein', lddt_intra_ligand', etc.),
            each with shape [batch_size].
    """
    device = outputs["pde_logits"].device

    # Compute pde (predicted distance error)
    pde = compute_predicted_distance_error(
        outputs["pde_logits"].detach(), **confidence_config.pde
    )["predicted_distance_error"]

    # Compute distogram-based contact probabilities (pij)
    # distogram_logits shape: [bs, n_samples, n_tokens, n_tokens, 38]
    distogram_logits = outputs["distogram_logits"].detach()
    distogram_probs = torch.softmax(distogram_logits, dim=-1)

    global_pde = compute_global_predicted_distance_error(
        pde=pde,
        distogram_probs=distogram_probs,
    )

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

    return final_metrics


def compute_final_model_selection_metric(metrics: dict, model_selection_weights: dict):
    """
    Computes aggregated model selection metric.

    Args:
        metrics:
            Dict of aggregated metrics for all targets
        model_selection_weights:
            Dict of weights for each metric to compute a weighted average

    Returns:
        model_selection: The final weighted model-selection metric

    """
    total_weighted = 0.0
    sum_weights = 0.0
    for name, weight in model_selection_weights.items():
        total_weighted += metrics[f"val/{name}"] * weight
        sum_weights += weight

    model_selection = total_weighted / sum_weights

    return model_selection
