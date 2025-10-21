# Copyright 2025 AlQuraishi Laboratory
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

import logging

import torch
from ml_collections import ConfigDict

from openfold3.core.metrics.confidence import (
    compute_global_predicted_distance_error,
    probs_to_expected_error,
)
from openfold3.projects.of3_all_atom.constants import METRICS_MAXIMIZE, METRICS_MINIMIZE

logger = logging.getLogger(__name__)


def compute_valid_model_selection_metrics(
    confidence_config: ConfigDict,
    outputs: dict,
    metrics: dict,
) -> dict:
    """
    Implements Model Selection (Section 5.7.3) LDDT metrics computation

    Args:
        confidence_config: Config for confidence metrics (needed for PDE)
        outputs: Output dictionary from the model
        metrics: Dict of metrics for all rollout samples

    Returns:
        final_metrics:
            Dictionary containing keys for various LDDT metrics
            (e.g., 'lddt_inter_protein_protein', lddt_intra_ligand', etc.),
            each with shape [batch_size].
    """
    device = outputs["pde_logits"].device

    num_atoms = outputs["atom_positions_predicted"].shape[-2]
    chunk_computation = (
        confidence_config.low_mem_validation
        and confidence_config.per_sample_atom_cutoff is not None
        and num_atoms > confidence_config.per_sample_atom_cutoff
    )

    # Compute pde (predicted distance error)
    pde_logits = outputs["pde_logits"].detach()
    if chunk_computation:
        pde = torch.zeros(
            pde_logits.shape[:-1], device=pde_logits.device, dtype=pde_logits.dtype
        )
        for i in range(pde_logits.shape[-4]):
            pde[..., i : i + 1, :, :] = probs_to_expected_error(
                torch.softmax(pde_logits[..., i : i + 1, :, :, :], dim=-1),
                **confidence_config.pde,
            )
    else:
        pde = probs_to_expected_error(
            torch.softmax(pde_logits, dim=-1),
            **confidence_config.pde,
        )

    # Compute distogram-based contact probabilities (pij)
    # distogram_logits shape: [bs, n_samples, n_tokens, n_tokens, 38]
    distogram_logits = outputs["distogram_logits"].detach()

    global_pde, _ = compute_global_predicted_distance_error(
        pde=pde,
        logits=distogram_logits,
        **confidence_config.distogram,
    )

    # Find the top-1 sample per batch based on global pde
    # top1_global_pde shape: [bs]
    top1_global_pde = torch.argmin(global_pde, dim=1)

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
