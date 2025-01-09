import logging

import torch

from openfold3.core.metrics.confidence import compute_predicted_distance_error
from openfold3.core.metrics.rasa import compute_rasa_batch

logger = logging.getLogger(__name__)


def compute_model_selection_metric(
    batch: dict,
    outputs: dict,
    metrics: dict,
    weights: dict,
    eps: float = 1e-8,
) -> dict:
    """
    Implements Model Selection (Section 5.7.3) LDDT metrics computation

    Args:
        batch: Updated batch dictionary post permutation alignment.
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
    # -------------------------------------------------------------------------
    # Compute pde (predicted distance error)
    # -------------------------------------------------------------------------
    # # pde_logits shape: [bs, n_samples, n_tokens, n_tokens, 64]
    # pde_logits = outputs["pde_logits"].detach()
    # # 64 bins equally spaced from 0 Å to 32 Å in
    # # 0.5 Å increments => centers in [0.25, 31.75]
    # bin_centers = torch.linspace(0.25, 31.75, 64, device=device)

    # # pde shape: [bs, n_samples, n_tokens, n_tokens]
    # pde = torch.sum(
    #     pde_logits * bin_centers, dim=-1
    # )  # expectation of distance for each pair
    pde = compute_predicted_distance_error(
        outputs["pde_logits"].detach(), max_bin=32, no_bins=64
    )["predicted_distance_error"]
    # -------------------------------------------------------------------------
    # Compute distogram-based contact probabilities (pij)
    # -------------------------------------------------------------------------
    # distogram_logits shape: [bs, n_samples, n_tokens, n_tokens, 38]
    distogram_logits = outputs["distogram_logits"].detach()
    # 38 bins in [3.25, 50.75] => we pick out bins <= 8 Å
    distogram_bins = torch.linspace(2, 22, 64, device=device)
    distogram_bins_8A = distogram_bins <= 8.0  # boolean mask for bins <= 8 Å

    # Probability of contact between tokens i and j (sum over bins <= 8 Å)
    # pij shape: [bs, n_samples, n_tokens, n_tokens]
    pij = torch.sum(distogram_logits[..., distogram_bins_8A], dim=-1)

    # -------------------------------------------------------------------------
    # Global pde: weighted by contact probability pij
    # -------------------------------------------------------------------------
    # weighted_pde shape: [bs, n_samples]
    weighted_pde = torch.sum(pij * pde, dim=[2, 3])
    sum_pij = torch.sum(pij, dim=[2, 3]) + eps  # avoid division by zero
    # global_pde shape: [bs, n_samples]
    global_pde = weighted_pde / sum_pij

    # -------------------------------------------------------------------------
    # Find the top-1 sample per batch based on global pde
    # -------------------------------------------------------------------------
    # top1_global_pde shape: [bs]
    top1_global_pde = torch.argmax(global_pde, dim=1)

    # -------------------------------------------------------------------------
    # Get the validation metrics, looping over each diffusion sample.
    # -------------------------------------------------------------------------

    # Add RASA metrics
    rasa = compute_rasa_batch(batch, outputs)
    if not torch.isnan(rasa).any():
        metrics["rasa"] = rasa
    # -------------------------------------------------------------------------
    # Select the top-1 metric values (across the sample dimension) per batch
    # -------------------------------------------------------------------------
    metrics_top_1 = {}
    for metric_name, metric_values in metrics.items():
        # metric_values shape: [bs, n_samples]
        # Index each batch by the top-1 sample
        batch_indices = torch.arange(metric_values.shape[0], device=device)
        metrics_top_1[metric_name] = metric_values[batch_indices, top1_global_pde]

    # -------------------------------------------------------------------------
    # Compute the best metric value (top) across all samples per batch
    # (referred to as "metric_top_5" in the original code—though it's just max)
    # -------------------------------------------------------------------------
    metric_top_max = {}
    for metric_name, metric_values in metrics.items():
        # Take the max across the sample dimension => shape [bs]
        metric_top_max[metric_name] = torch.max(metric_values, dim=1)[0]

    # -------------------------------------------------------------------------
    # Combine top-1 and top-max metrics by arithmetic mean
    # -------------------------------------------------------------------------
    final_metrics = {}
    for metric_name in metrics_top_1:
        final_metrics[metric_name] = 0.5 * (
            metrics_top_1[metric_name] + metric_top_max[metric_name]
        )

    # -------------------------------------------------------------------------
    # Compute weighted average of the selected metrics -> "model_selection_metric"
    # -------------------------------------------------------------------------
    metrics_to_average = set(final_metrics.keys()).intersection(set(weights.keys()))
    if len(metrics_to_average) == 0:
        # If no overlapping metrics, return a tensor of nan and put a warning
        final_metrics["model_selection_metric"] = torch.full(
            (outputs["pde_logits"].shape[0],), float("nan"), device=device
        )
        logging.warning(
            f"No overlapping metrics between the computed metrics: "
            f"{final_metrics.keys()} and the weights: {weights.keys()} "
            f"for pdb_id: {batch['pdb_id']}"
        )

        return final_metrics

    # Sum up the weighted metrics (shape: [bs])
    # and then divide by the sum of weights (scalar).
    total_weighted = 0.0
    sum_weights = 0.0
    for metric_name in metrics_to_average:
        total_weighted += final_metrics[metric_name] * weights[metric_name]
        sum_weights += weights[metric_name]

    final_metrics["model_selection"] = total_weighted / sum_weights

    return final_metrics
