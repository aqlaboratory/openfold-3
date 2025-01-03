from collections import defaultdict

import torch

from openfold3.core.metrics.validation_all_atom import (
    get_validation_metrics,
)
from openfold3.openfold3.openfold3.openfold3.core.metrics.rasa import compute_rasa_batch

# WIP: model selection should accept the output of
# get_validation_metrics and compute 3 additional LDDTs
#   - lddt_inter_ligand_rna
#   - lddt_inter_ligand_dna
#   - lddt_intra_modified_residues
# We may move the lddt computation of the three additional metrics to validation.py
# in a future PR to ensure that all the masks are computed in the same place.


def compute_model_selection_metric(
    batch: dict,
    outputs: dict,
    metrics: dict,
    weights: dict,
):
    """
    Implements Model Selection (Section 5.7.3) LDDT metrics computation

    Args:
        batch: Updated batch dictionary post permutation alignment.
        outputs: Output dictionary from the model.
        metrics: Dictionary containing validation metrics.
        weights: Dict of weights for each metric to compute a weighted average.

    Returns:
        metrics: Dictionary containing:
            - Keys for various LDDT metrics (e.g., 'lddt_inter_protein_protein',
              'lddt_intra_ligand', etc.), each with shape [batch_size].
            - "model_selection_metric" with shape [batch_size], representing
              the final weighted model-selection metric.
    """
    device = outputs["pde_logits"].device
    epsilon = 1e-8
    N_samples = outputs["pde_logits"].shape[1]
    # -------------------------------------------------------------------------
    # Compute PDE (predicted distance error)
    # -------------------------------------------------------------------------
    # pde_logits shape: [bs, n_samples, n_tokens, n_tokens, 64]
    pde_logits = outputs["pde_logits"].detach()
    # 64 bins equally spaced from 0 Å to 32 Å in 0.5 Å increments => centers in [0.25, 31.75]
    bin_centers = torch.linspace(0.25, 31.75, 64, device=device)

    # PDE shape: [bs, n_samples, n_tokens, n_tokens]
    PDE = torch.sum(
        pde_logits * bin_centers, dim=-1
    )  # expectation of distance for each pair

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
    # Global PDE: weighted by contact probability pij
    # -------------------------------------------------------------------------
    # weighted_pde shape: [bs, n_samples]
    weighted_pde = torch.sum(pij * PDE, dim=[2, 3])
    sum_pij = torch.sum(pij, dim=[2, 3]) + epsilon  # avoid division by zero
    # global_PDE shape: [bs, n_samples]
    global_PDE = weighted_pde / sum_pij

    # -------------------------------------------------------------------------
    # Find the top-1 sample per batch based on global PDE
    # -------------------------------------------------------------------------
    # top1_global_PDE shape: [bs]
    top1_global_PDE = torch.argmax(global_PDE, dim=1)

    # -------------------------------------------------------------------------
    # Get the validation metrics. We assume `metrics` already has shape [bs, n_samples]
    # -------------------------------------------------------------------------
    metrics = defaultdict(list)

    for i in range(N_samples):
        output_sample = {key: value[:, i, ...] for key, value in outputs.items()}
        metrics_sample = get_validation_metrics(
            batch, output_sample, superimposition_metrics=True, is_train=False
        )
        for metric_name, metric_values in metrics_sample.items():
            metrics[metric_name].append(metric_values)

    # Convert the lists to tensors
    for metric_name, metric_values in metrics.items():
        metrics[metric_name] = torch.stack(metric_values, dim=1)

    # For now, we'll create an empty dictionary
    # metrics = {}
    # Add RASA metrics
    # metrics["RASA"] = compute_rasa_batch(batch, outputs)
    # squeeze the sample dimension
    for metric_name, metric_values in metrics.items():
        metrics[metric_name] = metric_values.unsqueeze(1)

    # -------------------------------------------------------------------------
    # Select the top-1 metric values (across the sample dimension) per batch
    # -------------------------------------------------------------------------
    metrics_top_1 = {}
    for metric_name, metric_values in metrics.items():
        # metric_values shape: [bs, n_samples]
        # Index each batch by the top-1 sample
        batch_indices = torch.arange(metric_values.shape[0], device=device)
        metrics_top_1[metric_name] = metric_values[batch_indices, top1_global_PDE]

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
        # If no overlapping metrics, return empty or raise an error
        final_metrics["model_selection_metric"] = torch.zeros(
            pde_logits.shape[0], device=device
        )
        return final_metrics

    # Sum up the weighted metrics (shape: [bs])
    # and then divide by the sum of weights (scalar).
    total_weighted = torch.zeros(pde_logits.shape[0], device=device)
    sum_weights = 0.0
    for metric_name in metrics_to_average:
        total_weighted += final_metrics[metric_name] * weights[metric_name]
        sum_weights += weights[metric_name]

    final_metrics["model_selection_metric"] = total_weighted / sum_weights

    return final_metrics


# def model_selection_metric(
#     batch: dict,
#     outputs: dict,
#     weights: dict,
# ):
#     """
#     Implements Model Selection (Section 5.7.3) LDDT metrics computation

#     Args:
#             batch: Updated batch dictionary post permutation alignment
#             output: Output dictionary
#     Returns:
#             metrics: Dictionary containing lddts of following keys:
#                     'lddt_inter_protein_protein': Protein-protein interface LDDT
#                     'lddt_inter_protein_dna': DNA-protein interface LDDT
#                     'lddt_inter_protein_rna': Protein-RNA interface LDDT
#                     'lddt_inter_ligand_dna': DNA-ligand interface LDDT
#                     'lddt_inter_protein_ligand': Ligand-protein interface LDDT
#                     'lddt_inter_ligand_rna': Ligand-RNA interface LDDT
#                     'lddt_intra_protein': Protein intra-chain LDDT
#                     'lddt_intra_dna': DNA intra-chain LDDT
#                     'lddt_intra_rna': RNA intra-chain LDDT
#                     'lddt_intra_ligand': Ligand intra-chain LDDT
#                     'lddt_intra_modified_residues':Modified residue intra-chain LDDT

#             with [bs, n_samples] shape for all metrics

#     Note:
#             In the case where no valid atom_type is available, metric dict doesn't
#             contain the key.
#     """


#     return compute_model_selection_metric(
#         batch, outputs, metrics, weights
#     )


# probably should be in config
initial_training_weights = {
    "lddt_intra_modified_residues": 10.0,
    "lddt_inter_ligand_rna": 5.0,
    "lddt_inter_ligand_dna": 5.0,
    "lddt_intra_protein": 20.0,
    "lddt_intra_ligand": 20.0,
    "lddt_intra_dna": 4.0,
    "lddt_intra_rna": 16.0,
    "lddt_inter_protein_protein": 20.0,
    "lddt_inter_protein_ligand": 10.0,
    "lddt_inter_protein_dna": 10.0,
    "lddt_inter_protein_rna": 10.0,
    "RASA": 10.0,
}

fine_tuning_weights = {
    "lddt_inter_ligand_rna": 2.0,
    "lddt_inter_ligand_dna": 5.0,
    "lddt_intra_protein": 20.0,
    "lddt_intra_ligand": 20.0,
    "lddt_intra_dna": 4.0,
    "lddt_intra_rna": 16.0,
    "lddt_inter_protein_protein": 20.0,
    "lddt_inter_protein_ligand": 10.0,
    "lddt_inter_protein_dna": 10.0,
    "lddt_inter_protein_rna": 2.0,
    "RASA": 10.0,
}
