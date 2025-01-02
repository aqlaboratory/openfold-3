import torch

from openfold3.core.metrics.validation_all_atom import (
    get_validation_metrics,
    interface_lddt,
    lddt,
)
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms


# WIP: model selection should accept the output of
# get_validation_metrics and compute 3 additional LDDTs
#   - lddt_inter_ligand_rna
#   - lddt_inter_ligand_dna
#   - lddt_intra_modified_residues
# We may move the lddt computation of the three additional metrics to validation.py
# in a future PR to ensure that all the masks are computed in the same place.
def model_selection_metric(
    batch: dict,
    outputs: dict,
    weights: dict,
):
    """
    Implements Model Selection (Section 5.7.3) LDDT metrics computation

    Args:
            batch: Updated batch dictionary post permutation alignment
            output: Output dictionary
    Returns:
            metrics: Dictionary containing lddts of following keys:
                    'lddt_inter_protein_protein': Protein-protein interface LDDT
                    'lddt_inter_protein_dna': DNA-protein interface LDDT
                    'lddt_inter_protein_rna': Protein-RNA interface LDDT
                    'lddt_inter_ligand_dna': DNA-ligand interface LDDT
                    'lddt_inter_protein_ligand': Ligand-protein interface LDDT
                    'lddt_inter_ligand_rna': Ligand-RNA interface LDDT
                    'lddt_intra_protein': Protein intra-chain LDDT
                    'lddt_intra_dna': DNA intra-chain LDDT
                    'lddt_intra_rna': RNA intra-chain LDDT
                    'lddt_intra_ligand': Ligand intra-chain LDDT
                    'lddt_intra_modified_residues':Modified residue intra-chain LDDT

            with [bs, n_samples] shape for all metrics

    Note:
            In the case where no valid atom_type is available, metric dict doesn't
            contain the key.
    """
    metrics = {}

    gt_coords = batch["ground_truth"]["atom_positions"]
    pred_coords = outputs["x_pred"]
    all_atom_mask = batch["ref_mask"]
    token_mask = batch["token_mask"]
    num_atoms_per_token = batch["num_atoms_per_token"]
    bs = pred_coords.shape[:-2]

    # getting rid of modified residues
    is_protein = batch["is_protein"]
    is_rna = batch["is_rna"]
    is_dna = batch["is_dna"]
    not_atomized = 1 - batch["is_atomized"]

    is_protein = is_protein * not_atomized
    is_rna = is_rna * not_atomized
    is_dna = is_dna * not_atomized

    # broadcast token level features to atom level features
    is_protein_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, is_protein
    )
    is_ligand_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, batch["is_ligand"]
    )
    is_rna_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, is_rna
    )
    is_dna_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, is_dna
    )
    asym_id_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, batch["asym_id"]
    )

    # set up filters for validation metrics if present, otherwise pass ones
    use_for_intra = batch.get(
        "use_for_intra_validation", torch.ones_like(is_protein)
    ).to(token_mask.device)
    use_for_inter = (
        batch.get(
            "use_for_inter_validation",
            torch.ones((is_protein.size()[-1], is_protein.size()[-1])),
        )
        .unsqueeze(0)
        .to(token_mask.device)
    )

    intra_filter_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, use_for_intra
    )

    # convert use_for_inter: [*, n_token, n_token] into [*, n_atom, n_atom]
    inter_filter_atomized = broadcast_token_feat_to_atoms(
        token_mask,
        num_atoms_per_token,
        use_for_inter,
        token_dim=-2,
    )
    inter_filter_atomized = broadcast_token_feat_to_atoms(
        token_mask,
        num_atoms_per_token,
        inter_filter_atomized.transpose(-1, -2),
        token_dim=-2,
    )
    inter_filter_atomized = inter_filter_atomized.transpose(-1, -2)

    # get metrics
    protein_validation_metrics = get_validation_metrics(
        is_protein_atomized,
        asym_id_atomized,
        pred_coords,
        gt_coords,
        all_atom_mask,
        is_protein_atomized,
        substrate="protein",
        is_nucleic_acid=False,
    )
    metrics = metrics | protein_validation_metrics

    ligand_validation_metrics = get_validation_metrics(
        is_ligand_atomized,
        asym_id_atomized,
        pred_coords,
        gt_coords,
        all_atom_mask,
        is_protein_atomized,
        substrate="ligand",
        is_nucleic_acid=False,
    )
    metrics = metrics | ligand_validation_metrics

    rna_validation_metrics = get_validation_metrics(
        is_rna_atomized,
        asym_id_atomized,
        pred_coords,
        gt_coords,
        all_atom_mask,
        is_protein_atomized,
        substrate="rna",
        is_nucleic_acid=True,
    )
    metrics = metrics | rna_validation_metrics

    dna_validation_metrics = get_validation_metrics(
        is_dna_atomized,
        asym_id_atomized,
        pred_coords,
        gt_coords,
        all_atom_mask,
        is_protein_atomized,
        substrate="dna",
        is_nucleic_acid=True,
    )
    metrics = metrics | dna_validation_metrics

    # remaining LDDTs
    # interLDDT: ligand_dna
    if torch.any(is_ligand_atomized) and torch.any(is_dna_atomized):
        metrics.update(
            {
                "lddt_inter_ligand_dna": interface_lddt(
                    pred_coords[is_ligand_atomized.bool()].view((bs) + (-1, 3)),
                    pred_coords[is_dna_atomized.bool()].view((bs) + (-1, 3)),
                    gt_coords[is_ligand_atomized.bool()].view((bs) + (-1,)),
                    gt_coords[is_dna_atomized.bool()].view((bs) + (-1,)),
                    all_atom_mask[is_ligand_atomized.bool()].view((bs) + (-1,)),
                    all_atom_mask[is_dna_atomized.bool()].view((bs) + (-1,)),
                    cutoff=30.0,
                )
            }
        )

    # interLDDT: ligand_rna
    if torch.any(is_ligand_atomized) and torch.any(is_rna_atomized):
        is_rna_ligand_pair = (
            is_rna_atomized[..., None] * is_ligand_atomized[..., None, :]
        )
        n_rna_atoms = torch.sum(is_rna_atomized).to(int).item()
        n_ligand_atoms = torch.sum(is_ligand_atomized).to(int).item()
        # TODO: just works for bs = 1, sample size = 1
        inter_filter_mask_rna_ligand = torch.masked_select(
            inter_filter_atomized, is_rna_ligand_pair.bool()
        ).reshape((bs) + (n_rna_atoms, n_ligand_atoms))
        lddt_inter_ligand_rna = interface_lddt(
            pred_coords[is_rna_atomized.bool()].view((bs) + (-1, 3)),
            pred_coords[is_ligand_atomized.bool()].view((bs) + (-1, 3)),
            gt_coords[is_rna_atomized.bool()].view((bs) + (-1, 3)),
            gt_coords[is_ligand_atomized.bool()].view((bs) + (-1, 3)),
            all_atom_mask[is_rna_atomized.bool()].view((bs) + (-1,)),
            all_atom_mask[is_ligand_atomized.bool()].view((bs) + (-1,)),
            inter_filter_mask_rna_ligand,
            cutoff=30.0,
        )

        if not (torch.isnan(lddt_inter_ligand_rna)).all():
            lddt_inter_ligand_rna[~torch.isnan(lddt_inter_ligand_rna)]
            metrics.update({"lddt_inter_ligand_rna": lddt_inter_ligand_rna})

    if torch.any(is_ligand_atomized) and torch.any(is_dna_atomized):
        is_dna_ligand_pair = (
            is_dna_atomized[..., None] * is_ligand_atomized[..., None, :]
        )
        n_dna_atoms = torch.sum(is_dna_atomized).to(int).item()
        n_ligand_atoms = torch.sum(is_ligand_atomized).to(int).item()
        inter_filter_mask_dna_ligand = torch.masked_select(
            inter_filter_atomized, is_dna_ligand_pair.bool()
        ).reshape((bs) + (n_dna_atoms, n_ligand_atoms))

        lddt_inter_ligand_dna = interface_lddt(
            pred_coords[is_dna_atomized.bool()].view((bs) + (-1, 3)),
            pred_coords[is_ligand_atomized.bool()].view((bs) + (-1, 3)),
            gt_coords[is_dna_atomized.bool()].view((bs) + (-1, 3)),
            gt_coords[is_ligand_atomized.bool()].view((bs) + (-1, 3)),
            all_atom_mask[is_dna_atomized.bool()].view((bs) + (-1,)),
            all_atom_mask[is_ligand_atomized.bool()].view((bs) + (-1,)),
            inter_filter_mask_dna_ligand,
            cutoff=30.0,
        )

        if not (torch.isnan(lddt_inter_ligand_dna)).all():
            lddt_inter_ligand_dna[~torch.isnan(lddt_inter_ligand_dna)]
            metrics.update({"lddt_inter_ligand_dna": lddt_inter_ligand_dna})

    # intraLDDT: modified_residues
    is_modified_res = batch["is_atomized"]
    is_modified_res = is_modified_res * (1 - batch["is_ligand"])
    is_modified_res_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, is_modified_res
    )
    is_modified_res_atomized = is_modified_res_atomized.bool()
    if torch.any(is_modified_res):
        pred_mr = pred_coords[is_modified_res_atomized.bool()].view((bs) + (-1, 3))
        gt_mr = gt_coords[is_modified_res_atomized.bool()].view((bs) + (-1, 3))
        intra_mask_atomized_mr = intra_filter_atomized[
            is_modified_res_atomized.bool()
        ].view((bs) + (-1,))
        is_mr_atomized_pair = (
            is_modified_res_atomized[..., None] * is_modified_res_atomized[..., None, :]
        )

        n_mr_atoms = torch.sum(is_modified_res_atomized).to(int).item()
        inter_mask_atomized_mr = torch.masked_select(
            inter_filter_atomized, is_mr_atomized_pair.bool()
        ).reshape((bs) + (n_mr_atoms, n_mr_atoms))

        pred_mr_pair = torch.sqrt(
            torch.sum(
                (pred_mr.unsqueeze(-2) - pred_mr.unsqueeze(-3)) ** 2,
                dim=-1,
            )
        )
        gt_mr_pair = torch.sqrt(
            torch.sum(
                (gt_mr.unsqueeze(-2) - gt_mr.unsqueeze(-3)) ** 2,
                dim=-1,
            )
        )
        lddt_intra_modified_residues, _ = lddt(
            pred_mr_pair,
            gt_mr_pair,
            all_atom_mask[is_modified_res_atomized.bool()].view((bs) + (-1,)),
            intra_mask_atomized_mr,
            inter_mask_atomized_mr,
            asym_id_atomized[is_modified_res_atomized.bool()].view((bs) + (-1,)),
        )

        if not (torch.isnan(lddt_intra_modified_residues)).all():
            lddt_intra_modified_residues[~torch.isnan(lddt_intra_modified_residues)]
            metrics.update(
                {"lddt_intra_modified_residues": lddt_intra_modified_residues}
            )

    # Here we suppose that we only have a batch of size 1.
    # compute the PDE for each sample:
    PDE = outputs["confidence_scores"]["PDE"][0]  # [N_samples, N_token, N_token, 1]
    distogram_logits = outputs["distogram_logits"][
        0
    ]  # [N_samples, N_token, N_token, 38]

    # distogram_bins are divided into 38 bins between 3.25 and 50.75 Angstroms
    distogram_bins = torch.linspace(3.25, 50.75, 38)
    # get the distogram bins corresponding to distance <= 8 Angstroms
    distogram_bins_8A = torch.where(distogram_bins <= 8)[0]

    # probability of contact between token i and token j
    pij = torch.sum(
        distogram_logits[:, :, :, distogram_bins_8A], dim=-1
    )  # [N_samples, N_token, N_token]

    # global PDE
    global_PDE = torch.average(PDE, dim=0, weights=pij)

    # get top1 global PDE
    top1_global_PDE = torch.argmax(global_PDE, dim=-1)

    metrics_top_1 = {
        metric: value[top1_global_PDE] for metric, value in metrics.items()
    }

    # best value for each metric across all the samples
    metric_top_5 = {
        metric: torch.max(value, dim=0)[0] for metric, value in metrics.items()
    }

    # arithmetic mean of the metrics from metrics_top_1 and metric_top_5
    metrics = {
        metric: (metrics_top_1[metric] + metric_top_5[metric]) / 2
        for metric in metrics_top_1
    }
    model_selection_metric = sum(
        [metrics[metric] * weights[metric] for metric in metrics]
    )

    sum_weights = sum(weights[metric] for metric in metrics)

    metrics["model_selection_metric"] = model_selection_metric / sum_weights

    return metrics


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
}

initial_training_weights = {
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
}
