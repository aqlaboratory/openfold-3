from collections.abc import Sequence
from typing import Optional

import torch

from openfold3.core.metrics.confidence import compute_plddt
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms


def lddt(
    pair_dist_pred_pos: torch.Tensor,
    pair_dist_gt_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
    intra_mask_filter: torch.Tensor,
    inter_mask_filter: torch.Tensor,
    asym_id: torch.Tensor,
    threshold: Optional[Sequence] = (0.5, 1.0, 2.0, 4.0),
    cutoff: Optional[float] = 15.0,
    eps: Optional[float] = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates lddt scores from pair distances
    Compute on all atoms within the same chain_type (protein, ligand, rna, dna)

    Args:
        pair_dist_pred_pos: pairwise distance of prediction [*, n_atom, n_atom]
        pair_dist_gt_pos: pairwise distance of gt [*, n_atom, n_atom]
        all_atom_mask: atom level mask [*, n_atom]
        intra_mask_filter:[*, n_atom] filter for intra chain computations
        inter_mask_filter: [*, n_atom, n_atom] pairwise interaction filter
        asym_id: atomized asym_id feature [*, n_atom]
        threshold: a list of thresholds to apply for lddt computation
            - Standard values: [0.5, 1., 2., 4.]
            - lddt_uha (for ligands): [0.25, 0.5, 0.75, 1.]
        cutoff: distance cutoff (aka. inclusion radius)
            - Nucleic Acids (DNA/RNA) 30.
            - Other biomolecules (Protein/Ligands) 15.
        eps: epsilon

    Returns:
        intra_score: intra lddt scores [*]
        inter_score: inter lddt scores [*]

    Note:
        returns nan for inter_score if inter_lddt invalid
        (ie. single chain, no atom pair within cutoff)
    """
    # create a mask
    n_atom = pair_dist_gt_pos.shape[-2]
    dists_to_score = (pair_dist_gt_pos < cutoff) * (
        all_atom_mask[..., None]
        * all_atom_mask[..., None, :]
        * (1.0 - torch.eye(n_atom, device=all_atom_mask.device))
    )  # [*, n_atom, n_atom]

    # distinguish intra- and inter- pair indices based on asym_id
    intra_mask = torch.where(asym_id[..., None] == asym_id[..., None, :], 1, 0)
    inter_mask = 1 - intra_mask  # [*, n_atom, n_atom]

    # update masks with filters
    intra_mask = intra_mask * (
        intra_mask_filter[..., None] * intra_mask_filter[..., None, :]
    )
    inter_mask = inter_mask * inter_mask_filter

    # get lddt scores
    dist_l1 = torch.abs(pair_dist_gt_pos - pair_dist_pred_pos)  # [*, n_atom, n_atom]
    score = torch.zeros_like(dist_l1)
    for distance_threshold in threshold:
        score += (dist_l1 < distance_threshold).type(dist_l1.dtype)
    score = score / len(threshold)

    # normalize to get intra_lddt scores
    intra_score = torch.full(intra_mask.shape[:-1], torch.nan)
    if torch.any(intra_mask):
        intra_norm = 1.0 / (eps + torch.sum(dists_to_score * intra_mask, dim=(-1, -2)))
        intra_score = intra_norm * (
            eps + torch.sum(dists_to_score * intra_mask * score, dim=(-1, -2))
        )

    # inter_score only applies when there exist atom pairs with
    # different asym_id (inter_mask) and distance threshold (dists_to_score)
    inter_score = torch.full(intra_score.shape, torch.nan)
    inter_mask = dists_to_score * inter_mask
    if torch.any(inter_mask):
        inter_norm = 1.0 / (eps + torch.sum(inter_mask, dim=(-1, -2)))
        inter_score = inter_norm * (eps + torch.sum(inter_mask * score, dim=(-1, -2)))
    return intra_score, inter_score


def interface_lddt(
    all_atom_pred_pos_1: torch.Tensor,
    all_atom_pred_pos_2: torch.Tensor,
    all_atom_gt_pos_1: torch.Tensor,
    all_atom_gt_pos_2: torch.Tensor,
    all_atom_mask1: torch.Tensor,
    all_atom_mask2: torch.Tensor,
    filter_mask: torch.Tensor,
    cutoff: Optional[float] = 15.0,
    eps: Optional[float] = 1e-10,
) -> torch.Tensor:
    """
    Calculates interface_lddt (ilddt) score between two different molecules

    Args:
        all_atom_pred_pos_1: predicted protein coordinates [*, n_atom1, 3]
        all_atom_pred_pos_2: predicted interacting molecule coordinates [*, n_atom2, 3]
        all_atom_gt_pos_1: gt protein coordinates [*, n_atom1, 3]
        all_atom_gt_pos_2: gt interacting molecule coordinates  [*, n_atom2, 3]
        all_atom_mask1: protein atom mask [*, n_atom1]
        all_atom_mask2: interacting molecule atom maks [*, n_atom2]
        filter_mask: [*, n_atom1, n_atom2] pairwise filter for atom types
        cutoff: distance cutoff
            - Nucleic Acids (DNA/RNA) 30.
            - Others(Protein/Ligands) 15.
        eps: epsilon

    Returns:
        scores: ilddt scores [*]
    """
    # get pairwise distance
    pair_dist_true = torch.sqrt(
        torch.sum(
            (all_atom_gt_pos_1.unsqueeze(-2) - all_atom_gt_pos_2.unsqueeze(-3)) ** 2,
            dim=-1,
        )
    )  # [*, n_atom1, n_atom2]
    pair_dist_pred = torch.sqrt(
        torch.sum(
            (all_atom_pred_pos_1.unsqueeze(-2) - all_atom_pred_pos_2.unsqueeze(-3))
            ** 2,
            dim=-1,
        )
    )  # [*, n_atom1, n_atom2]

    # create a mask
    dists_to_score = (pair_dist_true < cutoff) * (
        all_atom_mask1[..., None] * all_atom_mask2[..., None, :]
    )  # [*, n_atom1, n_atom2]
    dists_to_score = dists_to_score * filter_mask

    if torch.any(dists_to_score):
        # get score
        dist_l1 = torch.abs(pair_dist_true - pair_dist_pred)
        score = (
            (dist_l1 < 0.5).type(dist_l1.dtype)
            + (dist_l1 < 1.0).type(dist_l1.dtype)
            + (dist_l1 < 2.0).type(dist_l1.dtype)
            + (dist_l1 < 4.0).type(dist_l1.dtype)
        )
        score = score * 0.25

        # normalize
        norm = 1.0 / (eps + torch.sum(dists_to_score, dim=(-1, -2)))
        score = norm * (eps + torch.sum(dists_to_score * score, dim=(-1, -2)))
    else:
        shape = all_atom_mask1.shape[:-1]
        if not shape:
            shape = (1,)
        score = torch.full(shape, torch.nan)
    return score


def drmsd(
    pair_dist_pred_pos: torch.Tensor,
    pair_dist_gt_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes drmsds from pair distances

    Args:
        pair_dist_pred_pos: predicted coordinates [*, n_atom, n_atom]
        pair_dist_gt_pos: gt coordinates [*, n_atom, n_atom]
        all_atom_mask: atom mask [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom]
    Returns:
        intra_drmsd: drmsd within chains
        inter_drmsd: drmsd across chains

    Note:
        returns nan if inter_drmsd is invalid (ie. single chain)
    """
    drmsd = pair_dist_pred_pos - pair_dist_gt_pos
    drmsd = drmsd**2

    # apply mask
    mask = all_atom_mask[..., None] * all_atom_mask[..., None, :]
    intra_mask = torch.where(asym_id[..., None] == asym_id[..., None, :], 1, 0)
    inter_mask = 1 - intra_mask

    intra_drmsd = drmsd * (mask * intra_mask)
    intra_drmsd = torch.sum(intra_drmsd, dim=(-1, -2))
    n_intra = torch.sum(intra_mask * mask, dim=(-1, -2))
    intra_drmsd = intra_drmsd * (1 / (n_intra))

    inter_drmsd = torch.full(intra_drmsd.shape, torch.nan)
    inter_mask = inter_mask * mask
    if torch.any(inter_mask):
        inter_drmsd = drmsd * (inter_mask)
        inter_drmsd = torch.sum(inter_drmsd, dim=(-1, -2))
        n_inter = torch.sum(inter_mask * mask, dim=(-1, -2))
        inter_drmsd = inter_drmsd * (1 / (n_inter))

    intra_drmsd = torch.sqrt(intra_drmsd)
    inter_drmsd = torch.sqrt(inter_drmsd)
    return intra_drmsd, inter_drmsd


def get_protein_metrics(
    is_protein_atomized: torch.Tensor,
    asym_id: torch.Tensor,
    intra_mask_atomized: torch.Tensor,
    inter_mask_atomized: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute validation metrics of protein

    Args:
        is_protein_atomized: broadcasted is_protein feature [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom]
        intra_mask_atomized:[*, n_atom] filter for intra chain computations
        inter_mask_atomized: [*, n_atom, n_atom] pairwise interaction filter
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
    Returns:
        out: dictionary containing validation metrics
            'lddt_intra_protein': intra protein lddt
            'lddt_inter_protein_protein: inter protein-protein lddt
            'drmsd_intra_protein: intra protein drmsd
    """
    out = {}

    if torch.any(is_protein_atomized):
        is_protein_atomized = is_protein_atomized.bool()
        bs = is_protein_atomized.shape[:-1]  # (bs, (n_sample),)

        gt_protein = gt_coords[is_protein_atomized].view((bs) + (-1, 3))
        pred_protein = pred_coords[is_protein_atomized].view((bs) + (-1, 3))
        asym_id_protein = asym_id[is_protein_atomized].view((bs) + (-1,))
        all_atom_mask_protein = all_atom_mask[is_protein_atomized].view((bs) + (-1,))
        intra_mask_atomized_protein = intra_mask_atomized[is_protein_atomized].view(
            (bs) + (-1,)
        )

        # Apply pairwise protein mask to get protein index values for inter_chain_mask
        is_protein_atomized_pair = (
            is_protein_atomized[..., None] * is_protein_atomized[..., None, :]
        )  # (1, n_protein, n_protein)
        n_protein_atoms = torch.sum(is_protein_atomized).item()  # n_protein

        inter_mask_atomized_protein = torch.masked_select(
            inter_mask_atomized, is_protein_atomized_pair
        ).reshape(bs + (n_protein_atoms, n_protein_atoms))

        # (bs,(n_sample), n_prot, n_prot)
        gt_protein_pair = torch.sqrt(
            torch.sum(
                (gt_protein.unsqueeze(-2) - gt_protein.unsqueeze(-3)) ** 2, dim=-1
            )
        )
        pred_protein_pair = torch.sqrt(
            torch.sum(
                (pred_protein.unsqueeze(-2) - pred_protein.unsqueeze(-3)) ** 2, dim=-1
            )
        )

        intra_lddt, inter_lddt = lddt(
            pred_protein_pair,
            gt_protein_pair,
            all_atom_mask_protein,
            intra_mask_atomized_protein,
            inter_mask_atomized_protein,
            asym_id_protein,
            cutoff=15.0,
        )
        out["lddt_intra_protein"] = intra_lddt
        out["lddt_inter_protein_protein"] = inter_lddt

        intra_drmsd, _ = drmsd(
            gt_protein_pair,
            pred_protein_pair,
            all_atom_mask_protein,
            asym_id_protein,
        )
        out["drmsd_intra_protein"] = intra_drmsd

        intra_clash, inter_clash = steric_clash(
            pred_protein_pair, all_atom_mask_protein, asym_id_protein, threshold=1.1
        )
        out["clash_intra_protein"] = intra_clash
        out["clash_inter_protein_protein"] = inter_clash
    return out


def get_nucleic_acid_metrics(
    is_nucleic_acid_atomized: torch.Tensor,
    asym_id: torch.Tensor,
    intra_mask_atomized: torch.Tensor,
    inter_mask_atomized: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
    is_protein_atomized: torch.Tensor,
    substrate: str,
) -> dict[str, torch.Tensor]:
    """
    Compute validation metrics of nucleic acids (dna/rna)

    Args:
        is_nucleic_acid_atomized: broadcasted is_dna/rna feature [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom]
        intra_mask_atomized:[*, n_atom] filter for intra chain computations
        inter_mask_atomized: [*, n_atom, n_atom] pairwise interaction filter
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
        is_protein_atomized: broadcasted is_protein feature [*, n_atom]
        substrate: 'rna', 'dna'
    Returns:
        out: dictionary containing validation metrics
            'lddt_intra_f'{dna/rna}': intra dna/rna lddt
            'lddt_inter_f'{dna/rna}'_f'{dna/rna}': inter dna/rna lddt
            'drmsd_intra_f'{dna/rna}': intra dna/rna drmsd
            'lddt_inter_protein_f'{dna/rna}': inter protein-dna/rna lddt

            'lddt_intra_{dna/rna}_15': intra dna/rna lddt with 15 A radius
            'lddt_inter_{dna/rna}_{dna/rna}_15': inter lddt with 15 A radius
            'lddt_inter_protein_{dna/rna}_15': inter protein-dna/rna lddt

    Notes:
        if there exists no appropriate substrate: returns an empty dict {}
        function is compatible with multiple samples,
            not compatible with batch with different number of atoms/substrates
    """
    out = {}

    if torch.any(is_nucleic_acid_atomized):
        is_nucleic_acid_atomized = is_nucleic_acid_atomized.bool()
        is_protein_atomized = is_protein_atomized.bool()

        bs = is_nucleic_acid_atomized.shape[:-1]  # (bs, (n_sample),)

        # getting appropriate atoms of shape (bs, (n_sample), n_na, (3)),
        gt_protein = gt_coords[is_protein_atomized].view((bs) + (-1, 3))
        gt_na = gt_coords[is_nucleic_acid_atomized].view((bs) + (-1, 3))
        pred_protein = pred_coords[is_protein_atomized].view((bs) + (-1, 3))
        pred_na = pred_coords[is_nucleic_acid_atomized].view((bs) + (-1, 3))
        asym_id_na = asym_id[is_nucleic_acid_atomized].view((bs) + (-1,))

        all_atom_mask_protein = all_atom_mask[is_protein_atomized].view((bs) + (-1,))
        all_atom_mask_na = all_atom_mask[is_nucleic_acid_atomized].view((bs) + (-1,))
        intra_mask_atomized_na = intra_mask_atomized[is_nucleic_acid_atomized].view(
            (bs) + (-1,)
        )

        # Apply pairwise na mask to get intra na interactions
        is_nucleic_acid_atomized_pair = (
            is_nucleic_acid_atomized[..., None] * is_nucleic_acid_atomized[..., None, :]
        )
        n_nucleic_acid_atoms = torch.sum(is_nucleic_acid_atomized).item()
        inter_mask_atomized_na = torch.masked_select(
            inter_mask_atomized, is_nucleic_acid_atomized_pair
        ).reshape(bs + (n_nucleic_acid_atoms, n_nucleic_acid_atoms))

        # Apply protein x na masks to select protein - na interactions
        is_protein_na_pair = (
            is_protein_atomized[..., None] * is_nucleic_acid_atomized[..., None, :]
        )
        n_protein_atoms = torch.sum(is_protein_atomized).item()
        inter_filter_mask = torch.masked_select(
            inter_mask_atomized, is_protein_na_pair
        ).reshape(bs + (n_protein_atoms, n_nucleic_acid_atoms))

        # (bs,(n_sample), n_na, n_na)
        gt_na_pair = torch.sqrt(
            torch.sum((gt_na.unsqueeze(-2) - gt_na.unsqueeze(-3)) ** 2, dim=-1)
        )
        pred_na_pair = torch.sqrt(
            torch.sum((pred_na.unsqueeze(-2) - pred_na.unsqueeze(-3)) ** 2, dim=-1)
        )

        intra_lddt, inter_lddt = lddt(
            pred_na_pair,
            gt_na_pair,
            all_atom_mask_na,
            intra_mask_atomized_na,
            inter_mask_atomized_na,
            asym_id_na,
            cutoff=30.0,
        )
        out["lddt_intra_" + substrate] = intra_lddt
        out["lddt_inter_" + substrate + "_" + substrate] = inter_lddt

        intra_drmsd, _ = drmsd(
            pred_na_pair,
            gt_na_pair,
            all_atom_mask_na,
            asym_id_na,
        )
        out["drmsd_intra_" + substrate] = intra_drmsd

        intra_lddt_15, inter_lddt_15 = lddt(
            pred_na_pair,
            gt_na_pair,
            all_atom_mask_na,
            intra_mask_atomized_na,
            inter_mask_atomized_na,
            asym_id_na,
            cutoff=15.0,
        )
        out["lddt_intra_" + substrate + "_15"] = intra_lddt_15
        out["lddt_inter_" + substrate + "_" + substrate + "_15"] = inter_lddt_15

        # ilddt with protein
        inter_lddt_protein_na = interface_lddt(
            pred_protein,
            pred_na,
            gt_protein,
            gt_na,
            all_atom_mask_protein,
            all_atom_mask_na,
            inter_filter_mask,
            cutoff=30.0,
        )
        out["lddt_inter_protein_" + substrate] = inter_lddt_protein_na

        inter_lddt_protein_na_15 = interface_lddt(
            pred_protein,
            pred_na,
            gt_protein,
            gt_na,
            all_atom_mask_protein,
            all_atom_mask_na,
            inter_filter_mask,
            cutoff=15.0,
        )
        out["lddt_inter_protein_" + substrate + "_15"] = inter_lddt_protein_na_15

        intra_clash, inter_clash = steric_clash(
            pred_na_pair, all_atom_mask_na, asym_id_na, threshold=1.1
        )
        out["clash_intra_" + substrate] = intra_clash
        out["clash_inter_" + substrate + "_" + substrate] = inter_clash

        interface_clash = interface_steric_clash(
            pred_protein,
            pred_na,
            all_atom_mask_protein,
            all_atom_mask_na,
            threshold=1.1,
        )
        out["clash_inter_protein_" + substrate] = interface_clash
    return out


def get_ligand_metrics(
    is_ligand_atomized: torch.Tensor,
    asym_id: torch.Tensor,
    intra_mask_atomized: torch.Tensor,
    inter_mask_atomized: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
    is_protein_atomized: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute validation metrics of a ligand

    Args:
        is_ligand_atomized: broadcasted is_ligand feature [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom]
        intra_mask_atomized:
        inter_mask_atomized:
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
        is_protein_atomized: broadcasted is_protein feature [*, n_atom]
    Returns:
        out: dictionary containing validation metrics
            'lddt_intra_ligand: intra ligand lddt
            'lddt_inter_ligand_ligand: inter ligand-ligand lddt
            'lddt_inter_protein_ligand': inter protein-ligand lddt
            'drmsd_intra_ligand': intra ligand drmsd

            'lddt_intra_ligand_uha': intra ligand lddt with [0.25, 0.5, 0.75, 1.]
            'lddt_inter_ligand_ligand_uha': inter ligand lddt with above threshold

    Notes:
        if there exists no appropriate substrate: returns an empty dict {}
        function is compatible with multiple samples,
            not compatible with batch with different number of atoms/substrates
    """
    out = {}

    if torch.any(is_ligand_atomized):
        is_ligand_atomized = is_ligand_atomized.bool()
        is_protein_atomized = is_protein_atomized.bool()

        bs = is_ligand_atomized.shape[:-1]  # (bs, (n_sample),)

        # getting appropriate atoms of shape (bs, (n_sample), n_protein/ligand, (3)),
        gt_protein = gt_coords[is_protein_atomized].view((bs) + (-1, 3))
        gt_ligand = gt_coords[is_ligand_atomized].view((bs) + (-1, 3))
        pred_protein = pred_coords[is_protein_atomized].view((bs) + (-1, 3))
        pred_ligand = pred_coords[is_ligand_atomized].view((bs) + (-1, 3))
        asym_id_ligand = asym_id[is_ligand_atomized].view((bs) + (-1,))

        all_atom_mask_protein = all_atom_mask[is_protein_atomized].view((bs) + (-1,))
        all_atom_mask_ligand = all_atom_mask[is_ligand_atomized].view((bs) + (-1,))
        intra_mask_atomized_ligand = intra_mask_atomized[is_ligand_atomized].view(
            (bs) + (-1,)
        )

        # Apply pairwise na mask to get intra na interactions
        is_ligand_atomized_pair = (
            is_ligand_atomized[..., None] * is_ligand_atomized[..., None, :]
        )
        n_ligand_atoms = torch.sum(is_ligand_atomized).item()
        inter_mask_atomized_ligand = torch.masked_select(
            inter_mask_atomized, is_ligand_atomized_pair
        ).reshape(bs + (n_ligand_atoms, n_ligand_atoms))

        # Apply protein x na masks to select protein - na interactions
        is_protein_ligand_pair = (
            is_protein_atomized[..., None] * is_ligand_atomized[..., None, :]
        )
        n_protein_atoms = torch.sum(is_protein_atomized).item()
        inter_filter_mask = torch.masked_select(
            inter_mask_atomized, is_protein_ligand_pair
        ).reshape(bs + (n_protein_atoms, n_ligand_atoms))

        # (bs,(n_sample), n_lig, n_lig)
        gt_ligand_pair = torch.sqrt(
            torch.sum((gt_ligand.unsqueeze(-2) - gt_ligand.unsqueeze(-3)) ** 2, dim=-1)
        )
        pred_ligand_pair = torch.sqrt(
            torch.sum(
                (pred_ligand.unsqueeze(-2) - pred_ligand.unsqueeze(-3)) ** 2, dim=-1
            )
        )

        intra_lddt, inter_lddt = lddt(
            pred_ligand_pair,
            gt_ligand_pair,
            all_atom_mask_ligand,
            intra_mask_atomized_ligand,
            inter_mask_atomized_ligand,
            asym_id_ligand,
            cutoff=15.0,
        )
        out["lddt_intra_ligand"] = intra_lddt
        out["lddt_inter_ligand_ligand"] = inter_lddt

        # get tighter threshold lddts
        intra_lddt_uha, inter_lddt_uha = lddt(
            pred_ligand_pair,
            gt_ligand_pair,
            all_atom_mask_ligand,
            intra_mask_atomized_ligand,
            inter_mask_atomized_ligand,
            asym_id_ligand,
            threshold=[0.25, 0.5, 0.75, 1.0],
            cutoff=15.0,
        )
        out["lddt_intra_ligand_uha"] = intra_lddt_uha
        out["lddt_inter_ligand_ligand_uha"] = inter_lddt_uha

        # ilddt with protein
        inter_lddt_protein_ligand = interface_lddt(
            pred_protein,
            pred_ligand,
            gt_protein,
            gt_ligand,
            all_atom_mask_protein,
            all_atom_mask_ligand,
            inter_filter_mask,
            cutoff=15.0,
        )
        out["lddt_inter_protein_ligand"] = inter_lddt_protein_ligand

        intra_drmsd, _ = drmsd(
            pred_ligand_pair,
            gt_ligand_pair,
            all_atom_mask_ligand,
            asym_id_ligand,
        )
        out["drmsd_intra_ligand"] = intra_drmsd

        intra_clash, inter_clash = steric_clash(
            pred_ligand_pair, all_atom_mask_ligand, asym_id_ligand, threshold=1.1
        )
        out["clash_intra_ligand"] = intra_clash
        out["clash_inter_ligand_ligand"] = inter_clash

        interface_clash = interface_steric_clash(
            pred_protein,
            pred_ligand,
            all_atom_mask_protein,
            all_atom_mask_ligand,
            threshold=1.1,
        )
        out["clash_inter_protein_ligand"] = interface_clash
    return out


def steric_clash(
    pred_pair: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    threshold: Optional[float] = 1.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes steric clash score

    Args:
        pred_pair: pairwise distance of predicted positions [*, n_atom, n_atom]
        all_atom_mask: atom mask [*, n_atom]
        asym_id: asym id [*, n_atom]
        threshold: threshold to define if there is steric clash
            Based on AF3 (SI 5.9.), define threshold as 1.1 Angstrom
            By no means perfect, a good threshold to capture any heavy atoms clashes
    Returns:
        intra_clash_score: steric clash for atoms with same asym_id (intra-chain)
        inter_clash_score: steric clash for atoms with different asym_id (inter-chain)

    Note:
        clash_scores in range (0, 1) s.t.
            0 (no atom pair having distance less than threshold) to
            1 (all atoms having same coordinate)
    """
    # Create mask
    n_atom = pred_pair.shape[-2]
    mask = (1 - torch.eye(n_atom).to(all_atom_mask.device)) * (
        all_atom_mask.unsqueeze(-1) * all_atom_mask.unsqueeze(-2)
    )

    intra = torch.where(asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2), 1, 0).float()
    inter = 1 - intra

    # Compute the clash
    clash = torch.relu(threshold - pred_pair)

    intra_clash = torch.full(pred_pair.shape[:-2], torch.nan)
    intra_mask = mask * intra
    if torch.any(intra_mask):
        intra_clash = torch.sum(clash * intra_mask, dim=(-1, -2)) / torch.sum(
            intra_mask, dim=(-1, -2)
        )
        intra_clash = intra_clash / threshold

    inter_clash = torch.full(pred_pair.shape[:-2], torch.nan)
    inter_mask = mask * inter
    if torch.any(inter_mask):
        inter_clash = torch.sum(clash * inter_mask, dim=(-1, -2)) / torch.sum(
            inter_mask, dim=(-1, -2)
        )
        inter_clash = inter_clash / threshold

    return intra_clash, inter_clash


def interface_steric_clash(
    pred_protein: torch.Tensor,
    pred_substrate: torch.Tensor,
    all_atom_mask_protein: torch.Tensor,
    all_atom_mask_substrate: torch.Tensor,
    threshold: Optional[float] = 1.1,
) -> torch.Tensor:
    """
    Computes steric clash score across protein and substrate

    Args:
        pred_protein: predicted protein coordinates [*, n_protein, 3]
        pred_substrate: predicted substrate coordinates [*, n_substrate, 3]
        all_atom_mask_protein: protein atom mask
        all_atom_mask_substrate: substrate atom mask
        threshold: threshold definiing if two atoms have any steric clash
    Returns:
        interface_clash: clash between protein and substrate interface

    Note:
        interface_clash score in range (0, 1) s.t.
            0 (no atom pair having distance less than threshold) to
            1 (all atoms having same coordinate)
    """
    # pair distance
    pair_dist = torch.sqrt(
        torch.sum(
            (pred_protein.unsqueeze(-2) - pred_substrate.unsqueeze(-3)) ** 2, dim=-1
        )
    )
    mask = all_atom_mask_protein.unsqueeze(-1) * all_atom_mask_substrate.unsqueeze(-2)
    clash = torch.relu(threshold - pair_dist)

    interface_clash = torch.full(pair_dist.shape[:-2], torch.nan)
    if torch.any(mask):
        interface_clash = torch.sum(clash * mask, dim=(-1, -2)) / torch.sum(
            mask, dim=(-1, -2)
        )
        interface_clash = interface_clash / threshold

    return interface_clash


def gdt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_gt_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoffs: list,
) -> torch.Tensor:
    """
    Calculates gdt scores

    Args:
        all_atom_pred_pos: predicted structures [*, n_atom, 3]
        all_atom_gt_pos: gt structure [*, n_atom, 3]
        all_atom_mask: mask [*, n_atom]
        cutoffs: list of cutoffs
    Returns:
        gdt score: [*]
    """

    distances = torch.sqrt(
        torch.sum((all_atom_pred_pos - all_atom_gt_pos) ** 2, dim=-1)
    )

    n = torch.sum(all_atom_mask, dim=-1)
    scores = []
    for c in cutoffs:
        score = torch.sum((distances <= c) * all_atom_mask, dim=-1)
        score = score / n
        scores.append(score)
    return torch.sum(torch.stack(scores, dim=-1), dim=-1) / len(scores)


def gdt_ts(p1, p2, mask):
    return gdt(p1, p2, mask, [1.0, 2.0, 4.0, 8.0])


def gdt_ha(p1, p2, mask):
    return gdt(p1, p2, mask, [0.5, 1.0, 2.0, 4.0])


def batched_kabsch(
    all_atom_pred_pos: torch.Tensor,
    all_atom_gt_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes optimal rotation and translation via Kabsch algorithm

    Args:
        all_atom_pred_pos: [*, n_atom, 3]
        all_atom_gt_pos: [*, n_atom, 3]
        all_atom_mask: [*, n_atom]

    Returns:
        optimal translation: [*, 1, 3]
        optimal rotation: [*, 3, 3]
        rmsd: alignment rmsd [*]
    """

    n_atom = all_atom_gt_pos.shape[-2]
    predicted_coordinates = all_atom_pred_pos * all_atom_mask.unsqueeze(-1)
    gt_coordinates = all_atom_gt_pos * all_atom_mask.unsqueeze(-1)

    # translation: center two molecules
    centroid_predicted = torch.mean(predicted_coordinates, dim=-2, keepdim=True)
    centroid_gt = torch.mean(gt_coordinates, dim=-2, keepdim=True)
    translation = centroid_gt - centroid_predicted
    predicted_coordinates_centered = predicted_coordinates - centroid_predicted
    gt_coordinates_centered = gt_coordinates - centroid_gt

    # SVD
    H = predicted_coordinates_centered.transpose(-2, -1) @ gt_coordinates_centered
    # fp16 not supported
    with torch.cuda.amp.autocast(enabled=False):
        U, S, Vt = torch.linalg.svd(H.float())
        Ut, V = U.transpose(-1, -2), Vt.transpose(-1, -2)  #

    # determine handedness
    dets = torch.det(V @ Ut)  # just do U @ Vt
    batch_dims = H.shape[:-2]
    D = torch.eye(3).tile(*batch_dims, 1, 1).to(V.device)
    D[..., -1, -1] = torch.sign(dets).to(torch.float64)

    rotation = V @ D @ Ut
    rmsd = torch.sqrt(
        torch.sum(
            (
                predicted_coordinates_centered @ rotation.transpose(-2, -1)
                - gt_coordinates_centered
            )
            ** 2,
            dim=(-1, -2),
        )
        / n_atom
    )

    return translation, rotation, rmsd


def get_superimpose_metrics(
    all_atom_pred_pos: torch.Tensor,
    all_atom_gt_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Computes superimposition metrics

    Args:
        all_atom_pred_pos: pred coordinates [*, n_atom, 3]
        all_atom_gt_pos: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
    Returns:
        out: a dictionary containing following metrics
            superimpose_rmsd: rmsd after superimposition [*]
            gdt_ts: gdt_ts [*]
            gdt_ha: gdt_ha [*]
    """
    out = {}

    _, rotation, rmsd = batched_kabsch(
        all_atom_pred_pos,
        all_atom_gt_pos,
        all_atom_mask,
    )
    out["superimpose_rmsd"] = rmsd

    pred_centered = all_atom_pred_pos - torch.mean(
        all_atom_pred_pos, dim=-2, keepdim=True
    )
    gt_centered = all_atom_gt_pos - torch.mean(
        all_atom_gt_pos,
        dim=-2,
        keepdim=True,
    )
    pred_superimposed = pred_centered @ rotation.transpose(-1, -2)

    gdt_ts_score = gdt_ts(
        pred_superimposed,
        gt_centered,
        all_atom_mask,
    )
    gdt_ha_score = gdt_ha(
        pred_superimposed,
        gt_centered,
        all_atom_mask,
    )
    out["gdt_ts"] = gdt_ts_score
    out["gdt_ha"] = gdt_ha_score
    return out


def get_full_complex_lddt(
    asym_id: torch.Tensor,
    intra_filter_atomized: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Computes lddt for the full complex, subject to intra chain filters

    Args:
        asym_id: atomized asym_id feature [*, n_atom]
        intra_mask_atomized:[*, n_atom] filter for intra chain computations
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
    Returns:
        out: dictionary containing validation metrics
            'lddt_complex': full complex lddt score
    """
    out = {}
    if torch.any(intra_filter_atomized):
        # do the whole complex lddt
        gt_pair = torch.sqrt(
            torch.sum((gt_coords.unsqueeze(-2) - gt_coords.unsqueeze(-3)) ** 2, dim=-1)
        )
        pred_pair = torch.sqrt(
            torch.sum(
                (pred_coords.unsqueeze(-2) - pred_coords.unsqueeze(-3)) ** 2, dim=-1
            )
        )

        # mask out all inter chain computations
        inter_filter_atomized_zeros = torch.zeros(
            (intra_filter_atomized.shape[-1], intra_filter_atomized.shape[-1])
        ).to(asym_id.device)

        complex_lddt, _ = lddt(
            gt_pair,
            pred_pair,
            all_atom_mask,
            intra_filter_atomized,
            inter_filter_atomized_zeros,
            asym_id,
        )
        out["lddt_complex"] = complex_lddt
    return out


def get_plddt_metrics(
    is_protein_atomized: torch.Tensor,
    is_ligand_atomized: torch.Tensor,
    is_rna_atomized: torch.Tensor,
    is_dna_atomized: torch.Tensor,
    intra_filter_atomized: torch.Tensor,
    plddt_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute plddt metric and report for different atom types.
    Args:
        is_protein_atomized: broadcasted is_protein feature [*, n_atom]
        is_ligand_atomized: broadcasted is_ligand feature [*, n_atom]
        is_rna_atomized: broadcasted is_rna feature [*, n_atom]
        is_dna_atomized: broadcasted is_dna feature [*, n_atom]
        intra_mask_atomized:[*, n_atom] filter for intra chain computations
        plddt_logits: [*, n_atom, 50] prediction output of lddt from model
    Returns:
        out: dictionary containing validation metrics
            'lddt_intra_protein': intra protein lddt
            'lddt_inter_protein_protein: inter protein-protein lddt
            'drmsd_intra_protein: intra protein drmsd
    """

    out = {}

    if torch.any(intra_filter_atomized):
        plddt_complex = compute_plddt(plddt_logits)
        out["plddt_complex"] = torch.sum(
            plddt_complex * intra_filter_atomized, dim=-1
        ) / torch.sum(intra_filter_atomized)

    is_protein_atomized = is_protein_atomized * intra_filter_atomized
    is_ligand_atomized = is_ligand_atomized * intra_filter_atomized
    is_rna_atomized = is_rna_atomized * intra_filter_atomized
    is_dna_atomized = is_dna_atomized * intra_filter_atomized

    if torch.any(is_protein_atomized):
        plddt_logits_protein = plddt_complex[is_protein_atomized.bool()]
        out["plddt_protein"] = torch.sum(plddt_logits_protein) / torch.sum(
            is_protein_atomized
        )

    if torch.any(is_ligand_atomized):
        plddt_logits_ligand = plddt_complex[is_ligand_atomized.bool()]
        out["plddt_ligand"] = torch.sum(plddt_logits_ligand) / torch.sum(
            is_ligand_atomized
        )

    if torch.any(is_rna_atomized):
        plddt_logits_rna = plddt_complex[is_rna_atomized.bool()]
        out["plddt_rna"] = torch.sum(plddt_logits_rna) / torch.sum(is_rna_atomized)

    if torch.any(is_dna_atomized):
        plddt_logits_dna = plddt_complex[is_dna_atomized.bool()]
        out["plddt_dna"] = torch.sum(plddt_logits_dna) / torch.sum(is_dna_atomized)

    return out


def get_validation_lddt_metrics(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    is_ligand_atomized: torch.Tensor,
    is_rna_atomized: torch.Tensor,
    is_dna_atomized: torch.Tensor,
    is_modified_residue_atomized: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id_atomized: torch.Tensor,
    intra_filter_atomized: torch.Tensor,
    inter_filter_atomized: torch.Tensor,
):
    """Compute lddt metrics for ligand-RNA, ligand-DNA and modified residues.
    These extra metrics are required for model selection metric.

    Args:
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        is_ligand_acid_atomized: broadcasted is_ligand feature [*, n_atom]
        is_rna_atomized: broadcasted is_rna feature [*, n_atom]
        is_dna_acid_atomized: broadcasted is_dna feature [*, n_atom]
        is_modified_residue_atomized: broadcasted is_modified_residue feature [*, n_atom]
        all_atom_mask: atom mask [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom]
        intra_mask_atomized:[*, n_atom] filter for intra chain computations
        inter_mask_atomized: [*, n_atom, n_atom] pairwise interaction filter
    Returns:
        out: dictionary containing validation metrics, if applicable
            'lddt_inter_ligand_dna': inter ligand dna lddt
            'lddt_inter_ligand_rna': inter ligand rna lddt
            'lddt_intra_modified_residue': intra modified residue lddt

    Notes:
        if there exists no appropriate substrate: returns an empty dict {}
        function is compatible with multiple samples,
            not compatible with batch with different number of atoms/substrates

    """
    metrics = {}
    bs = is_ligand_atomized.shape[:-1]  # (bs, (n_sample),)

    if torch.any(is_ligand_atomized) and torch.any(is_rna_atomized):
        is_rna_ligand_pair = (
            is_rna_atomized[..., None] * is_ligand_atomized[..., None, :]
        )
        n_rna_atoms = torch.sum(is_rna_atomized).to(int).item()
        n_ligand_atoms = torch.sum(is_ligand_atomized).to(int).item()
        # TODO: just works for bs = 1, sample size = 1
        inter_filter_mask_rna_ligand = torch.masked_select(
            inter_filter_atomized, is_rna_ligand_pair.bool()
        ).reshape(bs + (n_rna_atoms, n_ligand_atoms))
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
        ).reshape(bs + (n_dna_atoms, n_ligand_atoms))

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
            metrics["lddt_inter_ligand_dna"] = lddt_inter_ligand_dna

    if torch.any(is_modified_residue_atomized):
        pred_mr = pred_coords[is_modified_residue_atomized.bool()].view((bs) + (-1, 3))
        gt_mr = gt_coords[is_modified_residue_atomized.bool()].view((bs) + (-1, 3))
        intra_mask_atomized_mr = intra_filter_atomized[
            is_modified_residue_atomized.bool()
        ].view((bs) + (-1,))
        is_mr_atomized_pair = (
            is_modified_residue_atomized[..., None]
            * is_modified_residue_atomized[..., None, :]
        )

        n_mr_atoms = torch.sum(is_modified_residue_atomized).to(int).item()
        inter_mask_atomized_mr = torch.masked_select(
            inter_filter_atomized, is_mr_atomized_pair.bool()
        ).reshape(bs + (n_mr_atoms, n_mr_atoms))

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
            all_atom_mask[is_modified_residue_atomized.bool()].view((bs) + (-1,)),
            intra_mask_atomized_mr,
            inter_mask_atomized_mr,
            asym_id_atomized[is_modified_residue_atomized.bool()].view((bs) + (-1,)),
        )

        if not (torch.isnan(lddt_intra_modified_residues)).all():
            lddt_intra_modified_residues[~torch.isnan(lddt_intra_modified_residues)]
            metrics["lddt_intra_modified_residues"] = lddt_intra_modified_residues

    return metrics


def get_metrics(
    batch,
    outputs,
    superimposition_metrics=False,
    is_train=True,
) -> dict[str, torch.Tensor]:
    """
    Compute validation metrics on all substrates

    Args:
        batch: ground truth and permutation applied features
        outputs: model outputs
        superimposition_metrics: computes superimposition metrics
    Returns:
        metrics: dict containing validation metrics across all substrates
            'lddt_intra_protein': intra protein lddt
            'lddt_intra_ligand': intra ligand lddt
            'lddt_intra_dna': intra dna lddt
            'lddt_intra_rna': intra rna lddt
            'lddt_inter_protein_protein': inter protein protein lddt
            'lddt_inter_protein_ligand': inter protein ligand lddt
            'lddt_inter_protein_dna;: inter protein dna lddt
            'lddt_inter_protein_rna': inter protein rna lddt
            'drmsd_intra_protein': intra protein drmsd
            'drmsd_intra_ligand': intra ligand drmsd
            'drmsd_intra_dna': intra dna drmsd
            'drmsd_intra_rna': intra rna drmsd

    Note:
        if no appropriate substrates, no corresponding metrics will be included
    """
    metrics = {}

    gt_coords = batch["ground_truth"]["atom_positions"].float()
    pred_coords = outputs["atom_positions_predicted"].float()
    all_atom_mask = batch["ground_truth"]["atom_resolved_mask"].bool()
    token_mask = batch["token_mask"]
    num_atoms_per_token = batch["num_atoms_per_token"]

    # getting rid of modified residues
    is_protein = batch["is_protein"]
    is_rna = batch["is_rna"]
    is_dna = batch["is_dna"]
    not_modified_res = 1 - batch["is_atomized"]
    is_protein = is_protein * not_modified_res
    is_rna = is_rna * not_modified_res
    is_dna = is_dna * not_modified_res

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
    is_modified_residue = batch["is_atomized"]
    is_modified_residue = is_modified_residue * (1 - batch["is_ligand"])
    is_modified_residue_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, is_modified_residue
    )
    is_modified_residue_atomized = is_modified_residue_atomized.bool()

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

    protein_validation_metrics = get_protein_metrics(
        is_protein_atomized,
        asym_id_atomized,
        intra_filter_atomized,
        inter_filter_atomized,
        pred_coords,
        gt_coords,
        all_atom_mask,
    )
    metrics = metrics | protein_validation_metrics

    ligand_validation_metrics = get_ligand_metrics(
        is_ligand_atomized,
        asym_id_atomized,
        intra_filter_atomized,
        inter_filter_atomized,
        pred_coords,
        gt_coords,
        all_atom_mask,
        is_protein_atomized,
    )
    metrics = metrics | ligand_validation_metrics

    rna_validation_metrics = get_nucleic_acid_metrics(
        is_rna_atomized,
        asym_id_atomized,
        intra_filter_atomized,
        inter_filter_atomized,
        pred_coords,
        gt_coords,
        all_atom_mask,
        is_protein_atomized,
        substrate="rna",
    )
    metrics = metrics | rna_validation_metrics

    dna_validation_metrics = get_nucleic_acid_metrics(
        is_dna_atomized,
        asym_id_atomized,
        intra_filter_atomized,
        inter_filter_atomized,
        pred_coords,
        gt_coords,
        all_atom_mask,
        is_protein_atomized,
        substrate="dna",
    )
    metrics = metrics | dna_validation_metrics

    full_complex_lddt_metrics = get_full_complex_lddt(
        asym_id_atomized, intra_filter_atomized, pred_coords, gt_coords, all_atom_mask
    )
    metrics = metrics | full_complex_lddt_metrics

    plddt_metrics = get_plddt_metrics(
        is_protein_atomized,
        is_ligand_atomized,
        is_rna_atomized,
        is_dna_atomized,
        intra_filter_atomized,
        outputs["plddt_logits"],
    )
    metrics = metrics | plddt_metrics

    if superimposition_metrics:
        superimpose_metrics = get_superimpose_metrics(
            pred_coords,
            gt_coords,
            all_atom_mask,
        )
        metrics = metrics | superimpose_metrics

    if not is_train:
        extra_lddt = get_validation_lddt_metrics(
            pred_coords,
            gt_coords,
            is_ligand_atomized,
            is_rna_atomized,
            is_dna_atomized,
            is_modified_residue_atomized,
            all_atom_mask,
            asym_id_atomized,
            intra_filter_atomized,
            inter_filter_atomized,
        )
        metrics = metrics | extra_lddt

    metrics = {
        k: v[~nan_mask]
        for k, v in metrics.items()
        if not (nan_mask := torch.isnan(v)).all()
    }

    return metrics
