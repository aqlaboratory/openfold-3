from typing import List, Optional

import torch


#suboptimal but works
def create_intra_mask(asym_id: torch.Tensor,
                      ):
    """
    create a mask of diagonal square blocks 

    Args: 
        asym_id: atomized asym_id [n_atom]
    Returns: 
        mask: pair mask of diagonal square blocks [n_atom, n_atom]
    """
    n_atom = asym_id.shape[-1]
    mask = torch.zeros(n_atom, n_atom)
    unique_ids = torch.unique(asym_id)
    for id in unique_ids:
        idx = torch.argwhere(asym_id == id.item()).squeeze(-1)
        mask[idx[..., None], idx] = 1
    return mask

def lddt(
    pair_dist_pred_pos: torch.Tensor,
    pair_dist_gt_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id: torch.Tensor, 
    cutoff: Optional[float] = 15.0,
    eps: Optional[float] = 1e-10,
) -> torch.Tensor:
    """
    Calculates lddt scores from pair distances

    Args: 
        pair_dist_pred_pos: pairwise distance of prediction [*, n_atom, n_atom, 3]
        all_atom_positions: pairwise distance of gt [*, n_atom, n_atom, 3]
        all_atom_mask: mask [*, n_atom]
        asym_id: entity id [*, n_atom]
        cutoff: distance cutoff  
            - Nucleic Acids (DNA/RNA) 30.
            - Other biomolecules (Protein/Ligands) 15.
        eps: epsilon 

    Returns: 
        intra_score: intra lddt scores [*]
        inter_score: inter lddt scores [*]
    """

    if len(pair_dist_pred_pos.shape) != len(pair_dist_gt_positions.shape):
        pair_dist_gt_positions = pair_dist_gt_positions.unsqueeze(-3)
        all_atom_mask = all_atom_mask.unsqueeze(-2)

    # create a mask
    n_atom = pair_dist_gt_positions.shape[-2]    
    dists_to_score = (
        (pair_dist_gt_positions < cutoff) * 
        (all_atom_mask[..., None] * all_atom_mask[..., None, :] * 
        (1.0 - torch.eye(n_atom, device=all_atom_mask.device))
        )
        )

    intra_mask = create_intra_mask(asym_id)
    inter_mask = 1 - intra_mask

    dist_l1 = torch.abs(pair_dist_gt_positions - pair_dist_pred_pos)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype) + 
        (dist_l1 < 1.0).type(dist_l1.dtype) + 
        (dist_l1 < 2.0).type(dist_l1.dtype) + 
        (dist_l1 < 4.0).type(dist_l1.dtype)
        )
    score = score * 0.25

    # normalize
    intra_norm = 1.0 / (eps + torch.sum(dists_to_score * intra_mask, dim= (-1, -2)))
    intra_score = intra_norm * (eps + torch.sum(dists_to_score * intra_mask * score, 
                                                dim= (-1, -2))
                                                )

    inter_norm = 1.0 / (eps + torch.sum(dists_to_score * inter_mask, dim= (-1, -2)))
    inter_score = inter_norm * (eps + torch.sum(dists_to_score * score * inter_mask, 
                                                dim= (-1, -2))
                                                )

    return intra_score, inter_score

def interface_lddt(
    all_atom_pred_pos_1: torch.Tensor,
    all_atom_pred_pos_2: torch.Tensor,
    all_atom_positions_1: torch.Tensor,
    all_atom_positions_2: torch.Tensor,
    all_atom_mask1: torch.Tensor,
    all_atom_mask2: torch.Tensor,
    cutoff: Optional[float] = 15.0,
    eps: Optional[float] = 1e-10,
) -> torch.Tensor:
    """
    Calculates interface_lddt score between two molecules (molecule1, molecule2)

    Args: 
        all_atom_pred_pos_1: predicted protein coordinates  [*, n_atom1, 3]
        all_atom_pred_pos_2: predicted interacting molecule coordinates [*, n_atom2, 3]
        all_atom_positions_1: gt protein coordinates [*, n_atom1, 3]
        all_atom_positions_2: gt interacting molecule coordinates  [*, n_atom2, 3]
        all_atom_mask1: protein atom mask [*, n_atom1]
        all_atom_mask2: interacting molecule atom maks [*, n_atom2]
        cutoff: distance cutoff  
            - Nucleic Acids (DNA/RNA) 30.  
            - Others(Protein/Ligands) 15. 
        eps: epsilon 

    Returns: 
        scores: ilddt scores [*]
    """ 

    if len(all_atom_pred_pos_1.shape) != len(all_atom_positions_1.shape):
        all_atom_positions_1 = all_atom_positions_1.unsqueeze(-3)
        all_atom_positions_2 = all_atom_positions_2.unsqueeze(-3)
        all_atom_mask1 = all_atom_mask1.unsqueeze(-2)
        all_atom_mask2 = all_atom_mask2.unsqueeze(-2)

    # get pairwise distance
    pair_dist_true = torch.cdist(all_atom_positions_1, all_atom_positions_2)
    pair_dist_pred = torch.cdist(all_atom_pred_pos_1, all_atom_pred_pos_2)

    # create a mask
    dists_to_score = (
        (pair_dist_true < cutoff) * 
        (all_atom_mask1[..., None] * all_atom_mask2[..., None, :]
         )
         )

    dist_l1 = torch.abs(pair_dist_true - pair_dist_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype) + 
        (dist_l1 < 1.0).type(dist_l1.dtype) + 
        (dist_l1 < 2.0).type(dist_l1.dtype) + 
        (dist_l1 < 4.0).type(dist_l1.dtype)
        )
    score = score * 0.25

    # normalize
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim= (-1, -2)))
    score = norm * (eps + torch.sum(dists_to_score * score, dim= (-1, -2)))

    return score

def drmsd(
        pair_dist_pred_pos: torch.Tensor,
        pair_dist_gt_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        asym_id: torch.Tensor,
        ) -> torch.Tensor:
    """ 
    Computes drmsds

    Args: 
        all_atom_pred_pos: predicted coordinates [*, n_atom, 3]
        all_atom_positions: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [n_atom]
        asym_id: asym_id [n_atom]

    Returns:
        intra_drmsd: computed intra_drmsd
        inter_drmsd: computed inter_drmsd
    """
    if pair_dist_pred_pos.shape != pair_dist_gt_positions.shape:
        pair_dist_gt_positions = pair_dist_gt_positions.unsqueeze(-3) 
        all_atom_mask = all_atom_mask.unsqueeze(-2) 

    drmsd = pair_dist_pred_pos - pair_dist_gt_positions 
    drmsd = drmsd ** 2 
    
    # apply mask
    mask = (all_atom_mask[..., None] * all_atom_mask[..., None, :]) 
    intra_mask = create_intra_mask(asym_id)
    inter_mask = 1 - intra_mask

    intra_drmsd = drmsd * (mask * intra_mask)
    inter_drmsd = drmsd * (mask * inter_mask)
    intra_drmsd = torch.sum(intra_drmsd, dim=(-1, -2))
    inter_drmsd = torch.sum(inter_drmsd, dim=(-1, -2))

    n_intra = torch.sum(intra_mask, dim = (-1, -2))
    n_inter = torch.sum(inter_mask, dim = (-1, -2))
    
    intra_drmsd = intra_drmsd * (1 / (n_intra))
    inter_drmsd = inter_drmsd * (1 / (n_inter))
    
    intra_drmsd = torch.sqrt(intra_drmsd) 
    inter_drmsd = torch.sqrt(inter_drmsd) 
    return intra_drmsd, inter_drmsd

def get_pair_dist(structure1: torch.Tensor, structure2: torch.Tensor):
    return torch.cdist(structure1, structure2)

def get_validation_metrics(
        is_ligand_atomized: torch.Tensor, 
        asym_id_atomized: torch.Tensor,
        pred_coords: torch.Tensor, 
        gt_coords: torch.Tensor, 
        all_atom_mask: torch.Tensor,
        protein_idx: torch.Tensor,
        ligand_type: str,
        is_nucleic_acid: Optional[bool] = False,
        ):
    """ 
    Args: 
        is_ligand_atomized: broadcasted is_ligand/rna/dna feature [n_atom]
        asym_id_atomized: broadcasted asym_id feature [n_atom] 
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [n_atom]
        protein_idx: broadcasted is_protein feature [n_atom]
        ligand_type: 'ligand', 'rna', 'dna' 
        is_nucleic_acid: boolean indicating if ligand type is nucleic acid
    Returns: 
        out: dictionary containing validation metrics
            intra_lddt: intra ligandtype lddt
            inter_lddt: inter ligandtype_ligandtype lddt
            intra_drmsd: intra ligandtype drmsd 
            inter_lddt_protein_ligand: inter protein-ligandtype lddt
    
    Notes: 
        if no ligands: returns an empty dict {}
    """
    
    out = {}

    if torch.any(is_ligand_atomized):
        ligand_idx = torch.nonzero(is_ligand_atomized).squeeze(-1) #[n_lig]
        gt_ligand = gt_coords[..., ligand_idx, :] #[*, n_lig, 3]
        pred_ligand = pred_coords[..., ligand_idx, :] #[*, n_lig, 3]
        ligand_asym_id = asym_id_atomized[..., ligand_idx] #[n_lig]

        gt_ligand_pair = get_pair_dist(gt_ligand, gt_ligand) #[*, nlig, nlig]
        pred_ligand_pair = get_pair_dist(pred_ligand, pred_ligand) #[*, nlig, nlig]

        cutoff = 30. if is_nucleic_acid else 15.

        intra_lddt, inter_lddt = lddt(pred_ligand_pair,
                                      gt_ligand_pair,
                                      all_atom_mask[..., ligand_idx],
                                      ligand_asym_id,
                                      cutoff = cutoff
                                      )
        out['lddt_intra_' + ligand_type] = intra_lddt
        out['lddt_inter_' + ligand_type + '_' + ligand_type] = inter_lddt
        
        intra_drmsd, inter_drmsd = drmsd(pred_ligand_pair,
                                         gt_ligand_pair,
                                         all_atom_mask[..., ligand_idx],
                                         ligand_asym_id,
                                         )
        out['drmsd_intra_' + ligand_type] = intra_drmsd

        if ligand_type != 'protein':
            inter_lddt_protein_ligand = interface_lddt(pred_coords[..., protein_idx, :],
                                                       pred_coords[..., ligand_idx, :],
                                                       gt_coords[..., protein_idx, :],
                                                       gt_coords[..., ligand_idx, :],
                                                       all_atom_mask[..., protein_idx],
                                                       all_atom_mask[..., ligand_idx],
                                                       cutoff = cutoff,
                                                       )
            out['lddt_inter_protein_' + ligand_type] = inter_lddt_protein_ligand
    
    return out

def gdt(
        all_atom_pred_pos: torch.Tensor,
        all_atom_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        cutoffs: List,
        ) -> torch.Tensor:
    """ 
    Calculates gdt

    Args: 
        all_atom_pred_pos: predicted structures [*, 48, n_atom, 3]
        all_atom_positions: gt structure [*, n_atom, 3]
        all_atom_mask: mask [*, n_atom]
        cutoffs: list of cutoffs 
    Returns:
        gdt score: [*]
    """

    if all_atom_pred_pos.shape != all_atom_positions.shape:
        all_atom_positions = all_atom_positions.unsqueeze(-3)
        all_atom_mask = all_atom_mask.unsqueeze(-2)

    distances = torch.sqrt(torch.sum((all_atom_pred_pos - 
                                      all_atom_positions) ** 2, 
                                      dim = -1)
                                      )
    
    n = torch.sum(all_atom_mask, dim = -1) 
    scores = [] 
    for c in cutoffs:
        score = torch.sum((distances <= c) * all_atom_mask, dim = -1) 
        score = score / n
        scores.append(score)
    return torch.sum(torch.stack(scores, dim = -1), dim = -1) / len(scores)

def gdt_ts(p1, p2, mask):
    return gdt(p1, p2, mask, [1., 2., 4., 8.])
def gdt_ha(p1, p2, mask):
    return gdt(p1, p2, mask, [0.5, 1., 2., 4.])


def batched_kabsch(
        all_atom_pred_pos: torch.Tensor, 
        all_atom_positions: torch.Tensor, 
        all_atom_mask: torch.Tensor, 
) -> torch.Tensor:
    """
    computes optimal rotation and translation via Kabsch algorithm

    Args: 
        all_atom_pred_pos: [*, n_atom, 3]
        all_atom_positions: [*, n_atom, 3]
        all_atom_mask: [*, n_atom]

    Returns:
        optimal translation: [*, 1, 3]
        optimal rotation: [*, 3, 3]
        rmsd: alignment rmsd [*]
    """

    if len(all_atom_pred_pos.shape) != len(all_atom_positions.shape):
        all_atom_positions = all_atom_positions.unsqueeze(-3)
        all_atom_mask = all_atom_mask.unsqueeze(-2)
    
    n_atom = all_atom_positions.shape[-2]
    predicted_coordinates = all_atom_pred_pos * all_atom_mask.unsqueeze(-1) 
    gt_coordinates = all_atom_positions * all_atom_mask.unsqueeze(-1)

    # translation: center two molecules
    centroid_predicted = torch.mean(predicted_coordinates, dim = -2, keepdim = True)
    centroid_gt = torch.mean(gt_coordinates, dim = -2, keepdim = True)
    translation = centroid_gt - centroid_predicted
    predicted_coordinates_centered = predicted_coordinates - centroid_predicted
    gt_coordinates_centered = gt_coordinates - centroid_gt

    # SVD 
    H = predicted_coordinates_centered.transpose(-2, -1) @ gt_coordinates_centered
    #fp16 not supported
    U, S, Vt = torch.linalg.svd(H.float()) 
    Ut, V = U.transpose(-1, -2), Vt.transpose(-1, -2)

    # determine handedness 
    dets = torch.det(V @ Ut)
    batch_dims = H.shape[:-2]
    D = torch.eye(3).tile(*batch_dims, 1, 1)
    D[..., -1, -1] = torch.sign(dets).to(torch.float64)

    rotation = V @ D @ Ut
    rmsd = torch.sqrt(
        torch.sum(
            (predicted_coordinates_centered @ rotation.transpose(-2, -1) - 
             gt_coordinates_centered) ** 2, 
             dim= (-1, -2))) / n_atom
    
    return translation, rotation, rmsd

def get_superimpose_metrics(       
        all_atom_pred_pos: torch.Tensor, 
        all_atom_positions: torch.Tensor, 
        all_atom_mask: torch.Tensor,       
):
    """ 
    Computes superimposition metrics

    Args: 
        all_atom_pred_pos: pred coordinates [*, n_atom, 3]
        all_atom_positions: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
    """
    out = {}

    translation, rotation, rmsd = batched_kabsch(all_atom_pred_pos,
                                                 all_atom_positions,
                                                 all_atom_mask,
                                                 )
    out['superimpose_rmsd'] = rmsd

    pred_centered = all_atom_pred_pos - torch.mean(all_atom_pred_pos, 
                                                   dim = -2, 
                                                   keepdim = True
                                                   )
    gt_centered = all_atom_positions - torch.mean(all_atom_positions, 
                                                  dim = -2, 
                                                  keepdim = True,
                                                  )
    pred_superimposed = pred_centered @ rotation.transpose(-1, -2)
    
    gdt_ts_score = gdt_ts(pred_superimposed, 
                          gt_centered, 
                          all_atom_mask,
                          )
    gdt_ha_score = gdt_ha(pred_superimposed, 
                          gt_centered, 
                          all_atom_mask,
                          )
    out['gdt_ts'] = gdt_ts_score
    out['gdt_ha'] = gdt_ha_score
    return out