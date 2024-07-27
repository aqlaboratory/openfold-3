from typing import List, Optional

import torch


def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: Optional[float] = 15.0,
    eps: Optional[float] = 1e-10,
) -> torch.Tensor:
    """
    Calculates lddt scores

    Args: 
        all_atom_pred_pos: predicted atom coordinates [*, n_atom, 3]
        all_atom_positions: gt atom coordinates [*, n_atom, 3]
        all_atom_mask: mask [*, n_atom]
        cutoff: distance cutoff  
            - Nucleic Acids (DNA/RNA) 30.
            - Other biomolecules (Protein/Ligands) 15.
        eps: epsilon 

    Returns: 
        scores: lddt scores [*]
    """

    if len(all_atom_pred_pos.shape) != len(all_atom_positions.shape):
        all_atom_positions = all_atom_positions.unsqueeze(-3)
        all_atom_mask = all_atom_mask.unsqueeze(-2)

    # get pairwise distances
    pair_dist_true = torch.cdist(all_atom_positions, all_atom_positions)
    pair_dist_pred = torch.cdist(all_atom_pred_pos, all_atom_pred_pos)

    # create a mask
    n_atom = all_atom_positions.shape[-2]    
    dists_to_score = (
        (pair_dist_true < cutoff) * 
        (all_atom_mask[..., None] * all_atom_mask[..., None, :] * 
        (1.0 - torch.eye(n_atom, device=all_atom_mask.device))
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
        all_atom_pred_pos: torch.Tensor,
        all_atom_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        ) -> torch.Tensor:
    """ 
    Computes drmsds

    Args: 
        all_atom_pred_pos: predicted coordinates [*, n_atom, 3]
        all_atom_positions: gt coordinates [*, n_atom, 3]
        all_atom_mask: [*, n_atom]

    Returns:
        drmsd: computed drmsds
    """
    # get pairwise distance
    d1 = torch.cdist(all_atom_pred_pos, all_atom_pred_pos) 
    d2 = torch.cdist(all_atom_positions, all_atom_positions)

    if d1.shape != d2.shape:
        d2 = d2.unsqueeze(-3) 
        all_atom_mask = all_atom_mask.unsqueeze(-2) 

    drmsd = d1 - d2 
    drmsd = drmsd ** 2 
    
    # apply mask
    drmsd = drmsd * (all_atom_mask[..., None] * all_atom_mask[..., None, :]) 
    drmsd = torch.sum(drmsd, dim=(-1, -2))

    n = d1.shape[-1] if all_atom_mask is None else torch.sum(all_atom_mask, dim = -1)
    drmsd = drmsd * (1 / (n * (n - 1)))
    drmsd = torch.sqrt(drmsd) 

    return drmsd

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