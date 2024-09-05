from typing import Dict, List, Optional

import torch


def lddt(
    pair_dist_pred_pos: torch.Tensor,
    pair_dist_gt_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id: torch.Tensor, 
    threshold: Optional[list] = [0.5, 1., 2., 4.],
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
    #create a mask
    n_atom = pair_dist_gt_pos.shape[-2]    
    dists_to_score = (
        (pair_dist_gt_pos < cutoff) * 
        (all_atom_mask[..., None] * all_atom_mask[..., None, :] * 
        (1.0 - torch.eye(n_atom, device=all_atom_mask.device))
        )
        ) #[*, n_atom, n_atom]
    
    #distinguish intra- and inter- pair indices based on asym_id
    intra_mask = torch.where(asym_id[..., None] == asym_id[..., None, :], 1, 0)
    inter_mask = 1 - intra_mask #[*, n_atom, n_atom]

    #get lddt scores
    dist_l1 = torch.abs(pair_dist_gt_pos - pair_dist_pred_pos) #[*, n_atom, n_atom]

    score = torch.zeros_like(dist_l1)
    for distance_threshold in threshold:
        score += (dist_l1 < distance_threshold).type(dist_l1.dtype)
    score /= len(threshold)

    #normalize to get intra_lddt scores
    intra_norm = 1.0 / (eps + torch.sum(dists_to_score * intra_mask, dim= (-1, -2)))
    intra_score = intra_norm * (eps + torch.sum(dists_to_score * intra_mask * score, 
                                                dim= (-1, -2))
                                                )
    
    #inter_score only applies when there exist atom pairs with 
    #different asym_id (inter_mask) and distance threshold (dists_to_score)
    inter_score = torch.full(intra_score.shape, torch.nan)
    inter_mask = dists_to_score * inter_mask
    if torch.any(inter_mask):
        inter_norm = 1.0 / (eps + torch.sum(inter_mask, dim= (-1, -2)))
        inter_score = inter_norm * (eps + torch.sum(inter_mask * score,
                                                    dim= (-1, -2)))
    return intra_score, inter_score

def interface_lddt(
    all_atom_pred_pos_1: torch.Tensor,
    all_atom_pred_pos_2: torch.Tensor,
    all_atom_gt_pos_1: torch.Tensor,
    all_atom_gt_pos_2: torch.Tensor,
    all_atom_mask1: torch.Tensor,
    all_atom_mask2: torch.Tensor,
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
        cutoff: distance cutoff  
            - Nucleic Acids (DNA/RNA) 30.  
            - Others(Protein/Ligands) 15. 
        eps: epsilon 

    Returns: 
        scores: ilddt scores [*]
    """ 
    #get pairwise distance
    pair_dist_true = torch.cdist(all_atom_gt_pos_1, all_atom_gt_pos_2)#[*, n_atom1, n_atom2]
    pair_dist_pred = torch.cdist(all_atom_pred_pos_1, all_atom_pred_pos_2)#[*, n_atom1, n_atom2]

    #create a mask
    dists_to_score = (
        (pair_dist_true < cutoff) * 
        (all_atom_mask1[..., None] * all_atom_mask2[..., None, :]
         )
         )#[*, n_atom1, n_atom2]
    
    #get score
    dist_l1 = torch.abs(pair_dist_true - pair_dist_pred)
    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype) + 
        (dist_l1 < 1.0).type(dist_l1.dtype) + 
        (dist_l1 < 2.0).type(dist_l1.dtype) + 
        (dist_l1 < 4.0).type(dist_l1.dtype)
        )
    score = score * 0.25

    #normalize
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim= (-1, -2)))
    score = norm * (eps + torch.sum(dists_to_score * score, dim= (-1, -2)))
    return score

def drmsd(
        pair_dist_pred_pos: torch.Tensor,
        pair_dist_gt_pos: torch.Tensor,
        all_atom_mask: torch.Tensor,
        asym_id: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """ 
    Computes drmsds

    Args: 
        pair_dist_pred_pos: predicted coordinates [*, n_atom, 3]
        pair_dist_gt_pos: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom]
    Returns:
        intra_drmsd: drmsd within chains
        inter_drmsd: drmsd across chains 

    Note: 
        returns nan if inter_drmsd is invalid
        (ie. single chain, no atom pair within threshold)
    """
    drmsd = pair_dist_pred_pos - pair_dist_gt_pos 
    drmsd = drmsd ** 2 
    
    #apply mask
    mask = (all_atom_mask[..., None] * all_atom_mask[..., None, :]) 
    intra_mask = torch.where(asym_id[..., None] == asym_id[..., None, :], 1, 0)
    inter_mask = 1 - intra_mask

    intra_drmsd = drmsd * (mask * intra_mask)
    intra_drmsd = torch.sum(intra_drmsd, dim=(-1, -2))
    n_intra = torch.sum(intra_mask * mask, dim = (-1, -2))
    intra_drmsd = intra_drmsd * (1 / (n_intra))

    inter_drmsd = torch.full(intra_drmsd.shape, torch.nan)
    inter_mask = inter_mask * mask
    if torch.any(inter_mask):
        inter_drmsd = drmsd * (inter_mask)
        inter_drmsd = torch.sum(inter_drmsd, dim=(-1, -2))
        n_inter = torch.sum(inter_mask * mask, dim = (-1, -2))
        inter_drmsd = inter_drmsd * (1 / (n_inter))
    
    intra_drmsd = torch.sqrt(intra_drmsd) 
    inter_drmsd = torch.sqrt(inter_drmsd) 
    return intra_drmsd, inter_drmsd

def get_validation_metrics(
        is_substrate_atomized: torch.Tensor, 
        asym_id: torch.Tensor,
        pred_coords: torch.Tensor, 
        gt_coords: torch.Tensor, 
        all_atom_mask: torch.Tensor,
        is_protein_atomized: torch.Tensor,
        substrate: str,
        is_nucleic_acid: Optional[bool] = False,
        ) -> Dict[str, torch.Tensor]:
    """ 
    Compute validation metrics with a given substrate (protein, ligand, rna, dna)

    Args: 
        is_substrate_atomized: broadcasted ligand/rna/dna/is_protein feature [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom] 
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
        is_protein_atomized: broadcasted is_protein feature [*, n_atom]
        substrate: 'protein', ligand', 'rna', 'dna' 
        is_nucleic_acid: boolean indicating if ligand type is nucleic acid
    Returns: 
        out: dictionary containing validation metrics
            'lddt_intra_f'{substrate}': intra ligand lddt
            'lddt_inter_f'{substrate}'_f'{substrate}': inter ligand-ligand lddt
            'drmsd_intra_f'{substrate}': intra ligand drmsd
            'lddt_inter_protein_f'{substrate}': inter protein-ligand lddt
                        
    Notes: 
        if no appropriate substrate: returns an empty dict {}
        for ligand: a few extra scores are calculated
            'lddt_intra_ligand_uha': intra ligand lddt with [0.25, 0.5, 0.75, 1.] 
            'lddt_inter_ligand_ligand_uha': inter ligand lddt with above threshold
        for dna/rna: lddts with 15 A inclusion radius added
            'lddt_intra_{dna/rna}_15': intra ligand lddt with 15 A radius
            'lddt_inter_{dna/rna}_{dna/rna}_15': inter lddt with 15 A radius
            'lddt_inter_protein_{dna/rna}_15': inter protein-dna/rna lddt 
     """
    out = {}

    if torch.any(is_substrate_atomized):
        is_substrate_atomized = is_substrate_atomized.bool() 
        is_protein_atomized = is_protein_atomized.bool() 

        bs = is_substrate_atomized.shape[: -1] #(bs, (n_sample),)

        #getting appropriate atoms of shape (bs, (n_sample), n_protein/ligand, (3)), 
        gt_protein = gt_coords[is_protein_atomized].view((bs) + (-1, 3))  
        gt_ligand = gt_coords[is_substrate_atomized].view((bs) + (-1, 3))
        pred_protein = pred_coords[is_protein_atomized].view((bs) + (-1, 3))
        pred_ligand = pred_coords[is_substrate_atomized].view((bs) + (-1, 3))
        asym_id_ligand = asym_id[is_substrate_atomized].view((bs) + (-1,))
        all_atom_mask_protein = all_atom_mask[is_protein_atomized].view((bs) + (-1,))
        all_atom_mask_ligand = all_atom_mask[is_substrate_atomized].view((bs) + (-1,))
        
        #(bs,(n_sample), n_lig, n_lig)
        gt_ligand_pair = torch.cdist(gt_ligand, gt_ligand)
        pred_ligand_pair = torch.cdist(pred_ligand, pred_ligand)

        cutoff = 30. if substrate == 'rna' or substrate == 'dna' else 15.
        intra_lddt, inter_lddt = lddt(pred_ligand_pair,
                                          gt_ligand_pair,
                                          all_atom_mask_ligand,
                                          asym_id_ligand,
                                          cutoff = cutoff
                                          )
        out['lddt_intra_' + substrate] = intra_lddt
        out['lddt_inter_' + substrate + '_' + substrate] = inter_lddt

        intra_drmsd, inter_drmsd = drmsd(pred_ligand_pair,
                                         gt_ligand_pair,
                                         all_atom_mask_ligand,
                                         asym_id_ligand,
                                         )
        out['drmsd_intra_' + substrate] = intra_drmsd

        #additional metrics
        #nucleic acid
        if substrate == 'rna' or substrate == 'dna':
            #get lddt with 15 A inclusion radius
            intra_lddt, inter_lddt = lddt(pred_ligand_pair,
                                          gt_ligand_pair,
                                          all_atom_mask_ligand,
                                          asym_id_ligand,
                                          cutoff = 15.
                                          )
            out['lddt_intra_' + substrate + '_15'] = intra_lddt
            out['lddt_inter_' + substrate + '_' + substrate + '_15'] = inter_lddt
            #ilddt with protein
            inter_lddt_protein_ligand = interface_lddt(pred_protein,
                                                       pred_ligand,
                                                       gt_protein,
                                                       gt_ligand,
                                                       all_atom_mask_protein,
                                                       all_atom_mask_ligand,
                                                       cutoff = 30.,
                                                       )
            out['lddt_inter_protein_' + substrate] = inter_lddt_protein_ligand

            inter_lddt_protein_ligand = interface_lddt(pred_protein,
                                                       pred_ligand,
                                                       gt_protein,
                                                       gt_ligand,
                                                       all_atom_mask_protein,
                                                       all_atom_mask_ligand,
                                                       cutoff = 15.,
                                                       )
            out['lddt_inter_protein_' + substrate + '_15'] = inter_lddt_protein_ligand

        elif substrate == 'ligand':
            #get tighter threshold lddts
            intra_lddt_uha, inter_lddt_uha = lddt(pred_ligand_pair,
                                                  gt_ligand_pair,
                                                  all_atom_mask_ligand,
                                                  asym_id_ligand,
                                                  threshold = [0.25, 0.5, 0.75, 1.],
                                                  cutoff = 15.
                                                  )
            out['lddt_intra_' + substrate + '_uha'] = intra_lddt_uha
            out['lddt_inter_' + substrate + '_' + substrate + '_uha'] = inter_lddt_uha
            
            #ilddt with protein
            inter_lddt_protein_ligand = interface_lddt(pred_protein,
                                                       pred_ligand,
                                                       gt_protein,
                                                       gt_ligand,
                                                       all_atom_mask_protein,
                                                       all_atom_mask_ligand,
                                                       cutoff = 15.,
                                                       )
            out['lddt_inter_protein_' + substrate] = inter_lddt_protein_ligand
    return out

def gdt(
        all_atom_pred_pos: torch.Tensor,
        all_atom_gt_pos: torch.Tensor,
        all_atom_mask: torch.Tensor,
        cutoffs: List,
        ) -> torch.Tensor:
    """ 
    Calculates gdt

    Args: 
        all_atom_pred_pos: predicted structures [*, n_atom, 3]
        all_atom_gt_pos: gt structure [*, n_atom, 3]
        all_atom_mask: mask [*, n_atom]
        cutoffs: list of cutoffs 
    Returns:
        gdt score: [*]
    """

    distances = torch.sqrt(torch.sum((all_atom_pred_pos - 
                                      all_atom_gt_pos) ** 2, 
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
        all_atom_gt_pos: torch.Tensor, 
        all_atom_mask: torch.Tensor, 
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    computes optimal rotation and translation via Kabsch algorithm

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
    centroid_predicted = torch.mean(predicted_coordinates, dim = -2, keepdim = True)
    centroid_gt = torch.mean(gt_coordinates, dim = -2, keepdim = True)
    translation = centroid_gt - centroid_predicted
    predicted_coordinates_centered = predicted_coordinates - centroid_predicted
    gt_coordinates_centered = gt_coordinates - centroid_gt

    # SVD 
    H = predicted_coordinates_centered.transpose(-2, -1) @ gt_coordinates_centered
    #fp16 not supported
    with torch.cuda.amp.autocast(enabled=False):
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
        all_atom_gt_pos: torch.Tensor, 
        all_atom_mask: torch.Tensor,       
) -> Dict[str, torch.Tensor]:
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

    translation, rotation, rmsd = batched_kabsch(all_atom_pred_pos,
                                                 all_atom_gt_pos,
                                                 all_atom_mask,
                                                 )
    out['superimpose_rmsd'] = rmsd

    pred_centered = all_atom_pred_pos - torch.mean(all_atom_pred_pos, 
                                                   dim = -2, 
                                                   keepdim = True
                                                   )
    gt_centered = all_atom_gt_pos - torch.mean(all_atom_gt_pos, 
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