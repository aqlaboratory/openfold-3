import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.linalg import vecdot

from openfold3.core.utils.tensor_utils import (
    masked_mean,
    permute_final_dims,
)


def pLDDT(
    batch: Dict[str, torch.Tensor],
    x: torch.Tensor,
    x_gt: torch.Tensor,
    logits: torch.Tensor,
    n_bins: int = 50,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Algorithm 27 of the AF3 supplementary.
    Computes the smooth LDDT loss.
    Args:
        batch:
            Batch dictionary.
        X:
            Predicted coordinates. [*, N, 3]
        X_gt:
            Ground truth coordinates per batch (B) and atom (N). [*, N, 3]
        logits:
            predicted lddt logits. [*, N, n_bins]
        n_bins:
            Number of bins for lddt. (int)
    Returns:
        loss:
            Smooth LDDT loss.
    """

    # distances between all pairs of atoms
    dx = torch.cdist(x, x)  # [*, N, N]
    dx_gt = torch.cdist(x_gt, x_gt)  # [*, N, N]
    delta = torch.abs(dx - dx_gt)  # [*, N, N]

    # Ca_mask = batch["Ca_mask"]  # [*, N]
    # C1_mask = batch["C1_mask"]  # [*, N]

    Ca_mask = batch['frame_idx'][..., 1] * batch['is_protein'] # [*, T]
    Ca_mask = F.one_hot(Ca_mask, num_classes=dx.shape[-1]).sum(dim=-2) # [*, N]

    C1_mask = batch['frame_idx'][..., 1] * (batch['is_dna'] + batch['is_rna']) # [*, T]
    C1_mask = F.one_hot(C1_mask, num_classes=dx.shape[-1]).sum(dim=-2) # [*, N]

    # Restrict to bespoke inclusion radius
    c = (
        (dx_gt < 30.0) * Ca_mask.unsqueeze(-2) + 
        (dx_gt < 15.0) * C1_mask.unsqueeze(-2)
      )  # [*, N, N]

    # atom mask and avoid self term
    atom_mask_gt = batch['atom_mask_gt']  # [*, N]
    # TODO check dtypes for masks
    mask = 1 - torch.eye(c.shape[-1], device=c.device)  # [N, N]
    mask = mask * atom_mask_gt[..., None] * atom_mask_gt[..., None, :]  # [*, N, N]
    c = c * mask  # [*, N, N]

    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0]).to(dx.device)
    lddt = torch.where(delta[..., None] < thresholds, 1.0, 0.0).sum(-1)  # [*, N, N]
    # TODO: shouldn't this be normalized? (normalized here...)
    lddt = 0.25 * (lddt * c).sum(dim=-1) / (c.sum(dim=-1) + eps)  # [*, N]

    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=lddt.device) # [n_bins + 1]
    bins = torch.bucketize(lddt, boundaries[1:-1])  # [*, N]

    # set bin to ignore value (=-100) of F.cross_entropy for masked atoms
    bins = bins.masked_fill(~atom_mask_gt.bool(), -100)  # [*, N]

    # compute cross entropy loss
    logits = permute_final_dims(logits, [1, 0])  # [*, n_bins, N]
    lddt_loss = F.cross_entropy(logits, bins)

    return lddt_loss

def get_Phi(
    batch: Dict[str, torch.Tensor],
    X_gt: torch.Tensor,
    X: torch.Tensor,
    angle_threshold: float = 25,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Subsection 4.3.2 of the AF3 supplementary.
    Extract coordinates of three atoms for each token, used to define the frame,
    according to the following rules:
    1. Proteins: N, C-alpha, C
    2. Nucleic acids: C1', C3', C4'
    3. Ligands: each token is a single atom to which we add the two closest neighbors.
    (If there are less than 3 atoms in the chain, the frame is invalid.
    If the three atoms are collinear to within 25 degrees, the frame is also invalid.)
    Args:
        batch:
            Batch dictionary.
        X_gt:
            Ground truth coordinates. [*, N, 3]
        X:
            Predicted coordinates. [*, N, 3]
        angle_threshold:
            Threshold for invalid frames. (float)
    Returns:
        Phi_gt:
            Ground truth coordinates of 3 atoms per token. [*, T, 3, 3]
        Phi:
            Predicted coordinates of 3 atoms per token T. [*, T, 3, 3]
        invalid_frame_mask:
            Mask for invalid frames. [*, T]
    """

    # ---------------------------------
    # ENOUGH ATOMS IN (LIGAND) CHAIN
    # ---------------------------------

    # mask for (ligand) tokens in chains which include more than 3 tokens
    chain_id = batch["asym_id"]  # [*, T]
    valid_frame_mask = []
    # TODO can this be done without splitting the batch?
    # TODO chain_id must be int type for bincount, is that the case?
    for b in chain_id:
        b_mask = (b[..., None] == torch.where(torch.bincount(b) >= 3)[0]).any(dim=-1)
        valid_frame_mask.append(b_mask)

    valid_frame_mask = torch.stack(valid_frame_mask)  # [*, T]
    valid_frame_mask = valid_frame_mask * batch["is_ligand"]  # [*, T]

    # -----------------------------
    # GET ATOMS FOR LIGAND FRAME
    # -----------------------------

    # mask for ligand atoms with valid frame
    # (*, N, T) @ (*, T, 1) -> (*, N, 1)
    is_ligand_atom_with_frame = (
        batch["atom_to_token_index"] @ valid_frame_mask[..., None]
    ).squeeze(-1)  # [*, N]

    # indices of three closest neighbors to each atom
    # TODO closest is assumed to be the atom itselt (d=0.0), can this lead to problems?
    d = torch.cdist(X_gt, X_gt)  # [*, N, N]
    _, idx = torch.topk(d, dim=-1, k=3, largest=False)  # [*, N, 3]

    # arrange as (closest neighbor, atom, 2nd closest)
    idx = idx[..., [1, 0, 2]]
    idx = idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # [*, N, 3, 3]

    # get the coordinates of atoms corresponding to idx
    Phi_ligand_gt = torch.gather(
        X_gt.unsqueeze(-3).expand(-1, X_gt.shape[-2], -1, -1), 
        dim=-2, 
        index=idx
    )  # [*, N, 3, 3]
    Phi_ligand_gt = Phi_ligand_gt * is_ligand_atom_with_frame[..., None, None]
    Phi_ligand_gt = (
        permute_final_dims(Phi_ligand_gt, [1, 2, 0]) @ 
        batch["atom_to_token_index"].unsqueeze(-3)
      ) # [*, 3, 3, T]
    Phi_ligand_gt = permute_final_dims(Phi_ligand_gt, [2, 0, 1])  # [*, T, 3, 3]

    Phi_ligand = torch.gather(
        X.unsqueeze(-3).expand(-1, X.shape[-2], -1, -1), 
        dim=-2, 
        index=idx
    ) # [*, N, 3, 3]
    Phi_ligand = Phi_ligand * is_ligand_atom_with_frame[..., None, None]
    Phi_ligand = (
        permute_final_dims(Phi_ligand, [1, 2, 0]) @ 
        batch["atom_to_token_index"].unsqueeze(-3)
      ) # [*, 3, 3, T]
    Phi_ligand = permute_final_dims(Phi_ligand, [2, 0, 1])  # [*, T, 3, 3]

    # -----------------------------
    # INVALID LIGAND FRAMES
    # -----------------------------

    def norm(x, eps):
        return torch.sqrt(torch.sum(x**2, dim=-1) + eps)

    COS_ANGLE_THRESHOLD = math.cos(angle_threshold * math.pi / 180)

    # check if the 3 atoms are collinear to within 25 degrees
    v1 = Phi_ligand[..., 0, :] - Phi_ligand[..., 1, :]  # [*, T, 3]
    v2 = Phi_ligand[..., 2, :] - Phi_ligand[..., 1, :]  # [*, T, 3]

    cos_angle = vecdot(v1, v2) / (norm(v1, eps) * norm(v2, eps))  # [*, T]

    valid_frame_mask = (cos_angle >= COS_ANGLE_THRESHOLD) * valid_frame_mask  # [*, T]
    invalid_frame_mask = (1 - valid_frame_mask) * batch["is_ligand"]  # [*, T]

    # ------------------------------------------------
    # GET ATOMS FOR PROTEIN AND NUCLEIC ACID FRAMES
    # ------------------------------------------------

    # TODO: update to correct batch dict convention
    # Let's say tentatively this gives the indices of the 3 atoms associated with the frame of token T.
    is_not_ligand = 1 - batch["is_ligand"]  # [*, T]
    frame_idx = batch["frame_idx"]  # [*, T, 3]
    frame_idx = frame_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # [*, T, 3, 3]

    Phi_gt = torch.gather(
        X_gt.unsqueeze(-3).expand(-1, is_not_ligand.shape[-1], -1, -1),
        dim=-2,
        index=frame_idx,
    )  # [*, T, 3, 3]
    Phi_gt = Phi_gt * is_not_ligand[..., None, None]

    Phi = torch.gather(
        X.unsqueeze(-3).expand(-1, is_not_ligand.shape[-1], -1, -1),
        dim=-2,
        index=frame_idx,
    )  # [*, T, 3, 3]
    Phi = Phi * is_not_ligand[..., None, None]

    # -----------------------------
    # COMBINED PHI
    # -----------------------------

    Phi_gt = Phi_gt + Phi_ligand_gt  # [*, T, 3, 3]
    Phi = Phi + Phi_ligand  # [*, T, 3, 3]

    return Phi_gt, Phi, invalid_frame_mask

def expressCoordinatesInFrame(
    X: torch.Tensor,
    Phi: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Algorithm 29 of the AF3 supplementary.
    Express coordinates in a frame.
    Args:
        X:
            Predicted token coordinates and token (T). [* T, 3]
        Phi:
            Coordinates of 3 atoms for each token. [*, T', 3, 3]
    Returns:
        X_frame:
            Coordinates expressed in frame. [*, T', T, 3]
            (More explicitly: R_i^{-1} (x_j - x_i)
            with frame index i --> T' and the token index j --> T.)

    """

    def norm(x, eps):
        return torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + eps)

    # Extract frame atoms
    a, b, c = Phi.unbind(dim=-2)  # [*, T', 3]

    w1 = a - b
    w1 = w1 / norm(w1, eps)

    w2 = c - b
    w2 = w2 / norm(w2, eps)

    # Build orthonormal basis
    e1 = w1 + w2
    e1 = e1 / norm(e1, eps)
    e1 = e1.unsqueeze(-2)  # [*, T', 1, 3]

    e2 = w2 - w1
    e2 = e2 / norm(e2, eps)
    e2 = e2.unsqueeze(-2)  # [*, T', 1, 3]

    e3 = torch.linalg.cross(e1, e2)  # [*, T', 1, 3]

    # Project onto frame basis
    X = X.unsqueeze(-3)  # [*, 1, T, 3]
    b = b.unsqueeze(-2)  # [*, T', 1, 3]

    d = X - b  # [*, T', T, 3]
    d1 = vecdot(d, e1)  # [*, T', T]
    d2 = vecdot(d, e2)  # [*, T', T]
    d3 = vecdot(d, e3)  # [*, T', T]

    X_frame = torch.stack([d1, d2, d3], dim=-1)  # [*, T', T, 3]

    return X_frame

def computeAlignmentError(
    X: torch.Tensor,
    X_gt: torch.Tensor,
    Phi: torch.Tensor,
    Phi_gt: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Algorithm 30 of the AF3 supplementary.
    Compute the alignment error.
    Args:
        X:
            Predicted token coordinates. [*, T, 3]
        X_gt:
            Ground truth token coordinates. [*, T, 3]
        Phi:
            Coordinates of 3 atoms associated with frame of token T. [*, T, 3, 3]
        Phi_gt:
            Ground truth coordinates of 3 atoms associated with frame of token T. [*, T, 3, 3]
    Returns:
        e:
            Alignment error per frame (dim=-2) and per token (dim=-1). [*, T, T]
    """

    X_frame = expressCoordinatesInFrame(X, Phi)  # [*, T, T, 3]
    X_gt_frame = expressCoordinatesInFrame(X_gt, Phi_gt)  # [*, T, T, 3]

    e = torch.sum((X_frame - X_gt_frame) ** 2, dim=-1)  # [*, T, T]
    e = torch.sqrt(e + eps)  # (*, T, T)

    return e

def predictedAlignmentError(
    batch: Dict[str, torch.Tensor],
    X: torch.Tensor,
    X_gt: torch.Tensor,
    pae_logits: torch.Tensor,
    angle_threshold: float = 25,
    n_bins: int = 64,
    d_min: float = 0.0,
    d_max: float = 32.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Compute the cross entropy loss between the predicted and computed alignment errors.
    Subsection 4.3.2 of the AF3 supplementary.
    Args:
        batch:
            Batch dictionary.
        X:
            Predicted coordinates. [*, N, 3]
        X_gt:
            Ground truth coordinates. [*, N, 3]
        pae_logits:
            Predicted alignmed error logits. [*, T, T, n_bins]
        angle_threshold:
            Threshold for invalid frames: float
        n_bins:
            Number of bins: int
        d_min:
            Minimum distance.
        d_max:
            Maximum distance.
    Returns:
        pae_loss:
            Predicted alignment error loss. ()
        PAE:
            Predicted alignment error. [*, T, T]
        invalid_frame_mask:
            Mask for invalid frames. [*, T, T]
    """

    Phi_gt, Phi, invalid_frame_mask = get_Phi(
        batch, X_gt, X, angle_threshold, eps
    )  # [*, T, 3, 3], [*, T]

    # get token atom coordinates
    # batch['token_atom'].shape = [*, T, N]
    X = batch["token_center_atom"] @ X  # [*, T, 3]
    X_gt = batch["token_center_atom"] @ X_gt  # [*, T, 3]

    e = computeAlignmentError(X, X_gt, Phi, Phi_gt, eps)  # [*, T, T]

    # -------------------------
    # BIN ALIGNMENT ERROR
    # -------------------------

    # boundaries = [0.0, 0.5, ...., 31.5, 32.0]
    boundaries = torch.linspace(
        d_min, d_max, n_bins + 1, device=e.device
    )  # [n_bins + 1]

    # index of bin for each computed alignment error
    # top bin: anything > 31.5, bottom bin: anything < 0.5
    bins = torch.bucketize(e, boundaries[1:-1])  # [*, T, T]

    # set bin to ignore value (=-100) of F.cross_entropy if invalid frame
    invalid_frame_mask = invalid_frame_mask.unsqueeze(-1).expand_as(bins)  # [*, T, T]
    bins = bins.masked_fill(invalid_frame_mask.bool(), -100)  # [*, T, T]

    # -------------------------
    # LOSS AND EXPECTED PAE
    # -------------------------

    # cross entropy loss
    pae_loss = F.cross_entropy(permute_final_dims(pae_logits, [2, 0, 1]), bins)

    # predicted alignment error (expectation over bins)
    pae = F.softmax(pae_logits, dim=-1)  # [*, T, T, n_bins]
    bin_centers = (boundaries[1:] + boundaries[:-1]) / 2  # [n_bins]
    PAE = (bin_centers * pae).sum(dim=-1)  # [*, T, T]

    # TODO I'm including the invalid frame mask in the return, because it must accompany
    # the PAE for whatever you do with it.
    # It could be easier to multiply it with the PAE before returning it, and include 
    # a normalization factor in the loss function.

    return {
        "pae_loss": pae_loss,
        "predicted_alignment_error": PAE,
        "invalid_frame_mask": invalid_frame_mask,
    }

def predictedDistanceError(
        batch: Dict[str, torch.Tensor],
        X: torch.Tensor,
        X_gt: torch.Tensor,
        pde_logits: torch.Tensor,
        n_bins: int = 64,
        d_min: float = 0.0,
        d_max: float = 32.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the cross entropy loss between the predicted and computed distance errors.
    Subsection 4.3.3 of the AF3 supplementary.
    Args:
        batch:
            Batch dictionary.
        X:
            Predicted atom coordinates. [*, N, 3]
        X_gt:
            Ground truth coordinates. [*, N, 3]
        pde_logits: 
            Predicted distance error logits. [*, T, T, n_bins]
        n_bins: 
            Number of bins.
        d_min: 
            Minimum distance.
        d_max: 
            Maximum distance.
    Returns:
        pae_loss: 
            Predicted alignment error loss.
    """
    # get token atom coordinates
    # batch['token_atom'].shape = [*, T, N]
    X = batch['token_center_atom'] @ X # [*, T, 3]
    X_gt = batch['token_center_atom'] @ X_gt # [*, T, 3]

    # token atom distances errors
    d = torch.cdist(X, X) # [*, T, T]
    d_gt = torch.cdist(X_gt, X_gt) # [*, T, T]
    e = torch.abs(d - d_gt) # [*, T, T]

    # boundaries = [0.0, 0.5, ...., 31.5, 32.0]
    boundaries = torch.linspace(d_min, d_max, n_bins + 1, device=e.device) # [n_bins + 1]
    bins = torch.bucketize(e, boundaries[1:-1]) # [*, T, T]
    
    # cross entropy loss
    pde_loss = F.cross_entropy(permute_final_dims(pde_logits, [2, 0, 1]), bins)

    # predicted distance error (expectation over bins)
    bin_centers = (boundaries[1:] + boundaries[:-1]) / 2 # [n_bins]
    pde = F.softmax(pde_logits, dim=-1) # [*, T, T, n_bins]
    PDE = (bin_centers * pde).sum(dim=-1) # [*, T, T]

    return pde_loss, PDE

def experimentally_resolved_prediction(
        batch: Dict[str, torch.Tensor],
        logits_resolved: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the experimentally resolved prediction, subsection 4.3.4 of the AF3.
    Args:
        batch:
            Batch dictionary.
        logits_resolved: 
            Predicted value. [*, N, 2]
    Returns:
        erp_loss: 
            Experimentally resolved prediction. 
    """

    # set bin to ignore value (=-100) of F.cross_entropy for masked atoms
    atom_mask_gt = batch['atom_mask_gt'].long() # [*, N]
    atom_mask = batch['atom_mask'].long() # [*, N]
    atom_mask_gt = atom_mask_gt.masked_fill(~atom_mask.bool(), -100) # [*, N]

    erp_loss = F.cross_entropy(permute_final_dims(logits_resolved, [1, 0]),atom_mask_gt)
    
    return erp_loss


