from typing import Dict, Tuple

import torch

from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_atom_index_offset,
    get_token_frame_atoms,
    get_token_representative_atoms,
)
from openfold3.core.utils.tensor_utils import binned_one_hot


def express_coords_in_frames(
    x: torch.Tensor, phi: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], eps: float
):
    """
    Implements AF3 Algorithm 29.

    Args:
        x:
            [*, N_token, 3] Atom positions
        phi:
            A tuple of atom positions used for frame construction, each
            has a shape of [*, N_token, 3]
        eps:
            Small float for numerical stability
    Returns:
        xij:
            [*, N_token, N_token, 3] Coordinates projected into frame basis
    """
    a, b, c = phi
    w1 = a - b
    w2 = c - b
    w1_norm = (eps + torch.sum(w1**2, dim=-1, keepdim=True)) ** 0.5
    w2_norm = (eps + torch.sum(w2**2, dim=-1, keepdim=True)) ** 0.5
    w1 = w1 / w1_norm
    w2 = w2 / w2_norm

    # Build orthonormal basis
    # [*, N_token, 3]
    e1 = w1 + w2
    e2 = w2 - w1
    e1_norm = (eps + torch.sum(e1**2, dim=-1, keepdim=True)) ** 0.5
    e2_norm = (eps + torch.sum(e2**2, dim=-1, keepdim=True)) ** 0.5
    e1 = e1 / e1_norm
    e2 = e2 / e2_norm
    e3 = torch.linalg.cross(e1, e2, dim=-1)

    # Project onto frame basis
    # [*, N_token, N_token, 3]
    # TODO: check this
    d = x[..., None, :, :] - b[..., None, :]
    xij = torch.stack(
        [
            torch.einsum("...ik,...ijk->...ij", e1, d),
            torch.einsum("...ik,...ijk->...ij", e2, d),
            torch.einsum("...ik,...ijk->...ij", e3, d),
        ],
        dim=-1,
    )

    return xij


def compute_alignment_error(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    phi: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    phi_gt: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float,
):
    """
    Implements AF3 Algorithm 30.

    Args:
        x:
            [*, N_token, 3] Atom positions
        x_gt:
            [*, N_token, 3] Groundtruth atom positions
        phi:
            A tuple of atom positions used for frame construction,
            each has a shape of [*, N_token, 3]
        phi_gt:
            A tuple of groundtruth atom positions used for frame
            construction, each has a shape of [*, N_token, 3]
        eps:
            Small float for numerical stability
    Returns:
        [*, N_token, N_token] Alignment error matrix
    """
    xij = express_coords_in_frames(x=x, phi=phi, eps=eps)
    xij_gt = express_coords_in_frames(x=x_gt, phi=phi_gt, eps=eps)
    return (torch.sum((xij - xij_gt) ** 2, dim=-1) + eps) ** 0.5


def plddt_loss(
    batch: Dict,
    x: torch.Tensor,
    p_b: torch.Tensor,
    no_bins: int,
    bin_min: float,
    bin_max: float,
    eps: float,
) -> torch.Tensor:
    """
    Compute loss on predicted local distance difference test (pLDDT).

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Predicted atom positions
        p_b:
            [*, N_atom, no_bins] Predicted probabilities
        no_bins:
            Number of bins
        bin_min:
            Minimum bin value
        bin_max:
            Maximum bin value
        eps:
            Small float for numerical stability
    Returns:
        [*] Losses on pLDDT
    """
    # Compute difference in distances
    # [*, N_atom, N_atom]
    x_gt = batch["gt_atom_positions"]
    dx = torch.sum(eps + (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1) ** 0.5
    dx_gt = (
        torch.sum((eps + x_gt[..., None, :] - x_gt[..., None, :, :]) ** 2, dim=-1)
        ** 0.5
    )
    d = torch.abs(dx_gt - dx)

    # Compute pair mask based on distance and type of atom m
    # [*, N_atom, N_atom]
    protein_atom_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["is_protein"],
    )
    nucleotide_atom_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["is_dna"] + batch["is_rna"],
    )
    pair_mask = (d < 15) * protein_atom_mask[..., None, :] + (
        d < 30
    ) * nucleotide_atom_mask[..., None, :]

    # Construct indices for representative atoms
    # Note that N_atom is used to denote masked out atoms (including missing
    # representative atoms and ligand representative atoms)
    n_atom = x.shape[-2]
    start_atom_index = batch["start_atom_index"].long()
    is_standard_protein = batch["is_protein"] * (1 - batch["is_atomized"])
    is_standard_nucleotide = (batch["is_dna"] + batch["is_rna"]) * (
        1 - batch["is_atomized"]
    )
    ca_atom_index_offset, ca_atom_mask = get_token_atom_index_offset(
        atom_name="CA", restype=batch["restype"]
    )
    c1p_atom_index_offset, c1p_atom_mask = get_token_atom_index_offset(
        atom_name="C1'", restype=batch["restype"]
    )
    rep_index = (
        ((start_atom_index + ca_atom_index_offset) * is_standard_protein * ca_atom_mask)
        + (
            (start_atom_index + c1p_atom_index_offset)
            * is_standard_nucleotide
            * c1p_atom_mask
        )
        + n_atom
        * (
            1
            - is_standard_protein * ca_atom_mask
            - is_standard_nucleotide * c1p_atom_mask
        )
    )

    # Construct atom mask for lddt computation
    # Note that additional dimension is padded and later removed for masked out atoms
    # [*, N_atom]
    atom_mask_shape = list(batch["gt_atom_mask"].shape)
    padded_atom_mask_shape = list(atom_mask_shape)
    padded_atom_mask_shape[-1] = padded_atom_mask_shape[-1] + 1
    atom_mask = torch.zeros(padded_atom_mask_shape, device=x.device).scatter_(
        index=rep_index.long(), value=1, dim=-1
    )[..., :-1]
    atom_mask = atom_mask * batch["gt_atom_mask"]

    # Construct pair atom selection mask for lddt computation
    # [*, N_atom, N_atom]
    pair_atom_mask = batch["gt_atom_mask"][..., None] * atom_mask[..., None, :]
    pair_mask = pair_mask * pair_atom_mask

    # Compute lddt
    # [*, N_atom]
    lddt = torch.sum(
        0.25
        * ((d < 0.5).int() + (d < 1).int() + (d < 2).int() + (d < 4).int())
        * pair_mask,
        dim=-1,
    ) / (torch.sum(pair_mask, dim=-1) + eps)

    # Compute binned lddt
    # [*, N_atom, no_bins]
    bin_size = (bin_max - bin_min) / no_bins
    v_bins = bin_min + torch.arange(no_bins, device=x.device) * bin_size
    lddt_b = binned_one_hot(lddt, v_bins)

    # Compute loss on plddt
    l_plddt = -torch.sum(
        torch.sum(lddt_b * torch.log(p_b + eps), dim=-1) * batch["gt_atom_mask"], dim=-1
    ) / (torch.sum(batch["gt_atom_mask"], dim=-1) + eps)

    return l_plddt


def pae_loss(
    batch: Dict,
    x: torch.Tensor,
    p_b: torch.Tensor,
    angle_threshold: float,
    no_bins: int,
    bin_min: float,
    bin_max: float,
    eps: float,
    inf: float,
):
    """
    Compute loss on predicted aligned error (PAE).

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Predicted atom positions
        p_b:
            [*, N_token, N_token, no_bins] Predicted probabilities
        angle_threshold:
            Angle threshold for filtering co-linear atoms
        no_bins:
            Number of bins
        bin_min:
            Minimum bin value
        bin_max:
            Maximum bin value
        eps:
            Small float for numerical stability
        inf:
            Large float for numerical stability
    Returns:
        [*] Losses on PAE
    """
    # Extract atom coordinates for frame construction
    atom_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["token_mask"],
    )
    phi, valid_frame_mask = get_token_frame_atoms(
        batch=batch,
        x=x,
        atom_mask=atom_mask,
        angle_threshold=angle_threshold,
        eps=eps,
        inf=inf,
    )
    phi_gt, valid_frame_mask_gt = get_token_frame_atoms(
        batch=batch,
        x=batch["gt_atom_positions"],
        atom_mask=batch["gt_atom_mask"],
        angle_threshold=angle_threshold,
        eps=eps,
        inf=inf,
    )

    # Extract representative atom coordinates
    rep_x, rep_atom_mask = get_token_representative_atoms(
        batch=batch, x=x, atom_mask=atom_mask
    )
    rep_x_gt, rep_atom_mask_gt = get_token_representative_atoms(
        batch=batch, x=batch["gt_atom_positions"], atom_mask=batch["gt_atom_mask"]
    )

    # Compute alignment error
    # [*, N_token, N_token]
    e = compute_alignment_error(x=rep_x, x_gt=rep_x_gt, phi=phi, phi_gt=phi_gt, eps=eps)

    # Compute binned alignment error
    # [*, N_token, N_token, no_bins]
    bin_size = (bin_max - bin_min) / no_bins
    v_bins = bin_min + torch.arange(no_bins, device=p_b.device) * bin_size
    e_b = binned_one_hot(e, v_bins)

    # Compute predicted alignment error
    pair_mask = (valid_frame_mask[..., None] * rep_atom_mask[..., None, :]) * (
        valid_frame_mask_gt[..., None] * rep_atom_mask_gt[..., None, :]
    )

    # Compute loss on pae
    l_pae = -torch.sum(
        torch.sum(e_b * torch.log(p_b + eps), dim=-1) * pair_mask, dim=(-1, -2)
    ) / (torch.sum(pair_mask, dim=(-1, -2)) + eps)

    return l_pae


def pde_loss(
    batch: Dict,
    x: torch.Tensor,
    p_b: torch.Tensor,
    no_bins: int,
    bin_min: float,
    bin_max: float,
    eps: float,
):
    """
    Implements AF3 Equation 12.

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Atom positions
        p_b:
            [*, N_token, N_token, no_bins] Predicted probabilites for errors in absolute
            distances (projected into bins)
        no_bins:
            Number of distance bins
        bin_size:
            Size of each distance bin (in Å)
        eps:
            Small constant for numerical stability
    Returns:
        l_pde:
            [*] Loss on predicted distance error
    """
    # Extract representative atoms
    atom_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["token_mask"],
    )
    rep_x, _ = get_token_representative_atoms(batch=batch, x=x, atom_mask=atom_mask)
    rep_x_gt, rep_atom_mask_gt = get_token_representative_atoms(
        batch=batch, x=batch["gt_atom_positions"], atom_mask=batch["gt_atom_mask"]
    )

    # Compute prediction target
    d = (
        torch.sum(eps + (rep_x[..., None, :] - rep_x[..., None, :, :]) ** 2, dim=-1)
        ** 0.5
    )
    d_gt = (
        torch.sum(
            eps + (rep_x_gt[..., None, :] - rep_x_gt[..., None, :, :]) ** 2, dim=-1
        )
        ** 0.5
    )
    e = torch.abs(d - d_gt)

    # Compute binned prediction target
    bin_size = (bin_max - bin_min) / no_bins
    v_bins = bin_min + torch.arange(no_bins, device=e.device) * bin_size
    e_b = binned_one_hot(e, v_bins)

    # Compute loss on predicted distance error
    pair_mask = rep_atom_mask_gt[..., None] * rep_atom_mask_gt[..., None, :]
    l_pde = -torch.sum(
        torch.sum(e_b * torch.log(p_b + eps), dim=-1) * pair_mask, dim=(-1, -2)
    ) / (torch.sum(pair_mask, dim=(-1, -2)) + eps)

    return l_pde


def resolved_loss(batch: Dict, p_b: torch.Tensor, no_bins: int, eps: float):
    """
    Implements AF3 Equation 14.

    Args:
        batch:
            Feature dictionary
        p_b:
            [*, N_atom, no_bins] Predicted probabilites on whether the atom is
            resolved in the ground truth
        no_bins:
            Number of bins
        eps:
            Small constant for numerical stability
    Returns:
        l_resolved:
            [*] Loss on predictions for whether the atom is resolved
            in the ground truth
    """
    # Compute binned prediction target
    v_bins = torch.arange(no_bins, device=p_b.device)
    y_b = binned_one_hot(batch["gt_atom_mask"], v_bins)

    # Compute loss on experimentally resolved prediction
    atom_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["token_mask"],
    )
    l_resolved = -torch.sum(
        torch.sum(y_b * torch.log(p_b + eps), dim=-1) * atom_mask, dim=-1
    ) / (torch.sum(atom_mask, dim=-1) + eps)

    return l_resolved


def confidence_loss(
    batch: Dict,
    output: Dict,
    no_bins_plddt: int,
    bin_min_plddt: float,
    bin_max_plddt: float,
    angle_threshold: float,
    no_bins_pae: int,
    bin_min_pae: float,
    bin_max_pae: float,
    no_bins_pde: int,
    bin_min_pde: float,
    bin_max_pde: float,
    no_bins_resolved: int,
    alpha_pae: float,
    eps: float,
    inf: float,
    **kwargs,
):
    """
    Compute loss on confidence module.

    Args:
        batch:
            Feature dictionary
        output:
            Output dictionary
        no_bins_plddt:
            Number of bins for pLDDT
        bin_min_plddt:
            Minimum bin value for pLDDT
        bin_max_plddt:
            Maximum bin value for pLDDT
        angle_threshold:
            Angle threshold for filtering co-linear atoms
        no_bins_pae:
            Number of bins for PAE
        bin_min_pae:
            Minimum bin value for PAE
        bin_max_pae:
            Maximum bin value for PAE
        no_bins_pde
            Number of bins for PDE
        bin_min_pde:
            Minimum bin value for PDE
        bin_max_pde:
            Maximum bin value for PDE
        no_bins_resolved:
            Number of bins for predictions on whether the
            atom is resolved in the ground truth
        alpha_pae:
            Weight on PAE loss
        eps:
            Small float for numerical stability
        inf:
            Large float for numerical stability
    Returns:
        Loss on confidence module
    """
    l_plddt = plddt_loss(
        batch=batch,
        x=output["x_pred"],
        p_b=output["plddt"],
        no_bins=no_bins_plddt,
        bin_min=bin_min_plddt,
        bin_max=bin_max_plddt,
        eps=eps,
    )

    l_pde = pde_loss(
        batch=batch,
        x=output["x_pred"],
        p_b=output["pde"],
        no_bins=no_bins_pde,
        bin_min=bin_min_pde,
        bin_max=bin_max_pde,
        eps=eps,
    )

    l_resolved = resolved_loss(
        batch=batch,
        p_b=output["resolved"],
        no_bins=no_bins_resolved,
        eps=eps,
    )

    l = l_plddt + l_pde + l_resolved
    if alpha_pae > 0:
        l_pae = pae_loss(
            batch=batch,
            x=output["x_pred"],
            p_b=output["pae"],
            angle_threshold=angle_threshold,
            no_bins=no_bins_pae,
            bin_min=bin_min_pae,
            bin_max=bin_max_pae,
            eps=eps,
            inf=inf,
        )

        l = l + alpha_pae * l_pae

    return torch.mean(l)
