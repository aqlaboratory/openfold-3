# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

"""Diffusion losses."""

from typing import Dict

import torch

from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.core.utils.tensor_utils import tensor_tree_map


def weighted_rigid_align(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    w: torch.Tensor,
    atom_mask_gt: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    Implements AF3 Algorithm 28.

    Args:
        x:
            [*, N_atom, 3] Atom positions (point clouds to be aligned)
        x_gt:
            [*, N_atom, 3] Groundtruth atom positions (reference point clouds)
        w:
            [*, N_atom] Weights based on molecule type
        atom_mask_gt:
            [*, N_atom] Atom mask
        eps:
            Small constant for stability
    Returns:
        [*, N_atom, 3] Aligned atom positions
    """
    # Mean-centre positions
    w_mean = torch.sum(w * atom_mask_gt, dim=-1, keepdim=True) / torch.sum(
        atom_mask_gt + eps, dim=-1, keepdim=True
    )
    wx_mean = torch.sum(x * w[..., None] * atom_mask_gt[..., None], dim=-2) / torch.sum(
        atom_mask_gt + eps, dim=-1, keepdim=True
    )
    wx_gt_mean = torch.sum(
        x_gt * w[..., None] * atom_mask_gt[..., None], dim=-2
    ) / torch.sum(atom_mask_gt + eps, dim=-1, keepdim=True)
    mu = wx_mean / w_mean
    mu_gt = wx_gt_mean / w_mean
    x = x - mu[..., None, :]
    x_gt = x_gt - mu_gt[..., None, :]

    # Construct covariance matrix
    H = x_gt[..., None] * x[..., None, :]
    H = H * w[..., None, None] * atom_mask_gt[..., None, None]
    H = torch.sum(H, dim=-3)

    dtype = H.dtype

    # TODO: Check why autocast did not work in test
    # Find optimal rotation from single value decomposition
    # SVD (cast to float because doesn't work with bf16/fp16)
    U, _, V = torch.linalg.svd(H.float())

    dets = torch.linalg.det(U @ V).to(dtype=dtype)
    U = U.to(dtype=dtype)
    V = V.to(dtype=dtype)

    # Remove reflection
    F = torch.eye(3, device=x.device, dtype=x.dtype).tile((*H.shape[:-2], 1, 1))
    F[..., -1, -1] = torch.sign(dets)
    R = U @ F @ V

    # Apply alignment
    x_align = x @ R.transpose(-1, -2) + mu_gt[..., None, :]

    return x_align.detach()


def mse_loss(
    batch: Dict,
    x: torch.Tensor,
    dna_weight: float,
    rna_weight: float,
    ligand_weight: float,
    eps: float,
) -> torch.Tensor:
    """
    Implements AF3 Equation 3.

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Atom positions
        dna_weight:
            Upweight factor for DNA atoms
        rna_weight:
            Upweight factor for RNA atoms
        ligand_weight:
            Upweight factor for ligand atoms
        eps:
            Small constant for stability
    Returns:
        [*] Weighted MSE between groundtruth and denoised structures
    """
    # Construct per-token weights based on molecule types
    # [*, n_token]
    w_dna = batch["is_dna"] * dna_weight
    w_rna = batch["is_rna"] * rna_weight
    w_ligand = batch["is_ligand"] * ligand_weight
    w = torch.ones_like(batch["is_dna"]) + w_dna + w_rna + w_ligand

    # Convert per-token weights to per-atom weights
    # [*, n_atom]
    w = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=w,
    )

    atom_positions_gt = batch["ground_truth"]["atom_positions"]
    atom_mask_gt = batch["ground_truth"]["atom_resolved_mask"]

    # Perform weighted rigid alignment
    x_gt_aligned = weighted_rigid_align(
        x=atom_positions_gt,
        x_gt=x,
        w=w,
        atom_mask_gt=atom_mask_gt,
        eps=eps,
    )

    mse = (
        (1 / 3.0)
        * torch.sum(
            torch.sum((x - x_gt_aligned) ** 2, dim=-1) * w * atom_mask_gt,
            dim=-1,
        )
        / torch.sum(atom_mask_gt + eps, dim=-1)
    )

    return mse


def bond_loss(batch: Dict, x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Implements AF3 Equation 5.

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Atom positions
        eps:
            Small constant for stability
    Returns:
        [*] Auxiliary loss for bonded ligands
    """
    x_gt = batch["ground_truth"]["atom_positions"]
    atom_mask_gt = batch["ground_truth"]["atom_resolved_mask"]

    # Compute pairwise distances
    dx = torch.cdist(x, x)
    dx_gt = torch.cdist(x_gt, x_gt)

    # Construct polymer-ligand per-token bond mask
    # TODO: double check this
    # [*, N_token, N_token]
    is_polymer = batch["is_protein"] + batch["is_dna"] + batch["is_rna"]
    bond_mask = batch["token_bonds"] * (
        is_polymer[..., None, :] * batch["is_ligand"][..., None]
    )

    # Construct polymer-ligand per-atom bond mask
    # [*, N_atom, N_atom]
    bond_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=bond_mask,
        token_dim=-2,
    )
    bond_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=bond_mask.transpose(-1, -2),
        token_dim=-2,
    )
    bond_mask = bond_mask.transpose(-1, -2)

    # Compute polymer-ligand bond loss
    mask = bond_mask * (atom_mask_gt[..., None] * atom_mask_gt[..., None, :])

    loss = torch.sum((dx - dx_gt) ** 2 * mask, dim=(-1, -2)) / torch.sum(
        mask + eps, dim=(-1, -2)
    )

    return loss


def smooth_lddt_loss(batch: Dict, x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Implements AF3 Algorithm 27.

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Atom positions
        eps:
            Small constant for stability
    Returns:
        [*] Auxiliary structure-based loss based on smooth LDDT
    """
    x_gt = batch["ground_truth"]["atom_positions"]
    atom_mask_gt = batch["ground_truth"]["atom_resolved_mask"]

    # [*, N_atom, N_atom]
    dx = torch.cdist(x, x)
    dx_gt = torch.cdist(x_gt, x_gt)

    # [*, N_atom, N_atom]
    d = torch.abs(dx_gt - dx)
    e = 0.25 * (
        torch.sigmoid(0.5 - d)
        + torch.sigmoid(1.0 - d)
        + torch.sigmoid(2.0 - d)
        + torch.sigmoid(4.0 - d)
    )

    # [*, N_atom]
    is_nucleotide = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["is_dna"] + batch["is_rna"],
    )

    # [*, N_atom, N_atom]
    c = (dx_gt < 30) * is_nucleotide[..., None] + (dx_gt < 15) * (
        1 - is_nucleotide[..., None]
    )

    # [*]
    mask = 1 - torch.eye(x.shape[-2], device=x.device, dtype=x.dtype).tile(
        (*x.shape[:-2], 1, 1)
    )
    mask = mask * (atom_mask_gt[..., None] * atom_mask_gt[..., None, :])
    ce_mean = torch.sum(c * e * mask, dim=(-1, -2)) / torch.sum(
        mask + eps, dim=(-1, -2)
    )
    c_mean = torch.sum(c * mask, dim=(-1, -2)) / torch.sum(mask + eps, dim=(-1, -2))
    lddt = ce_mean / c_mean

    return 1 - lddt


def diffusion_loss(
    batch: Dict,
    x: torch.Tensor,
    t: torch.Tensor,
    sigma_data: float,
    bond_weight: float,
    smooth_lddt_weight: float,
    dna_weight: float = 5.0,
    rna_weight: float = 5.0,
    ligand_weight: float = 10.0,
    eps: float = 1e-8,
    **kwargs,
) -> [torch.Tensor, Dict]:
    """
    Implements AF3 Equation 6.

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Atom positions
        t:
            [*] Noise level at a diffusion step
        sigma_data:
            Constant determined by data variance
        bond_weight:
            Weight for auxiliary loss for bonded ligands
        smooth_lddt_weight
            Weight for smooth lddt loss
        dna_weight:
            Upweight factor for DNA atoms
        rna_weight:
            Upweight factor for RNA atoms
        ligand_weight:
            Upweight factor for ligand atoms
        eps:
            Small constant for stability
    Returns:
        mean_loss:
            Diffusion loss
        loss_breakdown:
            Dict of individual component losses
    """
    # Expand to broadcast to number of samples
    batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)

    l_mse = mse_loss(
        batch=batch,
        x=x,
        dna_weight=dna_weight,
        rna_weight=rna_weight,
        ligand_weight=ligand_weight,
        eps=eps,
    )
    loss_breakdown = {"mse_loss": l_mse}

    l_bond = 0.0
    if bond_weight > 0:
        l_bond = bond_weight * bond_loss(batch=batch, x=x, eps=eps)
        loss_breakdown["bond_loss"] = l_bond

    l_smooth_lddt = 0.0
    if smooth_lddt_weight > 0:
        l_smooth_lddt = smooth_lddt_weight * smooth_lddt_loss(batch=batch, x=x, eps=eps)
        loss_breakdown["smooth_lddt_loss"] = l_smooth_lddt

    loss_breakdown = {
        k: torch.mean(v).detach().clone() for k, v in loss_breakdown.items()
    }

    w = (t**2 + sigma_data**2) / (t + sigma_data) ** 2
    l = w * (l_mse + l_bond) + l_smooth_lddt

    mean_loss = torch.mean(l)

    # Return to original batch size
    tensor_tree_map(lambda t: t.squeeze(1), batch)

    return mean_loss, loss_breakdown
