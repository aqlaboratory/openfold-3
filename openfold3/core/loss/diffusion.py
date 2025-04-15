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

import logging
from functools import partial
from typing import Callable, Optional

import torch

from openfold3.core.loss.loss_utils import loss_masked_batch_mean
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.core.utils.checkpointing import checkpoint_section

logger = logging.getLogger(__name__)


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
    atom_mask_gt = atom_mask_gt.bool()

    # Mean-centre positions
    w_mean = torch.sum(w * atom_mask_gt, dim=-1, keepdim=True) / (
        torch.sum(atom_mask_gt, dim=-1, keepdim=True) + eps
    )
    wx_mean = torch.sum(x * w[..., None] * atom_mask_gt[..., None], dim=-2) / (
        torch.sum(atom_mask_gt, dim=-1, keepdim=True) + eps
    )
    wx_gt_mean = torch.sum(x_gt * w[..., None] * atom_mask_gt[..., None], dim=-2) / (
        torch.sum(atom_mask_gt, dim=-1, keepdim=True) + eps
    )
    mu = wx_mean / w_mean
    mu_gt = wx_gt_mean / w_mean
    x = x - mu[..., None, :]
    x_gt = x_gt - mu_gt[..., None, :]

    # Construct covariance matrix
    H = x_gt[..., None] * x[..., None, :]
    H = H * w[..., None, None] * atom_mask_gt[..., None, None]
    H = torch.sum(H, dim=-3)

    # SVD (cast to float because doesn't work with bf16/fp16)
    dtype = x.dtype
    with torch.amp.autocast("cuda", dtype=torch.float32):
        try:
            U, _, V = torch.linalg.svd(H)
            dets = torch.linalg.det(U @ V)

            # Remove reflection
            F = torch.eye(3, device=U.device, dtype=U.dtype).tile((*H.shape[:-2], 1, 1))
            F[..., -1, -1] = torch.sign(dets)
            R = U @ F @ V
        except Exception as e:
            logger.warning(
                f"Error in computing rotation matrix in weighted rigid align. "
                f"Matrix:\n{H}\nError: {e}\n"
                "Returning identity matrix instead."
            )
            # Use identity rotation
            R = torch.eye(3, device=x.device, dtype=torch.float32).tile(
                (*H.shape[:-2], 1, 1)
            )

        # Apply alignment
        x_align = x @ R.transpose(-1, -2) + mu_gt[..., None, :]

    return x_align.to(dtype=dtype).detach()


def mse_loss(
    x: torch.Tensor,
    batch: dict,
    loss_token_mask: torch.Tensor,
    dna_weight: float,
    rna_weight: float,
    ligand_weight: float,
    eps: float,
) -> torch.Tensor:
    """
    Implements AF3 Equation 3.

    Args:
        x:
            [*, N_atom, 3] Atom positions
        batch:
            Feature dictionary
        loss_token_mask:
            [*, N_tokens] token-wise mask indicating whether to apply loss
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

    loss_atom_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=loss_token_mask,
    ).bool()
    loss_atom_mask = loss_atom_mask * atom_mask_gt

    mse = (
        (1 / 3.0)
        * torch.sum(
            torch.sum((x - x_gt_aligned) ** 2, dim=-1) * w * loss_atom_mask,
            dim=-1,
        )
        / (torch.sum(loss_atom_mask, dim=-1) + eps)
    )

    return mse


def bond_loss(x: torch.Tensor, batch: dict, eps: float) -> torch.Tensor:
    """
    Implements AF3 Equation 5.

    Args:
        x:
            [*, N_atom, 3] Atom positions
        batch:
            Feature dictionary
        eps:
            Small constant for stability
    Returns:
        [*] Auxiliary loss for bonded ligands
    """
    x_gt = batch["ground_truth"]["atom_positions"]
    atom_mask_gt = batch["ground_truth"]["atom_resolved_mask"]

    # Compute pairwise distances
    dx = torch.sqrt(
        eps + torch.sum((x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1)
    )
    dx_gt = torch.sqrt(
        eps + torch.sum((x_gt[..., None, :] - x_gt[..., None, :, :]) ** 2, dim=-1)
    )

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
    mask = (bond_mask * (atom_mask_gt[..., None] * atom_mask_gt[..., None, :])).bool()

    loss = torch.sum((dx - dx_gt) ** 2 * mask, dim=(-1, -2)) / (
        torch.sum(mask, dim=(-1, -2)) + eps
    )

    return loss


def smooth_lddt_loss(
    x: torch.Tensor, batch: dict, loss_token_mask: torch.Tensor, eps: float
) -> torch.Tensor:
    """
    Implements AF3 Algorithm 27.

    Args:
        x:
            [*, N_atom, 3] Atom positions
        batch:
            Feature dictionary
        loss_token_mask:
            [*, N_tokens] token-wise mask indicating whether to apply loss
        eps:
            Small constant for stability
    Returns:
        [*] Auxiliary structure-based loss based on smooth LDDT
    """
    x_gt = batch["ground_truth"]["atom_positions"]
    atom_mask_gt = batch["ground_truth"]["atom_resolved_mask"]

    # [*, N_atom, N_atom]
    dx = torch.sqrt(
        eps + torch.sum((x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1)
    )
    dx_gt = torch.sqrt(
        eps + torch.sum((x_gt[..., None, :] - x_gt[..., None, :, :]) ** 2, dim=-1),
    )

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
    loss_atom_mask = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=loss_token_mask,
    )
    loss_atom_mask = loss_atom_mask * atom_mask_gt

    # [*, N_atom, N_atom]
    c = (dx_gt < 30) * is_nucleotide[..., None] + (dx_gt < 15) * (
        1 - is_nucleotide[..., None]
    )

    # [*]
    mask = 1 - torch.eye(x.shape[-2], device=x.device, dtype=x.dtype).tile(
        (*x.shape[:-2], 1, 1)
    )

    mask = (mask * (loss_atom_mask[..., None] * loss_atom_mask[..., None, :])).bool()

    ce_mean = torch.sum(c * e * mask, dim=(-1, -2)) / (
        torch.sum(mask, dim=(-1, -2)) + eps
    )
    c_mean = torch.sum(c * mask, dim=(-1, -2)) / (torch.sum(mask, dim=(-1, -2)) + eps)
    lddt = ce_mean / (c_mean + eps)

    return 1 - lddt


def run_low_mem_loss_fn(
    loss_fn: Callable, x: torch.Tensor, kwargs: dict, chunk_size: int
) -> torch.Tensor:
    """
    Run a loss function in low memory mode by chunking over the sample dimension with
    activation checkpointing.

    Args:
        loss_fn:
            The loss function to run
        x:
            [*, N_atom, 3] Atom positions
        kwargs:
            Keyword arguments for the loss function
        chunk_size:
            Chunk size over sample dimension

    Returns:
        [*, no_samples] Loss for each sample
    """
    loss_fn_partial = partial(loss_fn, **kwargs)
    chunks = []
    for i in range(0, x.shape[-3], chunk_size):
        x_chunk = x[..., i : i + chunk_size, :, :]
        l_chunk = checkpoint_section(
            fn=loss_fn_partial, args=(x_chunk,), apply_ckpt=True, use_reentrant=False
        )
        chunks.append(l_chunk)

    return torch.cat(chunks, dim=-1)


def diffusion_loss(
    batch: dict,
    x: torch.Tensor,
    t: torch.Tensor,
    sigma_data: float,
    dna_weight: float = 5.0,
    rna_weight: float = 5.0,
    ligand_weight: float = 10.0,
    eps: float = 1e-8,
    chunk_size: Optional[int] = None,
    **kwargs,
) -> [torch.Tensor, dict]:
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
        dna_weight:
            Upweight factor for DNA atoms
        rna_weight:
            Upweight factor for RNA atoms
        ligand_weight:
            Upweight factor for ligand atoms
        eps:
            Small constant for stability
        chunk_size:
            Chunk size over sample dimension for large loss computation
            for smooth lddt and bond loss. Defaults to no chunking.
    Returns:
        mean_loss:
            Diffusion loss
        loss_breakdown:
            Dict of individual component losses (unweighted)
    """
    loss_weights = batch["loss_weights"]
    mse_weight = loss_weights["mse"]
    bond_weight = loss_weights["bond"]
    smooth_lddt_weight = loss_weights["smooth_lddt"]

    # Create a mask for non-protein tokens (used for PDB disordered set):
    disable_non_protein_loss = loss_weights.get(
        "disable_non_protein_diffusion_weights", None
    )
    loss_token_mask = (
        batch["is_protein"]
        if disable_non_protein_loss is not None and disable_non_protein_loss.any()
        else torch.ones_like(batch["is_protein"])
    )

    l_mse = mse_loss(
        x=x,
        batch=batch,
        loss_token_mask=loss_token_mask,
        dna_weight=dna_weight,
        rna_weight=rna_weight,
        ligand_weight=ligand_weight,
        eps=eps,
    )

    # Mean over diffusion sample dimension
    loss_breakdown = {"mse": l_mse.detach().clone().mean(dim=-1)}

    l_bond = 0.0
    l_smooth_lddt = 0.0
    if chunk_size is None:
        if bond_weight.any():
            l_bond = bond_loss(x=x, batch=batch, eps=eps)
            loss_breakdown["bond"] = l_bond.detach().clone().mean(dim=-1)

        if smooth_lddt_weight.any():
            l_smooth_lddt = smooth_lddt_loss(
                x=x, batch=batch, loss_token_mask=loss_token_mask, eps=eps
            )

            # Mean over diffusion sample dimension
            loss_breakdown["smooth_lddt"] = l_smooth_lddt.detach().clone().mean(dim=-1)
    else:
        if bond_weight.any():
            l_bond = run_low_mem_loss_fn(
                loss_fn=bond_loss,
                x=x,
                kwargs={"batch": batch, "eps": eps},
                chunk_size=chunk_size,
            )

            # Mean over diffusion sample dimension
            loss_breakdown["bond"] = l_bond.detach().clone().mean(dim=-1)

        if smooth_lddt_weight.any():
            l_smooth_lddt = run_low_mem_loss_fn(
                loss_fn=smooth_lddt_loss,
                x=x,
                kwargs={
                    "batch": batch,
                    "eps": eps,
                    "loss_token_mask": loss_token_mask,
                },
                chunk_size=chunk_size,
            )
            loss_breakdown["smooth_lddt"] = l_smooth_lddt.detach().clone().mean(dim=-1)

    # Mean over batch dimension for individual losses
    # Mask out samples where the loss is disabled
    loss_breakdown = {
        f"{name}_loss": loss_masked_batch_mean(
            loss=loss,
            weight=loss_weights[name].squeeze(-1),
            apply_weight=False,
            nan_zero_weights=True,
            eps=eps,
        )
        for name, loss in loss_breakdown.items()
    }

    l_mse = l_mse * mse_weight
    l_bond = l_bond * bond_weight
    l_smooth_lddt = l_smooth_lddt * smooth_lddt_weight

    # Note: Changed from SI, denominator (t + sigma_data) ** 2 changed
    #  to (t * sigma_data) ** 2.
    w = (t**2 + sigma_data**2) / (t * sigma_data) ** 2
    l = w * (l_mse + l_bond) + l_smooth_lddt

    # Mean over diffusion sample dimension
    mean_loss = torch.mean(l, dim=-1)

    # Mean over batch dimension, only for samples with diffusion losses enabled
    mean_loss = loss_masked_batch_mean(
        loss=mean_loss,
        weight=mse_weight.squeeze(-1),
        apply_weight=False,
        nan_zero_weights=False,
        eps=eps,
    )

    return mean_loss, loss_breakdown
