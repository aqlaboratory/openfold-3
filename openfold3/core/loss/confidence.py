# Copyright 2021 AlQuraishi Laboratory
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

"""Confidence losses from predicted logits in the Confidence Module."""

from functools import partial

import torch
import torch.nn.functional as F

from openfold3.core.loss.loss_utils import (
    loss_masked_batch_mean,
    sigmoid_cross_entropy,
    softmax_cross_entropy,
)
from openfold3.core.metrics.validation import lddt
from openfold3.core.np import residue_constants
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_atom_index_offset,
    get_token_frame_atoms,
    get_token_representative_atoms,
)
from openfold3.core.utils.rigid_utils import Rigid
from openfold3.core.utils.tensor_utils import binned_one_hot, tensor_tree_map

########################
# AF2 Confidence Losses
########################


def ca_plddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos : (ca_pos + 1)]  # keep dim

    score = lddt(
        all_atom_pred_pos, all_atom_positions, all_atom_mask, cutoff=cutoff, eps=eps
    )

    score[score < 0] = 0

    score = score.detach()
    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = F.one_hot(bin_index, num_classes=no_bins)

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    all_atom_mask = all_atom_mask.squeeze(-1)
    loss = torch.sum(errors * all_atom_mask, dim=-1) / (
        eps + torch.sum(all_atom_mask, dim=-1)
    )

    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def atom37_experimentally_resolved_loss(
    logits: torch.Tensor,
    atom37_atom_exists: torch.Tensor,
    all_atom_mask: torch.Tensor,
    resolution: torch.Tensor,
    min_resolution: float,
    max_resolution: float,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    errors = sigmoid_cross_entropy(logits, all_atom_mask)
    loss = torch.sum(errors * atom37_atom_exists, dim=-1)
    loss = loss / (eps + torch.sum(atom37_atom_exists, dim=(-1, -2)).unsqueeze(-1))
    loss = torch.sum(loss, dim=-1)

    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    loss = torch.mean(loss)

    return loss


def masked_msa_loss(logits, true_msa, bert_mask, num_classes, eps=1e-8, **kwargs):
    """
    Computes BERT-style masked MSA loss. Implements subsection 1.9.9.

    Args:
        logits: [*, N_seq, N_res, 23] predicted residue distribution
        true_msa: [*, N_seq, N_res] true MSA
        bert_mask: [*, N_seq, N_res] MSA mask
    Returns:
        Masked MSA loss
    """
    errors = softmax_cross_entropy(logits, F.one_hot(true_msa, num_classes=num_classes))

    # FP16-friendly averaging. Equivalent to:
    # loss = (
    #     torch.sum(errors * bert_mask, dim=(-1, -2)) /
    #     (eps + torch.sum(bert_mask, dim=(-1, -2)))
    # )
    loss = errors * bert_mask
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = eps + torch.sum(scale * bert_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = torch.mean(loss)

    return loss


def tm_loss(
    logits,
    final_affine_tensor,
    backbone_rigid_tensor,
    backbone_rigid_mask,
    resolution,
    max_bin=31,
    no_bins=64,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps=1e-8,
    **kwargs,
):
    # first check whether this is a tensor_7 or tensor_4*4
    if final_affine_tensor.shape[-1] == 7:
        pred_affine = Rigid.from_tensor_7(final_affine_tensor)
    elif final_affine_tensor.shape[-1] == 4:
        pred_affine = Rigid.from_tensor_4x4(final_affine_tensor)
    backbone_rigid = Rigid.from_tensor_4x4(backbone_rigid_tensor)

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum((_points(pred_affine) - _points(backbone_rigid)) ** 2, dim=-1)

    sq_diff = sq_diff.detach()

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)
    boundaries = boundaries**2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(logits, F.one_hot(true_bins, no_bins))

    square_mask = backbone_rigid_mask[..., None] * backbone_rigid_mask[..., None, :]

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = loss * ((resolution >= min_resolution) & (resolution <= max_resolution))

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


########################
# AF3 Confidence Losses
########################


def express_coords_in_frames(
    x: torch.Tensor, phi: tuple[torch.Tensor, torch.Tensor, torch.Tensor], eps: float
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
    w1_norm = torch.sqrt(eps + torch.sum(w1**2, dim=-1, keepdim=True))
    w2_norm = torch.sqrt(eps + torch.sum(w2**2, dim=-1, keepdim=True))
    w1 = w1 / w1_norm
    w2 = w2 / w2_norm

    # Build orthonormal basis
    # [*, N_token, 3]
    e1 = w1 + w2
    e2 = w2 - w1
    e1_norm = torch.sqrt(eps + torch.sum(e1**2, dim=-1, keepdim=True))
    e2_norm = torch.sqrt(eps + torch.sum(e2**2, dim=-1, keepdim=True))
    e1 = e1 / e1_norm
    e2 = e2 / e2_norm

    # BF16-friendly cross product
    e3 = [
        e1[..., 1] * e2[..., 2] - e1[..., 2] * e2[..., 1],
        e1[..., 2] * e2[..., 0] - e1[..., 0] * e2[..., 2],
        e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0],
    ]

    e3 = torch.stack(e3, dim=-1)

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
    phi: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    phi_gt: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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
    return torch.sqrt(eps + torch.sum((xij - xij_gt) ** 2, dim=-1))


def all_atom_plddt_loss(
    batch: dict,
    x: torch.Tensor,
    logits: torch.Tensor,
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
        logits:
            [*, N_atom, no_bins] Predicted logits
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
    x_gt = batch["ground_truth"]["atom_positions"]
    dx = torch.sqrt(
        eps + torch.sum((x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1)
    )
    dx_gt = torch.sqrt(
        eps + torch.sum((x_gt[..., None, :] - x_gt[..., None, :, :]) ** 2, dim=-1)
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

    restype = batch["restype"]
    ca_atom_index_offset, ca_atom_mask = get_token_atom_index_offset(
        atom_name="CA", restype=restype
    )
    c1p_atom_index_offset, c1p_atom_mask = get_token_atom_index_offset(
        atom_name="C1'", restype=restype
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
    atom_mask_gt = batch["ground_truth"]["atom_resolved_mask"].bool()
    atom_mask_shape = list(atom_mask_gt.shape)
    padded_atom_mask_shape = list(atom_mask_shape)
    padded_atom_mask_shape[-1] = padded_atom_mask_shape[-1] + 1

    # TODO: Revisit this to see if this happens anywhere else
    # Rep index is padded for shorter sequences, remove it match ground truth
    rep_index_unpadded = rep_index.long()[..., : atom_mask_shape[-1]]

    # We need to expand the rep_index_unpadded to match the
    # shape of (bs, n_samples, ...)
    rep_index_unpadded = rep_index_unpadded.expand(
        (*atom_mask_shape[:-1], rep_index_unpadded.shape[-1])
    )

    atom_mask = torch.zeros(padded_atom_mask_shape, device=x.device, dtype=x.dtype)
    atom_mask = atom_mask.scatter_(
        index=rep_index_unpadded, src=torch.ones_like(atom_mask), dim=-1
    )[..., :-1]
    atom_mask = atom_mask * atom_mask_gt

    # Construct pair atom selection mask for lddt computation
    # [*, N_atom, N_atom]
    pair_atom_mask = atom_mask_gt[..., None] * atom_mask[..., None, :]
    pair_mask = (
        pair_mask * pair_atom_mask * (1.0 - torch.eye(n_atom, device=atom_mask.device))
    ).bool()

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
    lddt_b = binned_one_hot(lddt, v_bins).to(dtype=x.dtype)

    errors = softmax_cross_entropy(logits, lddt_b)

    # Compute loss on plddt
    l_plddt = torch.sum(errors * atom_mask_gt, dim=-1) / (
        torch.sum(atom_mask_gt, dim=-1) + eps
    )

    return l_plddt


def per_sample_all_atom_plddt_loss(
    batch: dict,
    x: torch.Tensor,
    logits: torch.Tensor,
    no_bins: int,
    bin_min: float,
    bin_max: float,
    eps: float,
) -> torch.Tensor:
    """
    Compute loss per sample on predicted local distance difference test (pLDDT).

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Predicted atom positions
        logits:
            [*, N_atom, no_bins] Predicted logits
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

    all_atom_plddt_partial = partial(
        all_atom_plddt_loss, no_bins=no_bins, bin_min=bin_min, bin_max=bin_max, eps=eps
    )
    chunks = []
    for i in range(0, x.shape[-3], 1):

        def index_batch(t: torch.Tensor):
            no_samples = t.shape[1]
            if no_samples == 1:
                return t
            return t[:, i : i + 1]  # noqa: B023

        batch_chunk = tensor_tree_map(index_batch, batch)
        x_chunk = x[:, i : i + 1]
        logits_chunk = logits[:, i : i + 1]
        l_chunk = all_atom_plddt_partial(batch_chunk, x_chunk, logits_chunk)
        chunks.append(l_chunk)

    return torch.cat(chunks, dim=-1)


def pae_loss(
    batch: dict,
    x: torch.Tensor,
    logits: torch.Tensor,
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
        logits:
            [*, N_token, N_token, no_bins] Predicted logits
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
    atom_mask = batch["atom_mask"]
    phi, valid_frame_mask = get_token_frame_atoms(
        batch=batch,
        x=x,
        atom_mask=atom_mask,
        angle_threshold=angle_threshold,
        eps=eps,
        inf=inf,
    )

    atom_positions_gt = batch["ground_truth"]["atom_positions"]
    atom_mask_gt = batch["ground_truth"]["atom_resolved_mask"]
    phi_gt, valid_frame_mask_gt = get_token_frame_atoms(
        batch=batch,
        x=atom_positions_gt,
        atom_mask=atom_mask_gt,
        angle_threshold=angle_threshold,
        eps=eps,
        inf=inf,
    )

    # Extract representative atom coordinates
    rep_x, rep_atom_mask = get_token_representative_atoms(
        batch=batch, x=x, atom_mask=atom_mask
    )
    rep_x_gt, rep_atom_mask_gt = get_token_representative_atoms(
        batch=batch, x=atom_positions_gt, atom_mask=atom_mask_gt
    )

    # Compute alignment error
    # [*, N_token, N_token]
    e = compute_alignment_error(x=rep_x, x_gt=rep_x_gt, phi=phi, phi_gt=phi_gt, eps=eps)

    # Compute binned alignment error
    # [*, N_token, N_token, no_bins]
    bin_size = (bin_max - bin_min) / no_bins
    v_bins = bin_min + torch.arange(no_bins, device=logits.device) * bin_size
    e_b = binned_one_hot(e, v_bins).to(dtype=logits.dtype)

    # Compute predicted alignment error
    pair_mask = (valid_frame_mask[..., None] * rep_atom_mask[..., None, :]) * (
        valid_frame_mask_gt[..., None] * rep_atom_mask_gt[..., None, :]
    ).bool()

    errors = softmax_cross_entropy(logits, e_b)

    # Compute loss on pae
    l_pae = torch.sum(errors * pair_mask, dim=(-1, -2)) / (
        torch.sum(pair_mask, dim=(-1, -2)) + eps
    )

    return l_pae


def pde_loss(
    batch: dict,
    x: torch.Tensor,
    logits: torch.Tensor,
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
        logits:
            [*, N_token, N_token, no_bins] Predicted logits for errors in absolute
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
    atom_mask = batch["atom_mask"]
    rep_x, _ = get_token_representative_atoms(batch=batch, x=x, atom_mask=atom_mask)
    rep_x_gt, rep_atom_mask_gt = get_token_representative_atoms(
        batch=batch,
        x=batch["ground_truth"]["atom_positions"],
        atom_mask=batch["ground_truth"]["atom_resolved_mask"],
    )

    # Compute prediction target
    d = torch.sqrt(
        eps + torch.sum((rep_x[..., None, :] - rep_x[..., None, :, :]) ** 2, dim=-1)
    )
    d_gt = torch.sqrt(
        eps
        + torch.sum((rep_x_gt[..., None, :] - rep_x_gt[..., None, :, :]) ** 2, dim=-1)
    )
    e = torch.abs(d - d_gt)

    # Compute binned prediction target
    bin_size = (bin_max - bin_min) / no_bins
    v_bins = bin_min + torch.arange(no_bins, device=e.device) * bin_size
    e_b = binned_one_hot(e, v_bins).to(dtype=e.dtype)

    pair_mask = (rep_atom_mask_gt[..., None] * rep_atom_mask_gt[..., None, :]).bool()

    errors = softmax_cross_entropy(logits, e_b)

    # Compute loss on predicted distance error
    l_pde = torch.sum(errors * pair_mask, dim=(-1, -2)) / (
        torch.sum(pair_mask, dim=(-1, -2)) + eps
    )

    return l_pde


def all_atom_experimentally_resolved_loss(
    batch: dict, logits: torch.Tensor, no_bins: int, eps: float
):
    """
    Implements AF3 Equation 14.

    Args:
        batch:
            Feature dictionary
        logits:
            [*, N_atom, no_bins] Predicted logits for whether the atom is
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
    atom_mask_gt = batch["ground_truth"]["atom_resolved_mask"]
    y_b = F.one_hot(atom_mask_gt.long(), num_classes=no_bins)

    atom_mask = batch["atom_mask"].bool()

    # Compute loss on experimentally resolved prediction
    errors = softmax_cross_entropy(logits, y_b)

    l_resolved = torch.sum(errors * atom_mask, dim=-1) / (
        torch.sum(atom_mask, dim=-1) + eps
    )

    return l_resolved


def confidence_loss(
    batch: dict,
    output: dict,
    plddt: dict,
    pde: dict,
    experimentally_resolved: dict,
    pae: dict,
    eps: float,
    inf: float,
    **kwargs,
) -> [torch.Tensor, dict]:
    """
    Compute loss on confidence module.

    Args:
        batch:
            Feature dictionary
        output:
            Output dictionary
        plddt:
            Dict for pLDDT loss containing the following:
                "no_bins": Number of bins
                "bin_min": Minimum bin value
                "bin_max": Maximum bin value
        pde:
            Dict for PDE loss containing the following:
                "no_bins": Number of bins
                "bin_min": Minimum bin value
                "bin_max": Maximum bin value
        experimentally_resolved:
            Dict for experimentally resolved loss containing the following:
                "no_bins": Number of bins
        pae:
            Dict for PAE loss containing the following:
                "angle_threshold": Angle threshold for filtering co-linear atoms
                "no_bins": Number of bins
                "bin_min": Minimum bin value
                "bin_max": Maximum bin value
        eps:
            Small float for numerical stability
        inf:
            Large float for numerical stability
    Returns:
        mean_loss:
            Loss on confidence module
        loss_breakdown:
            Dict of individual component losses (unweighted)
    """
    loss_weights = batch["loss_weights"]

    # For more than one sample, calculate per-sample losses
    # This will happen in validation, where 5 samples are generated
    # for the rollout.
    n_samples = output["atom_positions_predicted"].shape[-3]
    n_atom = output["atom_positions_predicted"].shape[-2]
    if n_samples > 1 and n_atom > 5e4:
        l_plddt = per_sample_all_atom_plddt_loss(
            batch=batch,
            x=output["atom_positions_predicted"],
            logits=output["plddt_logits"],
            no_bins=plddt["no_bins"],
            bin_min=plddt["bin_min"],
            bin_max=plddt["bin_max"],
            eps=eps,
        )
    else:
        l_plddt = all_atom_plddt_loss(
            batch=batch,
            x=output["atom_positions_predicted"],
            logits=output["plddt_logits"],
            no_bins=plddt["no_bins"],
            bin_min=plddt["bin_min"],
            bin_max=plddt["bin_max"],
            eps=eps,
        )

    l_pde = pde_loss(
        batch=batch,
        x=output["atom_positions_predicted"],
        logits=output["pde_logits"],
        no_bins=pde["no_bins"],
        bin_min=pde["bin_min"],
        bin_max=pde["bin_max"],
        eps=eps,
    )

    l_resolved = all_atom_experimentally_resolved_loss(
        batch=batch,
        logits=output["experimentally_resolved_logits"],
        no_bins=experimentally_resolved["no_bins"],
        eps=eps,
    )

    loss_breakdown = {
        "plddt": l_plddt,
        "pde": l_pde,
        "experimentally_resolved": l_resolved,
    }

    pae_weight = loss_weights["pae"]
    if pae_weight.any():
        l_pae = pae_loss(
            batch=batch,
            x=output["atom_positions_predicted"],
            logits=output["pae_logits"],
            angle_threshold=pae["angle_threshold"],
            no_bins=pae["no_bins"],
            bin_min=pde["bin_min"],
            bin_max=pde["bin_max"],
            eps=eps,
            inf=inf,
        )

        loss_breakdown["pae"] = l_pae

    # Calculate total confidence loss
    # Mask out samples where the loss is disabled
    conf_loss = sum(
        [
            loss_masked_batch_mean(
                loss=loss,
                weight=loss_weights[name],
                apply_weight=True,
                eps=eps,
            )
            for name, loss in loss_breakdown.items()
        ]
    )

    # Unweighted mean over batch dimension for individual losses
    valid_loss_breakdown = {}
    for name, loss in loss_breakdown.items():
        if loss_weights[name].any():
            valid_loss_breakdown[f"{name}_loss"] = loss_masked_batch_mean(
                loss=loss.detach().clone(),
                weight=loss_weights[name],
                apply_weight=False,
                eps=eps,
            )

    return conf_loss, valid_loss_breakdown
