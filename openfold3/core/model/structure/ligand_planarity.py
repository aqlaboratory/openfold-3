"""Inference-time guidance for assigned C=C planarity."""

import math
from typing import NamedTuple

import torch

PLANARITY_START_FRACTION = 0.725
_PLANARITY_BUFFER = math.radians(15.0)
_PLANARITY_STEP_SIZE = 0.05
_PLANARITY_STEPS = 20
_GEOMETRY_EPSILON = 1e-6


class LigandPlanarityRestraints(NamedTuple):
    """Device-local atom quartets and their reference orientations."""

    atom_indices: torch.Tensor
    trans_orientations: torch.Tensor


def _strip_leading_singletons(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    while tensor.dim() > ndim and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor


def prepare_ligand_planarity_restraints(
    batch: dict, atom_mask: torch.Tensor
) -> LigandPlanarityRestraints | None:
    """Validate and move restraint features to the sampling device."""
    if "ligand_planarity_index" not in batch:
        return None
    if batch["ligand_planarity_index"].numel() == 0:
        return None
    if atom_mask.shape[0] != 1:
        raise ValueError("Ligand planarity restraints require one query per batch.")

    indices = _strip_leading_singletons(batch["ligand_planarity_index"], 2)
    if indices.dim() != 2 or 4 not in indices.shape:
        raise ValueError("'ligand_planarity_index' must have atom-index arity 4.")
    if indices.shape[0] == 4:
        indices = indices.T
    if int(indices.min()) < 0 or int(indices.max()) >= atom_mask.shape[-1]:
        raise ValueError("'ligand_planarity_index' contains an invalid atom index.")

    trans = _strip_leading_singletons(batch["ligand_planarity_trans"], 1)
    if trans.dim() != 1 or trans.shape[0] != indices.shape[0]:
        raise ValueError(
            "'ligand_planarity_trans' must contain one value per restraint."
        )
    return LigandPlanarityRestraints(
        atom_indices=indices.to(device=atom_mask.device, dtype=torch.long),
        trans_orientations=trans.to(device=atom_mask.device, dtype=torch.bool),
    )


def _absolute_dihedrals_and_gradients(
    coords: torch.Tensor, atom_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    points = coords[..., atom_indices, :]
    p0, p1, p2, p3 = points.unbind(dim=-2)
    r01 = p0 - p1
    r21 = p2 - p1
    r23 = p2 - p3
    normal_1 = torch.cross(r01, r21, dim=-1)
    normal_2 = torch.cross(r21, r23, dim=-1)
    central_norm = torch.linalg.norm(r21, dim=-1).clamp_min(_GEOMETRY_EPSILON)
    normal_1_norm = torch.linalg.norm(normal_1, dim=-1).clamp_min(_GEOMETRY_EPSILON)
    normal_2_norm = torch.linalg.norm(normal_2, dim=-1).clamp_min(_GEOMETRY_EPSILON)

    sine = (r21 * torch.cross(normal_1, normal_2, dim=-1)).sum(dim=-1) / (
        central_norm * normal_1_norm * normal_2_norm
    )
    cosine = (normal_1 * normal_2).sum(dim=-1) / (normal_1_norm * normal_2_norm)
    signed_angles = torch.atan2(sine, cosine)

    central_norm_sq = central_norm.square().unsqueeze(-1)
    projection_1 = (r01 * r21).sum(dim=-1, keepdim=True) / central_norm_sq
    projection_2 = (r23 * r21).sum(dim=-1, keepdim=True) / central_norm_sq
    grad_0 = normal_1 * (central_norm / normal_1_norm.square()).unsqueeze(-1)
    grad_3 = -normal_2 * (central_norm / normal_2_norm.square()).unsqueeze(-1)
    grad_1 = (projection_1 - 1) * grad_0 - projection_2 * grad_3
    grad_2 = (projection_2 - 1) * grad_3 - projection_1 * grad_0
    gradients = torch.stack((grad_0, grad_1, grad_2, grad_3), dim=-2)
    gradients = torch.where(
        (signed_angles < 0).unsqueeze(-1).unsqueeze(-1), -gradients, gradients
    )
    return signed_angles.abs(), gradients


def _planarity_violations(
    angles: torch.Tensor, trans_orientations: torch.Tensor
) -> torch.Tensor:
    return torch.where(
        trans_orientations,
        torch.relu(math.pi - _PLANARITY_BUFFER - angles),
        torch.relu(angles - _PLANARITY_BUFFER),
    )


def _planarity_gradient(
    coords: torch.Tensor, restraints: LigandPlanarityRestraints
) -> torch.Tensor:
    angles, angle_gradients = _absolute_dihedrals_and_gradients(
        coords, restraints.atom_indices
    )
    violations = _planarity_violations(angles, restraints.trans_orientations)
    derivatives = torch.where(
        restraints.trans_orientations,
        -violations,
        violations,
    )
    scaled = angle_gradients * derivatives.unsqueeze(-1).unsqueeze(-1)
    gradient = torch.zeros_like(coords)
    for atom_slot in range(4):
        index = restraints.atom_indices[:, atom_slot].view(
            *((1,) * (coords.dim() - 2)), -1, 1
        )
        atom_gradient = scaled[..., atom_slot, :]
        gradient.scatter_add_(-2, index.expand_as(atom_gradient), atom_gradient)
    return gradient


def apply_ligand_planarity_restraints(
    xl_denoised: torch.Tensor, restraints: LigandPlanarityRestraints
) -> torch.Tensor:
    """Move a denoised estimate into the restraints' flat-bottom region."""
    dtype = xl_denoised.dtype
    guided = xl_denoised.detach().float()
    for _ in range(_PLANARITY_STEPS):
        guided = guided - _PLANARITY_STEP_SIZE * _planarity_gradient(guided, restraints)
    return guided.to(dtype=dtype)
