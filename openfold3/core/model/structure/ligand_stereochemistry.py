"""Inference-time ligand stereochemistry guidance.

The guidance is applied to the denoised coordinate estimate during reverse diffusion.
"""

import math
from typing import NamedTuple

import torch

_RESTRAINT_SPECS = {
    "distance": 2,
    "signed_dihedral": 4,
    "stereo_dihedral": 4,
    "planar_dihedral": 4,
}
_GEOMETRY_EPSILON = 1e-6


class _FlatBottomRestraints(NamedTuple):
    """Atom indices and bounds for one analytical restraint family."""

    index: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    weight: float


class PreparedLigandStereochemistryGuidance(NamedTuple):
    """Validated, device-local guidance inputs shared across diffusion steps."""

    start_fraction: float
    num_gd_steps: int
    distance: _FlatBottomRestraints
    signed_dihedral: _FlatBottomRestraints
    stereo_dihedral: _FlatBottomRestraints
    planar_dihedral: _FlatBottomRestraints


def _strip_leading_singletons(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    """Strip collator/model singleton axes until the tensor has the expected rank."""
    while tensor.dim() > ndim and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor


def _normalize_index_tensor(
    tensor: torch.Tensor, arity: int, name: str
) -> torch.Tensor:
    """Normalize an index tensor to shape ``[arity, n_constraints]``."""
    tensor = _strip_leading_singletons(tensor, 2)
    if tensor.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError(f"'{name}' must contain integer atom indices.")
    if tensor.dim() != 2:
        raise ValueError(f"'{name}' must be a 2D index tensor.")
    if tensor.shape[0] == arity:
        return tensor
    if tensor.shape[1] == arity:
        return tensor.T
    raise ValueError(f"'{name}' must have an index arity of {arity}.")


def _required_batch_feature(batch: dict, name: str):
    """Return a required guidance feature or raise a descriptive error."""
    if name not in batch:
        raise ValueError(
            f"Ligand stereochemistry guidance requires batch feature {name!r}."
        )
    return batch[name]


def _required_batch_scalar(batch: dict, name: str, dtype: type):
    """Read a required scalar setting from a collated guidance batch."""
    value = _required_batch_feature(batch, name)
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                f"Ligand stereochemistry batch feature {name!r} must be scalar."
            )
        value = value.item()
    return dtype(value)


def ligand_stereochemistry_guidance_enabled(batch: dict) -> bool:
    """Return whether a batch requests ligand stereochemistry guidance."""
    value = batch.get("ligand_stereochemistry_guidance_enabled")
    if value is None:
        return False
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(
                "'ligand_stereochemistry_guidance_enabled' must be scalar."
            )
        value = value.item()
    return bool(value)


def _prepare_restraint(
    batch: dict,
    name: str,
    arity: int,
    device: torch.device,
    num_atoms: int,
) -> _FlatBottomRestraints:
    """Normalize and validate one emitted flat-bottom restraint family."""
    prefix = f"ligand_stereochemistry_{name}"
    index_name = f"{prefix}_index"
    index = _normalize_index_tensor(
        _required_batch_feature(batch, index_name), arity, index_name
    ).to(device)
    num_constraints = index.shape[1]
    if num_constraints > 0 and (int(index.min()) < 0 or int(index.max()) >= num_atoms):
        raise ValueError(f"'{index_name}' contains an out-of-range atom index.")

    values = {}
    for suffix in ("lower", "upper"):
        feature_name = f"{prefix}_{suffix}"
        value = _strip_leading_singletons(
            _required_batch_feature(batch, feature_name), 1
        )
        if value.dim() != 1:
            raise ValueError(f"'{feature_name}' must be a 1D constraint tensor.")
        if value.shape[0] != num_constraints:
            raise ValueError(
                f"'{feature_name}' must contain one value per {index_name} constraint."
            )
        values[suffix] = value.to(device)

    weight = _required_batch_scalar(batch, f"{prefix}_weight", float)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"'{prefix}_weight' must be a finite non-negative value.")

    return _FlatBottomRestraints(index=index, weight=weight, **values)


def prepare_ligand_stereochemistry_guidance(
    batch: dict,
    atom_mask: torch.Tensor,
) -> PreparedLigandStereochemistryGuidance | None:
    """Validate and prepare invariant guidance inputs before diffusion."""
    if not ligand_stereochemistry_guidance_enabled(batch):
        return None
    if atom_mask.shape[0] != 1:
        raise ValueError(
            "Ligand stereochemistry guidance currently supports one query per "
            "model batch."
        )

    start_fraction = _required_batch_scalar(
        batch, "ligand_stereochemistry_start_fraction", float
    )
    if not 0.0 <= start_fraction <= 1.0:
        raise ValueError(
            "'ligand_stereochemistry_start_fraction' must be between 0 and 1."
        )
    num_gd_steps = _required_batch_scalar(
        batch, "ligand_stereochemistry_num_gd_steps", int
    )
    if num_gd_steps < 1:
        raise ValueError("'ligand_stereochemistry_num_gd_steps' must be at least 1.")
    restraints = {
        name: _prepare_restraint(
            batch,
            name,
            arity,
            atom_mask.device,
            atom_mask.shape[-1],
        )
        for name, arity in _RESTRAINT_SPECS.items()
    }
    if not any(restraint.index.shape[1] > 0 for restraint in restraints.values()):
        return None

    return PreparedLigandStereochemistryGuidance(
        start_fraction=start_fraction,
        num_gd_steps=num_gd_steps,
        **restraints,
    )


def _signed_dihedral(
    r_kj: torch.Tensor,
    n_ijk: torch.Tensor,
    n_jkl: torch.Tensor,
    r_kj_norm: torch.Tensor,
    n_ijk_norm: torch.Tensor,
    n_jkl_norm: torch.Tensor,
) -> torch.Tensor:
    """Compute a signed angle while distinguishing exact cis and trans geometry."""
    denominator = r_kj_norm * n_ijk_norm * n_jkl_norm
    sin_phi = (r_kj * torch.cross(n_ijk, n_jkl, dim=-1)).sum(dim=-1) / denominator
    cos_phi = (n_ijk * n_jkl).sum(dim=-1) / (n_ijk_norm * n_jkl_norm)
    return torch.atan2(sin_phi, cos_phi)


def _flat_bottom_derivative(
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Derivative of the linear flat-bottom energy with respect to ``value``."""
    return (value > upper.expand_as(value)).to(value.dtype) - (
        value < lower.expand_as(value)
    ).to(value.dtype)


def _scatter_variable_gradient(
    coords: torch.Tensor,
    index: torch.Tensor,
    value_gradient: torch.Tensor,
    energy_derivative: torch.Tensor,
) -> torch.Tensor:
    """Scatter pair/dihedral gradients back to the full atom coordinate axis."""
    gradient = torch.zeros_like(coords)
    scaled_gradient = value_gradient * energy_derivative.unsqueeze(-2).unsqueeze(-1)
    for atom_slot in range(index.shape[0]):
        atom_index = index[atom_slot].view(
            *((1,) * (coords.dim() - 2)), index.shape[1], 1
        )
        atom_index = atom_index.expand(*scaled_gradient[..., atom_slot, :, :].shape)
        gradient.scatter_add_(-2, atom_index, scaled_gradient[..., atom_slot, :, :])
    return gradient


def _distance_value_and_gradient(
    coords: torch.Tensor,
    index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pair distances and analytical coordinate gradients."""
    r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    r_ij_norm = torch.linalg.norm(r_ij, dim=-1).clamp_min(_GEOMETRY_EPSILON)
    r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)
    return r_ij_norm, torch.stack((r_hat_ij, -r_hat_ij), dim=-3)


def _dihedral_value_and_gradient(
    coords: torch.Tensor,
    index: torch.Tensor,
    absolute: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return improper dihedrals and their analytical gradients."""
    r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
    r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])

    n_ijk = torch.cross(r_ij, r_kj, dim=-1)
    n_jkl = torch.cross(r_kj, r_kl, dim=-1)

    r_kj_norm = torch.linalg.norm(r_kj, dim=-1).clamp_min(_GEOMETRY_EPSILON)
    n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1).clamp_min(_GEOMETRY_EPSILON)
    n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1).clamp_min(_GEOMETRY_EPSILON)

    phi = _signed_dihedral(
        r_kj,
        n_ijk,
        n_jkl,
        r_kj_norm,
        n_ijk_norm,
        n_jkl_norm,
    )

    a = (
        (r_ij.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        / r_kj_norm.square()
    ).unsqueeze(-1)
    b = (
        (r_kl.unsqueeze(-2) @ r_kj.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        / r_kj_norm.square()
    ).unsqueeze(-1)

    grad_i = n_ijk * (r_kj_norm / n_ijk_norm.square()).unsqueeze(-1)
    grad_l = -n_jkl * (r_kj_norm / n_jkl_norm.square()).unsqueeze(-1)
    grad_j = (a - 1) * grad_i - b * grad_l
    grad_k = (b - 1) * grad_l - a * grad_i
    gradient = torch.stack((grad_i, grad_j, grad_k, grad_l), dim=-3)

    if absolute:
        gradient = torch.where(
            (phi < 0).unsqueeze(-2).unsqueeze(-1), -gradient, gradient
        )
        phi = torch.abs(phi)

    return phi, gradient


def _restraint_gradient(
    coords: torch.Tensor,
    restraints: _FlatBottomRestraints,
    *,
    dihedral: bool,
    absolute: bool = False,
) -> torch.Tensor:
    """Evaluate and scatter one prepared flat-bottom restraint family."""
    if dihedral:
        value, value_gradient = _dihedral_value_and_gradient(
            coords,
            restraints.index,
            absolute=absolute,
        )
    else:
        value, value_gradient = _distance_value_and_gradient(coords, restraints.index)
    derivative = restraints.weight * _flat_bottom_derivative(
        value, restraints.lower, restraints.upper
    )
    return _scatter_variable_gradient(
        coords, restraints.index, value_gradient, derivative
    )


def ligand_stereochemistry_gradient(
    coords: torch.Tensor,
    guidance: PreparedLigandStereochemistryGuidance,
) -> torch.Tensor:
    """Return the analytical ligand stereochemistry guidance gradient."""
    gradient = torch.zeros_like(coords)
    restraint_groups = (
        (guidance.distance, False, False),
        (guidance.signed_dihedral, True, False),
        (guidance.stereo_dihedral, True, True),
        (guidance.planar_dihedral, True, True),
    )
    for restraints, is_dihedral, absolute in restraint_groups:
        if restraints.index.shape[1] > 0:
            gradient += _restraint_gradient(
                coords,
                restraints,
                dihedral=is_dihedral,
                absolute=absolute,
            )
    return gradient


def apply_ligand_stereochemistry_guidance(
    xl_denoised: torch.Tensor,
    guidance: PreparedLigandStereochemistryGuidance | None,
    step_fraction: float,
) -> torch.Tensor:
    """Apply opt-in ligand stereochemistry guidance to the denoised estimate."""
    if guidance is None or step_fraction < guidance.start_fraction:
        return xl_denoised

    guided = xl_denoised.float()
    for _ in range(guidance.num_gd_steps):
        guided = guided - ligand_stereochemistry_gradient(guided, guidance)

    return guided.to(dtype=xl_denoised.dtype)
