"""Inference-time ligand stereochemistry guidance.

The guidance is applied to the denoised coordinate estimate during reverse diffusion.
It preserves local ligand chemistry encoded by the input molecule without adding any
binding-site bias.

Physical guidance is adapted from Boltz. See ``THIRD_PARTY_NOTICES.md``.
"""

from typing import NamedTuple

import torch

from openfold3.core.config import ligand_stereochemistry_defaults as defaults

_INDEX_ARITIES = {
    "rdkit_bounds_index": 2,
    "chiral_atom_index": 4,
    "stereo_bond_index": 4,
    "planar_bond_index": 6,
}

_VECTOR_KEYS = {
    "rdkit_lower_bounds",
    "rdkit_upper_bounds",
    "rdkit_bounds_bond_mask",
    "rdkit_bounds_angle_mask",
    "rdkit_bounds_pair_vdw_cutoff",
    "chiral_atom_orientations",
    "stereo_bond_orientations",
}


class PreparedLigandStereochemistryGuidance(NamedTuple):
    """Validated, device-local guidance inputs shared across diffusion steps."""

    features: dict[str, torch.Tensor]
    start_fraction: float
    num_gd_steps: int
    posebusters_bounds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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


def _guidance_features(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Collect stereochemistry feature tensors for a single inference structure."""
    features = {}
    for key, arity in _INDEX_ARITIES.items():
        tensor = _required_batch_feature(batch, key)
        features[key] = _normalize_index_tensor(tensor, arity, key).to(device)
    for key in _VECTOR_KEYS:
        tensor = _required_batch_feature(batch, key)
        tensor = _strip_leading_singletons(tensor, 1)
        if tensor.dim() != 1:
            raise ValueError(f"'{key}' must be a 1D constraint tensor.")
        features[key] = tensor.to(device)
    return features


def _validate_constraint_group(
    features: dict[str, torch.Tensor],
    index_name: str,
    vector_names: tuple[str, ...],
    num_atoms: int,
) -> None:
    """Validate index bounds and vector cardinality for one constraint group."""
    index = features[index_name]
    num_constraints = index.shape[1]
    if num_constraints > 0 and (int(index.min()) < 0 or int(index.max()) >= num_atoms):
        raise ValueError(f"'{index_name}' contains an out-of-range atom index.")
    for vector_name in vector_names:
        if features[vector_name].shape[0] != num_constraints:
            raise ValueError(
                f"'{vector_name}' must contain one value per {index_name} constraint."
            )


def _validate_guidance_features(
    features: dict[str, torch.Tensor], num_atoms: int
) -> None:
    """Validate every emitted stereochemistry constraint group."""
    _validate_constraint_group(
        features,
        "rdkit_bounds_index",
        (
            "rdkit_lower_bounds",
            "rdkit_upper_bounds",
            "rdkit_bounds_bond_mask",
            "rdkit_bounds_angle_mask",
            "rdkit_bounds_pair_vdw_cutoff",
        ),
        num_atoms,
    )
    _validate_constraint_group(
        features,
        "chiral_atom_index",
        ("chiral_atom_orientations",),
        num_atoms,
    )
    _validate_constraint_group(
        features,
        "stereo_bond_index",
        ("stereo_bond_orientations",),
        num_atoms,
    )
    _validate_constraint_group(features, "planar_bond_index", (), num_atoms)


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


def _buffered_rdkit_bounds(
    features: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Boltz PoseBusters buffers and atom-radius cutoffs to RDKit bounds."""
    index = features["rdkit_bounds_index"].long()
    lower = features["rdkit_lower_bounds"].clone()
    upper = features["rdkit_upper_bounds"].clone()
    bond = features["rdkit_bounds_bond_mask"].bool()
    angle = features["rdkit_bounds_angle_mask"].bool()
    pair_vdw_cutoff = features["rdkit_bounds_pair_vdw_cutoff"].to(lower.dtype)

    lower[bond & ~angle] *= 1.0 - defaults.BOND_BUFFER
    upper[bond & ~angle] *= 1.0 + defaults.BOND_BUFFER
    lower[~bond & angle] *= 1.0 - defaults.ANGLE_BUFFER
    upper[~bond & angle] *= 1.0 + defaults.ANGLE_BUFFER
    shared_buffer = min(defaults.BOND_BUFFER, defaults.ANGLE_BUFFER)
    lower[bond & angle] *= 1.0 - shared_buffer
    upper[bond & angle] *= 1.0 + shared_buffer
    lower[~bond & ~angle] *= 1.0 - defaults.CLASH_BUFFER
    upper[~bond & ~angle] = float("inf")

    lower[~bond] = torch.maximum(lower[~bond], pair_vdw_cutoff[~bond])
    upper[bond] = torch.minimum(upper[bond], pair_vdw_cutoff[bond])
    return index, lower, upper


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

    features = _guidance_features(batch, atom_mask.device)
    _validate_guidance_features(features, atom_mask.shape[-1])
    return PreparedLigandStereochemistryGuidance(
        features=features,
        start_fraction=start_fraction,
        num_gd_steps=num_gd_steps,
        posebusters_bounds=_buffered_rdkit_bounds(features),
    )


def _flat_bottom_derivative(
    value: torch.Tensor,
    lower: torch.Tensor | None,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Derivative of the linear flat-bottom energy with respect to ``value``."""
    derivative = torch.zeros_like(value)
    if lower is not None:
        derivative = derivative - (value < lower.expand_as(value)).to(value.dtype)
    derivative = derivative + (value > upper.expand_as(value)).to(value.dtype)
    return derivative


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
    coords: torch.Tensor, index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pair distances and analytic coordinate gradients."""
    r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    r_ij_norm = torch.linalg.norm(r_ij, dim=-1).clamp_min(defaults.GEOMETRY_EPS)
    r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)
    return r_ij_norm, torch.stack((r_hat_ij, -r_hat_ij), dim=-3)


def _dihedral_value_and_gradient(
    coords: torch.Tensor, index: torch.Tensor, absolute: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return improper dihedrals and Boltz-style analytic gradients."""
    r_ij = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    r_kj = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
    r_kl = coords.index_select(-2, index[2]) - coords.index_select(-2, index[3])

    n_ijk = torch.cross(r_ij, r_kj, dim=-1)
    n_jkl = torch.cross(r_kj, r_kl, dim=-1)

    r_kj_norm = torch.linalg.norm(r_kj, dim=-1).clamp_min(defaults.GEOMETRY_EPS)
    n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1).clamp_min(defaults.GEOMETRY_EPS)
    n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1).clamp_min(defaults.GEOMETRY_EPS)

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


def _posebusters_gradient(
    coords: torch.Tensor,
    features: dict[str, torch.Tensor],
    buffered_bounds: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Analytic RDKit bounds gradient for bond, angle, and clash distances."""
    if buffered_bounds is None:
        buffered_bounds = _buffered_rdkit_bounds(features)
    index, lower, upper = buffered_bounds
    if index.shape[-1] == 0:
        return None

    value, value_gradient = _distance_value_and_gradient(coords, index)
    energy_derivative = defaults.POSEBUSTERS_WEIGHT * _flat_bottom_derivative(
        value, lower, upper
    )
    return _scatter_variable_gradient(coords, index, value_gradient, energy_derivative)


def _chiral_atom_gradient(
    coords: torch.Tensor, features: dict[str, torch.Tensor]
) -> torch.Tensor | None:
    """Analytic improper-dihedral gradient for tetrahedral stereocenters."""
    index = features["chiral_atom_index"].long()
    if index.shape[-1] == 0:
        return None

    orientations = features["chiral_atom_orientations"].bool()
    lower = torch.zeros_like(orientations, dtype=coords.dtype)
    upper = torch.zeros_like(orientations, dtype=coords.dtype)
    lower[orientations] = defaults.CHIRAL_BUFFER
    upper[orientations] = float("inf")
    lower[~orientations] = float("-inf")
    upper[~orientations] = -defaults.CHIRAL_BUFFER

    value, value_gradient = _dihedral_value_and_gradient(coords, index)
    energy_derivative = defaults.CHIRAL_ATOM_WEIGHT * _flat_bottom_derivative(
        value, lower, upper
    )
    return _scatter_variable_gradient(coords, index, value_gradient, energy_derivative)


def _stereo_bond_gradient(
    coords: torch.Tensor, features: dict[str, torch.Tensor]
) -> torch.Tensor | None:
    """Analytic improper-dihedral gradient for assigned E/Z double bonds."""
    index = features["stereo_bond_index"].long()
    if index.shape[-1] == 0:
        return None

    orientations = features["stereo_bond_orientations"].bool()
    lower = torch.zeros_like(orientations, dtype=coords.dtype)
    upper = torch.zeros_like(orientations, dtype=coords.dtype)
    lower[orientations] = torch.pi - defaults.STEREO_BOND_BUFFER
    upper[orientations] = float("inf")
    lower[~orientations] = float("-inf")
    upper[~orientations] = defaults.STEREO_BOND_BUFFER

    value, value_gradient = _dihedral_value_and_gradient(coords, index, absolute=True)
    energy_derivative = defaults.STEREO_BOND_WEIGHT * _flat_bottom_derivative(
        value, lower, upper
    )
    return _scatter_variable_gradient(coords, index, value_gradient, energy_derivative)


def _planar_bond_gradient(
    coords: torch.Tensor, features: dict[str, torch.Tensor]
) -> torch.Tensor | None:
    """Analytic improper-dihedral gradient for double-bond planarity."""
    index = features["planar_bond_index"].long()
    if index.shape[-1] == 0:
        return None

    double_bond_index = index.T
    first_improper = double_bond_index[:, [1, 2, 3, 0]]
    second_improper = double_bond_index[:, [4, 5, 0, 3]]
    improper_index = torch.cat([first_improper, second_improper], dim=0).T
    upper = torch.full(
        (improper_index.shape[-1],),
        defaults.PLANAR_BOND_BUFFER,
        dtype=coords.dtype,
        device=coords.device,
    )

    value, value_gradient = _dihedral_value_and_gradient(
        coords, improper_index, absolute=True
    )
    energy_derivative = defaults.PLANAR_BOND_WEIGHT * _flat_bottom_derivative(
        value, None, upper
    )
    return _scatter_variable_gradient(
        coords, improper_index, value_gradient, energy_derivative
    )


def ligand_stereochemistry_gradient(
    coords: torch.Tensor,
    features: dict[str, torch.Tensor],
    posebusters_bounds: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Return the analytic ligand stereochemistry guidance gradient."""
    gradients = [
        _posebusters_gradient(coords, features, posebusters_bounds),
        _chiral_atom_gradient(coords, features),
        _stereo_bond_gradient(coords, features),
        _planar_bond_gradient(coords, features),
    ]
    gradients = [gradient for gradient in gradients if gradient is not None]
    if not gradients:
        return None
    return sum(gradients)


def _finite_guidance_update(
    coords: torch.Tensor, gradient: torch.Tensor, previous: torch.Tensor
) -> torch.Tensor:
    """Apply the raw analytic update while retaining finite per-particle states."""
    candidate = coords - gradient
    finite = torch.isfinite(candidate).all(dim=(-1, -2), keepdim=True)
    return torch.where(finite, candidate, previous)


def apply_ligand_stereochemistry_guidance(
    xl_denoised: torch.Tensor,
    guidance: PreparedLigandStereochemistryGuidance | None,
    step_fraction: float,
) -> torch.Tensor:
    """Apply opt-in ligand stereochemistry guidance to the denoised x0 estimate.

    Args:
        xl_denoised:
            Denoised atom coordinates with shape [B, S, N_atom, 3].
        guidance:
            Prepared guidance inputs, or ``None`` when guidance is disabled.
        step_fraction:
            Fraction of the reverse diffusion trajectory already completed.

    Returns:
        Guided coordinates with the same shape and dtype as ``xl_denoised``.
    """
    if guidance is None:
        return xl_denoised
    if step_fraction < guidance.start_fraction:
        return xl_denoised

    guided = xl_denoised.float()
    for _ in range(guidance.num_gd_steps):
        gradient = ligand_stereochemistry_gradient(
            guided,
            guidance.features,
            guidance.posebusters_bounds,
        )
        if gradient is None:
            return guided.to(dtype=xl_denoised.dtype)
        guided = _finite_guidance_update(guided, gradient, guided)

    return guided.to(dtype=xl_denoised.dtype)
