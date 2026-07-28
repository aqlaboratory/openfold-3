"""Ligand stereochemistry guidance feature extraction.

The guidance features are derived from the input ligand chemistry and address only
local ligand geometry: distance-geometry bounds, assigned tetrahedral chirality,
assigned E/Z alkene stereochemistry, and double-bond planarity. They are independent
of binding-site constraints.

Constraint extraction is adapted from Boltz. See ``THIRD_PARTY_NOTICES.md``.
"""

from dataclasses import dataclass, field, fields

import numpy as np
import torch
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, HybridizationType, Mol
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

from openfold3.core.config import ligand_stereochemistry_defaults as defaults
from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.structure.component import PERIODIC_TABLE
from openfold3.core.data.primitives.structure.labels import residue_view_iter
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import Query


@dataclass
class _LigandGeometryConstraints:
    """Mutable accumulator for constraints from one or more ligand chains."""

    bounds_index: list[tuple[int, int]] = field(default_factory=list)
    lower_bounds: list[float] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)
    bond_mask: list[bool] = field(default_factory=list)
    angle_mask: list[bool] = field(default_factory=list)
    pair_vdw_cutoffs: list[float] = field(default_factory=list)
    chiral_index: list[tuple[int, int, int, int]] = field(default_factory=list)
    chiral_orientations: list[bool] = field(default_factory=list)
    stereo_bond_index: list[tuple[int, int, int, int]] = field(default_factory=list)
    stereo_bond_orientations: list[bool] = field(default_factory=list)
    planar_bond_index: list[tuple[int, int, int, int, int, int]] = field(
        default_factory=list
    )

    def extend(self, other: "_LigandGeometryConstraints") -> None:
        """Append constraints while preserving their shared atom-axis mapping."""
        for constraint_field in fields(self):
            getattr(self, constraint_field.name).extend(
                getattr(other, constraint_field.name)
            )


def _compute_geometry_constraints(
    mol: Mol, idx_map: dict[int, int]
) -> tuple[
    list[tuple[int, int]],
    list[float],
    list[float],
    list[bool],
    list[bool],
    list[float],
]:
    """Compute RDKit distance-geometry bounds in the model atom axis."""
    if mol.GetNumAtoms() <= 1:
        return [], [], [], [], [], []

    mapped_atomic_numbers = {
        atom_idx: mol.GetAtomWithIdx(atom_idx).GetAtomicNum() for atom_idx in idx_map
    }
    if any(
        atomic_number < 1 or atomic_number > PERIODIC_TABLE.GetMaxAtomicNumber()
        for atomic_number in mapped_atomic_numbers.values()
    ):
        raise ValueError(
            "Ligand stereochemistry guidance requires atoms with supported atomic "
            "numbers."
        )

    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    bounds = GetMoleculeBoundsMatrix(
        mol,
        set15bounds=True,
        scaleVDW=True,
        doTriangleSmoothing=True,
        useMacrocycle14config=False,
    )
    bonds = {
        tuple(sorted(match))
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*"))
    }
    angles = {
        tuple(sorted((match[0], match[2])))
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*"))
    }

    bounds_index = []
    lower_bounds = []
    upper_bounds = []
    bond_mask = []
    angle_mask = []
    pair_vdw_cutoffs = []
    for i, j in zip(*np.triu_indices(mol.GetNumAtoms(), k=1), strict=True):
        if i not in idx_map or j not in idx_map:
            continue
        atomic_numbers = (mapped_atomic_numbers[i], mapped_atomic_numbers[j])
        atom_pair = tuple(sorted((i, j)))
        bounds_index.append((idx_map[i], idx_map[j]))
        upper_bounds.append(float(bounds[i, j]))
        lower_bounds.append(float(bounds[j, i]))
        bond_mask.append(atom_pair in bonds)
        angle_mask.append(atom_pair in angles)
        pair_radii = [PERIODIC_TABLE.GetRvdw(number) for number in atomic_numbers]
        pair_vdw_cutoffs.append(
            defaults.VDW_PAIR_CUTOFF_OFFSET + sum(pair_radii) / len(pair_radii)
        )

    return (
        bounds_index,
        lower_bounds,
        upper_bounds,
        bond_mask,
        angle_mask,
        pair_vdw_cutoffs,
    )


def _compute_chiral_atom_constraints(
    mol: Mol, idx_map: dict[int, int]
) -> tuple[list[tuple[int, int, int, int]], list[bool]]:
    """Compute improper-dihedral constraints for assigned tetrahedral stereocenters."""
    constraints = []
    orientations = []
    if not all(atom.HasProp("_CIPRank") for atom in mol.GetAtoms()):
        return constraints, orientations

    for center_idx, orientation in Chem.FindMolChiralCenters(
        mol, includeUnassigned=False
    ):
        center = mol.GetAtomWithIdx(center_idx)
        neighbors = [
            (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
            for neighbor in center.GetNeighbors()
        ]
        neighbors = sorted(neighbors, key=lambda neighbor: neighbor[1], reverse=True)
        neighbors = tuple(neighbor[0] for neighbor in neighbors)

        if len(neighbors) > 4 or center.GetHybridization() != HybridizationType.SP3:
            continue

        atom_idxs = (*neighbors[:3], center_idx)
        if all(i in idx_map for i in atom_idxs):
            constraints.append(tuple(idx_map[i] for i in atom_idxs))
            orientations.append(orientation == "R")

        if len(neighbors) == 4:
            for skip_idx in range(3):
                chiral_set = neighbors[:skip_idx] + neighbors[skip_idx + 1 :]
                if skip_idx % 2 == 0:
                    atom_idxs = chiral_set[::-1] + (center_idx,)
                else:
                    atom_idxs = chiral_set + (center_idx,)
                if all(i in idx_map for i in atom_idxs):
                    constraints.append(tuple(idx_map[i] for i in atom_idxs))
                    orientations.append(orientation == "R")

    return constraints, orientations


def _compute_stereo_bond_constraints(
    mol: Mol, idx_map: dict[int, int]
) -> tuple[list[tuple[int, int, int, int]], list[bool]]:
    """Compute improper-dihedral constraints for assigned E/Z double bonds."""
    constraints = []
    orientations = []
    if not all(atom.HasProp("_CIPRank") for atom in mol.GetAtoms()):
        return constraints, orientations

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in {BondStereo.STEREOE, BondStereo.STEREOZ}:
            continue

        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        start_neighbors = [
            (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
            for neighbor in mol.GetAtomWithIdx(start_atom_idx).GetNeighbors()
            if neighbor.GetIdx() != end_atom_idx
        ]
        start_neighbors = sorted(
            start_neighbors, key=lambda neighbor: neighbor[1], reverse=True
        )
        start_neighbors = [neighbor[0] for neighbor in start_neighbors]
        end_neighbors = [
            (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
            for neighbor in mol.GetAtomWithIdx(end_atom_idx).GetNeighbors()
            if neighbor.GetIdx() != start_atom_idx
        ]
        end_neighbors = sorted(
            end_neighbors, key=lambda neighbor: neighbor[1], reverse=True
        )
        end_neighbors = [neighbor[0] for neighbor in end_neighbors]
        is_e = stereo == BondStereo.STEREOE

        atom_idxs = (
            start_neighbors[0],
            start_atom_idx,
            end_atom_idx,
            end_neighbors[0],
        )
        if all(i in idx_map for i in atom_idxs):
            constraints.append(tuple(idx_map[i] for i in atom_idxs))
            orientations.append(is_e)

        if len(start_neighbors) == 2 and len(end_neighbors) == 2:
            atom_idxs = (
                start_neighbors[1],
                start_atom_idx,
                end_atom_idx,
                end_neighbors[1],
            )
            if all(i in idx_map for i in atom_idxs):
                constraints.append(tuple(idx_map[i] for i in atom_idxs))
                orientations.append(is_e)

    return constraints, orientations


def _compute_flatness_constraints(
    mol: Mol, idx_map: dict[int, int]
) -> list[tuple[int, int, int, int, int, int]]:
    """Compute Boltz-style double-bond planarity constraints."""
    planar_double_bond_smarts = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    constraints = []
    for match in mol.GetSubstructMatches(planar_double_bond_smarts):
        if all(i in idx_map for i in match):
            constraints.append(tuple(idx_map[i] for i in match))
    return constraints


def _compute_ligand_constraints(
    mol: Mol, idx_map: dict[int, int]
) -> _LigandGeometryConstraints:
    """Build all supported stereochemistry guidance constraints for one ligand."""
    if not idx_map:
        return _LigandGeometryConstraints()

    mol = Chem.Mol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    (
        bounds_index,
        lower_bounds,
        upper_bounds,
        bond_mask,
        angle_mask,
        pair_vdw_cutoffs,
    ) = _compute_geometry_constraints(mol, idx_map)
    chiral_index, chiral_orientations = _compute_chiral_atom_constraints(mol, idx_map)
    stereo_bond_index, stereo_bond_orientations = _compute_stereo_bond_constraints(
        mol, idx_map
    )
    planar_bond_index = _compute_flatness_constraints(mol, idx_map)

    return _LigandGeometryConstraints(
        bounds_index=bounds_index,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bond_mask=bond_mask,
        angle_mask=angle_mask,
        pair_vdw_cutoffs=pair_vdw_cutoffs,
        chiral_index=chiral_index,
        chiral_orientations=chiral_orientations,
        stereo_bond_index=stereo_bond_index,
        stereo_bond_orientations=stereo_bond_orientations,
        planar_bond_index=planar_bond_index,
    )


def _empty_feature_tensors() -> dict[str, torch.Tensor]:
    features = {}
    for name, arity in (
        ("distance", 2),
        ("signed_dihedral", 4),
        ("stereo_dihedral", 4),
        ("planar_dihedral", 4),
    ):
        prefix = f"ligand_stereochemistry_{name}"
        features[f"{prefix}_index"] = torch.empty((arity, 0), dtype=torch.long)
        for suffix in ("lower", "upper"):
            features[f"{prefix}_{suffix}"] = torch.empty((0,), dtype=torch.float32)
    return features


def _restraint_features(
    name: str,
    index: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Package one prepared flat-bottom restraint family as batch features."""
    prefix = f"ligand_stereochemistry_{name}"
    return {
        f"{prefix}_index": index,
        f"{prefix}_lower": lower,
        f"{prefix}_upper": upper,
    }


def _tensorize_constraints(
    constraints: _LigandGeometryConstraints,
) -> dict[str, torch.Tensor]:
    features = _empty_feature_tensors()

    if constraints.bounds_index:
        index = torch.tensor(constraints.bounds_index, dtype=torch.long).T
        lower = torch.tensor(constraints.lower_bounds, dtype=torch.float32)
        upper = torch.tensor(constraints.upper_bounds, dtype=torch.float32)
        bond = torch.tensor(constraints.bond_mask, dtype=torch.bool)
        angle = torch.tensor(constraints.angle_mask, dtype=torch.bool)
        pair_vdw_cutoff = torch.tensor(
            constraints.pair_vdw_cutoffs, dtype=torch.float32
        )

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
        features.update(_restraint_features("distance", index, lower, upper))

    chiral_index = (
        torch.tensor(constraints.chiral_index, dtype=torch.long).reshape(-1, 4).T
    )
    chiral_orientations = torch.tensor(
        constraints.chiral_orientations, dtype=torch.bool
    )
    chiral_lower = torch.full(
        chiral_orientations.shape, float("-inf"), dtype=torch.float32
    )
    chiral_upper = torch.full_like(chiral_lower, float("inf"))
    chiral_lower[chiral_orientations] = defaults.CHIRAL_BUFFER
    chiral_upper[~chiral_orientations] = -defaults.CHIRAL_BUFFER
    features.update(
        _restraint_features(
            "signed_dihedral",
            chiral_index,
            chiral_lower,
            chiral_upper,
        )
    )

    stereo_index = (
        torch.tensor(constraints.stereo_bond_index, dtype=torch.long).reshape(-1, 4).T
    )
    stereo_orientations = torch.tensor(
        constraints.stereo_bond_orientations, dtype=torch.bool
    )
    stereo_lower = torch.full(
        stereo_orientations.shape, float("-inf"), dtype=torch.float32
    )
    stereo_upper = torch.full_like(stereo_lower, float("inf"))
    stereo_lower[stereo_orientations] = torch.pi - defaults.STEREO_BOND_BUFFER
    stereo_upper[~stereo_orientations] = defaults.STEREO_BOND_BUFFER
    features.update(
        _restraint_features(
            "stereo_dihedral",
            stereo_index,
            stereo_lower,
            stereo_upper,
        )
    )

    planar_index = torch.tensor(
        constraints.planar_bond_index, dtype=torch.long
    ).reshape(-1, 6)
    planar_improper_index = torch.cat(
        (planar_index[:, [1, 2, 3, 0]], planar_index[:, [4, 5, 0, 3]]), dim=0
    ).T
    planar_lower = torch.full(
        (planar_improper_index.shape[1],), float("-inf"), dtype=torch.float32
    )
    planar_upper = torch.full_like(planar_lower, defaults.PLANAR_BOND_BUFFER)
    features.update(
        _restraint_features(
            "planar_dihedral",
            planar_improper_index,
            planar_lower,
            planar_upper,
        )
    )

    return features


def featurize_ligand_stereochemistry_guidance(
    query: Query,
    atom_array: AtomArray,
    processed_reference_molecules: list[ProcessedReferenceMolecule],
) -> dict[str, torch.Tensor]:
    """Create inference-time ligand stereochemistry guidance features.

    The query flag controls whether the sampler uses the emitted constraints. Empty
    tensors are returned when guidance is disabled so the downstream hook remains
    shape-stable and no-ops safely.
    """
    features = _empty_feature_tensors()
    features["ligand_stereochemistry_guidance_enabled"] = torch.tensor(
        [query.ligand_stereochemistry_guidance], dtype=torch.bool
    )
    features["ligand_stereochemistry_start_fraction"] = torch.tensor(
        [query.ligand_stereochemistry_start_fraction], dtype=torch.float32
    )
    features["ligand_stereochemistry_num_gd_steps"] = torch.tensor(
        [query.ligand_stereochemistry_num_gd_steps], dtype=torch.long
    )

    if not query.ligand_stereochemistry_guidance:
        return features

    all_constraints = _LigandGeometryConstraints()
    global_atom_indices = np.arange(len(atom_array))
    for residue, processed_mol in zip(
        residue_view_iter(atom_array), processed_reference_molecules, strict=True
    ):
        if not np.all(residue.molecule_type_id == MoleculeType.LIGAND):
            continue

        reference_indices = np.flatnonzero(processed_mol.in_crop_mask)
        residue_indices = global_atom_indices[residue.indices]
        if len(reference_indices) != len(residue_indices):
            raise ValueError(
                "Processed ligand reference atoms must match the OF3 residue."
            )
        idx_map = dict(
            zip(reference_indices.tolist(), residue_indices.tolist(), strict=True)
        )
        constraints = _compute_ligand_constraints(processed_mol.mol, idx_map)
        all_constraints.extend(constraints)

    features.update(_tensorize_constraints(all_constraints))
    return features
