"""Ligand stereochemistry guidance feature extraction.

The guidance features are derived from the input ligand chemistry and address only
local ligand geometry: distance-geometry bounds, assigned tetrahedral chirality,
assigned E/Z alkene stereochemistry, and double-bond planarity.
"""

from dataclasses import dataclass, field, fields

import numpy as np
import torch
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, HybridizationType, Mol
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

from openfold3.core.config.ligand_stereochemistry_config import (
    LigandStereochemistryGuidanceSettings,
)
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
        """Append constraints while preserving their shared atom-axis mapping.

        Args:
            other:
                Constraints to append to this accumulator.
        """
        for constraint_field in fields(self):
            getattr(self, constraint_field.name).extend(
                getattr(other, constraint_field.name)
            )


def _compute_geometry_constraints(
    mol: Mol,
    idx_map: dict[int, int],
    settings: LigandStereochemistryGuidanceSettings,
) -> _LigandGeometryConstraints:
    """Compute RDKit distance-geometry bounds in the model atom axis.

    Args:
        mol:
            RDKit reference molecule containing the ligand bond graph.
        idx_map:
            Mapping from reference-molecule atom indices to model atom indices.
        settings:
            Validated settings controlling geometry buffers and VDW cutoffs.

    Raises:
        ValueError:
            If a mapped ligand atom has an unsupported atomic number.

    Returns:
        Distance bounds, pair classifications, and VDW cutoffs for mapped atoms.
    """
    constraints = _LigandGeometryConstraints()
    if mol.GetNumAtoms() <= 1:
        return constraints

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

    for i, j in zip(*np.triu_indices(mol.GetNumAtoms(), k=1), strict=True):
        i_idx, j_idx = int(i), int(j)
        if i_idx not in idx_map or j_idx not in idx_map:
            continue
        atomic_numbers = (
            mapped_atomic_numbers[i_idx],
            mapped_atomic_numbers[j_idx],
        )
        atom_pair = tuple(sorted((i_idx, j_idx)))
        constraints.bounds_index.append((idx_map[i_idx], idx_map[j_idx]))
        constraints.upper_bounds.append(float(bounds[i_idx, j_idx]))
        constraints.lower_bounds.append(float(bounds[j_idx, i_idx]))
        constraints.bond_mask.append(atom_pair in bonds)
        constraints.angle_mask.append(atom_pair in angles)
        pair_radii = [PERIODIC_TABLE.GetRvdw(number) for number in atomic_numbers]
        constraints.pair_vdw_cutoffs.append(
            settings.vdw_pair_cutoff_offset + sum(pair_radii) / len(pair_radii)
        )

    return constraints


def _compute_chiral_atom_constraints(
    mol: Mol, idx_map: dict[int, int]
) -> _LigandGeometryConstraints:
    """Compute improper-dihedral constraints for assigned tetrahedral stereocenters.

    Args:
        mol:
            RDKit reference molecule with assigned CIP stereochemistry.
        idx_map:
            Mapping from reference-molecule atom indices to model atom indices.
    Returns:
        Mapped tetrahedral atom indices and their assigned orientations.
    """
    constraints = _LigandGeometryConstraints()
    if not all(atom.HasProp("_CIPRank") for atom in mol.GetAtoms()):
        return constraints

    for center_idx, orientation in Chem.FindMolChiralCenters(
        mol, includeUnassigned=False
    ):
        center = mol.GetAtomWithIdx(center_idx)
        ranked_neighbors = [
            (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
            for neighbor in center.GetNeighbors()
        ]
        ranked_neighbors.sort(key=lambda neighbor: neighbor[1], reverse=True)
        neighbor_indices = tuple(neighbor[0] for neighbor in ranked_neighbors)

        if (
            len(neighbor_indices) not in {3, 4}
            or center.GetHybridization() != HybridizationType.SP3
        ):
            continue

        first, second, third = neighbor_indices[:3]
        atom_idxs = (first, second, third, center_idx)
        if all(i in idx_map for i in atom_idxs):
            constraints.chiral_index.append(
                (
                    idx_map[first],
                    idx_map[second],
                    idx_map[third],
                    idx_map[center_idx],
                )
            )
            constraints.chiral_orientations.append(orientation == "R")

        if len(neighbor_indices) == 4:
            for skip_idx in range(3):
                chiral_set = (
                    neighbor_indices[:skip_idx] + neighbor_indices[skip_idx + 1 :]
                )
                if skip_idx % 2 == 0:
                    chiral_set = chiral_set[::-1]
                first, second, third = chiral_set
                atom_idxs = (first, second, third, center_idx)
                if all(i in idx_map for i in atom_idxs):
                    constraints.chiral_index.append(
                        (
                            idx_map[first],
                            idx_map[second],
                            idx_map[third],
                            idx_map[center_idx],
                        )
                    )
                    constraints.chiral_orientations.append(orientation == "R")

    return constraints


def _compute_stereo_bond_constraints(
    mol: Mol, idx_map: dict[int, int]
) -> _LigandGeometryConstraints:
    """Compute improper-dihedral constraints for assigned E/Z double bonds.

    Args:
        mol:
            RDKit reference molecule with assigned bond stereochemistry.
        idx_map:
            Mapping from reference-molecule atom indices to model atom indices.
    Returns:
        Mapped double-bond atom indices and their assigned E/Z orientations.
    """
    constraints = _LigandGeometryConstraints()
    if not all(atom.HasProp("_CIPRank") for atom in mol.GetAtoms()):
        return constraints

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in {BondStereo.STEREOE, BondStereo.STEREOZ}:
            continue

        start_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        ranked_start_neighbors = [
            (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
            for neighbor in mol.GetAtomWithIdx(start_atom_idx).GetNeighbors()
            if neighbor.GetIdx() != end_atom_idx
        ]
        ranked_start_neighbors.sort(key=lambda neighbor: neighbor[1], reverse=True)
        start_neighbors = [neighbor[0] for neighbor in ranked_start_neighbors]
        ranked_end_neighbors = [
            (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
            for neighbor in mol.GetAtomWithIdx(end_atom_idx).GetNeighbors()
            if neighbor.GetIdx() != start_atom_idx
        ]
        ranked_end_neighbors.sort(key=lambda neighbor: neighbor[1], reverse=True)
        end_neighbors = [neighbor[0] for neighbor in ranked_end_neighbors]
        is_e = stereo == BondStereo.STEREOE

        start_substituent = start_neighbors[0]
        end_substituent = end_neighbors[0]
        atom_idxs = (
            start_substituent,
            start_atom_idx,
            end_atom_idx,
            end_substituent,
        )
        if all(i in idx_map for i in atom_idxs):
            constraints.stereo_bond_index.append(
                (
                    idx_map[start_substituent],
                    idx_map[start_atom_idx],
                    idx_map[end_atom_idx],
                    idx_map[end_substituent],
                )
            )
            constraints.stereo_bond_orientations.append(is_e)

        if len(start_neighbors) == 2 and len(end_neighbors) == 2:
            start_substituent = start_neighbors[1]
            end_substituent = end_neighbors[1]
            atom_idxs = (
                start_substituent,
                start_atom_idx,
                end_atom_idx,
                end_substituent,
            )
            if all(i in idx_map for i in atom_idxs):
                constraints.stereo_bond_index.append(
                    (
                        idx_map[start_substituent],
                        idx_map[start_atom_idx],
                        idx_map[end_atom_idx],
                        idx_map[end_substituent],
                    )
                )
                constraints.stereo_bond_orientations.append(is_e)

    return constraints


def _compute_flatness_constraints(
    mol: Mol, idx_map: dict[int, int]
) -> _LigandGeometryConstraints:
    """Compute double-bond planarity constraints.

    Args:
        mol:
            RDKit reference molecule containing the ligand bond graph.
        idx_map:
            Mapping from reference-molecule atom indices to model atom indices.
    Returns:
        Six-atom patterns defining mapped planar double bonds.
    """
    planar_double_bond_smarts = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    constraints = _LigandGeometryConstraints()
    for match in mol.GetSubstructMatches(planar_double_bond_smarts):
        first, second, third, fourth, fifth, sixth = match
        if all(i in idx_map for i in match):
            constraints.planar_bond_index.append(
                (
                    idx_map[first],
                    idx_map[second],
                    idx_map[third],
                    idx_map[fourth],
                    idx_map[fifth],
                    idx_map[sixth],
                )
            )
    return constraints


def _compute_ligand_constraints(
    mol: Mol,
    idx_map: dict[int, int],
    settings: LigandStereochemistryGuidanceSettings,
) -> _LigandGeometryConstraints:
    """Build all supported stereochemistry guidance constraints for one ligand.

    Args:
        mol:
            Processed RDKit reference molecule for one ligand residue.
        idx_map:
            Mapping from reference-molecule atom indices to model atom indices.
        settings:
            Validated settings controlling restraint bounds.
    Returns:
        Combined geometry, chirality, E/Z, and planarity constraints.
    """
    if not idx_map:
        return _LigandGeometryConstraints()

    mol = Chem.Mol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    constraints = _compute_geometry_constraints(mol, idx_map, settings)
    constraints.extend(_compute_chiral_atom_constraints(mol, idx_map))
    constraints.extend(_compute_stereo_bond_constraints(mol, idx_map))
    constraints.extend(_compute_flatness_constraints(mol, idx_map))
    return constraints


def _empty_feature_tensors() -> dict[str, torch.Tensor]:
    """Create shape-stable empty tensors for every restraint family.

    Returns:
        Empty index and bound tensors keyed by batch feature name.
    """
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
    """Package one prepared flat-bottom restraint family as batch features.

    Args:
        name:
            Restraint-family name used in emitted feature keys.
        index:
            Atom indices for all constraints in the family.
        lower:
            Lower bound for each constraint.
        upper:
            Upper bound for each constraint.
    Returns:
        Index and bound tensors keyed for the inference batch.
    """
    prefix = f"ligand_stereochemistry_{name}"
    return {
        f"{prefix}_index": index,
        f"{prefix}_lower": lower,
        f"{prefix}_upper": upper,
    }


def _tensorize_constraints(
    constraints: _LigandGeometryConstraints,
    settings: LigandStereochemistryGuidanceSettings,
) -> dict[str, torch.Tensor]:
    """Convert accumulated constraints into flat-bottom restraint tensors.

    Args:
        constraints:
            Ligand constraints expressed in the model atom axis.
        settings:
            Validated settings controlling restraint buffers.
    Returns:
        Batch feature tensors for distance and dihedral restraint families.
    """
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

        lower[bond & ~angle] *= 1.0 - settings.bond_buffer
        upper[bond & ~angle] *= 1.0 + settings.bond_buffer
        lower[~bond & angle] *= 1.0 - settings.angle_buffer
        upper[~bond & angle] *= 1.0 + settings.angle_buffer
        shared_buffer = min(settings.bond_buffer, settings.angle_buffer)
        lower[bond & angle] *= 1.0 - shared_buffer
        upper[bond & angle] *= 1.0 + shared_buffer
        lower[~bond & ~angle] *= 1.0 - settings.clash_buffer
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
    chiral_lower[chiral_orientations] = settings.chiral_buffer
    chiral_upper[~chiral_orientations] = -settings.chiral_buffer
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
    stereo_lower[stereo_orientations] = torch.pi - settings.stereo_bond_buffer
    stereo_upper[~stereo_orientations] = settings.stereo_bond_buffer
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
    planar_upper = torch.full_like(planar_lower, settings.planar_bond_buffer)
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
    settings: LigandStereochemistryGuidanceSettings,
) -> dict[str, torch.Tensor]:
    """Create inference-time ligand stereochemistry guidance features.

    The query flag controls whether the sampler uses the emitted constraints. Empty
    tensors are returned when guidance is disabled so the downstream hook remains
    shape-stable and no-ops safely.

    Args:
        query:
            Validated inference query containing the per-query guidance flag.
        atom_array:
            Preprocessed structure whose atom axis is used by model features.
        processed_reference_molecules:
            Reference molecules aligned with residues in ``atom_array``.
        settings:
            Validated settings shared by enabled queries in the inference run.

    Raises:
        ValueError:
            If reference molecules do not align one-to-one with the atom-array
            residues, or if a mapped ligand atom has an unsupported atomic number.

    Returns:
        Shape-stable restraint and sampler-setting tensors for the inference batch.
    """
    features = _empty_feature_tensors()
    features["ligand_stereochemistry_guidance_enabled"] = torch.tensor(
        [query.ligand_stereochemistry_guidance], dtype=torch.bool
    )
    features["ligand_stereochemistry_start_fraction"] = torch.tensor(
        [settings.start_fraction], dtype=torch.float32
    )
    features["ligand_stereochemistry_num_gd_steps"] = torch.tensor(
        [settings.num_gd_steps], dtype=torch.long
    )
    for name, weight in (
        ("distance", settings.distance_weight),
        ("signed_dihedral", settings.chiral_atom_weight),
        ("stereo_dihedral", settings.stereo_bond_weight),
        ("planar_dihedral", settings.planar_bond_weight),
    ):
        features[f"ligand_stereochemistry_{name}_weight"] = torch.tensor(
            [weight], dtype=torch.float32
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
        constraints = _compute_ligand_constraints(processed_mol.mol, idx_map, settings)
        all_constraints.extend(constraints)

    features.update(_tensorize_constraints(all_constraints, settings))
    return features
