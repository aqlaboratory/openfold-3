"""Ligand chemical steering feature extraction.

The guidance features are derived from the input ligand chemistry and include
distance-geometry bounds, assigned tetrahedral chirality, assigned E/Z alkene
stereochemistry, double-bond planarity, and interchain VDW overlap.
"""

from dataclasses import dataclass, field, fields
from itertools import combinations

import numpy as np
import torch
from biotite.structure import AtomArray, BondType
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import BondStereo, HybridizationType, Mol
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

from openfold3.core.config.ligand_chemical_steering_config import (
    LigandChemicalSteeringSettings,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.structure.component import (
    PERIODIC_TABLE,
    find_cross_chain_bonds,
)
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
    vdw_overlap_index: list[tuple[int, int]] = field(default_factory=list)
    vdw_overlap_lower_bounds: list[float] = field(default_factory=list)
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
    settings: LigandChemicalSteeringSettings,
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
            "Ligand chemical steering requires atoms with supported atomic numbers."
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


def _compute_vdw_overlap_constraints(
    atom_array: AtomArray,
    settings: LigandChemicalSteeringSettings,
) -> _LigandGeometryConstraints:
    """Compute Boltz-style interchain VDW bounds for ligand-containing pairs.

    Args:
        atom_array:
            OF3 structure whose atom and chain axes define the model features.
        settings:
            Validated settings controlling the VDW overlap buffer.

    Raises:
        ValueError:
            If a participating atom has an unsupported element.

    Returns:
        Pair indices and lower distance bounds for non-covalently-connected chains.
    """
    constraints = _LigandGeometryConstraints()
    chain_ids, atom_chain_indices, chain_sizes = np.unique(
        atom_array.chain_id,
        return_inverse=True,
        return_counts=True,
    )
    ligand_atom_mask = (
        atom_array.get_annotation("molecule_type_id") == MoleculeType.LIGAND
    )
    ligand_chain_mask = np.asarray(
        [
            np.any(ligand_atom_mask[atom_chain_indices == i])
            for i in range(len(chain_ids))
        ]
    )

    connected_chain_pairs: set[tuple[int, int]] = set()
    for atom_i, atom_j, bond_type in find_cross_chain_bonds(atom_array):
        if BondType(int(bond_type)) == BondType.COORDINATION:
            continue
        connected_chain_pairs.add(
            tuple(
                sorted(
                    (atom_chain_indices[int(atom_i)], atom_chain_indices[int(atom_j)])
                )
            )
        )

    pair_indices: list[tuple[int, int]] = []
    for chain_i in range(len(chain_ids)):
        for chain_j in range(chain_i + 1, len(chain_ids)):
            if not (ligand_chain_mask[chain_i] or ligand_chain_mask[chain_j]):
                continue
            if chain_sizes[chain_i] == 1 or chain_sizes[chain_j] == 1:
                continue
            if (chain_i, chain_j) in connected_chain_pairs:
                continue

            atom_indices_i = np.flatnonzero(atom_chain_indices == chain_i)
            atom_indices_j = np.flatnonzero(atom_chain_indices == chain_j)
            pair_indices.extend(
                zip(
                    np.repeat(atom_indices_i, len(atom_indices_j)).tolist(),
                    np.tile(atom_indices_j, len(atom_indices_i)).tolist(),
                    strict=True,
                )
            )

    if not pair_indices:
        return constraints

    participating_atom_indices = sorted(
        {index for pair in pair_indices for index in pair}
    )
    vdw_radii: dict[int, float] = {}
    for atom_index in participating_atom_indices:
        element = str(atom_array.element[atom_index]).capitalize()
        atomic_number = PERIODIC_TABLE.GetAtomicNumber(element)
        if atomic_number < 1 or atomic_number > PERIODIC_TABLE.GetMaxAtomicNumber():
            raise ValueError(
                "Ligand chemical steering requires atoms with supported elements."
            )
        vdw_radii[atom_index] = PERIODIC_TABLE.GetRvdw(atomic_number)

    constraints.vdw_overlap_index.extend(pair_indices)
    constraints.vdw_overlap_lower_bounds.extend(
        (vdw_radii[i] + vdw_radii[j]) * (1.0 - settings.vdw_buffer)
        for i, j in pair_indices
    )
    return constraints


def _compute_chiral_atom_constraints(
    mol: Mol, idx_map: dict[int, int]
) -> _LigandGeometryConstraints:
    """Compute improper-dihedral constraints for assigned tetrahedral stereocenters.

    Args:
        mol:
            RDKit reference molecule with assigned tetrahedral stereochemistry and
            an OF3 reference conformer.
        idx_map:
            Mapping from reference-molecule atom indices to model atom indices.
    Raises:
        ValueError:
            If an assigned tetrahedral center has no reference conformer.
    Returns:
        Mapped tetrahedral atom indices and their assigned orientations.
    """
    constraints = _LigandGeometryConstraints()
    if not idx_map:
        return constraints
    assigned_tags = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    assigned_centers = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() in assigned_tags
        and atom.GetHybridization() == HybridizationType.SP3
        and atom.GetDegree() in {3, 4}
    ]
    if not assigned_centers:
        return constraints
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "Assigned ligand stereocenters require an OF3 reference conformer."
        )

    conformer = mol.GetConformer()
    for center in assigned_centers:
        center_idx = center.GetIdx()
        neighbor_indices = tuple(
            sorted(neighbor.GetIdx() for neighbor in center.GetNeighbors())
        )
        for first, second, third in combinations(neighbor_indices, 3):
            atom_idxs = (first, second, third, center_idx)
            if not all(atom_idx in idx_map for atom_idx in atom_idxs):
                continue

            reference_dihedral = rdMolTransforms.GetDihedralRad(
                conformer, first, second, third, center_idx
            )
            constraints.chiral_index.append(
                (
                    idx_map[first],
                    idx_map[second],
                    idx_map[third],
                    idx_map[center_idx],
                )
            )
            constraints.chiral_orientations.append(reference_dihedral > 0.0)

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
    settings: LigandChemicalSteeringSettings,
) -> _LigandGeometryConstraints:
    """Build all supported chemical steering constraints for one ligand.

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
        ("vdw_overlap", 2),
        ("signed_dihedral", 4),
        ("stereo_dihedral", 4),
        ("planar_dihedral", 4),
    ):
        prefix = f"ligand_chemical_steering_{name}"
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
    prefix = f"ligand_chemical_steering_{name}"
    return {
        f"{prefix}_index": index,
        f"{prefix}_lower": lower,
        f"{prefix}_upper": upper,
    }


def _tensorize_constraints(
    constraints: _LigandGeometryConstraints,
    settings: LigandChemicalSteeringSettings,
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

    vdw_overlap_index = (
        torch.tensor(constraints.vdw_overlap_index, dtype=torch.long).reshape(-1, 2).T
    )
    vdw_overlap_lower = torch.tensor(
        constraints.vdw_overlap_lower_bounds, dtype=torch.float32
    )
    vdw_overlap_upper = torch.full_like(vdw_overlap_lower, float("inf"))
    features.update(
        _restraint_features(
            "vdw_overlap",
            vdw_overlap_index,
            vdw_overlap_lower,
            vdw_overlap_upper,
        )
    )

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


def featurize_ligand_chemical_steering(
    query: Query,
    atom_array: AtomArray,
    processed_reference_molecules: list[ProcessedReferenceMolecule],
    settings: LigandChemicalSteeringSettings,
) -> dict[str, torch.Tensor]:
    """Create inference-time ligand chemical steering features.

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
    features["ligand_chemical_steering_enabled"] = torch.tensor(
        [query.ligand_chemical_steering], dtype=torch.bool
    )
    features["ligand_chemical_steering_start_fraction"] = torch.tensor(
        [settings.start_fraction], dtype=torch.float32
    )
    features["ligand_chemical_steering_num_gd_steps"] = torch.tensor(
        [settings.num_gd_steps], dtype=torch.long
    )
    features["ligand_chemical_steering_vdw_guidance_interval"] = torch.tensor(
        [settings.vdw_guidance_interval], dtype=torch.long
    )
    for name, weight in (
        ("distance", settings.distance_weight),
        ("vdw_overlap", settings.vdw_weight),
        ("signed_dihedral", settings.chiral_atom_weight),
        ("stereo_dihedral", settings.stereo_bond_weight),
        ("planar_dihedral", settings.planar_bond_weight),
    ):
        features[f"ligand_chemical_steering_{name}_weight"] = torch.tensor(
            [weight], dtype=torch.float32
        )

    if not query.ligand_chemical_steering:
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

    if settings.vdw_weight > 0.0:
        all_constraints.extend(_compute_vdw_overlap_constraints(atom_array, settings))
    features.update(_tensorize_constraints(all_constraints, settings))
    return features
