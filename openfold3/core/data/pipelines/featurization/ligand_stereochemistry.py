"""Ligand stereochemistry guidance feature extraction.

The guidance features are derived from the input ligand chemistry and address only
local ligand geometry: distance-geometry bounds, assigned tetrahedral chirality,
assigned E/Z alkene stereochemistry, and double-bond planarity. They are independent
of binding-site constraints.

Constraint extraction is adapted from Boltz. See ``THIRD_PARTY_NOTICES.md``.
"""

from dataclasses import dataclass

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
from openfold3.core.data.primitives.structure.conformer import get_name_match_argsort
from openfold3.core.data.primitives.structure.labels import uniquify_ids
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import Query


@dataclass(frozen=True)
class _LigandGeometryConstraints:
    bounds_index: list[tuple[int, int]]
    lower_bounds: list[float]
    upper_bounds: list[float]
    bond_mask: list[bool]
    angle_mask: list[bool]
    pair_vdw_cutoffs: list[float]
    chiral_index: list[tuple[int, int, int, int]]
    chiral_orientations: list[bool]
    stereo_bond_index: list[tuple[int, int, int, int]]
    stereo_bond_orientations: list[bool]
    planar_bond_index: list[tuple[int, int, int, int, int, int]]


def _empty_constraints() -> _LigandGeometryConstraints:
    return _LigandGeometryConstraints(
        bounds_index=[],
        lower_bounds=[],
        upper_bounds=[],
        bond_mask=[],
        angle_mask=[],
        pair_vdw_cutoffs=[],
        chiral_index=[],
        chiral_orientations=[],
        stereo_bond_index=[],
        stereo_bond_orientations=[],
        planar_bond_index=[],
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


def _resolve_ligand_reference_molecule(
    query: Query,
    processed_reference_molecules: list[ProcessedReferenceMolecule],
    ligand_chain_id: str,
) -> ProcessedReferenceMolecule | None:
    """Find a ligand reference molecule in query construction order."""
    ref_mol_idx = 0
    for chain in query.chains:
        for chain_id in chain.chain_ids:
            match chain.molecule_type:
                case MoleculeType.PROTEIN | MoleculeType.DNA | MoleculeType.RNA:
                    if chain.sequence is None:
                        raise ValueError(
                            f"Chain {chain_id} has no sequence but is required to "
                            "resolve ligand stereochemistry reference molecules"
                        )
                    ref_mol_idx += len(chain.sequence)
                case MoleculeType.LIGAND:
                    if ref_mol_idx >= len(processed_reference_molecules):
                        raise ValueError(
                            "Not enough processed reference molecules to resolve "
                            f"ligand chain {ligand_chain_id!r}"
                        )
                    if chain_id == ligand_chain_id:
                        return processed_reference_molecules[ref_mol_idx]
                    ref_mol_idx += 1
                case _:
                    raise ValueError(
                        f"Unsupported molecule type: {chain.molecule_type}"
                    )

    return None


def _ligand_atom_index_map(
    processed_reference_molecule: ProcessedReferenceMolecule,
    ligand_atom_array: AtomArray,
    global_atom_indices: np.ndarray,
) -> dict[int, int]:
    """Map reference-molecule atoms to the OF3 global atom axis by atom name."""
    mol = processed_reference_molecule.mol
    in_crop_mask = np.asarray(processed_reference_molecule.in_crop_mask, dtype=bool)
    if in_crop_mask.shape != (mol.GetNumAtoms(),):
        raise ValueError(
            "Ligand stereochemistry reference mask must match the reference molecule."
        )
    if len(ligand_atom_array) != len(global_atom_indices):
        raise ValueError(
            "Ligand stereochemistry atom indices must match the ligand atom array."
        )
    if not all(atom.HasProp("annot_atom_name") for atom in mol.GetAtoms()):
        raise ValueError(
            "Ligand stereochemistry reference atoms require annotated atom names."
        )

    reference_names = np.asarray(
        uniquify_ids([atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()]),
        dtype=object,
    )
    ligand_names = np.asarray(
        uniquify_ids(ligand_atom_array.atom_name.tolist()), dtype=object
    )
    cropped_reference_names = reference_names[in_crop_mask]
    if len(cropped_reference_names) != len(ligand_names):
        raise ValueError(
            "Ligand stereochemistry reference atom names do not match the OF3 "
            "ligand atom names."
        )

    reference_indices = np.flatnonzero(in_crop_mask)
    reorder_index = get_name_match_argsort(cropped_reference_names, ligand_names)
    reference_indices = reference_indices[reorder_index]
    cropped_reference_names = cropped_reference_names[reorder_index]
    if not np.array_equal(cropped_reference_names, ligand_names):
        raise ValueError(
            "Ligand stereochemistry reference atom names do not match the OF3 "
            "ligand atom names."
        )

    idx_map = {
        int(reference_idx): int(global_index)
        for reference_idx, global_index in zip(
            reference_indices, global_atom_indices, strict=True
        )
    }

    reference_elements = np.asarray(
        [
            mol.GetAtomWithIdx(int(index)).GetSymbol().upper()
            for index in reference_indices
        ]
    )
    ligand_elements = np.asarray(
        [str(element).upper() for element in ligand_atom_array.element]
    )
    if not np.array_equal(reference_elements, ligand_elements):
        raise ValueError(
            "Ligand stereochemistry reference elements do not match the OF3 ligand "
            "atom elements."
        )

    return idx_map


def _compute_ligand_constraints(
    mol: Mol, idx_map: dict[int, int]
) -> _LigandGeometryConstraints:
    """Build all supported stereochemistry guidance constraints for one ligand."""
    if not idx_map:
        return _empty_constraints()

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
    return {
        "rdkit_bounds_index": torch.empty((2, 0), dtype=torch.long),
        "rdkit_lower_bounds": torch.empty((0,), dtype=torch.float32),
        "rdkit_upper_bounds": torch.empty((0,), dtype=torch.float32),
        "rdkit_bounds_bond_mask": torch.empty((0,), dtype=torch.bool),
        "rdkit_bounds_angle_mask": torch.empty((0,), dtype=torch.bool),
        "rdkit_bounds_pair_vdw_cutoff": torch.empty((0,), dtype=torch.float32),
        "chiral_atom_index": torch.empty((4, 0), dtype=torch.long),
        "chiral_atom_orientations": torch.empty((0,), dtype=torch.bool),
        "stereo_bond_index": torch.empty((4, 0), dtype=torch.long),
        "stereo_bond_orientations": torch.empty((0,), dtype=torch.bool),
        "planar_bond_index": torch.empty((6, 0), dtype=torch.long),
    }


def _append_constraints(
    accumulator: _LigandGeometryConstraints, constraints: _LigandGeometryConstraints
) -> _LigandGeometryConstraints:
    return _LigandGeometryConstraints(
        bounds_index=accumulator.bounds_index + constraints.bounds_index,
        lower_bounds=accumulator.lower_bounds + constraints.lower_bounds,
        upper_bounds=accumulator.upper_bounds + constraints.upper_bounds,
        bond_mask=accumulator.bond_mask + constraints.bond_mask,
        angle_mask=accumulator.angle_mask + constraints.angle_mask,
        pair_vdw_cutoffs=accumulator.pair_vdw_cutoffs + constraints.pair_vdw_cutoffs,
        chiral_index=accumulator.chiral_index + constraints.chiral_index,
        chiral_orientations=accumulator.chiral_orientations
        + constraints.chiral_orientations,
        stereo_bond_index=accumulator.stereo_bond_index + constraints.stereo_bond_index,
        stereo_bond_orientations=accumulator.stereo_bond_orientations
        + constraints.stereo_bond_orientations,
        planar_bond_index=accumulator.planar_bond_index + constraints.planar_bond_index,
    )


def _tensorize_constraints(
    constraints: _LigandGeometryConstraints,
) -> dict[str, torch.Tensor]:
    features = _empty_feature_tensors()

    if constraints.bounds_index:
        features["rdkit_bounds_index"] = torch.tensor(
            constraints.bounds_index, dtype=torch.long
        ).T
        features["rdkit_lower_bounds"] = torch.tensor(
            constraints.lower_bounds, dtype=torch.float32
        )
        features["rdkit_upper_bounds"] = torch.tensor(
            constraints.upper_bounds, dtype=torch.float32
        )
        features["rdkit_bounds_bond_mask"] = torch.tensor(
            constraints.bond_mask, dtype=torch.bool
        )
        features["rdkit_bounds_angle_mask"] = torch.tensor(
            constraints.angle_mask, dtype=torch.bool
        )
        features["rdkit_bounds_pair_vdw_cutoff"] = torch.tensor(
            constraints.pair_vdw_cutoffs, dtype=torch.float32
        )

    if constraints.chiral_index:
        features["chiral_atom_index"] = torch.tensor(
            constraints.chiral_index, dtype=torch.long
        ).T
        features["chiral_atom_orientations"] = torch.tensor(
            constraints.chiral_orientations, dtype=torch.bool
        )

    if constraints.stereo_bond_index:
        features["stereo_bond_index"] = torch.tensor(
            constraints.stereo_bond_index, dtype=torch.long
        ).T
        features["stereo_bond_orientations"] = torch.tensor(
            constraints.stereo_bond_orientations, dtype=torch.bool
        )

    if constraints.planar_bond_index:
        features["planar_bond_index"] = torch.tensor(
            constraints.planar_bond_index, dtype=torch.long
        ).T

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

    all_constraints = _empty_constraints()
    for chain in query.chains:
        for chain_id in chain.chain_ids:
            if chain.molecule_type != MoleculeType.LIGAND:
                continue

            ligand_mask = (atom_array.chain_id == chain_id) & (
                atom_array.molecule_type_id == MoleculeType.LIGAND
            )
            if not ligand_mask.any():
                raise ValueError(
                    "Ligand stereochemistry guidance could not find ligand chain "
                    f"{chain_id!r} in the OF3 atom array."
                )
            processed_mol = _resolve_ligand_reference_molecule(
                query=query,
                processed_reference_molecules=processed_reference_molecules,
                ligand_chain_id=chain_id,
            )
            if processed_mol is None:
                raise ValueError(
                    "Ligand stereochemistry guidance could not find a processed "
                    f"reference molecule for ligand chain {chain_id!r}."
                )
            global_indices = np.flatnonzero(ligand_mask)
            idx_map = _ligand_atom_index_map(
                processed_reference_molecule=processed_mol,
                ligand_atom_array=atom_array[ligand_mask],
                global_atom_indices=global_indices,
            )
            constraints = _compute_ligand_constraints(processed_mol.mol, idx_map)
            all_constraints = _append_constraints(all_constraints, constraints)

    features.update(_tensorize_constraints(all_constraints))
    return features
