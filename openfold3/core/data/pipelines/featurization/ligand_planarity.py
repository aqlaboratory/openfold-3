"""Assigned C=C planarity restraints for inference."""

import itertools
import math

import numpy as np
import torch
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import BondStereo, BondType, HybridizationType, Mol

from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.structure.labels import residue_view_iter
from openfold3.core.data.resources.residues import MoleculeType


def _assigned_cc_planarity_restraints(
    mol: Mol, idx_map: dict[int, int]
) -> tuple[list[tuple[int, int, int, int]], list[bool]]:
    """Map every heavy-atom dihedral around assigned C=C bonds."""
    mol = Chem.Mol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    if mol.GetNumConformers() == 0:
        raise ValueError("Assigned C=C planarity restraints require a conformer.")

    conformer = mol.GetConformer()
    atom_indices: list[tuple[int, int, int, int]] = []
    trans_orientations: list[bool] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != BondType.DOUBLE or bond.GetIsAromatic():
            continue
        if bond.GetStereo() not in {BondStereo.STEREOE, BondStereo.STEREOZ}:
            continue

        start = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if (
            start.GetAtomicNum() != 6
            or end.GetAtomicNum() != 6
            or start.GetHybridization() != HybridizationType.SP2
            or end.GetHybridization() != HybridizationType.SP2
        ):
            continue

        start_neighbors = [
            atom.GetIdx()
            for atom in start.GetNeighbors()
            if atom.GetIdx() != end.GetIdx() and atom.GetAtomicNum() > 1
        ]
        end_neighbors = [
            atom.GetIdx()
            for atom in end.GetNeighbors()
            if atom.GetIdx() != start.GetIdx() and atom.GetAtomicNum() > 1
        ]
        for start_neighbor, end_neighbor in itertools.product(
            start_neighbors, end_neighbors
        ):
            quartet = (start_neighbor, start.GetIdx(), end.GetIdx(), end_neighbor)
            if not all(index in idx_map for index in quartet):
                continue
            reference_angle = abs(rdMolTransforms.GetDihedralRad(conformer, *quartet))
            mapped = tuple(idx_map[index] for index in quartet)
            atom_indices.append((mapped[0], mapped[1], mapped[2], mapped[3]))
            trans_orientations.append(reference_angle > math.pi / 2)

    return atom_indices, trans_orientations


def featurize_ligand_planarity(
    atom_array: AtomArray,
    processed_reference_molecules: list[ProcessedReferenceMolecule],
) -> dict[str, torch.Tensor]:
    """Create planarity restraints in the model atom axis."""
    atom_indices = []
    trans_orientations = []
    global_atom_indices = np.arange(len(atom_array))
    for residue, processed_mol in zip(
        residue_view_iter(atom_array), processed_reference_molecules, strict=True
    ):
        if not np.all(residue.molecule_type_id == MoleculeType.LIGAND):
            continue

        reference_indices = np.flatnonzero(processed_mol.in_crop_mask)
        residue_indices = global_atom_indices[residue.indices]
        if len(reference_indices) != len(residue_indices):
            raise ValueError("Processed ligand atoms must match the OF3 residue.")
        idx_map = dict(
            zip(reference_indices.tolist(), residue_indices.tolist(), strict=True)
        )
        residue_restraints, residue_orientations = _assigned_cc_planarity_restraints(
            processed_mol.mol, idx_map
        )
        atom_indices.extend(residue_restraints)
        trans_orientations.extend(residue_orientations)

    return {
        "ligand_planarity_index": torch.tensor(atom_indices, dtype=torch.long)
        .reshape(-1, 4)
        .T,
        "ligand_planarity_trans": torch.tensor(trans_orientations, dtype=torch.bool),
    }
