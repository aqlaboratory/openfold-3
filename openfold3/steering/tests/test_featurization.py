# Copyright 2026 AlQuraishi Laboratory
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

"""Golden-value coverage of the buffer rules, single-atom and cropped-pair
handling, unsupported-atomic-number rejection, and misaligned/missing
reference-molecule errors.

The posebusters aromatic-geometry test is distance-only by construction:
benzene's aromatic ring bonds are RDKit BondType.AROMATIC rather than
DOUBLE, so no planar-dihedral SMARTS would ever match them even if this
package implemented that term (it does not yet).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from rdkit import Chem

from openfold3.core.data.primitives.structure.labels import residue_view_iter
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import (
    add_token_positions,
    tokenize_atom_array,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import Query
from openfold3.steering import defaults
from openfold3.steering.featurization import (
    MissingReferenceMoleculeError,
    _compute_distance_constraints,
    _DistanceConstraints,
    _tensorize_distance_constraints,
    build_context,
)
from openfold3.steering.potentials import DistanceBoundsPotential
from openfold3.steering.tests._structures import (
    LAYOUTS,
    ligand_atom_indices,
    reference_coords_by_atom_name,
    structure_for,
)
from openfold3.steering.types import RestraintSet


def _ligand_query(smiles: str) -> Query:
    return Query.model_validate(
        {"chains": [{"molecule_type": "ligand", "chain_ids": ["L"], "smiles": smiles}]}
    )


def _structure_for_smiles(smiles: str):
    structure = structure_with_ref_mols_from_query(_ligand_query(smiles))
    tokenize_atom_array(structure.atom_array)
    add_token_positions(structure.atom_array)
    return structure


def test_tensorized_distance_bounds_match_pair_cutoff_rules():
    constraints = _DistanceConstraints(
        index=[(0, 1), (0, 2), (0, 3), (0, 4)],
        lower=[1.0] * 4,
        upper=[4.0] * 4,
        bond_mask=[True, False, True, False],
        angle_mask=[False, True, True, False],
        pair_vdw_cutoffs=[2.0] * 4,
    )

    restraints = _tensorize_distance_constraints(
        constraints,
        bond_buffer=defaults.BOND_BUFFER,
        angle_buffer=defaults.ANGLE_BUFFER,
        clash_buffer=defaults.CLASH_BUFFER,
    )

    torch.testing.assert_close(
        restraints.atom_index,
        torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=torch.int64),
    )
    torch.testing.assert_close(restraints.lower, torch.tensor([0.875, 2.0, 0.875, 2.0]))
    torch.testing.assert_close(
        restraints.upper, torch.tensor([2.0, 4.5, 2.0, float("inf")])
    )


def test_tensorize_empty_constraints_returns_empty_restraint_set():
    restraints = _tensorize_distance_constraints(
        _DistanceConstraints(),
        bond_buffer=defaults.BOND_BUFFER,
        angle_buffer=defaults.ANGLE_BUFFER,
        clash_buffer=defaults.CLASH_BUFFER,
    )
    assert restraints.atom_index.shape == (0, 2)
    assert restraints.lower.shape == (0,)
    assert restraints.upper.shape == (0,)


def test_geometry_constraints_handle_single_atom_and_cropped_pairs():
    single_atom = Chem.MolFromSmiles("[Na+]")
    empty = _compute_distance_constraints(
        single_atom, {0: 4}, vdw_pair_cutoff_offset=defaults.VDW_PAIR_CUTOFF_OFFSET
    )
    assert empty == _DistanceConstraints()

    mol = Chem.MolFromSmiles("CCO")
    constraints = _compute_distance_constraints(
        mol, {0: 10, 2: 12}, vdw_pair_cutoff_offset=defaults.VDW_PAIR_CUTOFF_OFFSET
    )
    assert constraints.index == [(10, 12)]
    periodic_table = Chem.GetPeriodicTable()
    expected_vdw_cutoff = (
        defaults.VDW_PAIR_CUTOFF_OFFSET
        + (periodic_table.GetRvdw(6) + periodic_table.GetRvdw(8)) / 2
    )
    assert constraints.pair_vdw_cutoffs == pytest.approx([expected_vdw_cutoff])


def test_geometry_constraints_reject_unsupported_atomic_numbers():
    mol = Chem.MolFromSmiles("*C")
    with pytest.raises(ValueError, match="supported atomic numbers"):
        _compute_distance_constraints(
            mol, {0: 0, 1: 1}, vdw_pair_cutoff_offset=defaults.VDW_PAIR_CUTOFF_OFFSET
        )


def test_build_context_rejects_misaligned_reference_atoms():
    structure = _structure_for_smiles("CCO")
    reference = structure.processed_reference_mols[0]
    malformed = replace(
        reference, in_crop_mask=np.arange(reference.mol.GetNumAtoms()) != 0
    )

    with pytest.raises(ValueError, match="must match the OF3 residue"):
        build_context(
            structure.atom_array, [malformed], n_atoms=len(structure.atom_array)
        )


def test_build_context_raises_on_missing_reference_molecule():
    """A missing reference molecule gets an explicit error naming both
    counts, not the incidental `zip() argument N is shorter` message the
    strict zip below would otherwise surface."""
    structure = _structure_for_smiles("CCO")
    with pytest.raises(
        MissingReferenceMoleculeError,
        match=r"one processed reference molecule per residue.*0 reference "
        r"molecule\(s\) for 1 residue\(s\)",
    ):
        build_context(structure.atom_array, [], n_atoms=len(structure.atom_array))


def test_build_context_raises_on_extra_reference_molecule():
    structure = _structure_for_smiles("CCO")
    extra = list(structure.processed_reference_mols) * 2

    with pytest.raises(MissingReferenceMoleculeError, match=r"2 reference"):
        build_context(structure.atom_array, extra, n_atoms=len(structure.atom_array))


def test_build_context_emits_distance_restraints_for_a_simple_ligand():
    structure = _structure_for_smiles("C[C@H](O)C(=O)O")
    ctx = build_context(
        structure.atom_array,
        structure.processed_reference_mols,
        n_atoms=len(structure.atom_array),
    )

    assert ctx.n_atoms == len(structure.atom_array)
    restraints = ctx.restraints["distance_bounds_potential"]
    assert restraints.atom_index.shape[0] > 0
    assert restraints.atom_index.dtype == torch.int64
    assert int(restraints.atom_index.max()) < ctx.n_atoms
    assert torch.all(restraints.lower <= restraints.upper)


def test_posebusters_bounds_penalize_out_of_plane_aromatic_geometry():
    """Benzene: zero energy on the ideal planar hexagon, positive once one
    atom is pushed out of plane. Distance-bounds-only: benzene's aromatic
    ring bonds are RDKit BondType.AROMATIC (not DOUBLE), so this exercises
    no planar-dihedral term -- there is none in this package yet."""
    structure = _structure_for_smiles("c1ccccc1")
    ctx = build_context(
        structure.atom_array,
        structure.processed_reference_mols,
        n_atoms=len(structure.atom_array),
    )
    restraints = ctx.restraints["distance_bounds_potential"]
    assert restraints.atom_index.shape[0] > 0

    angles = torch.arange(6) * torch.pi / 3
    planar = 1.4 * torch.stack(
        (torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1
    )
    planar = planar[None, None]
    distorted = planar.clone()
    distorted[..., 0, 2] = 1.0

    potential = DistanceBoundsPotential()
    planar_energy, _ = potential.energy_and_gradient(planar, restraints, 1.0)
    distorted_energy, _ = potential.energy_and_gradient(distorted, restraints, 1.0)

    torch.testing.assert_close(planar_energy, torch.zeros_like(planar_energy))
    assert distorted_energy.item() > planar_energy.item()


# ---------------------------------------------------------------------------
# Protein-ligand complexes and atom renumbering
#
# build_context maps RDKit-local atom indices onto the global atom axis by
# positionally zipping `in_crop_mask` against each ligand residue. In a
# complex that axis is renumbered relative to a ligand-only query -- the
# ligand no longer starts at 0, and where it starts depends on chain order,
# chain count, and how many residues precede it. These tests pin that
# restraints still land on the atoms they were computed for.
#
# The oracle is deliberately independent of the mapping under test: reference
# conformer coordinates are placed by matching *atom names* between the
# AtomArray and the RDKit molecule, not by the positional zip build_context
# uses. If the two disagree, the reference conformer stops satisfying its own
# restraints.
# ---------------------------------------------------------------------------


def _context_for(chains: list[dict]):
    structure = structure_for(chains)
    ctx = build_context(
        structure.atom_array,
        structure.processed_reference_mols,
        n_atoms=len(structure.atom_array),
    )
    return structure, ctx.restraints["distance_bounds_potential"]


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_complex_restraints_reference_only_ligand_atoms(layout: str):
    structure, restraints = _context_for(LAYOUTS[layout])
    ligand_atoms = set(ligand_atom_indices(structure).tolist())

    assert restraints.atom_index.shape[0] > 0
    referenced = set(restraints.atom_index.flatten().tolist())
    assert referenced <= ligand_atoms, (
        f"{sorted(referenced - ligand_atoms)} are not ligand atoms"
    )


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_complex_restraints_are_satisfied_by_the_reference_conformer(layout: str):
    """The conformer the bounds were derived from must satisfy them exactly.

    This is what catches a mis-mapped atom index: shift the ligand's global
    offset and the restraints start relating the wrong pairs of atoms, so the
    reference geometry violates its own bounds.
    """
    structure, restraints = _context_for(LAYOUTS[layout])
    coords = reference_coords_by_atom_name(structure)

    energy, _ = DistanceBoundsPotential().energy_and_gradient(coords, restraints, 1.0)

    assert float(energy) == pytest.approx(0.0, abs=1e-6)

    # Guard against the assertion above being vacuous: rotating every index by
    # one atom is exactly the renumbering error this is meant to detect, and
    # it must register.
    rotated = RestraintSet(
        atom_index=(restraints.atom_index + 1) % len(structure.atom_array),
        lower=restraints.lower,
        upper=restraints.upper,
    )
    rotated_energy, _ = DistanceBoundsPotential().energy_and_gradient(
        coords, rotated, 1.0
    )
    assert float(rotated_energy) > 1.0


def test_restraints_never_span_two_ligands() -> None:
    """Distance bounds are intramolecular: every restraint must join two
    atoms of the same ligand residue, never one atom from each."""
    structure, restraints = _context_for(LAYOUTS["two_ligands"])

    residue_of_atom: dict[int, int] = {}
    global_indices = np.arange(len(structure.atom_array))
    for residue_number, residue in enumerate(residue_view_iter(structure.atom_array)):
        for global_index in global_indices[residue.indices]:
            residue_of_atom[int(global_index)] = residue_number

    for first, second in restraints.atom_index.tolist():
        assert residue_of_atom[first] == residue_of_atom[second], (
            f"restraint {(first, second)} spans two residues"
        )


def test_each_ligand_contributes_its_own_restraints():
    """Both ligands in a two-ligand complex are steered, not just the first."""
    structure, restraints = _context_for(LAYOUTS["two_ligands"])
    referenced = set(restraints.atom_index.flatten().tolist())

    ligand_residue_atoms = [
        set(np.arange(len(structure.atom_array))[residue.indices].tolist())
        for residue in residue_view_iter(structure.atom_array)
        if np.all(residue.molecule_type_id == MoleculeType.LIGAND)
    ]
    assert len(ligand_residue_atoms) == 2
    for atoms in ligand_residue_atoms:
        assert referenced & atoms, "a ligand contributed no restraints"
