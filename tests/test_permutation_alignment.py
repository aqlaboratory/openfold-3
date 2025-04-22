import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import biotite.structure as struc
import numpy as np
import torch

from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz
from openfold3.core.data.primitives.permutation.mol_labels import (
    assign_mol_permutation_ids,
    chain_connected_molecule_iter,
)
from openfold3.core.data.resources.residues import RESTYPE_INDEX_3
from openfold3.core.utils.permutation_alignment import multi_chain_permutation_alignment
from tests.custom_assert_utils import assert_atomarray_equal

TEST_DIR = Path("tests/test_data/permutation_alignment")


def test_mol_symmetry_id_assignment():
    """Checks that permutation IDs are correctly assigned.

    This checks that the permutation IDs required to detect symmetry-equivalent parts of
    the AtomArray are working as expected.

    The test case here is a challenging structure with covalent ligands and symmetric
    molecules with different chain order that was manually verified.

    The input AtomArray was properly processed beforehand to have the additional IDs
    (like token IDs & component IDs) required for the permutation ID assignment.
    """
    atom_array_in = read_atomarray_from_npz(
        TEST_DIR / "inputs/npz/7pbd/7pbd_subset.npz"
    )
    atom_array_out = read_atomarray_from_npz(
        TEST_DIR / "outputs/npz/7pbd_subset_with-perm-ids.npz"
    )

    atom_array_out_test = assign_mol_permutation_ids(atom_array_in, retokenize=True)

    assert_atomarray_equal(atom_array_out, atom_array_out_test)

    # Assert retokenization
    assert np.array_equal(
        np.unique(np.diff(atom_array_out_test.token_id)), np.array([0, 1])
    )


def test_chain_connected_molecule_iter():
    """Checks that the chain-connected molecule iterator works as expected."""
    # Create AtomArray with three chains
    atom_array = struc.array(
        [
            struc.Atom([0, 0, 0], chain_id="A"),
            struc.Atom([0, 0, 0], chain_id="A"),
            struc.Atom([0, 0, 0], chain_id="A"),
            struc.Atom([0, 0, 0], chain_id="B"),
            struc.Atom([0, 0, 0], chain_id="B"),
            struc.Atom([0, 0, 0], chain_id="C"),
            struc.Atom([0, 0, 0], chain_id="C"),
        ]
    )

    # Add only a single connection between A and B
    atom_array.bonds = struc.BondList(7, np.array([(2, 3)]))

    atom_array_original = atom_array.copy()

    # Normal biotite molecule_iter returns 6 molecules
    assert len(struc.get_molecule_indices(atom_array)) == 6

    # Chain-connected molecule_iter should return chain-slices with A merged with B
    expected_slices = [atom_array[0:5], atom_array[5:7]]

    for mol_array, expected_array in zip(
        chain_connected_molecule_iter(atom_array), expected_slices
    ):
        assert_atomarray_equal(mol_array, expected_array)

    # Verify that original atom_array was unchanged
    assert_atomarray_equal(atom_array, atom_array_original)
