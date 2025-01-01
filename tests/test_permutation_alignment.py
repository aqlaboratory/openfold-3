from pathlib import Path

import numpy as np

from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz
from openfold3.core.data.primitives.permutation.mol_labels import (
    assign_mol_permutation_ids,
)
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
