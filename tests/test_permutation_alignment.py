from pathlib import Path

from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz
from openfold3.core.data.primitives.permutation.mol_labels import (
    assign_mol_permutation_ids,
)
from tests.custom_assert_utils import assert_atomarray_equal

TEST_DIR = Path("test_data/permutation_alignment")


def test_permutation_id_assignment():
    """Checks that permutation IDs are correctly assigned.

    This checks that the permutation IDs required to detect symmetry-equivalent parts of
    the AtomArray are working as expected.
    """
    atom_array_in = read_atomarray_from_npz(TEST_DIR / "inputs/npz/7pbd_subset.npz")
    atom_array_out = read_atomarray_from_npz(
        TEST_DIR / "outputs/npz/7pbd_subset_with-perm-ids.npz"
    )

    atom_array_out_test = assign_mol_permutation_ids(atom_array_in)

    assert_atomarray_equal(atom_array_out, atom_array_out_test)
