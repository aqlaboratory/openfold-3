import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, array

from openfold3.core.data.primitives.structure.labels import AtomArrayView
from tests.custom_assert_utils import assert_atomarray_equal


@pytest.fixture
def test_atom_array() -> AtomArray:
    atom1 = Atom([1, 2, 3], chain_id="A")
    atom2 = Atom([2, 3, 4], chain_id="A")
    atom3 = Atom([3, 4, 5], chain_id="B")
    atom4 = Atom([3, 4, 5], chain_id="B")
    return array([atom1, atom2, atom3, atom4])


def test_atom_array_slice_view(test_atom_array):
    slice_indices = slice(2, 4, 1)
    slice_view = AtomArrayView(test_atom_array, slice_indices)

    # This slice view has an underlying base
    assert slice_view.chain_id.base is not None
    assert len(slice_view) == 2
    np.testing.assert_equal(slice_view.chain_id, np.array(["B", "B"]))

    # If we materialize, we expect a new array
    materialized = slice_view.materialize()
    assert isinstance(materialized, AtomArray)
    assert_atomarray_equal(materialized, test_atom_array[slice_indices])


def test_atom_array_mask_view(test_atom_array):
    mask_indices = np.array([False, True, False, True])
    mask_view = AtomArrayView(test_atom_array, mask_indices)

    # When the index used is not basic indexing, we get new arrays
    assert mask_view.chain_id.base is None
    assert len(mask_view) == 2
    np.testing.assert_equal(mask_view.chain_id, np.array(["A", "B"]))

    # If we materialize, we expect a new array
    materialized = mask_view.materialize()
    assert isinstance(materialized, AtomArray)
    assert_atomarray_equal(materialized, test_atom_array[mask_indices])
