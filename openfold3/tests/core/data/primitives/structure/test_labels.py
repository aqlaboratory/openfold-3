# Copyright 2025 AlQuraishi Laboratory
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

import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, array

from openfold3.core.data.primitives.structure.labels import (
    AtomArrayView,
    assign_atom_indices,
    residue_view_iter,
)
from openfold3.tests.custom_assert_utils import assert_atomarray_equal


@pytest.fixture
def fake_atom_array() -> AtomArray:
    atom1 = Atom([1, 2, 3], chain_id="A")
    atom2 = Atom([2, 3, 4], chain_id="A")
    atom3 = Atom([3, 4, 5], chain_id="B")
    atom4 = Atom([3, 4, 5], chain_id="B")
    return array([atom1, atom2, atom3, atom4])


class TestAtomArrayView:
    """Tests for the AtomArrayView class."""

    def test_atom_array_slice_view(self, fake_atom_array):
        slice_indices = slice(2, 4, 1)
        slice_view = AtomArrayView(fake_atom_array, slice_indices)

        # This slice view has an underlying base
        assert slice_view.chain_id.base is not None
        assert len(slice_view) == 2
        np.testing.assert_equal(slice_view.chain_id, np.array(["B", "B"]))

        # If we materialize, we expect a new array
        materialized = slice_view.materialize()
        assert isinstance(materialized, AtomArray)
        assert_atomarray_equal(materialized, fake_atom_array[slice_indices])

    def test_atom_array_mask_view(self, fake_atom_array):
        mask_indices = np.array([False, True, False, True])
        mask_view = AtomArrayView(fake_atom_array, mask_indices)

        # When the index used is not basic indexing, we get new arrays
        assert mask_view.chain_id.base is None
        assert len(mask_view) == 2
        np.testing.assert_equal(mask_view.chain_id, np.array(["A", "B"]))

        # If we materialize, we expect a new array
        materialized = mask_view.materialize()
        assert isinstance(materialized, AtomArray)
        assert_atomarray_equal(materialized, fake_atom_array[mask_indices])


class TestAssignAtomIndices:
    """Tests for the assign_atom_indices function."""

    def test_assigns_indices_to_atom_array(self, dummy_atom_array):
        """Indices 0 to n-1 are assigned to the AtomArray."""
        assign_atom_indices(dummy_atom_array)

        assert "_atom_idx" in dummy_atom_array.get_annotation_categories()
        assert np.array_equal(
            dummy_atom_array._atom_idx, np.arange(len(dummy_atom_array))
        )

    def test_custom_label(self, dummy_atom_array):
        """A custom label can be used for the annotation."""
        assign_atom_indices(dummy_atom_array, label="_my_custom_idx")

        assert "_my_custom_idx" in dummy_atom_array.get_annotation_categories()
        assert np.array_equal(
            dummy_atom_array._my_custom_idx, np.arange(len(dummy_atom_array))
        )

    def test_raises_error_if_label_exists(self, dummy_atom_array):
        """Raises ValueError if the annotation already exists."""
        assign_atom_indices(dummy_atom_array)

        with pytest.raises(ValueError, match="already exists"):
            assign_atom_indices(dummy_atom_array)

    def test_overwrite_existing_label(self, dummy_atom_array):
        """Existing annotation can be overwritten with overwrite=True."""
        assign_atom_indices(dummy_atom_array)
        # Modify the array to change its length conceptually (we'll just verify overwrite works)
        assign_atom_indices(dummy_atom_array, overwrite=True)

        assert np.array_equal(
            dummy_atom_array._atom_idx, np.arange(len(dummy_atom_array))
        )


class TestResidueViewIter:
    """Tests for the residue_view_iter function."""

    def test_yields_correct_number_of_residues(self, mse_ala_atom_array):
        """Yields one view per residue in the AtomArray."""
        residue_views = list(residue_view_iter(mse_ala_atom_array))

        # mse_ala_atom_array has 2 residues: MSE (res_id=1) and ALA (res_id=2)
        assert len(residue_views) == 2

    def test_yields_atom_array_views(self, mse_ala_atom_array):
        """Each yielded item is an AtomArrayView."""
        for view in residue_view_iter(mse_ala_atom_array):
            assert isinstance(view, AtomArrayView)

    def test_each_view_contains_correct_atoms(self, mse_ala_atom_array):
        """Each view contains only atoms from one residue."""
        residue_views = list(residue_view_iter(mse_ala_atom_array))

        # First residue is MSE with 8 atoms
        mse_view = residue_views[0]
        assert len(mse_view) == 8
        assert np.all(mse_view.res_name == "MSE")
        assert np.all(mse_view.res_id == 1)

        # Second residue is ALA with 5 atoms
        ala_view = residue_views[1]
        assert len(ala_view) == 5
        assert np.all(ala_view.res_name == "ALA")
        assert np.all(ala_view.res_id == 2)

    def test_empty_atom_array_yields_nothing(self):
        """An empty AtomArray yields no residue views."""
        empty_array = AtomArray(0)

        residue_views = list(residue_view_iter(empty_array))

        assert len(residue_views) == 0
