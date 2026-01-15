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

from openfold3.core.data.primitives.structure.labels import assign_atom_indices


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
