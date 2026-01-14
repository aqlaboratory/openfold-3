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
from biotite.structure import AtomArray

from openfold3.core.data.primitives.structure.cleanup import (
    convert_MSE_to_MET,
    return_on_empty_atom_array,
)


class TestReturnOnEmptyAtomArrayDecorator:
    """Tests for the return_on_empty_atom_array decorator."""

    def test_empty_atom_array_returns_immediately(self):
        """When given an empty AtomArray, the decorator returns it without calling the wrapped function."""
        function_was_called = False

        @return_on_empty_atom_array
        def dummy_function(atom_array: AtomArray) -> AtomArray:
            nonlocal function_was_called
            function_was_called = True
            return atom_array

        empty_array = AtomArray(0)
        result = dummy_function(empty_array)

        assert result is empty_array
        assert not function_was_called

    def test_non_empty_atom_array_calls_function(self, dummy_atom_array):
        """When given a non-empty AtomArray, the decorator calls the wrapped function."""
        function_was_called = False

        @return_on_empty_atom_array
        def dummy_function(atom_array: AtomArray) -> AtomArray:
            nonlocal function_was_called
            function_was_called = True
            return atom_array

        result = dummy_function(dummy_atom_array)

        assert result is dummy_atom_array
        assert function_was_called


class TestConvertMSEtoMET:
    """Tests for the convert_MSE_to_MET function."""

    def test_returns_early_when_no_mse_residues(self, dummy_atom_array):
        """When no MSE residues are present, the function returns early without modifications."""
        # Add required attributes for the function
        dummy_atom_array.res_name = np.array(["ALA", "ALA", "GLY", "GLY", "GLY"])

        result = convert_MSE_to_MET(dummy_atom_array)

        # Function returns None for early exit
        assert result is None
        # res_name should be unchanged
        assert np.all(dummy_atom_array.res_name == ["ALA", "ALA", "GLY", "GLY", "GLY"])

    def test_converts_mse_to_met(self, mse_ala_atom_array):
        """MSE residues are converted to MET with correct element, atom_name, and hetero changes."""
        convert_MSE_to_MET(mse_ala_atom_array)

        # MSE should now be MET
        mse_mask = mse_ala_atom_array.res_id == 1
        assert np.all(mse_ala_atom_array.res_name[mse_mask] == "MET")

        # Selenium atom should be converted to sulfur
        se_atom_idx = 6  # The SE atom is at index 6
        assert mse_ala_atom_array.element[se_atom_idx] == "S"
        assert mse_ala_atom_array.atom_name[se_atom_idx] == "SD"

        # Hetero should be False for converted residue
        assert np.all(not mse_ala_atom_array.hetero[mse_mask])

    def test_ala_residue_unchanged(self, mse_ala_atom_array):
        """ALA residue should remain unchanged after MSE conversion."""
        # Store original ALA values
        ala_mask = mse_ala_atom_array.res_id == 2
        original_res_name = mse_ala_atom_array.res_name[ala_mask].copy()
        original_element = mse_ala_atom_array.element[ala_mask].copy()
        original_atom_name = mse_ala_atom_array.atom_name[ala_mask].copy()
        original_hetero = mse_ala_atom_array.hetero[ala_mask].copy()

        convert_MSE_to_MET(mse_ala_atom_array)

        # ALA should be unchanged
        assert np.all(mse_ala_atom_array.res_name[ala_mask] == original_res_name)
        assert np.all(mse_ala_atom_array.element[ala_mask] == original_element)
        assert np.all(mse_ala_atom_array.atom_name[ala_mask] == original_atom_name)
        assert np.all(mse_ala_atom_array.hetero[ala_mask] == original_hetero)
