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
from biotite.structure import AtomArray

from openfold3.core.data.primitives.structure.cleanup import (
    convert_MSE_to_MET,
    fix_arginine_naming,
    return_on_empty_atom_array,
)


@pytest.fixture
def bad_arginine_atom_array():
    """AtomArray with one ARG where NH2 is closer to CD than NH1 (needs fixing)."""
    # Arginine atoms: N, CA, C, O, CB, CG, CD, NE, CZ, NH1, NH2
    n_atoms = 11
    atom_array = AtomArray(n_atoms)

    atom_array.chain_id[:] = "A"
    atom_array.res_id[:] = 1
    atom_array.res_name[:] = "ARG"
    atom_array.atom_name[:] = [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
    ]
    atom_array.element[:] = ["N", "C", "C", "O", "C", "C", "C", "N", "C", "N", "N"]
    atom_array.hetero[:] = False

    # Set coordinates so NH2 is closer to CD than NH1
    # CD is at index 6, NH1 at index 9, NH2 at index 10
    atom_array.coord = np.zeros((n_atoms, 3))
    atom_array.coord[6] = [0.0, 0.0, 0.0]  # CD
    atom_array.coord[9] = [5.0, 0.0, 0.0]  # NH1 - far from CD
    atom_array.coord[10] = [1.0, 0.0, 0.0]  # NH2 - close to CD (bad naming)

    return atom_array


@pytest.fixture
def good_arginine_atom_array():
    """AtomArray with one ARG where NH1 is already closer to CD than NH2 (correct naming)."""
    n_atoms = 11
    atom_array = AtomArray(n_atoms)

    atom_array.chain_id[:] = "A"
    atom_array.res_id[:] = 1
    atom_array.res_name[:] = "ARG"
    atom_array.atom_name[:] = [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
    ]
    atom_array.element[:] = ["N", "C", "C", "O", "C", "C", "C", "N", "C", "N", "N"]
    atom_array.hetero[:] = False

    # Set coordinates so NH1 is closer to CD than NH2 (correct naming)
    # CD is at index 6, NH1 at index 9, NH2 at index 10
    atom_array.coord = np.zeros((n_atoms, 3))
    atom_array.coord[6] = [0.0, 0.0, 0.0]  # CD
    atom_array.coord[9] = [1.0, 0.0, 0.0]  # NH1 - close to CD (correct)
    atom_array.coord[10] = [5.0, 0.0, 0.0]  # NH2 - far from CD (correct)

    return atom_array


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


class TestFixArginineNaming:
    """Tests for the fix_arginine_naming function."""

    def test_swaps_nh1_nh2_when_nh2_closer_to_cd(self, bad_arginine_atom_array):
        """When NH2 is closer to CD than NH1, atom names are swapped."""
        # Verify initial state - NH2 closer to CD
        nh1_idx = np.where(bad_arginine_atom_array.atom_name == "NH1")[0][0]
        nh2_idx = np.where(bad_arginine_atom_array.atom_name == "NH2")[0][0]
        cd_idx = np.where(bad_arginine_atom_array.atom_name == "CD")[0][0]

        nh1_to_cd_dist = np.linalg.norm(
            bad_arginine_atom_array.coord[nh1_idx]
            - bad_arginine_atom_array.coord[cd_idx]
        )
        nh2_to_cd_dist = np.linalg.norm(
            bad_arginine_atom_array.coord[nh2_idx]
            - bad_arginine_atom_array.coord[cd_idx]
        )
        assert nh2_to_cd_dist < nh1_to_cd_dist, (
            "Test fixture should have NH2 closer to CD"
        )

        # Apply fix
        result = fix_arginine_naming(bad_arginine_atom_array)

        # After fix, the atom at position 9 (originally NH1) should now have name NH2
        # and atom at position 10 (originally NH2) should now have name NH1
        assert result.atom_name[9] == "NH2"
        assert result.atom_name[10] == "NH1"

    def test_no_change_when_nh1_already_closer_to_cd(self, good_arginine_atom_array):
        """When NH1 is already closer to CD than NH2, no changes are made."""
        # Store original atom names
        original_names = good_arginine_atom_array.atom_name.copy()

        # Apply fix
        result = fix_arginine_naming(good_arginine_atom_array)

        # Names should be unchanged
        np.testing.assert_array_equal(result.atom_name, original_names)

    def test_returns_early_when_no_arginine(self, mse_ala_atom_array):
        """When no ARG residues are present, no changes are made."""
        original_names = mse_ala_atom_array.atom_name.copy()

        result = fix_arginine_naming(mse_ala_atom_array)

        np.testing.assert_array_equal(result.atom_name, original_names)

    def test_cleans_up_temporary_annotation(self, bad_arginine_atom_array):
        """The temporary _atom_idx_arginine_fix annotation is removed after processing."""
        result = fix_arginine_naming(bad_arginine_atom_array)

        assert "_atom_idx_arginine_fix" not in result.get_annotation_categories()
