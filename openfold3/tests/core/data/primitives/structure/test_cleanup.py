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

import numpy as np
import pytest
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from openfold3.core.data.primitives.structure.cleanup import (
    check_ligand_atoms_in_ccd,
    convert_MSE_to_MET,
    fix_arginine_naming,
    remove_crystallization_aids,
    remove_non_CCD_atoms,
    return_on_empty_atom_array,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.tests.utils.custom_assert_utils import assert_atomarray_equal


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


def make_tracking_dummy():
    """Create a decorated dummy function that tracks whether it was called.

    Returns:
        tuple: (dummy_function, was_called) where was_called is a callable
               that returns True if dummy_function was invoked.
    """
    called = False

    @return_on_empty_atom_array
    def dummy_function(atom_array: AtomArray) -> AtomArray:
        nonlocal called
        called = True
        return atom_array

    return dummy_function, lambda: called


class TestReturnOnEmptyAtomArrayDecorator:
    """Tests for the return_on_empty_atom_array decorator."""

    def test_empty_atom_array_returns_immediately(self):
        """When given an empty AtomArray, the decorator returns it without calling the wrapped function."""
        dummy_function, was_called = make_tracking_dummy()

        empty_array = AtomArray(0)
        result = dummy_function(empty_array)

        assert result is empty_array
        assert not was_called()

    def test_non_empty_atom_array_calls_function(self, dummy_atom_array):
        """When given a non-empty AtomArray, the decorator calls the wrapped function."""
        dummy_function, was_called = make_tracking_dummy()

        result = dummy_function(dummy_atom_array)

        assert result is dummy_atom_array
        assert was_called()


class TestConvertMSEtoMET:
    """Tests for the convert_MSE_to_MET function."""

    def test_returns_early_when_no_mse_residues(self, dummy_atom_array):
        """When no MSE residues are present, the function returns early without modifications."""
        # Add required attributes for the function
        dummy_atom_array.res_name = np.array(["ALA", "ALA", "GLY", "GLY", "GLY"])
        original = dummy_atom_array.copy()

        result = convert_MSE_to_MET(dummy_atom_array)

        # Function returns None for early exit
        assert result is None
        # res_name should be unchanged
        assert_atomarray_equal(original, dummy_atom_array)

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
        assert not np.any(mse_ala_atom_array.hetero[mse_mask])

    def test_ala_residue_unchanged(self, mse_ala_atom_array):
        """ALA residue should remain unchanged after MSE conversion."""
        original = mse_ala_atom_array.copy()
        ala_mask = mse_ala_atom_array.res_name == "ALA"
        convert_MSE_to_MET(mse_ala_atom_array)

        # ALA should be unchanged
        assert_atomarray_equal(original[ala_mask], mse_ala_atom_array[ala_mask])


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
        original = good_arginine_atom_array.copy()

        # Apply fix
        result = fix_arginine_naming(good_arginine_atom_array)

        # Names should be unchanged
        assert_atomarray_equal(original, result)

    def test_returns_early_when_no_arginine(self, mse_ala_atom_array):
        """When no ARG residues are present, no changes are made."""
        original = mse_ala_atom_array.copy()

        result = fix_arginine_naming(mse_ala_atom_array)

        assert_atomarray_equal(original, result)

    def test_cleans_up_temporary_annotation(self, bad_arginine_atom_array):
        """The temporary _atom_idx_arginine_fix annotation is removed after processing."""
        result = fix_arginine_naming(bad_arginine_atom_array)

        assert "_atom_idx_arginine_fix" not in result.get_annotation_categories()

    @pytest.mark.parametrize(
        "atoms_to_remove",
        [
            ["NH1", "NH2"],  # Missing guanidinium nitrogens (e.g., PDB 4YVD)
            ["NH1", "NH2", "CD"],  # Missing CD and guanidinium (e.g., PDB 8DMQ)
            ["CD"],  # Missing only CD
            ["NH1"],  # Missing only NH1
            ["NH2"],  # Missing only NH2
        ],
    )
    def test_skips_arginine_with_missing_atoms(
        self, bad_arginine_atom_array, atoms_to_remove
    ):
        """When an arginine is missing required atoms, it is skipped without error.

        This represents real-world cases where an arginine residue is only partially
        resolved in the crystal structure. See PDB entries like 4YVD and 8DMQ.
        """
        mask = ~np.isin(bad_arginine_atom_array.atom_name, atoms_to_remove)
        incomplete_arg = bad_arginine_atom_array[mask]

        # Should not raise an error
        result = fix_arginine_naming(incomplete_arg)

        # Verify the array is returned unchanged (no swapping possible)
        assert len(result) == len(incomplete_arg)
        assert np.all(result.atom_name == incomplete_arg.atom_name)


@pytest.fixture
def atom_array_with_crystallization_aids():
    """AtomArray with ALA residue, SO4 (crystallization aid), and GOL (crystallization aid)."""
    # ALA: 5 atoms, SO4: 5 atoms (S + 4 O), GOL: 6 atoms (C3H8O3 heavy atoms)
    n_atoms = 16
    atom_array = AtomArray(n_atoms)

    atom_array.coord = np.zeros((n_atoms, 3))

    # ALA residue (res_id=1, chain A) - should be kept
    atom_array.chain_id[:5] = "A"
    atom_array.res_id[:5] = 1
    atom_array.res_name[:5] = "ALA"
    atom_array.atom_name[:5] = ["N", "CA", "C", "O", "CB"]
    atom_array.element[:5] = ["N", "C", "C", "O", "C"]
    atom_array.hetero[:5] = False

    # SO4 residue (res_id=2, chain B) - crystallization aid, should be removed
    atom_array.chain_id[5:10] = "B"
    atom_array.res_id[5:10] = 2
    atom_array.res_name[5:10] = "SO4"
    atom_array.atom_name[5:10] = ["S", "O1", "O2", "O3", "O4"]
    atom_array.element[5:10] = ["S", "O", "O", "O", "O"]
    atom_array.hetero[5:10] = True

    # GOL residue (res_id=3, chain C) - crystallization aid, should be removed
    atom_array.chain_id[10:] = "C"
    atom_array.res_id[10:] = 3
    atom_array.res_name[10:] = "GOL"
    atom_array.atom_name[10:] = ["C1", "C2", "C3", "O1", "O2", "O3"]
    atom_array.element[10:] = ["C", "C", "C", "O", "O", "O"]
    atom_array.hetero[10:] = True

    return atom_array


class TestRemoveCrystallizationAids:
    """Tests for the remove_crystallization_aids function."""

    def test_removes_crystallization_aids(self, atom_array_with_crystallization_aids):
        """Crystallization aids (SO4, GOL) are removed from the AtomArray."""
        original = atom_array_with_crystallization_aids.copy()
        ala_mask = atom_array_with_crystallization_aids.res_name == "ALA"
        ala_indices = np.nonzero(ala_mask)[0]

        result = remove_crystallization_aids(atom_array_with_crystallization_aids)

        # Only ALA should remain
        assert_atomarray_equal(original[ala_indices], result)

    def test_preserves_non_crystallization_aid_residues(
        self, atom_array_with_crystallization_aids
    ):
        """Non-crystallization aid residues are preserved unchanged."""
        original = atom_array_with_crystallization_aids.copy()
        ala_mask = atom_array_with_crystallization_aids.res_name == "ALA"
        ala_indices = np.nonzero(ala_mask)[0]

        result = remove_crystallization_aids(atom_array_with_crystallization_aids)

        assert_atomarray_equal(original[ala_indices], result[ala_indices])

    def test_no_change_when_no_crystallization_aids(self, mse_ala_atom_array):
        """When no crystallization aids are present, the array is unchanged."""
        original = mse_ala_atom_array.copy()

        result = remove_crystallization_aids(mse_ala_atom_array)

        assert_atomarray_equal(original, result)

    def test_custom_ccd_codes(self, atom_array_with_crystallization_aids):
        """Custom ccd_codes parameter allows specifying which residues to remove."""
        original = atom_array_with_crystallization_aids.copy()
        resname_mask = np.isin(
            atom_array_with_crystallization_aids.res_name, ["ALA", "GOL"]
        )
        resname_indicies = np.nonzero(resname_mask)[0]

        # Only remove SO4, keep GOL
        result = remove_crystallization_aids(
            atom_array_with_crystallization_aids, ccd_codes=["SO4"]
        )

        # ALA (5 atoms) + GOL (6 atoms) should remain
        assert_atomarray_equal(original[resname_indicies], result)

    def test_empty_atom_array_returns_empty(self):
        """Empty AtomArray returns empty (via decorator)."""
        empty_array = AtomArray(0)

        result = remove_crystallization_aids(empty_array)

        assert_atomarray_equal(empty_array, result)


def _make_ccd(comp_id_to_atom_ids):
    """Builds a minimal CCD that only defines the atom names of each component."""
    ccd = pdbx.CIFFile()
    for comp_id, atom_ids in comp_id_to_atom_ids.items():
        block = pdbx.CIFBlock()
        block["chem_comp_atom"] = pdbx.CIFCategory({"atom_id": np.array(atom_ids)})
        ccd[comp_id] = block
    return ccd


def _make_residues(residues):
    """Builds an AtomArray from (chain_id, res_id, res_name, molecule_type, atoms)."""
    atoms = [
        (chain_id, res_id, res_name, molecule_type, atom_name)
        for chain_id, res_id, res_name, molecule_type, atom_names in residues
        for atom_name in atom_names
    ]
    atom_array = AtomArray(len(atoms))
    atom_array.chain_id[:] = [atom[0] for atom in atoms]
    atom_array.res_id[:] = [atom[1] for atom in atoms]
    atom_array.res_name[:] = [atom[2] for atom in atoms]
    atom_array.atom_name[:] = [atom[4] for atom in atoms]
    atom_array.set_annotation(
        "molecule_type_id", np.array([atom[3] for atom in atoms], dtype=int)
    )
    return atom_array


CCD = _make_ccd(
    {
        "ALA": ["N", "CA", "C", "O", "CB", "OXT"],
        "LIG": ["C1", "C2", "O1", "O2"],
    }
)
ALA_ATOMS = ["N", "CA", "C", "O", "CB"]


class TestCheckLigandAtomsInCCD:
    """Tests for the check_ligand_atoms_in_ccd function."""

    def test_ligand_with_atoms_outside_ccd_entry_raises(self):
        """A ligand with atoms that its CCD entry lacks raises and names them."""
        atom_array = _make_residues(
            [
                ("A", 1, "ALA", MoleculeType.PROTEIN, ALA_ATOMS),
                ("B", 2, "LIG", MoleculeType.LIGAND, ["C1", "C2", "S1", "O3"]),
            ]
        )

        with pytest.raises(ValueError, match=r"\('B', 2, 'LIG', \['S1', 'O3'\]\)"):
            check_ligand_atoms_in_ccd(atom_array, CCD)

    def test_partially_resolved_ligand_passes(self):
        """Atoms missing from the structure are not a mismatch."""
        atom_array = _make_residues(
            [("B", 1, "LIG", MoleculeType.LIGAND, ["C1", "O1"])]
        )

        check_ligand_atoms_in_ccd(atom_array, CCD)

    def test_polymer_atoms_outside_ccd_entry_are_not_checked(self):
        """Polymer residues keep the AF3 behavior of removing non-CCD atoms."""
        atom_array = _make_residues(
            [("A", 1, "ALA", MoleculeType.PROTEIN, [*ALA_ATOMS, "XX"])]
        )

        check_ligand_atoms_in_ccd(atom_array, CCD)

    def test_unl_is_not_checked(self):
        """UNL has no atoms in the CCD and is removed by remove_non_CCD_atoms."""
        atom_array = _make_residues(
            [("B", 1, "UNL", MoleculeType.LIGAND, ["C1", "O1"])]
        )

        check_ligand_atoms_in_ccd(atom_array, CCD)

    def test_all_mismatched_ligands_reported_together(self):
        """Every mismatched ligand is listed in a single error."""
        atom_array = _make_residues(
            [
                ("B", 1, "LIG", MoleculeType.LIGAND, ["C1", "S1"]),
                ("C", 1, "LIG", MoleculeType.LIGAND, ["C1", "CL1"]),
            ]
        )

        with pytest.raises(ValueError) as exc_info:
            check_ligand_atoms_in_ccd(atom_array, CCD)

        message = str(exc_info.value)
        assert "('B', 1, 'LIG', ['S1'])" in message
        assert "('C', 1, 'LIG', ['CL1'])" in message


class TestRemoveNonCCDAtoms:
    """Tests for the remove_non_CCD_atoms function."""

    def test_removes_polymer_atoms_outside_ccd_entry(self):
        """Atoms that are not in the residue's CCD entry are removed (AF3 SI 2.5.4)."""
        atom_array = _make_residues(
            [("A", 1, "ALA", MoleculeType.PROTEIN, [*ALA_ATOMS, "XX"])]
        )

        result = remove_non_CCD_atoms(atom_array, CCD)

        assert result.atom_name.tolist() == ALA_ATOMS

    def test_removes_unl(self):
        """UNL residues are removed entirely."""
        atom_array = _make_residues(
            [
                ("A", 1, "ALA", MoleculeType.PROTEIN, ALA_ATOMS),
                ("B", 2, "UNL", MoleculeType.LIGAND, ["C1", "O1"]),
            ]
        )

        result = remove_non_CCD_atoms(atom_array, CCD)

        assert result.res_name.tolist() == ["ALA"] * len(ALA_ATOMS)

    def test_residue_not_in_ccd_raises(self):
        """A residue name that is not in the CCD raises a ValueError naming it."""
        atom_array = _make_residues([("B", 1, "LIG1", MoleculeType.LIGAND, ["C1"])])

        with pytest.raises(
            ValueError, match=r"Residue LIG1 \(chain B, residue 1\) is not in the CCD"
        ):
            remove_non_CCD_atoms(atom_array, CCD)
