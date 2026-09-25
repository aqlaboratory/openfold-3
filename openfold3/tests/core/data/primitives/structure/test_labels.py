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
from biotite.structure import Atom, AtomArray, array
from biotite.structure.io import pdbx

from openfold3.core.data.primitives.structure.labels import (
    AtomArrayView,
    assign_atom_indices,
    assign_molecule_type_ids,
    residue_view_iter,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.tests.utils.custom_assert_utils import assert_atomarray_equal


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


def _make_cif_file(ids, types):
    cif_file = pdbx.CIFFile()
    block = pdbx.CIFBlock()
    block["chem_comp"] = pdbx.CIFCategory(
        {
            "id": np.array(ids),
            "type": np.array(types),
        }
    )
    cif_file["test"] = block
    return cif_file


def _make_atom_array(res_names, chain_ids):
    atoms = AtomArray(len(res_names))
    atoms.chain_id = np.array(chain_ids)
    atoms.res_id = np.arange(1, len(res_names) + 1)
    atoms.res_name = np.array(res_names)
    atoms.atom_name = np.array(["CA"] * len(res_names))
    atoms.element = np.array(["C"] * len(res_names))
    return atoms


class TestAssignMoleculeTypeIds:
    """Tests for the assign_molecule_type_ids function."""

    def test_valid_types_protein_and_ligand(self):
        cif_file = _make_cif_file(
            ["ALA", "GLY", "LIG"],
            ["L-PEPTIDE LINKING", "L-PEPTIDE LINKING", "NON-POLYMER"],
        )
        atom_array = _make_atom_array(
            ["ALA", "GLY", "LIG"],
            ["A", "A", "B"],
        )

        assign_molecule_type_ids(atom_array, cif_file)

        assert atom_array.molecule_type_id[0] == MoleculeType.PROTEIN
        assert atom_array.molecule_type_id[1] == MoleculeType.PROTEIN
        assert atom_array.molecule_type_id[2] == MoleculeType.LIGAND

    @pytest.mark.parametrize("chem_comp_type", ["", ".", "?", "BOGUS_TYPE"])
    def test_unknown_chem_comp_type_raises(self, chem_comp_type):
        cif_file = _make_cif_file(["BAD"], [chem_comp_type])
        atom_array = _make_atom_array(["BAD"], ["A"])

        with pytest.raises(ValueError, match="BAD"):
            assign_molecule_type_ids(atom_array, cif_file)

    def test_missing_chem_comp_entry_raises(self):
        cif_file = _make_cif_file(
            ["ALA", "GLY"], ["L-PEPTIDE LINKING", "L-PEPTIDE LINKING"]
        )
        atom_array = _make_atom_array(["ALA", "GLY", "XYZ"], ["A", "A", "A"])

        with pytest.raises(ValueError, match="XYZ"):
            assign_molecule_type_ids(atom_array, cif_file)

    def test_lowercase_valid_type(self):
        cif_file = _make_cif_file(
            ["ALA", "GLY"], ["l-peptide linking", "l-peptide linking"]
        )
        atom_array = _make_atom_array(["ALA", "GLY"], ["A", "A"])

        assign_molecule_type_ids(atom_array, cif_file)
        assert atom_array.molecule_type_id[0] == MoleculeType.PROTEIN
        assert atom_array.molecule_type_id[1] == MoleculeType.PROTEIN

    def test_multiple_invalid_components_reported_together(self):
        cif_file = _make_cif_file(["BAD1", "BAD2"], ["BOGUS", ""])
        atom_array = _make_atom_array(["BAD1", "BAD2"], ["A", "A"])

        with pytest.raises(ValueError) as exc_info:
            assign_molecule_type_ids(atom_array, cif_file)

        message = str(exc_info.value)
        assert "BAD1" in message
        assert "BAD2" in message

    def test_valid_OTHER_type(self):
        cif_file = _make_cif_file(
            ["OTH1", "OTH2"],
            ["OTHER", "OTHER"],
        )
        atom_array = _make_atom_array(
            ["OTH1", "OTH2"],
            ["A", "A"],
        )

        assign_molecule_type_ids(atom_array, cif_file)
        assert np.all(atom_array.molecule_type_id == MoleculeType.LIGAND)

    def test_unreferenced_chem_comp_type_is_ignored(self):
        cif_file = _make_cif_file(
            ["ALA", "UNUSED"],
            ["L-PEPTIDE LINKING", "BOGUS_TYPE"],
        )
        atom_array = _make_atom_array(
            ["ALA", "ALA"],
            ["A", "A"],
        )

        assign_molecule_type_ids(atom_array, cif_file)

        assert np.all(atom_array.molecule_type_id == MoleculeType.PROTEIN)
