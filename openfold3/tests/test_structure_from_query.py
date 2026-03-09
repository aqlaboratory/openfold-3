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

# TODO: Add more tests for general inference inputs
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path

import biotite.structure.info.ccd as biotite_ccd
import numpy as np
import pytest
import torch
from rdkit import Chem

from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from openfold3.core.data.primitives.structure.biotite_ccd import (
    concatenate_ccd,
    update_biotite_ccd,
)
from openfold3.core.data.primitives.structure.conformer import ConformerGenerationError
from openfold3.core.data.primitives.structure.query import (
    processed_reference_molecule_from_ccd_code,
    processed_reference_molecule_from_mol,
    structure_with_ref_mol_from_ccd_code,
    structure_with_ref_mols_from_query,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Query,
)
from openfold3.tests.custom_assert_utils import (
    assert_atomarray_equal,
    assert_ref_mols_equal,
)

reference_data_path = Path(__file__).parent / "test_data" / "structure_from_query"

# Custom CCD with alanine renamed to "GLU", with atoms in deliberately non-standard row
# order to verify that this order is propagated consistently.
CUSTOM_CCD_CODE = "GLU"
CUSTOM_CCD_EXPECTED_ATOM_NAMES = ["OX1", "CB1", "N1", "C1", "O1", "CA1"]
CUSTOM_CCD_CIF = """
data_GLU
#
_chem_comp.id                                    GLU
_chem_comp.name                                  "ALANINE (BUT NAMED GLU TO TEST CCD-OVERRIDE)"
_chem_comp.type                                  "L-PEPTIDE LINKING"
_chem_comp.pdbx_type                             ATOMP
_chem_comp.formula                               "C3 H7 N O2"
_chem_comp.mon_nstd_parent_comp_id               ?
_chem_comp.pdbx_synonyms                         ?
_chem_comp.pdbx_formal_charge                    0
_chem_comp.pdbx_initial_date                     1999-07-08
_chem_comp.pdbx_modified_date                    2024-09-27
_chem_comp.pdbx_ambiguous_flag                   N
_chem_comp.pdbx_release_status                   REL
_chem_comp.pdbx_replaced_by                      ?
_chem_comp.pdbx_replaces                         ?
_chem_comp.formula_weight                        89.093
_chem_comp.one_letter_code                       A
_chem_comp.three_letter_code                     GLU
_chem_comp.pdbx_model_coordinates_details        ?
_chem_comp.pdbx_model_coordinates_missing_flag   N
_chem_comp.pdbx_ideal_coordinates_details        ?
_chem_comp.pdbx_ideal_coordinates_missing_flag   N
_chem_comp.pdbx_model_coordinates_db_code        ?
_chem_comp.pdbx_subcomponent_list                ?
_chem_comp.pdbx_processing_site                  RCSB
_chem_comp.pdbx_pcm                              N
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.alt_atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_align
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_stereo_config
_chem_comp_atom.pdbx_backbone_atom_flag
_chem_comp_atom.pdbx_n_terminal_atom_flag
_chem_comp_atom.pdbx_c_terminal_atom_flag
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
GLU OX1 OX1 O 0 1 N Y N Y N Y 0.523 29.194 13.997 0.661  0.439 -1.742 OX1 GLU 1
GLU CB1 CB1 C 0 1 N N N N N N 0.601 26.143 14.574 1.204  -0.620 1.296  CB1 GLU 2
GLU N1  N1  N 0 1 N N N Y Y N 2.281 26.213 12.804 -0.966 0.493 1.500  N1  GLU 3
GLU C1  C1  C 0 1 N N N Y N Y 1.539 28.344 13.874 -0.094 0.017 -0.716 C1  GLU 4
GLU O1  O1  O 0 1 N N N Y N Y 2.709 28.647 14.114 -1.056 -0.682 -0.923 O1  GLU 5
GLU CA1 CA1 C 0 1 N N S Y N N 1.169 26.942 13.411 0.257  0.418 0.692  CA1 GLU 6
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal
GLU N1  CA1 SING N N 1
GLU CA1 C1  SING N N 2
GLU CA1 CB1 SING N N 3
GLU C1  O1  DOUB N N 4
GLU C1  OX1 SING N N 5
#
"""

# A standard peptide query
standard_peptide_query = Query.model_validate(
    {
        "query_name": "std_peptide",
        "chains": [
            {
                "molecule_type": "protein",
                "chain_ids": "A",
                "sequence": "MACHINELEARNING",
            }
        ],
    }
)

# A peptide query with non-canonical residues methionine sulfoxide (MHO) and
# selenocysteine (SEC)
non_canonical_peptide_query = Query.model_validate(
    {
        "query_name": "non_std_peptide",
        "chains": [
            {
                "molecule_type": "protein",
                "chain_ids": "A",
                "sequence": "MACHINELEARNING",
                "non_canonical_residues": {
                    "1": "MHO",
                    "3": "SEC",
                },
            }
        ],
    }
)


@pytest.mark.parametrize(
    "query, ground_truth_file",
    [
        (
            standard_peptide_query,
            reference_data_path / "structure-w-ref-mols_std-peptide.pkl",
        ),
        (
            non_canonical_peptide_query,
            reference_data_path / "structure-w-ref-mols_non-std-peptide.pkl",
        ),
    ],
    ids=[
        "standard_peptide",
        "non_canonical_peptide",
    ],
)
def test_structure_from_query(query: Query, ground_truth_file: Path):
    """Tests that the generated structure and reference molecules matches gt."""
    structure_with_ref_mols = structure_with_ref_mols_from_query(query)

    # Get reference file
    structure_with_ref_mols_gt = pickle.loads(ground_truth_file.read_bytes())

    # Check that atom arrays match (for some reason the GT generation script generated a
    # different order of annotations but that's fine)
    assert_atomarray_equal(
        structure_with_ref_mols.atom_array,
        structure_with_ref_mols_gt.atom_array,
        strict_annot_order=False,
    )

    # Check that reference molecules match
    for ref_mol, ref_mol_gt in zip(
        structure_with_ref_mols.processed_reference_mols,
        structure_with_ref_mols_gt.processed_reference_mols,
        strict=False,
    ):
        assert_ref_mols_equal(ref_mol, ref_mol_gt)


def test_smiles_with_explicit_hydrogen():
    """Tests that SMILES with explicit hydrogens can be processed.

    Regression test for a bug where explicit hydrogens in the input molecule
    caused a length mismatch between the atom mask and the molecule after
    conformer generation (which removes hydrogens).
    """
    # SMILES with explicit hydrogen - this triggered the bug
    smiles_with_explicit_h = "[H]/C=C\\Cl"
    mol = Chem.MolFromSmiles(smiles_with_explicit_h)

    # Should not raise an error
    ref_mol = processed_reference_molecule_from_mol(mol)

    # Verify mask length matches mol atom count
    assert ref_mol.mol.GetNumAtoms() == len(ref_mol.in_crop_mask)

    # Featurization should also succeed
    features = featurize_reference_conformers_of3(
        [ref_mol],
        add_ref_space_uid_to_perm=False,
    )
    assert "ref_pos" in features


def test_hem_ligand_generates_reference_conformer():
    """Ensure HEM can be converted into a usable inference reference conformer.

    This is a regression test for previous versions that had problems with
    organometallics-conformer generation, due to not using pdbeccdutils and its internal
    sanitization.
    """
    structure_with_ref_mols = structure_with_ref_mol_from_ccd_code(
        ccd_code="HEM",
        chain_id="A",
    )

    atom_names = structure_with_ref_mols.atom_array.atom_name.tolist()
    reference_mol = structure_with_ref_mols.processed_reference_mols[0].mol
    reference_atom_names = [
        atom.GetProp("annot_atom_name") for atom in reference_mol.GetAtoms()
    ]

    assert reference_mol.GetNumConformers() == 1
    assert len(atom_names) == reference_mol.GetNumAtoms()
    assert atom_names == reference_atom_names
    assert np.isfinite(reference_mol.GetConformer(0).GetPositions()).all()


@contextmanager
def _custom_biotite_ccd_context():
    """Temporarily point Biotite to the hard-coded custom CCD entry."""
    original_ccd_path = Path(biotite_ccd._CCD_FILE)

    with tempfile.TemporaryDirectory(prefix="of3_test_ccd_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        ccd_cif_path = tmp_dir / "custom_components.cif"
        ccd_bcif_path = tmp_dir / "custom_components.bcif"
        ccd_cif_path.write_text(CUSTOM_CCD_CIF)
        concatenate_ccd(
            ccd_path=ccd_cif_path,
        ).write(ccd_bcif_path)
        update_biotite_ccd(ccd_bcif_path)
        try:
            yield
        finally:
            update_biotite_ccd(original_ccd_path)


def _assert_atom_names_align_with_reference_mol(structure_with_ref_mols):
    """Assert both outputs use the expected custom atom-name order."""
    atom_names = structure_with_ref_mols.atom_array.atom_name.tolist()
    reference_mol = structure_with_ref_mols.processed_reference_mols[0].mol
    reference_atom_names = [
        atom.GetProp("annot_atom_name") for atom in reference_mol.GetAtoms()
    ]

    assert atom_names == CUSTOM_CCD_EXPECTED_ATOM_NAMES
    assert reference_atom_names == CUSTOM_CCD_EXPECTED_ATOM_NAMES
    assert atom_names == reference_atom_names


@pytest.mark.parametrize(
    "input_type",
    [
        "query",
        "ccd_code",
    ],
    ids=[
        "query_uses_global_biotite_ccd",
        "structure_with_ref_mol_from_ccd_code_aligns_atom_order",
    ],
)
def test_ligand_ccd_paths_respect_custom_ccd_and_atom_order(input_type):
    """Validate custom CCD use and atom-order alignment for both ligand input paths."""
    with _custom_biotite_ccd_context():
        # Build single amino acid structure and ref mol
        if input_type == "query":
            structure_with_ref_mols = structure_with_ref_mols_from_query(
                Query.model_validate(
                    {
                        "query_name": "ligand_custom_ccd",
                        "chains": [
                            {
                                "molecule_type": "ligand",
                                "chain_ids": "A",
                                "ccd_codes": [CUSTOM_CCD_CODE],
                            }
                        ],
                    }
                )
            )
        else:
            structure_with_ref_mols = structure_with_ref_mol_from_ccd_code(
                ccd_code=CUSTOM_CCD_CODE,
                chain_id="A",
            )

    _assert_atom_names_align_with_reference_mol(structure_with_ref_mols)


def test_conformer_fallback_to_ideal_with_partial_nan(monkeypatch):
    """Verify the Ideal-coordinate fallback produces valid features even with partial NaN.

    Mocks conformer generation to always fail, then injects an Ideal conformer
    with one NaN atom position. Checks that:
    - The fallback is used without raising
    - NaN positions are zeroed out
    - used_atom_mask correctly marks the NaN atom as False
    - Featurization succeeds without NaN in the output
    """
    from unittest.mock import patch

    from openfold3.core.data.primitives.structure.component import (
        mol_from_biotite_ccd_cached,
    )

    # Get a real mol (ALA) with its CCD Ideal conformer and corrupt one atom
    mol = mol_from_biotite_ccd_cached("ALA")
    ideal_conf = mol.GetConformer(0)
    assert ideal_conf.GetProp("name") == "Ideal"
    ideal_conf.SetAtomPosition(0, (float("nan"), float("nan"), float("nan")))

    # Patch the cached mol getter to return our modified mol, and conformer
    # generation to always fail
    with (
        patch(
            "openfold3.core.data.primitives.structure.query.mol_from_biotite_ccd_cached",
            return_value=Chem.Mol(mol),
        ),
        patch(
            "openfold3.core.data.primitives.structure.query.multistrategy_compute_conformer",
            side_effect=ConformerGenerationError("mocked failure"),
        ),
    ):
        proc_ref_mol = processed_reference_molecule_from_ccd_code("ALA")

    ref_mol = proc_ref_mol.mol
    positions = ref_mol.GetConformer(0).GetPositions()

    # NaN positions should be zeroed, not NaN
    assert np.isfinite(positions).all()
    assert np.allclose(positions[0], [0, 0, 0])

    # used_atom_mask should be False for the NaN atom
    masks = [atom.GetBoolProp("annot_used_atom_mask") for atom in ref_mol.GetAtoms()]
    assert masks[0] is False
    assert all(masks[1:])

    # Featurization should succeed without NaN
    features = featurize_reference_conformers_of3(
        [proc_ref_mol],
        add_ref_space_uid_to_perm=False,
    )
    assert torch.isfinite(features["ref_pos"]).all()
