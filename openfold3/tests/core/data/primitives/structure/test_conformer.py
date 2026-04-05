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

"""Tests for conformer fallback resolution and source annotation.

Tests the ``resolve_and_format_fallback_conformer`` function, which selects a
single fallback conformer for a molecule and annotates its source. The four
possible fallback sources are:

    - "rdkit": Conformer generated successfully by RDKit's ETKDGv3
    - "ccd-ideal": Ideal coordinates from the CCD entry
    - "ccd-model": Model coordinates from the CCD entry
    - "all-nan": No conformer available; all-NaN coordinates (replaced with zeros)
"""

import numpy as np
import pytest
from biotite.structure.io.pdbx import CIFFile
from rdkit import Chem

from openfold3.core.data.primitives.structure.component import (
    mol_from_ccd_entry,
)
from openfold3.core.data.primitives.structure.conformer import (
    ConformerGenerationError,
    resolve_and_format_fallback_conformer,
)

# ---------------------------------------------------------------------------
# CCD CIF test fixture for GLY (Glycine)
# ---------------------------------------------------------------------------
# Realistic CCD CIF entry based on the real Chemical Component Dictionary
# format. Contains both Ideal and Model conformers for all 9 atoms (5 heavy
# atoms: N, CA, C, O, OXT; 4 hydrogens: H, H2, HA2, HA3).

GLY_CIF_STRING = """\
data_GLY
#
_chem_comp.id                                    GLY
_chem_comp.name                                  GLYCINE
_chem_comp.type                                  "L-PEPTIDE LINKING"
_chem_comp.pdbx_type                             ATOMP
_chem_comp.formula                               "C2 H5 N O2"
_chem_comp.mon_nstd_flag                         y
_chem_comp.pdbx_synonyms                         ?
_chem_comp.pdbx_formal_charge                    0
_chem_comp.pdbx_initial_date                     1999-07-08
_chem_comp.pdbx_modified_date                    2023-11-03
_chem_comp.pdbx_ambiguous_flag                   N
_chem_comp.pdbx_release_status                   REL
_chem_comp.pdbx_replaced_by                      ?
_chem_comp.pdbx_replaces                         ?
_chem_comp.formula_weight                        75.032
_chem_comp.one_letter_code                       G
_chem_comp.three_letter_code                     GLY
_chem_comp.pdbx_model_coordinates_details        ?
_chem_comp.pdbx_model_coordinates_missing_flag   N
_chem_comp.pdbx_ideal_coordinates_details        Corina
_chem_comp.pdbx_ideal_coordinates_missing_flag   N
_chem_comp.pdbx_model_coordinates_db_code        1EJG
_chem_comp.pdbx_subcomponent_list                ?
_chem_comp.pdbx_processing_site                  RCSB
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
_chem_comp_atom.model_Cartn_x
_chem_comp_atom.model_Cartn_y
_chem_comp_atom.model_Cartn_z
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
_chem_comp_atom.pdbx_component_atom_id
_chem_comp_atom.pdbx_component_comp_id
_chem_comp_atom.pdbx_ordinal
GLY N    N    N  0 1 N N N  -1.549  -0.660   0.000   0.411   1.392   0.000 N    GLY 1
GLY CA   CA   C  0 1 N N N  -0.401   0.232   0.000  -0.006   0.000   0.000 CA   GLY 2
GLY C    C    C  0 1 N N N   0.927  -0.412   0.000   1.482  -0.156   0.000 C    GLY 3
GLY O    O    O  0 1 N N N   1.016  -1.638   0.000   2.081  -1.219   0.000 O    GLY 4
GLY OXT  OXT  O -1 1 N Y N   1.919   0.302   0.000   2.102   0.945   0.000 OXT  GLY 5
GLY H    H    H  0 1 N N N  -2.467  -0.229   0.000  -0.089   1.886   0.815 H    GLY 6
GLY H2   HN2  H  0 1 N Y N  -1.504  -1.263   0.000   0.411   1.978  -0.818 H2   GLY 7
GLY HA2  HA1  H  0 1 N N N  -0.499   0.853   0.900  -1.076   0.000  -0.213 HA2  GLY 8
GLY HA3  HA2  H  0 1 N N N  -0.499   0.853  -0.900  -0.431  -0.478  -0.892 HA3  GLY 9
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
_chem_comp_bond.pdbx_stereo_config
_chem_comp_bond.pdbx_ordinal
GLY N   CA   SING N N 1
GLY N   H    SING N N 2
GLY N   H2   SING N N 3
GLY CA  C    SING N N 4
GLY CA  HA2  SING N N 5
GLY CA  HA3  SING N N 6
GLY C   O    DOUB N N 7
GLY C   OXT  SING N N 8
#
"""

# Variant with ideal coordinates missing — forces fallback to Model conformer
GLY_CIF_IDEAL_MISSING = GLY_CIF_STRING.replace(
    "_chem_comp.pdbx_ideal_coordinates_missing_flag   N",
    "_chem_comp.pdbx_ideal_coordinates_missing_flag   Y",
)

# Expected heavy-atom ideal coordinates (after H removal, order: N, CA, C, O, OXT)
EXPECTED_IDEAL_COORDS = {
    "N": (0.411, 1.392, 0.000),
    "CA": (-0.006, 0.000, 0.000),
    "C": (1.482, -0.156, 0.000),
    "O": (2.081, -1.219, 0.000),
    "OXT": (2.102, 0.945, 0.000),
}

# Expected heavy-atom model coordinates
EXPECTED_MODEL_COORDS = {
    "N": (-1.549, -0.660, 0.000),
    "CA": (-0.401, 0.232, 0.000),
    "C": (0.927, -0.412, 0.000),
    "O": (1.016, -1.638, 0.000),
    "OXT": (1.919, 0.302, 0.000),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_cif_from_string(cif_string: str, tmp_path) -> CIFFile:
    """Write CIF string to a temp file and read it back as a CIFFile."""
    cif_path = tmp_path / "component.cif"
    cif_path.write_text(cif_string)
    return CIFFile.read(str(cif_path))


def _get_atom_names(mol) -> list[str]:
    """Extract atom names from an AnnotatedMol."""
    return [atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()]


def _get_used_atom_mask(mol) -> list[bool]:
    """Extract the used_atom_mask annotation from an AnnotatedMol."""
    return [atom.GetBoolProp("annot_used_atom_mask") for atom in mol.GetAtoms()]


def _get_coords_by_name(mol) -> dict[str, tuple[float, float, float]]:
    """Get conformer coordinates indexed by atom name."""
    conf = mol.GetConformer()
    atom_names = _get_atom_names(mol)
    coords = {}
    for idx, name in enumerate(atom_names):
        pos = conf.GetAtomPosition(idx)
        coords[name] = (pos.x, pos.y, pos.z)
    return coords


def _assert_coords_match(
    actual: dict[str, tuple], expected: dict[str, tuple], atol: float = 1e-3
):
    """Assert that actual coordinates match expected, keyed by atom name."""
    for name, exp_xyz in expected.items():
        assert name in actual, f"Atom {name} not found in molecule"
        np.testing.assert_allclose(
            actual[name],
            exp_xyz,
            atol=atol,
            err_msg=f"Coordinates mismatch for atom {name}",
        )


def _raise_conformer_generation_error(*args, **kwargs):
    raise ConformerGenerationError("mocked")


def _mock_conformer_generation_failure(monkeypatch):
    """Patch multistrategy_compute_conformer to always raise."""
    monkeypatch.setattr(
        "openfold3.core.data.primitives.structure.conformer"
        ".multistrategy_compute_conformer",
        _raise_conformer_generation_error,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gly_mol(tmp_path):
    """GLY mol with both Ideal and Model conformers (from CCD CIF)."""
    cif = _read_cif_from_string(GLY_CIF_STRING, tmp_path)
    return mol_from_ccd_entry("GLY", cif)


@pytest.fixture
def gly_mol_ideal_missing(tmp_path):
    """GLY mol with only Model conformer (ideal coordinates missing)."""
    cif = _read_cif_from_string(GLY_CIF_IDEAL_MISSING, tmp_path)
    return mol_from_ccd_entry("GLY", cif)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rdkit_conformer_source(gly_mol):
    """RDKit conformer generation succeeds for GLY."""
    mol, strategy, source = resolve_and_format_fallback_conformer(gly_mol)

    assert source == "rdkit"
    assert strategy in ("default", "random_init")
    assert len(list(mol.GetConformers())) == 1

    # Coordinates should not be all-zero (RDKit generated real 3D coords)
    coords = mol.GetConformer().GetPositions()
    assert not np.allclose(coords, 0.0)

    # All atoms should have valid coordinates (mask all True)
    assert all(_get_used_atom_mask(mol))


def test_ccd_ideal_fallback_source(monkeypatch, gly_mol):
    """Falls back to CCD Ideal conformer when RDKit generation fails."""
    _mock_conformer_generation_failure(monkeypatch)

    mol, strategy, source = resolve_and_format_fallback_conformer(gly_mol)

    assert source == "ccd-ideal"
    assert strategy == "use_fallback"
    assert len(list(mol.GetConformers())) == 1

    # Coordinates should match the ideal coordinates from the CIF
    _assert_coords_match(_get_coords_by_name(mol), EXPECTED_IDEAL_COORDS)
    assert all(_get_used_atom_mask(mol))


def test_ccd_model_fallback_source(monkeypatch, gly_mol_ideal_missing):
    """Falls back to CCD Model conformer when Ideal is missing and RDKit fails."""
    _mock_conformer_generation_failure(monkeypatch)

    mol, strategy, source = resolve_and_format_fallback_conformer(gly_mol_ideal_missing)

    assert source == "ccd-model"
    assert strategy == "use_fallback"
    assert len(list(mol.GetConformers())) == 1

    # Coordinates should match the model coordinates from the CIF
    _assert_coords_match(_get_coords_by_name(mol), EXPECTED_MODEL_COORDS)
    assert all(_get_used_atom_mask(mol))

    # The source PDB ID of the model coordinates should be preserved on the mol
    assert mol.GetProp("model_pdb_id") == "1EJG"


def test_allnan_fallback_source(monkeypatch, gly_mol):
    """Falls back to all-NaN conformer when no stored conformers exist."""
    _mock_conformer_generation_failure(monkeypatch)

    # Remove all conformers to simulate a mol with no stored fallback
    gly_mol.RemoveAllConformers()

    mol, strategy, source = resolve_and_format_fallback_conformer(gly_mol)

    assert source == "all-nan"
    assert strategy == "use_fallback"
    assert len(list(mol.GetConformers())) == 1

    # All coordinates should be 0.0 (NaN replaced with zeros)
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), 0.0)

    # All atoms should be masked as invalid (all-NaN -> all-False)
    assert not any(_get_used_atom_mask(mol))


def test_unexpected_conformer_name_raises(monkeypatch, gly_mol):
    """Raises ValueError when conformer has an unrecognized name."""
    _mock_conformer_generation_failure(monkeypatch)

    gly_mol.RemoveAllConformers()
    bad_conf = Chem.Conformer(gly_mol.GetNumAtoms())
    bad_conf.SetProp("name", "Unknown")
    for i in range(gly_mol.GetNumAtoms()):
        bad_conf.SetAtomPosition(i, (0.0, 0.0, 0.0))
    gly_mol.AddConformer(bad_conf, assignId=True)

    with pytest.raises(ValueError, match="Unexpected conformer name"):
        resolve_and_format_fallback_conformer(gly_mol)
