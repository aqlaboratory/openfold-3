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

import dataclasses
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
import pytest
from biotite.structure.io.pdbx import CIFFile
from func_timeout import FunctionTimedOut
from rdkit import Chem

from openfold3.core.data.primitives.structure.component import (
    mol_from_ccd_entry,
)
from openfold3.core.data.primitives.structure.conformer import (
    CONFORMER_STRATEGIES,
    ConformerGenerationError,
    ConformerResult,
    _compute_conformer,
    multistrategy_compute_conformer,
    resolve_and_format_fallback_conformer,
)


@pytest.fixture
def ethanol():
    return Chem.MolFromSmiles("CCO")


def _stub_with_outcomes(outcomes):
    """Build a side_effect callable that consumes `outcomes` in order.

    Each entry is either a return value `(mol, conf_id)` or a BaseException to raise.
    Note: FunctionTimedOut extends BaseException (not Exception).
    """
    iterator = iter(outcomes)

    def side_effect(mol, **kwargs):
        outcome = next(iterator)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return side_effect


def test_all_strategies_fail_raises(ethanol):
    timeout_exc = FunctionTimedOut("timed out")
    # One outcome per strategy in the chain.
    outcomes = [timeout_exc] * len(CONFORMER_STRATEGIES)
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer._compute_conformer",
            side_effect=_stub_with_outcomes(outcomes),
        ),
        pytest.raises(ConformerGenerationError) as excinfo,
    ):
        multistrategy_compute_conformer(ethanol)

    assert excinfo.value.__cause__ is timeout_exc


@pytest.mark.parametrize(
    ("start_from", "expected_strategy"),
    [
        pytest.param("default", "default", id="start_from_default"),
        pytest.param(
            "small_ring_torsions",
            "small_ring_torsions",
            id="start_from_small_ring_torsions",
        ),
        pytest.param(
            "random_init", "random_init", id="start_from_random_init_skips_earlier"
        ),
    ],
)
def test_start_from_slices_chain(ethanol, start_from, expected_strategy):
    with patch(
        "openfold3.core.data.primitives.structure.conformer._compute_conformer",
        side_effect=_stub_with_outcomes([(ethanol, 0)]),
    ) as mock_cc:
        result = multistrategy_compute_conformer(ethanol, start_from=start_from)

    assert result.strategy == expected_strategy
    assert mock_cc.call_count == 1
    expected_kwargs = next(
        s.kwargs for s in CONFORMER_STRATEGIES if s.name == start_from
    )
    for k, v in expected_kwargs.items():
        assert mock_cc.call_args.kwargs[k] == v


def test_start_from_unknown_raises(ethanol):
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer._compute_conformer",
            side_effect=_stub_with_outcomes([]),
        ),
        pytest.raises(ValueError, match="Unknown conformer strategy"),
    ):
        multistrategy_compute_conformer(ethanol, start_from="bogus")


def test_result_is_frozen_dataclass():
    result = ConformerResult(mol=Chem.MolFromSmiles("C"), conf_id=0, strategy="default")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.strategy = "random_init"


def test_strategy_names_match_cache_wire_values():
    # Hard constraint: cached `conformer_gen_strategy` strings are these literals.
    names = [s.name for s in CONFORMER_STRATEGIES]
    assert "default" in names
    assert "random_init" in names
    assert "small_ring_torsions" in names


def test_end_to_end_real_rdkit(ethanol):
    # No patching: exercises the real RDKit path on a small molecule.
    result = multistrategy_compute_conformer(ethanol)
    assert result.strategy in {s.name for s in CONFORMER_STRATEGIES}
    assert result.mol.GetNumConformers() >= 1
    result.mol.GetConformer(result.conf_id)  # would raise if id is invalid


# ----------------------------------------------------------------------------
# compute_conformer tests
# ----------------------------------------------------------------------------

# A fixed seed for the random-coord initializer so snapshots are reproducible
# regardless of where this test sits in the seeded_rng-derived global RNG state.
_FIXED_RDKIT_SEED = 424242


@pytest.mark.parametrize(
    ("smiles", "use_random_coord_init", "remove_hs"),
    [
        pytest.param("CCO", False, True, id="ethanol_default_no_hs"),
        pytest.param("CCO", True, True, id="ethanol_random_init_no_hs"),
        pytest.param("CCO", False, False, id="ethanol_default_keep_hs"),
        pytest.param("c1ccccc1", False, True, id="benzene_default_no_hs"),
        pytest.param("CC(=O)Nc1ccc(O)cc1", False, True, id="paracetamol_default_no_hs"),
    ],
)
def test_compute_conformer_snapshot(
    smiles, use_random_coord_init, remove_hs, assert_conformer_snapshot
):
    mol = Chem.MolFromSmiles(smiles)

    # Pin the RDKit seed (sourced from random.randint inside compute_conformer) so the
    # generated coordinates are reproducible across test runs and parametrize ordering.
    with patch(
        "openfold3.core.data.primitives.structure.conformer.random.randint",
        return_value=_FIXED_RDKIT_SEED,
    ):
        out_mol, conf_id = _compute_conformer(
            mol,
            use_random_coord_init=use_random_coord_init,
            remove_hs=remove_hs,
            timeout=None,
        )

    assert conf_id == 0
    elements = np.array([a.GetSymbol() for a in out_mol.GetAtoms()], dtype="U2")
    if remove_hs:
        assert "H" not in set(elements.tolist())
    else:
        assert "H" in set(elements.tolist())

    assert_conformer_snapshot(out_mol, conf_id)


@pytest.mark.parametrize(
    ("use_random_coord_init",),
    [
        pytest.param(False, id="default_init"),
        pytest.param(True, id="random_init"),
    ],
)
def test_compute_conformer_embed_returns_minus_one_raises(
    ethanol, use_random_coord_init
):
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer.AllChem.EmbedMolecule",
            return_value=-1,
        ),
        pytest.raises(ConformerGenerationError, match="Failed to generate"),
    ):
        _compute_conformer(
            ethanol,
            use_random_coord_init=use_random_coord_init,
            timeout=None,
        )


def test_compute_conformer_timeout_propagates(ethanol):
    timeout_exc = FunctionTimedOut("timed out")
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer.func_timeout",
            side_effect=timeout_exc,
        ) as mock_ft,
        pytest.raises(FunctionTimedOut),
    ):
        _compute_conformer(ethanol, timeout=0.001)

    assert mock_ft.call_count == 1
    # The wrapper should have been handed the actual EmbedMolecule callable + the
    # supplied timeout value.
    assert mock_ft.call_args.kwargs["timeout"] == 0.001


def test_compute_conformer_timeout_none_skips_func_timeout(ethanol):
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer.func_timeout"
        ) as mock_ft,
        patch(
            "openfold3.core.data.primitives.structure.conformer.AllChem.EmbedMolecule",
            return_value=0,
        ) as mock_embed,
    ):
        _compute_conformer(ethanol, timeout=None)

    mock_ft.assert_not_called()
    mock_embed.assert_called_once()


def test_compute_conformer_addhs_failure_is_non_fatal(ethanol, caplog):
    # AddHs raising must not abort conformer generation — it should log and proceed.
    with (
        patch(
            "openfold3.core.data.primitives.structure.conformer.Chem.AddHs",
            side_effect=RuntimeError("addhs blew up"),
        ),
        patch(
            "openfold3.core.data.primitives.structure.conformer.AllChem.EmbedMolecule",
            return_value=0,
        ),
        caplog.at_level("WARNING"),
    ):
        mol_out, conf_id = _compute_conformer(ethanol, timeout=None)

    assert conf_id == 0
    assert mol_out is not None
    assert any("Failed to add hydrogens" in rec.message for rec in caplog.records), (
        caplog.text
    )


def test_compute_conformer_empty_mol_raises():
    # An empty Mol has no atoms; RDKit's EmbedMolecule raises ValueError for that
    # rather than returning -1, so the error propagates as-is. Either way the
    # function must not return silently. ConformerGenerationError subclasses
    # ValueError, so this assertion accepts either.
    empty_mol = Chem.Mol()
    with pytest.raises(ValueError):
        _compute_conformer(empty_mol, timeout=None)


@pytest.mark.parametrize(
    "use_small_ring_torsions",
    [
        pytest.param(False, id="flag_off"),
        pytest.param(True, id="flag_on"),
    ],
)
def test_compute_conformer_small_ring_torsions_flag_propagates(
    ethanol, use_small_ring_torsions
):
    # Verify the kwarg actually flips the corresponding ETKDGv3 attribute on the
    # strategy object handed to EmbedMolecule.
    with patch(
        "openfold3.core.data.primitives.structure.conformer.AllChem.EmbedMolecule",
        return_value=0,
    ) as mock_embed:
        _compute_conformer(
            ethanol,
            use_small_ring_torsions=use_small_ring_torsions,
            timeout=None,
        )

    embed_strategy = mock_embed.call_args.args[1]
    assert embed_strategy.useSmallRingTorsions is use_small_ring_torsions


# ----------------------------------------------------------------------------
# PDB-CCD metallotetrapyrrole exercise of the multistrategy chain (real RDKit)
# ----------------------------------------------------------------------------

_DEFAULT_FAILURE_EXCEPTIONS = (ConformerGenerationError, FunctionTimedOut)


class _TetrapyrroleCase(NamedTuple):
    ccd_code: str
    smiles: str
    default_succeeds: bool


_PDB_TETRAPYRROLES = [
    pytest.param(
        *_TetrapyrroleCase(
            ccd_code="HEM",
            smiles=(
                "Cc1c2n3c(c1CCC(=O)O)C=C4C(=C(C5=[N]4[Fe]36[N]7=C(C=C8N6C(=C5)"
                "C(=C8C)C=C)C(=C(C7=C2)C)C=C)C)CCC(=O)O"
            ),
            default_succeeds=False,
        ),
        id="hem_fe_porphyrin",
    ),
    pytest.param(
        *_TetrapyrroleCase(
            ccd_code="CLA",
            smiles=(
                "CCC1=C(C2=Cc3c(c(c4n3[Mg]56[N]2=C1C=C7N5C8=C([C@H](C(=O)C8=C7C)"
                "C(=O)OC)C9=[N]6C(=C4)[C@H]([C@@H]9CCC(=O)OC/C=C(\\C)/CCC[C@H](C)"
                "CCC[C@H](C)CCCC(C)C)C)C)C=C)C"
            ),
            default_succeeds=True,
        ),
        id="cla_chlorophyll_a",
    ),
    pytest.param(
        *_TetrapyrroleCase(
            ccd_code="CHL",
            smiles=(
                "CCC1=C(c2cc3c(c(c4n3[Mg]56[n+]2c1cc7n5c8c(c9[n+]6c(c4)"
                "C(C9CCC(=O)OC/C=C(\\C)/CCC[C@H](C)CCC[C@H](C)CCCC(C)C)C)"
                "[C@H](C(=O)c8c7C)C(=O)OC)C)C=C)C=O"
            ),
            default_succeeds=True,
        ),
        id="chl_chlorophyll_b",
    ),
    pytest.param(
        *_TetrapyrroleCase(
            ccd_code="BCL",
            smiles=(
                "CC[C@@H]1[C@H](C2=CC3=C(C(=C4[N-]3[Mg+2]56[N]2=C1C=C7[N-]5C8="
                "C([C@H](C(=O)C8=C7C)C(=O)OC)C9=[N]6C(=C4)[C@H]([C@@H]9CCC(=O)OC"
                "/C=C(\\C)/CCC[C@H](C)CCC[C@H](C)CCCC(C)C)C)C)C(=O)C)C"
            ),
            default_succeeds=True,
        ),
        id="bcl_bacteriochlorophyll_a",
    ),
]


@pytest.mark.parametrize(("ccd_code", "smiles", "default_succeeds"), _PDB_TETRAPYRROLES)
def test_pdb_tetrapyrrole_default_strategy_outcome(ccd_code, smiles, default_succeeds):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"failed to parse PDB CCD {ccd_code!r} SMILES"

    if default_succeeds:
        mol_out, conf_id = _compute_conformer(
            mol, use_small_ring_torsions=False, timeout=15.0
        )
        assert conf_id == 0
        assert mol_out.GetNumConformers() >= 1
    else:
        with pytest.raises(_DEFAULT_FAILURE_EXCEPTIONS):
            _compute_conformer(mol, use_small_ring_torsions=False, timeout=5.0)


@pytest.mark.parametrize(("ccd_code", "smiles", "default_succeeds"), _PDB_TETRAPYRROLES)
def test_pdb_tetrapyrrole_small_ring_torsions_succeeds(
    ccd_code,
    smiles,
    default_succeeds,
):
    del (
        ccd_code,
        default_succeeds,
    )  # unused since small_ring_torsions should succeed for all cases
    mol = Chem.MolFromSmiles(smiles)
    # Pin the RDKit seed so the test is reproducible: tetrapyrroles are large
    # macrocycles and ETKDGv3 + small-ring-torsions has high variance — some
    # seeds succeed in < 5s, others time out or return -1. 60s headroom on top
    # because x86_64 CI under `pytest-xdist -n auto` is slower than local aarch64.
    with patch(
        "openfold3.core.data.primitives.structure.conformer.random.randint",
        return_value=_FIXED_RDKIT_SEED,
    ):
        mol_out, conf_id = _compute_conformer(
            mol, use_small_ring_torsions=True, timeout=60.0
        )
    assert conf_id == 0
    assert mol_out.GetNumConformers() >= 1
    coords = mol_out.GetConformer(conf_id).GetPositions()
    assert np.isfinite(coords).all()


@pytest.mark.parametrize(("ccd_code", "smiles", "default_succeeds"), _PDB_TETRAPYRROLES)
def test_pdb_tetrapyrrole_chain_resolves(ccd_code, smiles, default_succeeds):
    # Chain order is default → small_ring_torsions → random_init. Whichever
    # entry first delivers a conformer wins; for current RDKit that's `default`
    # on the chlorin/bacteriochlorin cores and `small_ring_torsions` on HEM.
    mol = Chem.MolFromSmiles(smiles)
    # Pin seed + 60s timeouts: see note on the small_ring_torsions test above.
    with patch(
        "openfold3.core.data.primitives.structure.conformer.random.randint",
        return_value=_FIXED_RDKIT_SEED,
    ):
        result = multistrategy_compute_conformer(
            mol,
            timeouts={
                "default": 60.0,
                "small_ring_torsions": 60.0,
                "random_init": 30.0,
            },
        )
    assert isinstance(result, ConformerResult)
    expected = "default" if default_succeeds else "small_ring_torsions"
    assert result.strategy == expected


# ===========================================================================
# resolve_and_format_fallback_conformer — fallback source annotation tests
# (complements the multistrategy_compute_conformer / _compute_conformer tests
# above; covers the 3-tuple fallback_conformer_source behavior)
# ===========================================================================
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
