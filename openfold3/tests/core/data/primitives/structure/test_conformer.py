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
from func_timeout import FunctionTimedOut
from rdkit import Chem

from openfold3.core.data.primitives.structure.conformer import (
    CONFORMER_STRATEGIES,
    ConformerGenerationError,
    ConformerResult,
    _compute_conformer,
    multistrategy_compute_conformer,
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
    smiles, use_random_coord_init, remove_hs, ndarrays_regression
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
    coords = out_mol.GetConformer(conf_id).GetPositions().astype(np.float64)
    elements = np.array([a.GetSymbol() for a in out_mol.GetAtoms()], dtype="U2")
    if remove_hs:
        assert "H" not in set(elements.tolist())
    else:
        assert "H" in set(elements.tolist())

    ndarrays_regression.check(
        {"coords": coords},
        # ETKDGv3 is deterministic for a fixed seed within one RDKit build, but
        # numerics drift across RDKit minor versions and CPU archs (aarch64 vs
        # x86_64). Observed drift on paracetamol is ~1.5e-3 Å; give it headroom
        # while staying ≪ chemistry scale (~0.1 Å) to still catch real regressions.
        default_tolerance=dict(atol=5e-3, rtol=5e-3),
    )


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
