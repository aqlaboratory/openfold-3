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

"""Tests for `get_processed_reference_conformer`.

Two layers:
  - Unit tests with deps mocked, exercising orchestration / branching only.
  - Integration test with real RDKit + biotite AtomArray, end-to-end.
"""

from unittest.mock import patch

import numpy as np
import pytest
from biotite.structure import AtomArray
from func_timeout import FunctionTimedOut
from rdkit import Chem
from rdkit.Chem import AllChem

from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
    get_processed_reference_conformer,
)
from openfold3.core.data.primitives.structure.component import set_atomwise_annotation
from openfold3.core.data.primitives.structure.conformer import (
    ConformerGenerationError,
    ConformerResult,
)

# ----------------------------------------------------------------------------
# Helpers / Fixtures
# ----------------------------------------------------------------------------


_ACETIC_ACID_ATOM_NAMES = ["C1", "C2", "O1", "O2"]


def _build_acetic_acid_mol(atom_names=_ACETIC_ACID_ATOM_NAMES, conformer_seed=42):
    """RDKit Mol of acetic acid with `annot_atom_name` and a single 3D conformer.

    Mirrors the shape of a reference molecule produced by
    `resolve_and_format_fallback_conformer`: heavy-atom-only mol with one stored
    "fallback" conformer and per-atom `annot_atom_name` properties.
    """
    mol = Chem.MolFromSmiles("CC(=O)O")
    assert mol.GetNumAtoms() == len(atom_names)
    embed_status = AllChem.EmbedMolecule(mol, randomSeed=conformer_seed)
    assert embed_status == 0
    set_atomwise_annotation(mol, "atom_name", atom_names)
    return mol


def _build_acetic_acid_atom_array(
    atom_names=_ACETIC_ACID_ATOM_NAMES,
    *,
    crop_mask=None,
    component_id=0,
):
    """Biotite AtomArray with the fields `get_processed_reference_conformer` reads.

    Specifically: `component_id`, `atom_name_unique`, `crop_mask`. Already-unique
    atom names get `_1` appended (mirroring `uniquify_ids`).
    """
    n = len(atom_names)
    aa = AtomArray(n)
    aa.coord = np.zeros((n, 3))
    aa.chain_id[:] = "A"
    aa.res_id[:] = 1
    aa.res_name[:] = "ACT"
    aa.atom_name[:] = atom_names
    aa.set_annotation("component_id", np.full(n, fill_value=component_id, dtype=int))
    aa.set_annotation(
        "atom_name_unique",
        np.array([f"{name}_1" for name in atom_names], dtype=object),
    )
    aa.set_annotation(
        "crop_mask",
        np.ones(n, dtype=bool)
        if crop_mask is None
        else np.asarray(crop_mask, dtype=bool),
    )
    return aa


@pytest.fixture
def acetic_acid_mol():
    return _build_acetic_acid_mol()


@pytest.fixture
def acetic_acid_atom_array():
    return _build_acetic_acid_atom_array(component_id=7)


# Path to the symbol *as imported into the module under test* — patching here
# replaces the binding `get_processed_reference_conformer` actually calls.
_MULTISTRATEGY_PATH = (
    "openfold3.core.data.pipelines.sample_processing.conformer"
    ".multistrategy_compute_conformer"
)


# ----------------------------------------------------------------------------
# Unit tests — orchestration / branching with deps mocked
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("preferred_confgen_strategy", "expect_multistrategy_call"),
    [
        pytest.param("default", True, id="default_calls_multistrategy"),
        pytest.param(
            "small_ring_torsions", True, id="small_ring_torsions_calls_multistrategy"
        ),
        pytest.param("random_init", True, id="random_init_calls_multistrategy"),
        pytest.param("use_fallback", False, id="use_fallback_skips_multistrategy"),
    ],
)
def test_unit_strategy_dispatch(
    acetic_acid_mol,
    acetic_acid_atom_array,
    preferred_confgen_strategy,
    expect_multistrategy_call,
):
    # `preferred_confgen_strategy` either flows through verbatim as `start_from`,
    # or — for `"use_fallback"` — short-circuits the regeneration entirely.
    fake_result = ConformerResult(
        mol=Chem.Mol(acetic_acid_mol),
        conf_id=0,
        strategy=preferred_confgen_strategy,
    )
    with patch(_MULTISTRATEGY_PATH, return_value=fake_result) as mock_ms:
        out = get_processed_reference_conformer(
            mol=acetic_acid_mol,
            mol_atom_array=acetic_acid_atom_array,
            preferred_confgen_strategy=preferred_confgen_strategy,
        )

    if expect_multistrategy_call:
        assert mock_ms.call_count == 1
        assert mock_ms.call_args.kwargs["start_from"] == preferred_confgen_strategy
        # `remove_hs=True` is the contract this caller has with multistrategy —
        # guarded so we don't accidentally invert it.
        assert mock_ms.call_args.kwargs["remove_hs"] is True
    else:
        mock_ms.assert_not_called()

    assert isinstance(out, ProcessedReferenceMolecule)
    assert out.component_id == 7
    assert out.mol.GetNumConformers() == 1


@pytest.mark.parametrize(
    "exception",
    [
        pytest.param(ConformerGenerationError("regen failed"), id="conformer_error"),
        pytest.param(FunctionTimedOut("regen timed out"), id="timeout"),
    ],
)
def test_unit_regen_failure_keeps_fallback_conformer(
    acetic_acid_mol, acetic_acid_atom_array, exception
):
    # Both exception types from multistrategy are suppressed so the stored
    # fallback conformer can stand in. component_id / in_crop_mask still populated.
    with patch(_MULTISTRATEGY_PATH, side_effect=exception):
        out = get_processed_reference_conformer(
            mol=acetic_acid_mol,
            mol_atom_array=acetic_acid_atom_array,
            preferred_confgen_strategy="default",
        )

    assert out.mol.GetNumConformers() == 1
    assert out.component_id == 7
    assert out.in_crop_mask.tolist() == [True, True, True, True]


@pytest.mark.parametrize(
    ("crop_mask", "expected_in_crop_mask"),
    [
        pytest.param(
            [True, True, True, True], [True, True, True, True], id="all_in_crop"
        ),
        pytest.param(
            [True, True, False, False],
            [True, True, False, False],
            id="first_half_in_crop",
        ),
        pytest.param(
            [False, True, True, False], [False, True, True, False], id="middle_in_crop"
        ),
        pytest.param(
            [False, False, False, True],
            [False, False, False, True],
            id="single_atom_in_crop",
        ),
    ],
)
def test_unit_in_crop_mask_and_permutations_track_crop(
    acetic_acid_mol, crop_mask, expected_in_crop_mask
):
    aa = _build_acetic_acid_atom_array(crop_mask=crop_mask)
    with patch(_MULTISTRATEGY_PATH) as mock_ms:
        out = get_processed_reference_conformer(
            mol=acetic_acid_mol,
            mol_atom_array=aa,
            preferred_confgen_strategy="use_fallback",
        )

    mock_ms.assert_not_called()
    assert out.in_crop_mask.tolist() == expected_in_crop_mask
    # Permutations are over in-crop atoms only — width matches the in-crop count.
    assert out.permutations.shape[1] == sum(crop_mask)


def test_unit_set_fallback_to_nan_marks_all_atoms_unused(
    acetic_acid_mol, acetic_acid_atom_array
):
    # set_fallback_to_nan=True with use_fallback strategy → mol coordinates are
    # painted all-NaN (then written back as 0 since SDF can't store NaN), and
    # `annot_used_atom_mask` is all-False.
    out = get_processed_reference_conformer(
        mol=acetic_acid_mol,
        mol_atom_array=acetic_acid_atom_array,
        preferred_confgen_strategy="use_fallback",
        set_fallback_to_nan=True,
    )

    used_mask = [a.GetBoolProp("annot_used_atom_mask") for a in out.mol.GetAtoms()]
    assert used_mask == [False] * acetic_acid_mol.GetNumAtoms()


def test_unit_atom_reorder_when_mol_order_differs_from_atom_array(
    acetic_acid_atom_array,
):
    # Build a mol whose annot_atom_name order is the REVERSE of the atom_array's
    # order. The function must reorder mol atoms via `get_name_match_argsort` so
    # that the output mol's atoms align with the atom_array.
    mol = Chem.MolFromSmiles("CC(=O)O")
    AllChem.EmbedMolecule(mol, randomSeed=11)
    set_atomwise_annotation(mol, "atom_name", list(reversed(_ACETIC_ACID_ATOM_NAMES)))

    with patch(_MULTISTRATEGY_PATH) as mock_ms:
        out = get_processed_reference_conformer(
            mol=mol,
            mol_atom_array=acetic_acid_atom_array,
            preferred_confgen_strategy="use_fallback",
        )

    mock_ms.assert_not_called()
    out_atom_names = [a.GetProp("annot_atom_name") for a in out.mol.GetAtoms()]
    # After reorder, output atom names match the atom_array's (forward) order.
    assert out_atom_names == _ACETIC_ACID_ATOM_NAMES


# ----------------------------------------------------------------------------
# Integration test — end-to-end on real RDKit + biotite, no mocks
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("preferred_confgen_strategy", "coords_should_match_fallback"),
    [
        pytest.param("default", False, id="default_replaces_fallback"),
        pytest.param("use_fallback", True, id="use_fallback_keeps_coords"),
    ],
)
def test_integration_strategy_effect_on_coords(
    preferred_confgen_strategy, coords_should_match_fallback
):
    # Build inputs end-to-end and call with no patches.
    # For "default" the regenerated coords should *differ* from the stored
    # fallback (proving multistrategy actually fired). For "use_fallback" they
    # should be byte-identical.
    mol = _build_acetic_acid_mol(conformer_seed=42)
    fallback_coords = mol.GetConformer(0).GetPositions().copy()

    aa = _build_acetic_acid_atom_array()  # all-True crop_mask, single component

    out = get_processed_reference_conformer(
        mol=mol,
        mol_atom_array=aa,
        preferred_confgen_strategy=preferred_confgen_strategy,
    )

    assert isinstance(out, ProcessedReferenceMolecule)
    assert out.component_id == 0
    assert out.in_crop_mask.tolist() == [True, True, True, True]
    assert out.mol.GetNumConformers() == 1

    out_coords = out.mol.GetConformer(0).GetPositions()
    assert np.isfinite(out_coords).all()

    if coords_should_match_fallback:
        np.testing.assert_allclose(out_coords, fallback_coords, atol=1e-6)
    else:
        assert not np.allclose(out_coords, fallback_coords)

    # annot_atom_name preserved through reorder; permutations are real.
    out_names = [a.GetProp("annot_atom_name") for a in out.mol.GetAtoms()]
    assert out_names == _ACETIC_ACID_ATOM_NAMES
    assert out.permutations.shape == (out.permutations.shape[0], 4)
    assert out.permutations.shape[0] >= 1
