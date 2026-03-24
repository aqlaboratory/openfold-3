"""Tests for the sym_id legacy patch in process_target_structure_of3."""

from pathlib import Path
from unittest.mock import patch

import biotite.structure as struc
import numpy as np
import pytest


def _make_atom_array(n_atoms, sym_ids=None):
    """Create a minimal AtomArray with the given sym_id values."""
    aa = struc.AtomArray(n_atoms)
    aa.chain_id[:] = "A"
    aa.res_id[:] = np.arange(1, n_atoms + 1)
    aa.ins_code[:] = ""
    aa.res_name[:] = "ALA"
    aa.atom_name[:] = "CA"
    aa.element[:] = "C"
    aa.coord[:] = np.random.randn(n_atoms, 3)
    aa.set_annotation("token_id", np.arange(n_atoms))
    if sym_ids is not None:
        aa.set_annotation("sym_id", np.array(sym_ids))
    return aa


MODULE = "openfold3.core.data.pipelines.sample_processing.structure"


@pytest.mark.parametrize(
    "sym_ids, expect_removed",
    [
        ([0, 0, -1, -1], True),
        ([0, 0, 0, 0], False),
        (None, None),
    ],
    ids=[
        "stale_npz_with_dummy_sym_id",
        "fresh_npz_with_clean_sym_id",
        "no_sym_id_annotation",
    ],
)
@patch(f"{MODULE}.assign_uniquified_atom_names", side_effect=lambda aa: aa)
@patch(f"{MODULE}.assign_mol_permutation_ids", side_effect=lambda aa, **kw: aa)
@patch(f"{MODULE}.crop_chainwise_and_set_crop_mask")
@patch(f"{MODULE}.tokenize_atom_array")
@patch(f"{MODULE}.assign_component_ids_from_metadata")
@patch(f"{MODULE}.parse_target_structure")
def test_sym_id_legacy_patch(
    mock_parse, _comp, _tok, mock_crop, _perm, _uniq, sym_ids, expect_removed
):
    from openfold3.core.data.pipelines.sample_processing.structure import (
        process_target_structure_of3,
    )

    aa = _make_atom_array(4, sym_ids=sym_ids)
    mock_parse.return_value = aa
    mock_crop.return_value = (aa, "whole")

    process_target_structure_of3(
        target_structures_directory=Path("/fake"),
        pdb_id="test",
        crop_config={"token_crop": {"enabled": False}},
        preferred_chain_or_interface=None,
        structure_format="npz",
        per_chain_metadata={},
    )

    has_sym_id = "sym_id" in aa.get_annotation_categories()

    if expect_removed is True:
        assert not has_sym_id, "sym_id should have been removed (stale NPZ)"
    elif expect_removed is False:
        assert has_sym_id, "sym_id should be kept (clean NPZ)"
    else:
        assert not has_sym_id, "sym_id was never present"
