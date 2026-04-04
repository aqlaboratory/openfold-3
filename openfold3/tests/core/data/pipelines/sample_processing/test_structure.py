"""Tests for the sym_id legacy patch in process_target_structure_of3."""

from pathlib import Path
from unittest.mock import patch

import biotite.structure as struc
import numpy as np

MODULE = "openfold3.core.data.pipelines.sample_processing.structure"


def _make_atom_array(n_atoms, sym_ids=None):
    """Create a minimal AtomArray, optionally with a sym_id annotation."""
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


def _run_pipeline(atom_array):
    """Run process_target_structure_of3 with all downstream steps mocked out."""
    with (
        patch(f"{MODULE}.parse_target_structure", return_value=atom_array),
        patch(f"{MODULE}.assign_component_ids_from_metadata"),
        patch(f"{MODULE}.tokenize_atom_array"),
        patch(
            f"{MODULE}.crop_chainwise_and_set_crop_mask",
            return_value=(atom_array, "whole"),
        ),
        patch(
            f"{MODULE}.assign_mol_permutation_ids",
            side_effect=lambda aa, **kw: aa,
        ),
        patch(f"{MODULE}.assign_uniquified_atom_names", side_effect=lambda aa: aa),
    ):
        from openfold3.core.data.pipelines.sample_processing.structure import (
            process_target_structure_of3,
        )

        process_target_structure_of3(
            target_structures_directory=Path("/fake"),
            pdb_id="test",
            crop_config={"token_crop": {"enabled": False}},
            preferred_chain_or_interface=None,
            structure_format="npz",
            per_chain_metadata={},
        )


def test_sym_id_removed_when_dummy_values_present():
    """Legacy patch fires: sym_id with -1 values is removed."""
    aa = _make_atom_array(4, sym_ids=[0, 0, -1, -1])
    _run_pipeline(aa)
    assert "sym_id" not in aa.get_annotation_categories()


def test_sym_id_kept_when_no_dummy_values():
    """Legacy patch skipped: sym_id with only 0 values is kept."""
    aa = _make_atom_array(4, sym_ids=[0, 0, 0, 0])
    _run_pipeline(aa)
    assert "sym_id" in aa.get_annotation_categories()


def test_no_sym_id_annotation_is_fine():
    """No crash when sym_id annotation is absent entirely."""
    aa = _make_atom_array(4)
    _run_pipeline(aa)
    assert "sym_id" not in aa.get_annotation_categories()
