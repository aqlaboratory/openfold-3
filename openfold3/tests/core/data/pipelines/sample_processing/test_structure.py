"""Tests for the sym_id legacy patch in process_target_structure_of3."""

from pathlib import Path
from unittest.mock import patch

MODULE = "openfold3.core.data.pipelines.sample_processing.structure"


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


def test_sym_id_removed_when_dummy_values_present(_make_atom_array):
    """Legacy patch fires: sym_id with -1 values is removed."""
    aa = _make_atom_array(4, sym_ids=[0, 0, -1, -1])
    _run_pipeline(aa)
    assert "sym_id" not in aa.get_annotation_categories()


def test_sym_id_kept_when_no_dummy_values(_make_atom_array):
    """Legacy patch skipped: sym_id with only 0 values is kept."""
    aa = _make_atom_array(4, sym_ids=[0, 0, 0, 0])
    _run_pipeline(aa)
    assert "sym_id" in aa.get_annotation_categories()


def test_no_sym_id_annotation_is_fine(_make_atom_array):
    """No crash when sym_id annotation is absent entirely."""
    aa = _make_atom_array(4)
    _run_pipeline(aa)
    assert "sym_id" not in aa.get_annotation_categories()
