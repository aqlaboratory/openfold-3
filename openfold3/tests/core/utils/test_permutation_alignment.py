"""Tests for the defensive size check in find_greedy_optimal_mol_permutation."""

import biotite.structure as struc
import numpy as np
import pytest
import torch

from openfold3.core.utils.permutation_alignment import (
    find_greedy_optimal_mol_permutation,
)


def _make_homodimer_inputs(n_tokens_a, n_tokens_b):
    """Build permutation alignment inputs mimicking a homodimer.

    Creates two symmetric instances of the same entity with the given token
    counts. When n_tokens_a != n_tokens_b this reproduces the mismatch caused
    by the sym_id residue-splitting bug.
    """
    # Build a minimal atom array with two chains
    n_atoms_a, n_atoms_b = n_tokens_a, n_tokens_b  # 1 atom per token
    n_total = n_atoms_a + n_atoms_b

    aa = struc.AtomArray(n_total)
    aa.chain_id[:n_atoms_a] = "A"
    aa.chain_id[n_atoms_a:] = "B"
    aa.res_id[:n_atoms_a] = np.arange(1, n_atoms_a + 1)
    aa.res_id[n_atoms_a:] = np.arange(1, n_atoms_b + 1)
    aa.ins_code[:] = ""
    aa.res_name[:] = "ALA"
    aa.atom_name[:] = "CA"
    aa.element[:] = "C"
    aa.coord[:] = np.random.randn(n_total, 3)

    # Derive tensor inputs from the atom array
    entity_ids = torch.ones(n_total, dtype=torch.long)
    sym_ids = torch.tensor([1] * n_atoms_a + [2] * n_atoms_b)
    sym_token_index = torch.tensor(list(range(n_tokens_a)) + list(range(n_tokens_b)))
    gt_coords = torch.tensor(aa.coord, dtype=torch.float32).unsqueeze(0)
    gt_resolved = torch.ones(n_total)
    pred_coords = torch.randn(n_total, 3)

    return dict(
        gt_token_center_positions_transformed=gt_coords,
        gt_token_center_resolved_mask=gt_resolved,
        gt_mol_entity_ids=entity_ids,
        gt_mol_sym_ids=sym_ids,
        gt_mol_sym_token_index=sym_token_index,
        pred_token_center_positions=pred_coords,
        pred_mol_entity_ids=entity_ids.clone(),
        pred_mol_sym_ids=sym_ids.clone(),
        pred_mol_sym_token_index=sym_token_index.clone(),
    )


def test_mismatched_token_counts_raises():
    """Symmetric instances with different token counts raise a clear error.

    This reproduces the crash caused by partially-unresolved residues where
    biotite's get_residue_starts() splits residues on sym_id boundaries,
    producing different token counts for symmetric chains.
    """
    inputs = _make_homodimer_inputs(n_tokens_a=5, n_tokens_b=3)

    with pytest.raises(
        ValueError,
        match="symmetric instances with different token counts",
    ):
        find_greedy_optimal_mol_permutation(**inputs)


def test_matched_token_counts_succeeds():
    """Symmetric instances with equal token counts produce a valid mapping."""
    inputs = _make_homodimer_inputs(n_tokens_a=4, n_tokens_b=4)

    result = find_greedy_optimal_mol_permutation(**inputs)

    assert isinstance(result, dict)
    assert len(result) == 2
