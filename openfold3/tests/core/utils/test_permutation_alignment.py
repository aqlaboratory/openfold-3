"""Tests for the defensive size check in find_greedy_optimal_mol_permutation."""

from contextlib import nullcontext

import pytest
import torch

from openfold3.core.utils.permutation_alignment import (
    find_greedy_optimal_mol_permutation,
)


def _make_symmetric_inputs(n_tokens_per_instance):
    """Create permutation alignment inputs for one entity with two sym instances.

    Args:
        n_tokens_per_instance: list of two ints, token count per sym instance.
            Equal values = valid input; unequal = triggers the size mismatch error.
    """
    n_a, n_b = n_tokens_per_instance
    n_total = n_a + n_b

    entity_ids = torch.ones(n_total, dtype=torch.long)
    sym_ids = torch.tensor([1] * n_a + [2] * n_b)
    sym_token_index = torch.tensor(list(range(n_a)) + list(range(n_b)))
    gt_coords = torch.randn(1, n_total, 3)
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


@pytest.mark.parametrize(
    "token_counts, expectation",
    [
        (
            [5, 3],
            pytest.raises(
                ValueError, match="symmetric instances with different token counts"
            ),
        ),
        ([4, 4], nullcontext()),
    ],
    ids=["mismatched", "matched"],
)
def test_symmetric_instance_token_counts(token_counts, expectation):
    """Symmetric instances must have equal token counts to be stackable."""
    inputs = _make_symmetric_inputs(token_counts)

    with expectation:
        result = find_greedy_optimal_mol_permutation(**inputs)

    if isinstance(expectation, nullcontext):
        assert isinstance(result, dict)
        assert len(result) == 2
