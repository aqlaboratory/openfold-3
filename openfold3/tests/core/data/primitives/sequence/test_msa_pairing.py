"""Regression tests for paired MSA row ranking"""

import math

import numpy as np
import pytest

from openfold3.core.data.primitives.sequence.msa import (
    sort_by_row_id_product,
    sort_subsample_paired_row_ids,
)


@pytest.fixture(params=[0, 1, 2])
def seed(request):
    return request.param


@pytest.fixture(
    params=[
        # Cover int64-safe products, rare and widespread overflow, and overflow
        # that changes ordering without changing the retained set. Partial rows
        # exercise odd/even -1 counts and mixed safe/overflowing pairing blocks.
        pytest.param((3, 2000, (0,)), id="int64-safe"),
        pytest.param((4, 60_000, (0,)), id="four-chains-rare-overflow"),
        pytest.param((5, 7000, (0,)), id="five-chains-rare-overflow"),
        pytest.param((6, 20_000, (0,)), id="issue-reproducer"),
        pytest.param((6, 2000, (0,)), id="six-chains-near-overflow"),
        pytest.param((8, 300, (0,)), id="eight-chains-order-only-overflow"),
        pytest.param((6, 20_000, (1,)), id="partial-odd-unpaired"),
        pytest.param((8, 20_000, (2,)), id="partial-even-unpaired"),
        pytest.param((7, 2000, (1,)), id="partial-near-overflow"),
        pytest.param((8, 20_000, tuple(range(9))), id="mixed-pairing-blocks"),
    ]
)
def paired_row_case(request, seed, monkeypatch):
    n_chains, max_row_id, unpaired_counts = request.param
    rng = np.random.default_rng(seed)
    n_rows = 12_000
    row_ids = rng.integers(1, max_row_id, size=(n_rows, n_chains), dtype=np.int64)
    n_unpaired = np.resize(unpaired_counts, n_rows)
    for count in unpaired_counts:
        if count:
            row_ids[n_unpaired == count, -count:] = -1
    per_chain = {f"C{i}": row_ids[:, i].copy() for i in range(n_chains)}

    # Tag each row to track which ones survive sorting and truncation.
    species = np.zeros((n_rows, n_chains), dtype=np.int64)
    species[:, 0] = np.arange(n_rows)

    # Independent Python-integer oracle; sort each block in its original slots.
    exact = np.array([abs(math.prod(map(int, row))) for row in row_ids], dtype=object)
    expected_order = np.arange(n_rows)
    expected_dtypes = []
    for count in np.unique(n_unpaired):
        positions = np.flatnonzero(n_unpaired == count)
        keys = exact[positions]
        dtype = np.int64 if max(keys) <= np.iinfo(np.int64).max else object
        expected_dtypes.append(np.dtype(dtype))
        # Preserve the existing dtype-specific argsort behavior for equal keys;
        # the sorter does not promise stable ties.
        expected_order[positions] = positions[np.argsort(keys.astype(dtype))]

    product_dtypes = []
    original_prod = np.prod

    def track_product_dtype(array, *args, **kwargs):
        dtype = kwargs.get("dtype")
        product_dtypes.append(np.dtype(dtype if dtype is not None else array.dtype))
        return original_prod(array, *args, **kwargs)

    monkeypatch.setattr(np, "prod", track_product_dtype)
    return row_ids, per_chain, species, expected_order, expected_dtypes, product_dtypes


def test_sort_paired_rows_uses_exact_product_order(paired_row_case):
    row_ids, per_chain, species, expected_order, expected_dtypes, product_dtypes = (
        paired_row_case
    )
    sorted_rows, sorted_species = sort_by_row_id_product(per_chain, species.copy())
    np.testing.assert_array_equal(sorted_species, species[expected_order])
    for chain_idx, chain_id in enumerate(per_chain):
        np.testing.assert_array_equal(
            sorted_rows[chain_id], row_ids[expected_order, chain_idx]
        )
    assert product_dtypes == expected_dtypes


def test_subsample_paired_rows_uses_exact_product_order(paired_row_case):
    row_ids, per_chain, species, expected_order, expected_dtypes, product_dtypes = (
        paired_row_case
    )
    max_rows_paired = 8191
    rows_out, species_out, n_rows_actual = sort_subsample_paired_row_ids(
        per_chain, species.copy(), max_rows_paired
    )
    assert n_rows_actual == max_rows_paired
    expected_kept = expected_order[:max_rows_paired]
    assert set(species_out[:, 0]) == set(expected_kept)
    np.testing.assert_array_equal(species_out, species[expected_kept])
    for chain_idx, chain_id in enumerate(per_chain):
        np.testing.assert_array_equal(
            rows_out[chain_id], row_ids[expected_kept, chain_idx]
        )
    assert product_dtypes == expected_dtypes


@pytest.mark.benchmark(group="paired-msa-ranking")
@pytest.mark.parametrize(
    "n_chains,max_row_id",
    [
        pytest.param(3, 2000, id="int64-safe"),
        pytest.param(6, 20_000, id="overflow-prone"),
    ],
)
def test_subsample_paired_rows_benchmark(benchmark, n_chains, max_row_id):
    """Benchmark truncation through the int64 and Python-integer paths."""
    rng = np.random.default_rng(0)
    n_rows, max_rows_paired = 12_000, 8191
    row_ids = rng.integers(1, max_row_id, size=(n_rows, n_chains), dtype=np.int64)
    species = np.zeros((n_rows, n_chains), dtype=np.int64)
    species[:, 0] = np.arange(n_rows)

    def setup():
        # Reset inputs outside timing: sorting permutes species in place.
        per_chain = {f"C{i}": row_ids[:, i].copy() for i in range(n_chains)}
        return (per_chain, species.copy(), max_rows_paired), {}

    rows_out, species_out, n_rows_actual = benchmark.pedantic(
        sort_subsample_paired_row_ids,
        setup=setup,
        rounds=50,
        iterations=1,
        warmup_rounds=5,
    )

    assert n_rows_actual == max_rows_paired
    assert species_out.shape == (max_rows_paired, n_chains)
    assert all(rows.shape == (max_rows_paired,) for rows in rows_out.values())
