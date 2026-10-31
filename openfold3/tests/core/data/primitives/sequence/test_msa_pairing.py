"""Regression tests for paired MSA row ranking"""

import math
from dataclasses import dataclass
from unittest.mock import Mock, patch

import numpy as np
import pytest

from openfold3.core.data.primitives.sequence.msa import (
    sort_by_row_id_product,
    sort_subsample_paired_row_ids,
)


@dataclass(frozen=True)
class PairedRowCase:
    """Inputs defining one paired-row ranking case.

    Attributes:
        n_chains (int):
            Number of chains (columns) in the synthetic row IDs.
        max_row_id (int):
            Exclusive upper bound for the sampled row IDs.
        unpaired_counts (tuple[int, ...]):
            Number of trailing -1 entries each pairing block gets.
    """

    n_chains: int
    max_row_id: int
    unpaired_counts: tuple[int, ...]


@dataclass(frozen=True)
class PairedRowCaseData:
    """Generated inputs for a case plus the expected ranking.

    Attributes:
        row_ids (np.ndarray):
            The (n_rows, n_chains) row IDs the case is built from.
        per_chain (dict[str, np.ndarray]):
            Per-chain views passed to the sorter.
        species (np.ndarray):
            Row tags tracking which rows survive sorting and truncation.
        expected_order (np.ndarray):
            Expected sort order from the Python-integer oracle.
        expected_dtypes (list[np.dtype]):
            Expected np.prod dtype per pairing block.
    """

    row_ids: np.ndarray
    per_chain: dict[str, np.ndarray]
    species: np.ndarray
    expected_order: np.ndarray
    expected_dtypes: list[np.dtype]


# Cover int64-safe products, rare and widespread overflow, and overflow that
# changes ordering without changing the retained set. Partial rows exercise
# odd/even -1 counts and mixed safe/overflowing pairing blocks.
PAIRED_ROW_CASES = [
    pytest.param(
        PairedRowCase(n_chains=3, max_row_id=2000, unpaired_counts=(0,)),
        id="int64-safe",
    ),
    pytest.param(
        PairedRowCase(n_chains=4, max_row_id=60_000, unpaired_counts=(0,)),
        id="four-chains-rare-overflow",
    ),
    pytest.param(
        PairedRowCase(n_chains=5, max_row_id=7000, unpaired_counts=(0,)),
        id="five-chains-rare-overflow",
    ),
    pytest.param(
        PairedRowCase(n_chains=6, max_row_id=20_000, unpaired_counts=(0,)),
        id="issue-reproducer",
    ),
    pytest.param(
        PairedRowCase(n_chains=6, max_row_id=2000, unpaired_counts=(0,)),
        id="six-chains-near-overflow",
    ),
    pytest.param(
        PairedRowCase(n_chains=8, max_row_id=300, unpaired_counts=(0,)),
        id="eight-chains-order-only-overflow",
    ),
    pytest.param(
        PairedRowCase(n_chains=6, max_row_id=20_000, unpaired_counts=(1,)),
        id="partial-odd-unpaired",
    ),
    pytest.param(
        PairedRowCase(n_chains=8, max_row_id=20_000, unpaired_counts=(2,)),
        id="partial-even-unpaired",
    ),
    pytest.param(
        PairedRowCase(n_chains=7, max_row_id=2000, unpaired_counts=(1,)),
        id="partial-near-overflow",
    ),
    pytest.param(
        PairedRowCase(n_chains=8, max_row_id=20_000, unpaired_counts=tuple(range(9))),
        id="mixed-pairing-blocks",
    ),
]

SEEDS = [0, 1, 2]


def build_paired_row_case(spec: PairedRowCase, seed: int) -> PairedRowCaseData:
    """Generate the row IDs for a case plus its expected ranking."""
    n_rows = 12_000
    rng = np.random.default_rng(seed)
    row_ids = rng.integers(
        1, spec.max_row_id, size=(n_rows, spec.n_chains), dtype=np.int64
    )
    n_unpaired = np.resize(spec.unpaired_counts, n_rows)
    for count in spec.unpaired_counts:
        if count:
            row_ids[n_unpaired == count, -count:] = -1
    per_chain = {f"C{i}": row_ids[:, i].copy() for i in range(spec.n_chains)}

    # Tag each row to track which ones survive sorting and truncation.
    species = np.zeros((n_rows, spec.n_chains), dtype=np.int64)
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

    return PairedRowCaseData(
        row_ids=row_ids,
        per_chain=per_chain,
        species=species,
        expected_order=expected_order,
        expected_dtypes=expected_dtypes,
    )


@pytest.fixture
def product_dtypes():
    """Records the dtype of every array passed to `np.prod` during a test."""
    recorded = []
    original_prod = np.prod

    def record_dtype(array, *args, **kwargs):
        # Numpy also calls np.prod on non-array inputs internally (e.g. on shape
        # tuples from Generator.integers), so only record real array inputs.
        if not isinstance(array, np.ndarray):
            return original_prod(array, *args, **kwargs)
        dtype = kwargs.get("dtype")
        recorded.append(np.dtype(dtype if dtype is not None else array.dtype))
        return original_prod(array, *args, **kwargs)

    with patch.object(np, "prod", Mock(side_effect=record_dtype)):
        yield recorded


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("spec", PAIRED_ROW_CASES)
def test_sort_paired_rows_uses_exact_product_order(spec, seed, product_dtypes):
    case = build_paired_row_case(spec, seed)
    sorted_rows, sorted_species = sort_by_row_id_product(
        case.per_chain, case.species.copy()
    )
    np.testing.assert_array_equal(sorted_species, case.species[case.expected_order])
    for chain_idx, chain_id in enumerate(case.per_chain):
        np.testing.assert_array_equal(
            sorted_rows[chain_id], case.row_ids[case.expected_order, chain_idx]
        )
    assert product_dtypes == case.expected_dtypes


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("spec", PAIRED_ROW_CASES)
def test_subsample_paired_rows_uses_exact_product_order(spec, seed, product_dtypes):
    case = build_paired_row_case(spec, seed)
    max_rows_paired = 8191
    rows_out, species_out, n_rows_actual = sort_subsample_paired_row_ids(
        case.per_chain, case.species.copy(), max_rows_paired
    )
    assert n_rows_actual == max_rows_paired
    expected_kept = case.expected_order[:max_rows_paired]
    assert set(species_out[:, 0]) == set(expected_kept)
    np.testing.assert_array_equal(species_out, case.species[expected_kept])
    for chain_idx, chain_id in enumerate(case.per_chain):
        np.testing.assert_array_equal(
            rows_out[chain_id], case.row_ids[expected_kept, chain_idx]
        )
    assert product_dtypes == case.expected_dtypes


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
