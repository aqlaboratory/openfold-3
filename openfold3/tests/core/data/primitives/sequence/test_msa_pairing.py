"""Regression tests for paired MSA row ranking (GH-372)."""

import numpy as np

from openfold3.core.data.primitives.sequence.msa import sort_subsample_paired_row_ids


def test_subsample_paired_rows_uses_exact_product_order():
    rng = np.random.default_rng(0)
    n_rows, n_chains, max_rows_paired = 12_000, 6, 8191

    # Six chains with MSA row IDs large enough to overflow an int64 product.
    row_ids = rng.integers(1, 20_000, size=(n_rows, n_chains), dtype=np.int64)
    per_chain = {f"C{i}": row_ids[:, i].copy() for i in range(n_chains)}

    # Tag each row to track which ones survive sorting and truncation.
    species = np.zeros((n_rows, n_chains), dtype=np.int64)
    species[:, 0] = np.arange(n_rows)

    _, species_out, n_rows_actual = sort_subsample_paired_row_ids(
        per_chain, species, max_rows_paired
    )
    kept = set(species_out[:, 0].tolist())

    # Python integers preserve the intended product ranking without overflow.
    exact = np.prod(row_ids.astype(object), axis=1)
    kept_intended = set(np.argsort(exact, kind="stable")[:max_rows_paired].tolist())

    assert n_rows_actual == max_rows_paired
    assert len(kept & kept_intended) == max_rows_paired
