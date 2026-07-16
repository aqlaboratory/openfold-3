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

"""Tests for calculate_profile function.

Regression coverage for a column-misattribution bug in the chunked bincount
implementation: `msa_chunk.ravel()` flattens each chunk row-major (walks
columns fastest, within each row), but `col_indices_local` was built with
`np.repeat(np.arange(block_n_cols), n_rows)`, which assumes column-major
grouping (walks rows fastest, within each column). The two orderings
disagree, so counts get accumulated under the wrong column.

The correct construction is `np.tile(np.arange(block_n_cols), n_rows)`,
which reproduces the same row-major reading order `.ravel()` used on the
real data (see test_calculate_profile_matches_hand_computed_reference and
the module-level note in each test for why).

Critically: for every entry that gets misattributed, it is still counted
*somewhere* -- so `counts[col, :].sum() == n_rows` holds for every column
regardless of whether the implementation is correct. A test that only
checks row sums (or output shape) will not catch this bug; the tests below
check the actual per-column, per-symbol values against independently
constructed ground truth.
"""

import numpy as np
import pytest

from openfold3.core.data.primitives.sequence.msa import calculate_profile
from openfold3.core.data.resources.residues import (
    STANDARD_RESIDUES_WITH_GAP_1,
    MoleculeType,
    map_str_array_to_idx_array,
)

N_SYMBOLS = len(STANDARD_RESIDUES_WITH_GAP_1)
_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"  # 20 standard AAs + gap; all valid for PROTEIN


def _naive_profile_reference(
    msa_array: np.ndarray, molecule_type: MoleculeType
) -> np.ndarray:
    """Obviously-correct, unchunked, un-vectorized reference implementation.

    Deliberately structured nothing like calculate_profile (no ravel, no
    bincount-with-offset trick, no chunking) so it can't share its bug.
    Column c's counts come directly from column c and nowhere else, by
    construction.
    """
    msa_index = map_str_array_to_idx_array(msa_array, molecule_type)
    n_rows, n_cols = msa_index.shape
    counts = np.zeros((n_cols, N_SYMBOLS), dtype=int)
    for c in range(n_cols):
        counts[c] = np.bincount(msa_index[:, c], minlength=N_SYMBOLS)
    return counts / n_rows


def _random_msa(rng: np.random.Generator, n_rows: int, n_cols: int) -> np.ndarray:
    letters = rng.choice(list(_ALPHABET), size=(n_rows, n_cols))
    return letters.astype("<U1")


def test_calculate_profile_matches_hand_computed_reference():
    """Small, fully hand-verifiable example -- no trust in any reference
    implementation required, the expected array is computed by hand from
    the function's own documented contract ("fraction of residue
    occurrences per character per column").

    msa:
        row 0: A C D
        row 1: A A A
    column 0 = [A, A] -> all A
    column 1 = [C, A] -> half C, half A
    column 2 = [D, A] -> half D, half A

    Deliberately uses THREE DISTINCT symbols in an asymmetric arrangement
    (not e.g. a 2-symbol pattern like A/C repeated) -- an earlier draft of
    this test used ["AAC","ACC"], which by coincidence produces the same
    per-column symbol *counts* even when values are scrambled across
    columns by the bug (col1 ends up with {C,A} sourced from the wrong
    cells, but that's still one C and one A). Verified by hand-tracing
    which cells the buggy repeat-based indexing actually reads for each
    output column before relying on this data: with A/C/D all distinct,
    any cross-column scrambling changes the observed symbol at at least
    one position, so this example cannot pass by accident.
    """
    msa_array = np.array([["A", "C", "D"], ["A", "A", "A"]], dtype="<U1")

    a_idx = STANDARD_RESIDUES_WITH_GAP_1.index("A")
    c_idx = STANDARD_RESIDUES_WITH_GAP_1.index("C")
    d_idx = STANDARD_RESIDUES_WITH_GAP_1.index("D")

    expected = np.zeros((3, N_SYMBOLS))
    expected[0, a_idx] = 1.0
    expected[1, c_idx] = 0.5
    expected[1, a_idx] = 0.5
    expected[2, d_idx] = 0.5
    expected[2, a_idx] = 0.5

    # Exercise both the single-chunk path (chunk_size >= n_cols) and the
    # maximally-chunked path (chunk_size=1, one chunk per column) against
    # the SAME hand-computed expectation -- the bug affects both.
    for chunk_size in (1, 3, 1000):
        result = calculate_profile(
            msa_array=msa_array,
            molecule_type=MoleculeType.PROTEIN,
            chunk_size=chunk_size,
        )
        np.testing.assert_array_almost_equal(
            result, expected, err_msg=f"mismatch at chunk_size={chunk_size}"
        )


def test_calculate_profile_column_sum_alone_does_not_prove_correctness():
    """Documents why a row/column-sum check is not a sufficient regression
    guard for this function: the buggy implementation also satisfies it.

    Every entry counted by the (possibly buggy) chunking logic is counted
    under *some* column, so each column's fractions always sum to 1
    regardless of whether the column assignment is correct. This test
    exists so nobody "fixes" this test suite later by replacing the
    hand-computed / reference-implementation checks with a cheaper sum
    check -- that would silently stop catching the bug.
    """
    rng = np.random.default_rng(0)
    msa_array = _random_msa(rng, n_rows=37, n_cols=17)
    result = calculate_profile(
        msa_array=msa_array, molecule_type=MoleculeType.PROTEIN, chunk_size=5
    )
    np.testing.assert_allclose(result.sum(axis=1), np.ones(17))


@pytest.mark.parametrize(
    "n_rows,n_cols,chunk_size,seed",
    [
        pytest.param(5, 8, 1000, 1, id="single_chunk_more_cols_than_rows"),
        pytest.param(8, 5, 1000, 2, id="single_chunk_more_rows_than_cols"),
        pytest.param(37, 17, 5, 3, id="multiple_chunks_uneven_final_chunk"),
        pytest.param(37, 20, 5, 4, id="multiple_chunks_exact_final_chunk"),
        pytest.param(1000, 1000, 1000, 5, id="large_square_chunk"),
        pytest.param(
            1,
            10,
            3,
            6,
            id="single_row_msa_degenerate_cannot_distinguish_repeat_vs_tile",
        ),
        pytest.param(
            50,
            1,
            1000,
            7,
            id="single_column_msa_degenerate_cannot_distinguish_repeat_vs_tile",
        ),
        pytest.param(200, 2500, 1000, 8, id="realistic_deep_msa_multi_chunk"),
        pytest.param(16384, 1000, 1000, 9, id="max_rows_realistic_chunk_size"),
        pytest.param(23, 23, 7, 10, id="chunk_size_does_not_divide_n_cols_evenly"),
    ],
)
def test_calculate_profile_matches_naive_reference(n_rows, n_cols, chunk_size, seed):
    """Randomized cross-check against an independently-written reference
    implementation, across shapes chosen to exercise: large square chunks,
    n_rows far from n_cols in both directions, single-chunk and
    multi-chunk paths, and chunk boundaries that divide n_cols evenly and
    unevenly.

    Note on the two "degenerate" cases: with n_rows=1, `np.repeat(arange(k), 1)`
    and `np.tile(arange(k), 1)` are identical (both just `arange(k)`); with
    block_n_cols=1, `np.repeat(arange(1), n)` and `np.tile(arange(1), n)`
    are likewise identical (both just `[0]*n`). Those two cases genuinely
    cannot distinguish the buggy implementation from the fixed one --
    they're included for basic shape/degenerate-input sanity, not as
    bug-catching cases. Every other case here has both n_rows > 1 and
    block_n_cols > 1 specifically so it can.
    """
    rng = np.random.default_rng(seed)
    msa_array = _random_msa(rng, n_rows=n_rows, n_cols=n_cols)

    result = calculate_profile(
        msa_array=msa_array, molecule_type=MoleculeType.PROTEIN, chunk_size=chunk_size
    )
    reference = _naive_profile_reference(msa_array, MoleculeType.PROTEIN)

    np.testing.assert_array_almost_equal(result, reference)


def test_calculate_profile_chunking_is_invariant_to_chunk_size():
    """The chunk_size parameter is a performance/memory knob, not a
    semantic one -- the output must not depend on it. Cross-checks several
    chunk sizes against each other directly (not just against the
    reference), so this also fails if a future change makes correctness
    chunk-size-dependent in some new way.
    """
    rng = np.random.default_rng(42)
    msa_array = _random_msa(rng, n_rows=53, n_cols=211)

    results = [
        calculate_profile(
            msa_array=msa_array, molecule_type=MoleculeType.PROTEIN, chunk_size=cs
        )
        for cs in (1, 7, 50, 211, 1000)
    ]
    for other in results[1:]:
        np.testing.assert_array_almost_equal(results[0], other)
