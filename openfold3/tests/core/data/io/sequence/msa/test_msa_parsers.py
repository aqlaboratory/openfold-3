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

"""Regression tests for MSA parsing and MsaArray truncation.

Covers the ``max_seq_count`` cap in ``parse_a3m`` / ``parse_stockholm`` and the
``MsaArray.truncate`` int/slice handling.
"""

import numpy as np
import pytest

from openfold3.core.data.io.sequence.msa import parse_a3m, parse_stockholm
from openfold3.core.data.primitives.sequence.msa import MsaArray

QUERY_LEN = 4


def _a3m_string(n_seqs: int) -> str:
    """Build a minimal a3m with ``n_seqs`` gap-free, equal-length sequences."""
    return "".join(f">seq{i}\nABCD\n" for i in range(n_seqs))


def _stockholm_string(n_seqs: int) -> str:
    """Build a minimal Stockholm alignment with ``n_seqs`` sequences."""
    body = "".join(f"seq{i} ABCD\n" for i in range(n_seqs))
    return f"# STOCKHOLM 1.0\n{body}//\n"


def _msa_array(n_rows: int = 5) -> MsaArray:
    """Build an MsaArray whose rows are distinguishable by their first column."""
    tags = list("ABCDEFGHIJ")[:n_rows]
    msa = np.array([[tag, "-", "-", "-"] for tag in tags])
    deletion_matrix = np.zeros((n_rows, QUERY_LEN), dtype=int)
    return MsaArray(msa=msa, deletion_matrix=deletion_matrix)


@pytest.mark.parametrize("max_seq_count, expected_rows", [(None, 5), (3, 3), (1, 1)])
def test_parse_a3m_respects_max_seq_count(max_seq_count, expected_rows):
    """parse_a3m must cap the MSA at max_seq_count.

    Regression: truncate() was called without inplace=True, so the cap was a
    no-op and every sequence was kept.
    """
    parsed = parse_a3m(_a3m_string(5), max_seq_count=max_seq_count)

    assert len(parsed) == expected_rows
    assert parsed.msa.shape[0] == expected_rows
    assert parsed.deletion_matrix.shape[0] == expected_rows


@pytest.mark.parametrize("max_seq_count, expected_rows", [(None, 5), (3, 3), (1, 1)])
def test_parse_stockholm_respects_max_seq_count(max_seq_count, expected_rows):
    """parse_stockholm must cap the MSA at max_seq_count.

    Regression: same no-op truncate bug as parse_a3m. Stockholm is the format of
    the main protein databases (uniref90, mgnify, bfd), so this path matters most
    in practice.
    """
    parsed = parse_stockholm(_stockholm_string(5), max_seq_count=max_seq_count)

    assert len(parsed) == expected_rows
    assert parsed.msa.shape[0] == expected_rows
    assert parsed.deletion_matrix.shape[0] == expected_rows


def test_truncate_int_returns_new_first_n_rows():
    """truncate(n) returns a new MsaArray of the first n rows and leaves the
    original untouched (default inplace=False)."""
    arr = _msa_array(5)

    out = arr.truncate(2)

    assert out is not None
    assert out.msa[:, 0].tolist() == ["A", "B"]
    assert len(out) == 2
    assert len(arr) == 5  # original unchanged when inplace=False


def test_truncate_inplace_mutates_and_returns_none():
    """truncate(n, inplace=True) mutates in place and returns None -- the mode the
    parsers rely on."""
    arr = _msa_array(5)

    result = arr.truncate(2, inplace=True)

    assert result is None
    assert len(arr) == 2
    assert arr.deletion_matrix.shape[0] == 2


def test_truncate_accepts_slice():
    """truncate() also accepts a slice, honoring its start and stop."""
    arr = _msa_array(5)

    out = arr.truncate(slice(1, 4))

    assert out is not None
    assert out.msa[:, 0].tolist() == ["B", "C", "D"]
    assert len(out) == 3


def test_truncate_over_length_keeps_all_rows():
    """Requesting more rows than exist returns every row, no error."""
    arr = _msa_array(5)

    out = arr.truncate(100)

    assert out is not None
    assert len(out) == 5
