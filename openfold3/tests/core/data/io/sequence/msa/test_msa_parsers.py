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


@pytest.mark.parametrize(
    "parser, build_msa_string",
    [(parse_a3m, _a3m_string), (parse_stockholm, _stockholm_string)],
)
@pytest.mark.parametrize(
    "max_seq_count, expected_rows", [(None, 5), (3, 3), (1, 1), (0, 0), (-1, 4)]
)
def test_parsers_respect_max_seq_count(
    parser, build_msa_string, max_seq_count, expected_rows
):
    """The parsers must cap the MSA at max_seq_count, keeping the query width.

    Regression: truncate() was called without inplace=True, so the cap was a
    no-op and every sequence was kept.
    """
    parsed = parser(build_msa_string(5), max_seq_count=max_seq_count)

    assert len(parsed) == expected_rows
    assert parsed.msa.shape == (expected_rows, QUERY_LEN)
    assert parsed.deletion_matrix.shape == (expected_rows, QUERY_LEN)


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


def test_truncate_over_length_keeps_all_rows():
    """Requesting more rows than exist returns every row, no error."""
    arr = _msa_array(5)

    out = arr.truncate(100)

    assert out is not None
    assert len(out) == 5


def test_parse_a3m_counts_insertions_as_deletions():
    """Lowercase residues are removed from the MSA and counted as deletions before
    the next aligned column. Insertions after the last column are dropped."""
    parsed = parse_a3m(">query\nACDE\n>hit\naaAcCD\nEzz\n")

    assert parsed.msa.tolist() == [list("ACDE"), list("ACDE")]
    assert parsed.deletion_matrix.tolist() == [[0, 0, 0, 0], [2, 1, 0, 0]]
    assert parsed.metadata == ["query", "hit"]


def test_parse_stockholm_removes_query_gap_columns():
    """Columns with a gap in the query are removed, and residues in them are
    counted as deletions. Residues after the last kept column are dropped.
    Sequences can be split over several blocks."""
    msa_string = (
        "# STOCKHOLM 1.0\nquery AC-D\nhit   AcGD\n\nquery E.F-\nhit   EkFw\n//\n"
    )

    parsed = parse_stockholm(msa_string)

    assert parsed.msa.tolist() == [list("ACDEF"), list("ACDEF")]
    assert parsed.deletion_matrix.tolist() == [[0, 0, 0, 0, 0], [0, 0, 1, 0, 1]]
    assert parsed.metadata == ["query", "hit"]


def test_parse_stockholm_custom_gap_symbols():
    """Only the given single-character symbols count as gaps."""
    parsed = parse_stockholm("query A-.D\nhit   AGkD\n", gap_symbols={"-", "--"})

    assert parsed.msa.tolist() == [list("A.D"), list("AKD")]
    assert parsed.deletion_matrix.tolist() == [[0, 0, 0], [0, 1, 0]]


@pytest.mark.parametrize(
    "parser, msa_string",
    [
        (parse_a3m, ">query\nACDE\n>hit\nACD\n"),
        (parse_stockholm, "query ACDE\nhit   ACD\n"),
    ],
)
def test_parsers_reject_rows_of_different_length(parser, msa_string):
    """Rows of different length fail the file, unless they come after
    max_seq_count: those rows are not parsed."""
    with pytest.raises(ValueError):
        parser(msa_string)

    assert parser(msa_string, max_seq_count=1).msa.tolist() == [list("ACDE")]


@pytest.mark.parametrize(
    "parser, msa_string",
    [(parse_a3m, ""), (parse_stockholm, "# STOCKHOLM 1.0\n//\n")],
)
def test_parsers_reject_files_without_sequences(parser, msa_string):
    with pytest.raises(ValueError, match="no sequences"):
        parser(msa_string)


@pytest.mark.parametrize(
    "parser, msa_string",
    [
        (parse_a3m, ">query\nAÅC\n>hit\nAÅbC\n"),
        (parse_stockholm, "query AÅ-C\nhit   AÅbC\n"),
    ],
)
def test_parsers_accept_non_ascii(parser, msa_string):
    """Non-ASCII text takes the slower code point path and gives the same result."""
    parsed = parser(msa_string)

    assert parsed.msa.tolist() == [list("AÅC"), list("AÅC")]
    assert parsed.deletion_matrix.tolist() == [[0, 0, 0], [0, 0, 1]]
