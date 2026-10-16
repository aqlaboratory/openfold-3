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

"""Tests for parse_msas_preparsed."""

import io

import numpy as np
import pytest

from openfold3.core.data.io.sequence.msa import parse_msas_preparsed


def _make_npz(msa_dicts: dict[str, dict]) -> io.BytesIO:
    """Build an in-memory npz buffer matching the pre-parsing pipeline format."""
    packed = {key: np.array(d, dtype=object) for key, d in msa_dicts.items()}
    buf = io.BytesIO()
    np.savez(buf, **packed)
    buf.seek(0)
    return buf


def _make_msa_dict(
    sequences: list[str],
    deletion_matrix: list[list[int]] | None = None,
    metadata: list[str] | None = None,
) -> dict:
    """Build a single MSA entry dict from readable string lists."""
    msa = np.array([list(seq) for seq in sequences])
    if deletion_matrix is None:
        deletion_matrix = np.zeros(msa.shape, dtype=int)
    else:
        deletion_matrix = np.array(deletion_matrix)
    if metadata is None:
        metadata = np.array([])
    else:
        metadata = np.array(metadata)
    return {"msa": msa, "deletion_matrix": deletion_matrix, "metadata": metadata}


class TestParseMsasPreparsed:
    @pytest.mark.parametrize(
        "sequences,expected_upper",
        [
            pytest.param(
                ["ACG", "A-G"],
                [list("ACG"), list("A-G")],
                id="already_uppercase",
            ),
            pytest.param(
                ["acg", "a-g"],
                [list("ACG"), list("A-G")],
                id="all_lowercase",
            ),
            pytest.param(
                ["aCg", "A-g"],
                [list("ACG"), list("A-G")],
                id="mixed_case",
            ),
        ],
    )
    def test_case_normalization(self, sequences, expected_upper):
        buf = _make_npz({"uniref": _make_msa_dict(sequences)})
        result = parse_msas_preparsed([buf])
        np.testing.assert_array_equal(result["uniref"].msa, np.array(expected_upper))

    def test_single_file_multiple_keys(self):
        buf = _make_npz(
            {
                "uniref": _make_msa_dict(["ACG", "A-G"]),
                "mgnify": _make_msa_dict(["TT", "GG"]),
            },
        )
        result = parse_msas_preparsed([buf])
        assert set(result.keys()) == {"uniref", "mgnify"}
        assert result["uniref"].msa.shape == (2, 3)
        assert result["mgnify"].msa.shape == (2, 2)

    def test_multiple_files(self):
        buf1 = _make_npz({"uniref": _make_msa_dict(["ACG"])})
        buf2 = _make_npz({"mgnify": _make_msa_dict(["TT"])})
        result = parse_msas_preparsed([buf1, buf2])
        assert set(result.keys()) == {"uniref", "mgnify"}

    def test_duplicate_key_warns_and_keeps_last(self):
        buf1 = _make_npz({"uniref": _make_msa_dict(["ACG"])})
        buf2 = _make_npz({"uniref": _make_msa_dict(["TTT"])})
        with pytest.warns(UserWarning, match="duplicate key uniref"):
            result = parse_msas_preparsed([buf1, buf2])
        # second file wins
        np.testing.assert_array_equal(result["uniref"].msa, np.array([list("TTT")]))

    def test_deletion_matrix_preserved(self):
        del_mat = [[1, 0, 2], [0, 0, 0]]
        buf = _make_npz(
            {"uniref": _make_msa_dict(["ACG", "A-G"], deletion_matrix=del_mat)}
        )
        result = parse_msas_preparsed([buf])
        np.testing.assert_array_equal(
            result["uniref"].deletion_matrix, np.array(del_mat)
        )

    def test_metadata_preserved(self):
        buf = _make_npz(
            {"uniref": _make_msa_dict(["ACG", "A-G"], metadata=["seq1", "seq2"])},
        )
        result = parse_msas_preparsed([buf])
        np.testing.assert_array_equal(
            result["uniref"].metadata, np.array(["seq1", "seq2"])
        )

    def test_empty_file_list(self):
        result = parse_msas_preparsed([])
        assert result == {}
