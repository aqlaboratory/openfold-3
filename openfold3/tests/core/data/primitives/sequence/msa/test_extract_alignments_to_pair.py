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

"""Tests for extract_alignments_to_pair.

Covers two bugs in the same code block:
    1. A crash: the "exclude query from subsequent MSAs" step called len() on an
       int (m.msa.shape[0] is already an int, not something to len()), so any
       representative with hits in more than one pairable MSA source raised
       TypeError immediately.
    2. A silent off-by-one: after fixing (1), the "does this source have a real
       hit" check (`len(m) > 1`) was applied unchanged to sources after the
       first, even though their query row had already been stripped by the
       exclude-query step. A second-or-later source with exactly one real hit
       therefore got silently dropped from pairing entirely.
"""

import numpy as np
import pytest

from openfold3.core.data.primitives.sequence.msa import (
    MsaArray,
    MsaArrayCollection,
    extract_alignments_to_pair,
)
from openfold3.core.data.resources.residues import MoleculeType

# Every row label used across this file's fixtures maps to a distinct
# deletion value, so a mutation that permutes/misaligns deletion_matrix
# relative to msa/metadata (but leaves metadata itself correct) is caught by
# _assert_consistent, instead of passing silently against an all-zeros
# deletion_matrix that looks the same regardless of row order.
_LABEL_TO_DELETION_VALUE = {
    "query": 0,
    "hit1": 1,
    "hit2": 2,
    "uniprot1": 3,
    "uniprot2": 4,
}


def _msa(rows: list[str], n_cols: int = 2, ndarray_metadata: bool = False) -> MsaArray:
    """Build an MsaArray with one distinguishable row-label per row.

    ndarray_metadata selects between metadata as a plain list (what the real
    a3m/stockholm parsers populate, io/sequence/msa.py) or as an np.ndarray
    (what the pre-parsed npz alignment loader in the same module populates)
    -- MsaArray.truncate()/.subset() handle the two differently.
    """
    metadata = np.array(rows) if ndarray_metadata else list(rows)
    deletion_values = [_LABEL_TO_DELETION_VALUE[label] for label in rows]
    return MsaArray(
        msa=np.array([[label] * n_cols for label in rows]),
        deletion_matrix=np.array([[v] * n_cols for v in deletion_values]),
        metadata=metadata,
    )


def _collection(rep_msa_map: dict[str, MsaArray]) -> MsaArrayCollection:
    return MsaArrayCollection(
        chain_id_to_rep_id={"A": "repA"},
        chain_id_to_mol_type={"A": MoleculeType.PROTEIN},
        rep_id_to_chain_id={"repA": "A"},
        rep_id_to_mol_type={"repA": MoleculeType.PROTEIN},
        rep_id_to_main_msa={"repA": rep_msa_map},
    )


def _assert_consistent(result: MsaArray, expected_labels: list[str]) -> None:
    """Assert msa, deletion_matrix, and metadata all agree on which rows
    survived pairing, not just metadata.
    """
    assert list(result.metadata) == expected_labels
    assert result.msa[:, 0].tolist() == expected_labels
    expected_deletion = [_LABEL_TO_DELETION_VALUE[label] for label in expected_labels]
    assert result.deletion_matrix[:, 0].tolist() == expected_deletion


def test_does_not_crash_with_multiple_pairable_sources():
    """Regression test for the len()-on-int crash.

    Any representative with hits in more than one of the configured
    msas_to_pair sources used to raise TypeError unconditionally.
    """
    collection = _collection(
        {
            "uniprot_hits": _msa(["query", "hit1", "hit2"]),
            "uniprot": _msa(["query", "uniprot1"]),
        }
    )

    result = extract_alignments_to_pair(
        collection, msas_to_pair=["uniprot_hits", "uniprot"]
    )

    assert "repA" in result


@pytest.mark.parametrize(
    "second_source_rows, expected_metadata",
    [
        pytest.param(
            ["query"],
            ["query", "hit1"],
            id="second-source-zero-real-hits-excluded",
        ),
        pytest.param(
            ["query", "uniprot1"],
            ["query", "hit1", "uniprot1"],
            id="second-source-one-real-hit-included",
        ),
        pytest.param(
            ["query", "uniprot1", "uniprot2"],
            ["query", "hit1", "uniprot1", "uniprot2"],
            id="second-source-two-real-hits-included",
        ),
    ],
)
def test_second_source_included_based_on_real_hit_count(
    second_source_rows, expected_metadata
):
    """Regression test for the off-by-one on sources after the first.

    A second pairing source should be included whenever it contributes at
    least one real (non-query) hit -- not only when it contributes at least
    two, which is what `len(m) > 1` incorrectly required once its query row
    had already been stripped.
    """
    collection = _collection(
        {
            "uniprot_hits": _msa(["query", "hit1"]),
            "uniprot": _msa(second_source_rows),
        }
    )

    result = extract_alignments_to_pair(
        collection, msas_to_pair=["uniprot_hits", "uniprot"]
    )

    _assert_consistent(result["repA"], expected_metadata)


def test_first_pairable_source_tracking_is_content_based_not_positional():
    """The "is this the first pairable source" check must track whether
    anything has actually been added so far, not just loop position.

    If the first-listed source contributes no real hits (and so contributes
    nothing), the next source that does contribute must still be treated as
    the *first* pairable source: its own query row should be kept, and it
    should be included with only one real hit.
    """
    collection = _collection(
        {
            # First in msas_to_pair, but contributes nothing.
            "uniprot_hits": _msa(["query"]),
            # Second in msas_to_pair, but is effectively the first real
            # contributor -- should be treated the same as if it were listed
            # first, i.e. its query row is kept and one real hit is enough.
            "uniprot": _msa(["query", "uniprot1"]),
        }
    )

    result = extract_alignments_to_pair(
        collection, msas_to_pair=["uniprot_hits", "uniprot"]
    )

    _assert_consistent(result["repA"], ["query", "uniprot1"])


def test_three_or_more_pairable_sources_all_contribute():
    """Regression coverage for more than two pairable sources.

    Existing tests only exercise two sources. Verify the "first pairable
    source keeps its query row, every later one has it stripped" bookkeeping
    still holds when a third source also contributes a real hit.
    """
    collection = _collection(
        {
            "uniprot_hits": _msa(["query", "hit1"]),
            "uniprot": _msa(["query", "uniprot1"]),
            "third_source": _msa(["query", "hit2"]),
        }
    )

    result = extract_alignments_to_pair(
        collection, msas_to_pair=["uniprot_hits", "uniprot", "third_source"]
    )

    _assert_consistent(result["repA"], ["query", "hit1", "uniprot1", "hit2"])


def test_pairing_with_ndarray_metadata():
    """Regression coverage for the pre-parsed npz alignment loader path.

    io/sequence/msa.py's npz loader builds MsaArrays with metadata taken
    directly from the npz file (an np.ndarray), not the plain list the
    a3m/stockholm parsers -- and every other test in this file -- use.
    truncate() and subset() handle ndarray metadata differently from list
    metadata, so this exercises a branch nothing else here covers.
    """
    collection = _collection(
        {
            "uniprot_hits": _msa(["query", "hit1"], ndarray_metadata=True),
            "uniprot": _msa(["query", "uniprot1"], ndarray_metadata=True),
        }
    )

    result = extract_alignments_to_pair(
        collection, msas_to_pair=["uniprot_hits", "uniprot"]
    )

    _assert_consistent(result["repA"], ["query", "hit1", "uniprot1"])


def test_second_source_with_zero_rows_does_not_crash():
    """Regression test: a pairing source with zero rows at all (not even a
    query row) must not crash.

    The old exclude-query step built a boolean mask from m.msa.shape[0] and
    unconditionally set index 0 to False, which raised IndexError for a
    genuinely empty (0-row) source. truncate(slice(1, None)) handles an
    empty array the same way it handles any other slice: no rows survive.
    """
    collection = _collection(
        {
            "uniprot_hits": _msa(["query", "hit1"]),
            "uniprot": MsaArray(
                msa=np.empty((0, 2), dtype=str),
                deletion_matrix=np.empty((0, 2)),
                metadata=[],
            ),
        }
    )

    result = extract_alignments_to_pair(
        collection, msas_to_pair=["uniprot_hits", "uniprot"]
    )

    _assert_consistent(result["repA"], ["query", "hit1"])
