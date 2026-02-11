from unittest.mock import MagicMock

import numpy as np
import pytest

from openfold3.core.data.primitives.structure.metadata import (
    get_author_to_label_chain_ids,
    get_label_to_author_chain_id_dict,
    resolve_author_to_label_chain_id,
)


def _make_cif_file(asym_ids: list[str], author_ids: list[str]) -> MagicMock:
    """Build a mock CIFFile whose pdbx_poly_seq_scheme has the given columns."""
    poly_scheme = MagicMock()
    poly_scheme.__getitem__ = lambda self, key: {
        "asym_id": MagicMock(as_array=lambda: np.array(asym_ids)),
        "pdb_strand_id": MagicMock(as_array=lambda: np.array(author_ids)),
    }[key]

    block = MagicMock()
    block.__getitem__ = lambda self, key: {"pdbx_poly_seq_scheme": poly_scheme}[key]

    cif_file = MagicMock()
    cif_file.block = block
    return cif_file


class TestGetLabelToAuthorChainIdDict:
    @pytest.mark.parametrize(
        ("asym_ids", "author_ids", "expected"),
        [
            pytest.param(
                ["A", "A", "A"], ["X", "X", "X"], {"A": "X"}, id="single_chain"
            ),
            pytest.param(
                ["A", "A", "B", "B", "C"],
                ["X", "X", "Y", "Y", "Z"],
                {"A": "X", "B": "Y", "C": "Z"},
                id="multiple_distinct_chains",
            ),
            pytest.param(
                ["A", "A", "B", "B"],
                ["X", "X", "X", "X"],
                {"A": "X", "B": "X"},
                id="homomeric_chains",
            ),
        ],
    )
    def test_label_to_author(self, asym_ids, author_ids, expected):
        cif_file = _make_cif_file(asym_ids=asym_ids, author_ids=author_ids)
        assert get_label_to_author_chain_id_dict(cif_file) == expected


class TestGetAuthorToLabelChainIds:
    def test_single_chain(self):
        """Single label → author entry produces a single-element list."""
        result = get_author_to_label_chain_ids({"A": "X"})
        assert result == {"X": ["A"]}

    def test_multiple_distinct_chains(self):
        """Distinct author IDs each get their own list."""
        result = get_author_to_label_chain_ids({"A": "X", "B": "Y", "C": "Z"})
        assert result == {"X": ["A"], "Y": ["B"], "Z": ["C"]}

    def test_homomeric_chains(self):
        """Multiple label asym_ids mapping to the same author ID are grouped."""
        result = get_author_to_label_chain_ids({"A": "X", "B": "X"})
        assert result == {"X": ["A", "B"]}

    def test_homomeric_chains_sorted(self):
        """Grouped label IDs are sorted regardless of input order."""
        result = get_author_to_label_chain_ids({"C": "X", "A": "X", "B": "X"})
        assert result == {"X": ["A", "B", "C"]}


class TestResolveAuthorToLabelChainId:
    def test_single_label(self):
        """Single matching label is returned directly."""
        result = resolve_author_to_label_chain_id(
            matching_labels=["A"],
            author_chain_id="X",
            chain_id_seq_map={"A": "MSEQ"},
        )
        assert result == "A"

    def test_homomeric_returns_first_sorted(self):
        """For homomeric chains with identical sequences, returns the first label."""
        result = resolve_author_to_label_chain_id(
            matching_labels=["A", "B", "C"],
            author_chain_id="X",
            chain_id_seq_map={"A": "MSEQ", "B": "MSEQ", "C": "MSEQ"},
        )
        assert result == "A"

    def test_homomeric_differing_sequences_raises(self):
        """Raises ValueError when homomeric chains have different sequences."""
        with pytest.raises(ValueError, match="got 2 distinct sequences"):
            resolve_author_to_label_chain_id(
                matching_labels=["A", "B"],
                author_chain_id="X",
                chain_id_seq_map={"A": "MSEQ", "B": "MOTHER"},
            )

    def test_homomeric_three_labels_two_distinct_raises(self):
        """Raises ValueError even when only some of the sequences differ."""
        with pytest.raises(ValueError, match="got 2 distinct sequences"):
            resolve_author_to_label_chain_id(
                matching_labels=["A", "B", "C"],
                author_chain_id="X",
                chain_id_seq_map={"A": "MSEQ", "B": "MSEQ", "C": "OTHER"},
            )
