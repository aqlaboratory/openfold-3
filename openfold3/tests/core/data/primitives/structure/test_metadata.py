import pytest

from openfold3.core.data.primitives.structure.metadata import (
    get_author_to_label_chain_ids,
)


class TestGetAuthorToLabelChainIds:
    @pytest.mark.parametrize(
        ("label_to_author", "expected"),
        [
            pytest.param({"A": "X"}, {"X": ["A"]}, id="single_chain"),
            pytest.param(
                {"A": "X", "B": "Y", "C": "Z"},
                {"X": ["A"], "Y": ["B"], "Z": ["C"]},
                id="multiple_distinct_chains",
            ),
            pytest.param(
                {"A": "X", "B": "X"}, {"X": ["A", "B"]}, id="homomeric_chains"
            ),
            pytest.param(
                {"C": "X", "A": "X", "B": "X"},
                {"X": ["A", "B", "C"]},
                id="homomeric_chains_sorted",
            ),
        ],
    )
    def test_author_to_labels(self, label_to_author, expected):
        assert get_author_to_label_chain_ids(label_to_author) == expected
