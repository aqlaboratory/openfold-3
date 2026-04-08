import numpy as np
import pytest
from biotite.structure.io.pdbx import CIFBlock, CIFCategory

from openfold3.core.data.primitives.structure.metadata import (
    get_author_to_label_chain_ids,
    get_resolution,
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


def _cif_block(categories: dict[str, dict[str, list[str]]]) -> CIFBlock:
    """Build a CIFBlock from {category: {column: [values]}}."""
    return CIFBlock(
        {
            cat: CIFCategory({col: np.array(vals) for col, vals in cols.items()})
            for cat, cols in categories.items()
        }
    )


@pytest.mark.parametrize(
    "cif_data, expected",
    [
        pytest.param(
            _cif_block({"refine": {"ls_d_res_high": ["2.5"]}}),
            2.5,
            id="refine-first-priority",
        ),
        pytest.param(
            _cif_block({"em_3d_reconstruction": {"resolution": ["3.1"]}}),
            3.1,
            id="em-second-priority",
        ),
        pytest.param(
            _cif_block({"reflns": {"d_resolution_high": ["4.0"]}}),
            4.0,
            id="reflns-third-priority",
        ),
        pytest.param(
            _cif_block(
                {
                    "refine": {"ls_d_res_high": ["1.8"]},
                    "em_3d_reconstruction": {"resolution": ["3.0"]},
                }
            ),
            1.8,
            id="refine-takes-precedence-over-em",
        ),
        pytest.param(
            _cif_block(
                {
                    "refine": {"ls_d_res_high": ["?"]},
                    "em_3d_reconstruction": {"resolution": ["3.0"]},
                }
            ),
            3.0,
            id="question-mark-skipped-falls-through",
        ),
        pytest.param(
            _cif_block(
                {
                    "refine": {"ls_d_res_high": ["."]},
                    "reflns": {"d_resolution_high": ["5.5"]},
                }
            ),
            5.5,
            id="dot-skipped-falls-through",
        ),
        pytest.param(
            _cif_block({}),
            None,
            id="no-categories-returns-none",
        ),
        pytest.param(
            _cif_block(
                {
                    "refine": {"ls_d_res_high": ["?"]},
                    "em_3d_reconstruction": {"resolution": ["."]},
                }
            ),
            None,
            id="all-missing-markers-returns-none",
        ),
    ],
)
def test_get_resolution(cif_data, expected):
    assert get_resolution(cif_data) == expected
