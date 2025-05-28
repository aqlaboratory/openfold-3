import pandas as pd
from openfold3.core.data.framework.single_datasets.dataset_utils import (
    pad_to_world_size,
)
import pytest

fewer_examples_than_world_size = {
    "label": "fewer_examples_than_world_size",
    "dp_cache": pd.DataFrame(
        {
            "sample_id": ["sample1", "sample2", "sample3"] * 5,
            "seeds": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        }
    ),
    "world_size": 16,
    "expected": pd.DataFrame(
        {
            "sample_id": ["sample1", "sample2", "sample3"] * 5 + ["sample1"],
            "seeds": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 1],
            "repeated_sample": [False] * 15 + [True],
        }
    ),
}

no_world_size = {
    "label": "no_world_size",
    "dp_cache": pd.DataFrame({"a": [1]}),
    "world_size": None,
    "expected": pd.DataFrame({"a": [1], "repeated_sample": [False]}),
}

more_examples_than_world_size = {
    "label": "more_examples_than_world_size",
    "dp_cache": pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    ),
    "world_size": 4,
    "expected": pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2],
            "repeated_sample": [False] * 10 + [True] * 2,
        }
    ),
}


@pytest.mark.parametrize(
    "data",
    [fewer_examples_than_world_size, no_world_size, more_examples_than_world_size],
    ids=lambda d: d["label"],
)
def test_example_with_seeds(data):
    dp_cache = data["dp_cache"]
    world_size = data["world_size"]
    expected = data["expected"]

    padded_df = pad_to_world_size(dp_cache, world_size)

    pd.testing.assert_frame_equal(padded_df, expected)
