# Copyright 2025 AlQuraishi Laboratory
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

import pandas as pd
import pytest

from openfold3.core.data.framework.single_datasets.dataset_utils import (
    pad_to_world_size,
)

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

world_size_is_zero = {
    "label": "world_size_is_zero",
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

same_examples_as_world_size = {
    "label": "same_number_of_examples_as_world_size",
    "dp_cache": pd.DataFrame(
        {
            "a": [1],
        }
    ),
    "world_size": 1,
    "expected": pd.DataFrame(
        {
            "a": [1],
            "repeated_sample": [False],
        }
    ),
}

num_examples_is_multiple_of_world_size = {
    "label": "num_examples_is_multiple_of_world_size",
    "dp_cache": pd.DataFrame(
        {
            "a": [1, 2, 3],
        }
    ),
    "world_size": 1,
    "expected": pd.DataFrame(
        {
            "a": [1, 2, 3],
            "repeated_sample": [False, False, False],
        }
    ),
}

num_examples_one_world_size_four = {
    "label": "num_examples_is_one",
    "dp_cache": pd.DataFrame(
        {
            "a": [1],
        }
    ),
    "world_size": 4,
    "expected": pd.DataFrame(
        {
            "a": [1, 1, 1, 1],
            "repeated_sample": [False, True, True, True],
        }
    ),
}


@pytest.mark.parametrize(
    "data",
    [
        fewer_examples_than_world_size,
        world_size_is_zero,
        more_examples_than_world_size,
        same_examples_as_world_size,
        num_examples_is_multiple_of_world_size,
        num_examples_one_world_size_four,
    ],
    ids=lambda d: d["label"],
)
def test_example_with_seeds(data):
    dp_cache = data["dp_cache"]
    world_size = data["world_size"]
    expected = data["expected"]

    padded_df = pad_to_world_size(dp_cache, world_size)

    pd.testing.assert_frame_equal(padded_df, expected)
