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

"""Tests for get_pairing_masks function."""

import numpy as np
import pytest

from openfold3.core.data.primitives.sequence.msa import get_pairing_masks


@pytest.mark.parametrize(
    "count_array,mask_keys,expected",
    [
        # Without shared_by_two - all species pass
        pytest.param(
            np.array(
                [
                    # 4 species (S0-S3) across 2 chains
                    [2, 1, 1, 3],
                    [1, 2, 1, 0],
                ]
            ),
            [],
            np.array([True, True, True, True]),
            id="no_filters_unshared_species_passes",
        ),
        pytest.param(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            [],
            np.array([True, True, True]),
            id="no_filters_all_pass",
        ),
        # With shared_by_two - unshared species filtered
        pytest.param(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ["shared_by_two"],
            np.array([False, False, False]),
            id="shared_by_two_no_shared_species",
        ),
        pytest.param(
            np.array([[2, 1, 1, 3], [1, 2, 1, 0]]),
            ["shared_by_two"],
            np.array([True, True, True, False]),
            id="shared_by_two_filters_unshared",
        ),
        pytest.param(
            np.array([[5, 3, 2]]),
            ["shared_by_two"],
            np.array([False, False, False]),
            id="shared_by_two_single_chain",
        ),
        pytest.param(
            np.array([[2, 1, 0, 0], [1, 2, 1, 0], [1, 0, 2, 3]]),
            ["shared_by_two"],
            np.array([True, True, True, False]),
            id="shared_by_two_three_chains",
        ),
        # less_than_600 filter alone
        pytest.param(
            np.array([[601, 10], [10, 10]]),
            ["less_than_600"],
            np.array([False, True]),
            id="less_than_600_only",
        ),
        # Combined filters
        pytest.param(
            np.array([[601, 10], [10, 10]]),
            ["shared_by_two", "less_than_600"],
            np.array([False, True]),
            id="combined_high_frequency_filtered",
        ),
        pytest.param(
            np.array([[600, 10], [10, 10]]),
            ["shared_by_two", "less_than_600"],
            np.array([True, True]),
            id="combined_exactly_600_passes",
        ),
    ],
)
def test_get_pairing_masks(count_array, mask_keys, expected):
    result = get_pairing_masks(count_array, mask_keys)
    np.testing.assert_array_equal(result, expected)
