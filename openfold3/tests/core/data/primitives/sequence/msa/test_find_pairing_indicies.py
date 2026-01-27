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

"""Tests for find_pairing_indices function.

Output format:
    - Shape: (num_paired_rows, num_chains)
    - Each row represents a paired position across chains
    - Each cell contains the species index that chain uses at that row
    - -1 indicates the chain doesn't have that species (partial pair)
"""

import numpy as np
import pytest

from openfold3.core.data.primitives.sequence.msa import find_pairing_indices


@pytest.mark.parametrize(
    "count_array,pairing_masks,expected",
    [
        pytest.param(
            # Chain A: 1 seq from S0, 1 seq from S1
            # Chain B: 1 seq from S0, 1 seq from S1
            # Result: 2 rows, each species paired once
            np.array([[1, 1], [1, 1]]),  # count_array
            np.array([True, True]),  # pairing_masks
            np.array([[0, 0], [1, 1]]),  # expected
            id="two_chains_one_each",
        ),
        pytest.param(
            # Pairing uses min count: S0→min(2,1)=1, S1→min(1,2)=1
            np.array([[2, 1], [1, 2]]),  # count_array
            np.array([True, True]),  # pairing_masks
            np.array([[0, 0], [1, 1]]),  # expected
            id="two_chains_min_count",
        ),
        pytest.param(
            # min(3,2)=2 paired rows from species 0
            np.array([[3], [2]]),  # count_array
            np.array([True]),  # pairing_masks
            np.array([[0, 0], [0, 0]]),  # expected
            id="multiple_pairs_same_species",
        ),
        pytest.param(
            # S0 included, S1 filtered out by mask
            np.array([[2, 2], [2, 2]]),  # count_array
            np.array([True, False]),  # pairing_masks
            np.array([[0, 0], [0, 0]]),  # expected
            id="filtered_species_excluded",
        ),
        pytest.param(
            # S0: in all 3 chains (fully paired)
            # S1: only in A and B (partial, C gets -1)
            np.array([[1, 1], [1, 1], [1, 0]]),  # count_array
            np.array([True, True]),  # pairing_masks
            np.array([[0, 0, 0], [1, 1, -1]]),  # expected
            id="three_chains_partial",
        ),
        pytest.param(
            # S0: 3 chains, paired first (fully paired)
            # S1: 2 chains, paired second (partial)
            np.array([[2, 2], [2, 2], [2, 0]]),  # count_array
            np.array([True, True]),  # pairing_masks
            np.array(
                [
                    [0, 0, 0],  # S0 fully paired
                    [0, 0, 0],  # S0 fully paired
                    [1, 1, -1],  # S1 partial (C missing)
                    [1, 1, -1],  # S1 partial (C missing)
                ]
            ),  # expected
            id="priority_full_before_partial",
        ),
    ],
)
def test_find_pairing_indices(count_array, pairing_masks, expected):
    """Standard cases with default max_rows=100, min_chains=2."""
    result = find_pairing_indices(
        count_array=count_array,
        pairing_masks=pairing_masks,
        max_rows_paired=100,
        min_chains_paired_partial=2,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "max_rows_paired,expected",
    [
        pytest.param(
            2,
            np.array([[0, 0], [0, 0]]),
            id="max_rows_2",
        ),
        pytest.param(
            4,
            np.array([[0, 0], [0, 0], [1, 1], [1, 1]]),
            id="max_rows_4",
        ),
    ],
)
def test_max_rows_truncation(max_rows_paired, expected):
    """Output truncated when exceeding max_rows_paired."""
    result = find_pairing_indices(
        count_array=np.array([[2, 2], [2, 2]]),  # Could make 4 pairs
        pairing_masks=np.array([True, True]),
        max_rows_paired=max_rows_paired,
        min_chains_paired_partial=2,
    )
    np.testing.assert_array_equal(result, expected)


def test_min_chains_excludes_partial():
    """Higher min_chains_paired_partial excludes partial pairs."""
    result = find_pairing_indices(
        count_array=np.array([[1, 1], [1, 1], [1, 0]]),  # S1 only in 2 chains
        pairing_masks=np.array([True, True]),
        max_rows_paired=100,
        min_chains_paired_partial=3,  # Require all 3 chains
    )
    # Only S0 (in all 3 chains) is included
    np.testing.assert_array_equal(result, np.array([[0, 0, 0]]))


def test_heterotrimer_with_varied_msa_coverage():
    """
    Realistic scenario: ABC heterotrimer with varied MSA coverage across species.

    Protein complex with 3 different chains (Alpha, Beta, Gamma).
    MSA searches found sequences from 5 species with different coverage:

    Species layout (columns):
        0: Human     - well-studied, found in all chains
        1: Mouse     - model organism, found in all chains
        2: Zebrafish - found in Alpha and Beta only (Gamma diverged)
        3: Fly       - found in Alpha and Gamma only (Beta lost the domain)
        4: Yeast     - ancient, only found in Alpha (others diverged too much)

    Sequence counts per chain:
                    Human  Mouse  Zebrafish  Fly  Yeast
        Alpha:        3      2        1       2     1
        Beta:         2      1        2       0     0
        Gamma:        1      2        0       1     0

    Expected pairing behavior:
        1. First, fully paired species (in all 3 chains): Human, Mouse
           - Human: min(3,2,1) = 1 pair
           - Mouse: min(2,1,2) = 1 pair
           After this pass, counts are subtracted:
             Human: [2,1,0], Mouse: [1,0,1] - now both only in 2 chains!

        2. Then, partially paired (in 2 chains) - includes leftover Human/Mouse:
           - Human (Alpha+Beta): min(2,1) = 1 pair, Gamma gets -1
           - Mouse (Alpha+Gamma): min(1,1) = 1 pair, Beta gets -1
           - Zebrafish (Alpha+Beta): min(1,2) = 1 pair, Gamma gets -1
           - Fly (Alpha+Gamma): min(2,1) = 1 pair, Beta gets -1

        3. Yeast excluded by mask: only in 1 chain, can't pair
    """
    # Species indices: Human=0, Mouse=1, Zebrafish=2, Fly=3, Yeast=4
    count_array = np.array(
        [
            #  Human Mouse Zebra Fly Yeast
            [3, 2, 1, 2, 1],  # Alpha
            [2, 1, 2, 0, 0],  # Beta
            [1, 2, 0, 1, 0],  # Gamma
        ]
    )

    # Yeast (index 4) filtered out - only in one chain
    pairing_masks = np.array([True, True, True, True, False])

    result = find_pairing_indices(
        count_array=count_array,
        pairing_masks=pairing_masks,
        max_rows_paired=100,
        min_chains_paired_partial=2,
    )

    expected = np.array(
        [
            # 3-chain pass (fully paired)
            [0, 0, 0],  # Human: all chains
            [1, 1, 1],  # Mouse: all chains
            # 2-chain pass (partial pairs, including leftovers)
            [0, 0, -1],  # Human leftover: Alpha+Beta
            [1, -1, 1],  # Mouse leftover: Alpha+Gamma
            [2, 2, -1],  # Zebrafish: Alpha+Beta
            [3, -1, 3],  # Fly: Alpha+Gamma
        ]
    )

    np.testing.assert_array_equal(result, expected)
