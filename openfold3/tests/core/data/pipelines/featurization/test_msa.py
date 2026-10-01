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

"""Tests for MsaFeaturizerOF3.create_features, specifically deletion_value.

deletion_value should equal (2/pi) * arctan(deletion_count / 3), per AF2/AF3.
A prior implementation computed the (2/pi) coefficient as
`2.0 / torch.acos(zeros) * 2`, which due to operator precedence evaluates to
`(2.0 / (pi/2)) * 2 == 8/pi` instead of `2.0 / (pi * 2 / 2) == 2/pi` -- exactly
4x too large.
"""

import math

import numpy as np
import pytest
import torch

from openfold3.core.data.pipelines.featurization.msa import (
    MsaFeaturizerOF3,
    MsaFeaturizerOF3Config,
)
from openfold3.core.data.primitives.featurization.msa import MsaFeaturePrecursorOF3


def _build_precursor(deletion_counts: list[int]) -> MsaFeaturePrecursorOF3:
    n_cols = len(deletion_counts)
    return MsaFeaturePrecursorOF3(
        msa=np.full((1, n_cols), "A"),
        msa_index=np.zeros((1, n_cols), dtype=int),
        deletion_matrix=np.array([deletion_counts]),
        n_rows_paired=0,
        msa_mask=np.ones((1, n_cols)),
        msa_profile=np.zeros((n_cols, 1)),
        deletion_mean=np.zeros(n_cols),
    )


def _featurizer() -> MsaFeaturizerOF3:
    return MsaFeaturizerOF3(
        MsaFeaturizerOF3Config(
            max_rows=1, max_rows_paired=0, subsample_with_bands=False
        )
    )


@pytest.mark.parametrize(
    "deletion_count, expected",
    [
        pytest.param(0, 0.0, id="no-deletion-is-zero"),
        # atan(3/3) = atan(1) = pi/4, so (2/pi) * (pi/4) == 0.5 exactly.
        # The prior buggy coefficient (8/pi) would give 2.0 here instead.
        pytest.param(3, 0.5, id="hand-computed-exact-value"),
    ],
)
def test_deletion_value_matches_af3_formula(deletion_count, expected):
    precursor = _build_precursor([deletion_count])
    features = _featurizer().create_features(precursor)

    assert features["deletion_value"].item() == pytest.approx(expected, abs=1e-6)


def test_deletion_value_matches_reference_formula_across_range():
    deletion_counts = [0, 1, 2, 3, 5, 10, 50, 200]
    precursor = _build_precursor(deletion_counts)
    features = _featurizer().create_features(precursor)

    expected = torch.tensor(
        [(2.0 / math.pi) * math.atan(d / 3.0) for d in deletion_counts]
    )
    torch.testing.assert_close(
        features["deletion_value"][0], expected, atol=1e-6, rtol=1e-6
    )
