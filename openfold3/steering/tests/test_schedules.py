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

from __future__ import annotations

import math

import pytest

from openfold3.steering.schedules import (
    Constant,
    ExponentialInterpolation,
    PiecewiseStepFunction,
)


def test_constant_ignores_t():
    schedule = Constant(0.5)
    assert schedule.at(0.0) == pytest.approx(0.5)
    assert schedule.at(1.0) == pytest.approx(0.5)


def test_exponential_interpolation_alpha_zero_is_linear():
    schedule = ExponentialInterpolation(start=0.0, end=1.0, alpha=0.0)
    assert schedule.at(0.0) == pytest.approx(0.0)
    assert schedule.at(1.0) == pytest.approx(1.0)
    assert schedule.at(0.5) == pytest.approx(0.5)


def test_exponential_interpolation_matches_closed_form():
    schedule = ExponentialInterpolation(start=1.0, end=2.0, alpha=-3.0)
    t = 0.4
    expected = 1.0 + 1.0 * (math.exp(-3.0 * t) - 1.0) / (math.exp(-3.0) - 1.0)
    assert schedule.at(t) == pytest.approx(expected)


def test_piecewise_step_function_selects_the_containing_bin():
    """A late-only guidance gate under this project's time convention
    (t = 1 - step/N): active only for t <= 0.275, i.e. the final 27.5% of
    denoising."""
    schedule = PiecewiseStepFunction(thresholds=(0.275,), values=(0.01, 0.0))
    assert schedule.at(0.0) == pytest.approx(0.01)
    assert schedule.at(0.274) == pytest.approx(0.01)
    assert schedule.at(0.275) == pytest.approx(0.0)
    assert schedule.at(1.0) == pytest.approx(0.0)


def test_piecewise_step_function_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="values must have"):
        PiecewiseStepFunction(thresholds=(0.5,), values=(1.0, 2.0, 3.0))
