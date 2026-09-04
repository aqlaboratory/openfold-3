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

"""Time-varying scalar schedules for per-term weights.

See THIRD_PARTY_NOTICES.md for provenance.

A schedule maps normalized diffusion time ``t`` in [0, 1] to a float, where
``t = 1`` is the noisiest step and ``t = 0`` the final (cleanest) step — the
Boltz/Protenix convention adopted by ``StepState.steering_t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class Schedule:
    """Base class: evaluate a weight at normalized time t."""

    def at(self, t: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class Constant(Schedule):
    value: float

    def at(self, t: float) -> float:
        return float(self.value)


@dataclass(frozen=True)
class ExponentialInterpolation(Schedule):
    """Exponential interpolation from ``start`` (t=0) to ``end`` (t=1).

    ``alpha == 0`` degenerates to linear interpolation. Negative ``alpha``
    front-loads the change near t=0.
    """

    start: float
    end: float
    alpha: float = 0.0

    def at(self, t: float) -> float:
        if self.alpha == 0.0:
            return float(self.start + (self.end - self.start) * t)
        num = math.exp(self.alpha * t) - 1.0
        den = math.exp(self.alpha) - 1.0
        return float(self.start + (self.end - self.start) * (num / den))


@dataclass(frozen=True)
class PiecewiseStepFunction(Schedule):
    """Piecewise-constant schedule.

    ``values`` has exactly one more entry than ``thresholds``; the value used
    is the one whose bin contains ``t``. This is how a late-only guidance
    gate is expressed: under the adopted time convention
    (``t = 1 - step/N``), "active only in the final 27.5% of denoising"
    is ``PiecewiseStepFunction(thresholds=(0.275,), values=(weight, 0.0))``.
    """

    thresholds: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.thresholds) + 1:
            raise ValueError(
                "values must have len(thresholds)+1 entries, got "
                f"{len(self.values)} and {len(self.thresholds)}"
            )

    def at(self, t: float) -> float:
        idx = 0
        for thr in self.thresholds:
            if t < thr:
                break
            idx += 1
        return float(self.values[idx])
