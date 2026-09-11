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

"""Run-level settings for chemical steering.

Every numeric default is referenced from ``defaults.py`` rather than
restated here, so the derived parameter table stays in exactly one place
(see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.steering import defaults
from openfold3.steering.potentials import CLASS_REGISTRY, DistanceBoundsPotential


class TermSettings(BaseModel):
    """Settings for one restraint family.

    Attributes:
        enabled (bool):
            Whether this term contributes to the guidance gradient. A
            disabled term emits no restraints, so the sampler never sees it.
        weight (float):
            Coordinate-update weight. Because the flat-bottom subgradient is
            exactly +-1 and a distance has |dv/dx| = 1, this is literally the
            step size: a violating atom moves `weight` Angstrom per
            gradient-descent step.
        interval (int):
            Apply this term every Nth gradient-descent step. Intended for
            terms too expensive to evaluate every step; 1 means every step.
    """

    model_config = PydanticConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = True
    weight: float = Field(default=defaults.DISTANCE_WEIGHT, ge=0.0)
    interval: int = Field(default=1, ge=1)


class SteeringSettings(BaseModel):
    """Settings for inference-time chemical steering of ligand geometry.

    Consumed by `maybe_create_steering_features` in
    `openfold3.steering.featurization`. Reachable from an inference
    experiment config via `dataset_config_kwargs.steering`.

    Steering is off by default and applies to every query in the run when
    enabled: whether ligands are steered is a property of the run, not of an
    individual query, which also makes a steered/unsteered ablation a yaml
    toggle rather than a second set of query JSONs.

    Attributes:
        enabled (bool):
            Whether chemical steering runs at all. Off by default.
        num_gd_steps (int):
            Gradient-descent steps applied to the denoised coordinates at
            each denoising step. With the default weight this bounds the
            per-step correction to `num_gd_steps * weight` Angstrom.
        terms (dict[str, TermSettings]):
            Per-restraint-family settings, keyed by the potential's
            snake_case registry name (a key of
            `openfold3.steering.potentials.CLASS_REGISTRY`, e.g.
            `distance_bounds_potential`). Unknown keys are rejected.
    """

    model_config = PydanticConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    num_gd_steps: int = Field(default=defaults.NUM_GD_STEPS, ge=1)
    terms: dict[str, TermSettings] = Field(
        default_factory=lambda: {DistanceBoundsPotential.name: TermSettings()}
    )

    @field_validator("terms")
    @classmethod
    def _validate_term_names(
        cls, value: dict[str, TermSettings]
    ) -> dict[str, TermSettings]:
        unknown = sorted(set(value) - set(CLASS_REGISTRY))
        if unknown:
            raise ValueError(
                f"unknown steering term(s) {unknown}; available terms are "
                f"{sorted(CLASS_REGISTRY)}"
            )
        return value

    def active_terms(self) -> dict[str, TermSettings]:
        """Terms that are enabled and carry a non-zero weight."""
        return {
            name: term
            for name, term in self.terms.items()
            if term.enabled and term.weight > 0.0
        }
