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

"""The gradient-descent guidance loop.

Follows Boltz's guidance step (``src/boltz/model/modules/diffusion.py``,
MIT). See THIRD_PARTY_NOTICES.md for full provenance.

The weight is the step size: the energy's gradient with respect to the
restrained variable is +-w (see ``potentials.Potential.energy_and_gradient``),
and for a distance restraint ``|dv/dx| = 1``, so an update of
``x -= grad`` moves a violating atom by exactly ``w`` Angstrom per gradient
step, with no separate learning rate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from openfold3.steering.potentials import Potential
from openfold3.steering.schedules import Schedule
from openfold3.steering.types import SteeringUpdate, StepState


@dataclass(frozen=True)
class Term:
    """One potential plus its weight schedule and evaluation cadence."""

    potential: Potential
    weight: Schedule
    interval: int = 1  # apply every Nth gradient-descent step


class ChemicalSteering:
    """Gradient-descent guidance on the denoised coordinate estimate.

    Terms are keyed by potential class name, matching
    ``potentials.CLASS_REGISTRY`` and the restraint-family keys on
    ``SteeringContext.restraints`` — so a term with no matching restraints in
    the context (or with a restraint family that produced zero restraints
    per-query) is silently skipped rather than erroring, and a disabled or
    all-zero-weight configuration returns ``None`` structurally.
    """

    def __init__(self, terms: Mapping[str, Term], num_gd_steps: int):
        self.terms = terms
        self.num_gd_steps = num_gd_steps

    def on_denoised(self, x0: torch.Tensor, step: StepState) -> SteeringUpdate | None:
        t = step.steering_t

        # Resolve once per call: weight and interval are both functions of
        # `t` (fixed for this call) and the context (fixed for this call),
        # never of `gd_step`. A term stays out of every remaining gd_step
        # once excluded here -- there is no scenario where it would start
        # contributing partway through. This also gives the structural
        # no-op: an empty `usable` returns None without looping.
        #
        # NOTE: an earlier version of this loop instead checked "did any
        # term fire on THIS gd_step" and `break`-ed the whole loop when it
        # hadn't. That is wrong whenever a term's `interval > 1` and no
        # other term covers the gap: the loop would terminate at the first
        # skipped step instead of resuming on the next step the interval
        # permits (e.g. interval=2 firing on steps 0, 2, 4, ... 18 of 20).
        usable = []
        for name, term in self.terms.items():
            w = term.weight.at(t)
            if w <= 0.0:
                continue
            restraints = step.ctx.restraints.get(name)
            if restraints is None or restraints.atom_index.numel() == 0:
                continue
            usable.append((term, restraints, w))

        if not usable:
            return None

        update = torch.zeros_like(x0)
        for gd_step in range(self.num_gd_steps):
            grad = torch.zeros_like(x0)
            step_active = False
            for term, restraints, w in usable:
                if gd_step % max(1, term.interval):
                    continue
                _, g = term.potential.energy_and_gradient(x0 + update, restraints, w)
                grad = grad + g
                step_active = True
            if step_active:
                update = update - grad

        return SteeringUpdate(delta=update, n_active_terms=len(usable))
