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

"""Core types for chemical steering.

RestraintSet and SteeringContext are built once at featurization, on CPU, from
RDKit. Everything downstream of that boundary — StepState, SteeringUpdate, and
the engine in ``engine.py`` — sees only tensors and never imports RDKit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RestraintSet:
    """One flat-bottom restraint family. Device-local, built at featurization."""

    atom_index: Tensor  # int64   [n_restraints, arity]
    lower: Tensor  # float32 [n_restraints]           may be -inf
    upper: Tensor  # float32 [n_restraints]           may be +inf

    def to(self, device: torch.device) -> RestraintSet:
        return RestraintSet(
            atom_index=self.atom_index.to(device),
            lower=self.lower.to(device),
            upper=self.upper.to(device),
        )


@dataclass(frozen=True)
class SteeringContext:
    """Everything steering needs at sampling time. No RDKit, no AtomArray."""

    restraints: Mapping[str, RestraintSet]  # keyed by potential class name
    n_atoms: int  # must match the model's atom axis

    def to(self, device: torch.device) -> SteeringContext:
        return SteeringContext(
            restraints={name: r.to(device) for name, r in self.restraints.items()},
            n_atoms=self.n_atoms,
        )


@dataclass(frozen=True)
class StepState:
    """What the sampler knows at the moment the hook fires."""

    xl_noisy: Tensor  # [B, S, N, 3] denoiser input
    noise: Tensor  # [B, S, N, 3] epsilon drawn this step (FK will need it)
    t: Tensor  # sigma at this step
    c_tau: Tensor  # sigma at the next step
    step_index: int  # 0-based within THIS rollout pass
    num_steps: int
    start_step: int  # non-zero in the pocket-refinement pass
    pass_name: str  # "primary" | "pocket_refine"
    atom_mask: Tensor
    ctx: SteeringContext

    @property
    def steering_t(self) -> float:
        """Normalized time, Boltz/Protenix convention: 1.0 at the start -> 0."""
        return 1.0 - self.step_index / max(1, self.num_steps)


@dataclass(frozen=True)
class SteeringUpdate:
    """An ADDITIVE correction to x0.

    Apply as:  x0_guided = x0 + update.delta
    """

    delta: Tensor  # [B, S, N, 3], same shape as x0, float32
    n_active_terms: int = 0  # diagnostics; 0 means steering was inert
    energy: Tensor | None = None  # per-sample; reserved for Feynman-Kac
