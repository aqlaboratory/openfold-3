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

"""Coverage for ChemicalSteering.on_denoised: the exact per-step
displacement and monotone energy decrease on a violated restraint, and the
structural no-op (`on_denoised(...) is None`) when nothing is usable.
"""

from __future__ import annotations

import torch

from openfold3.steering.engine import ChemicalSteering, Term
from openfold3.steering.potentials import DistanceBoundsPotential
from openfold3.steering.schedules import Constant
from openfold3.steering.types import RestraintSet, SteeringContext, StepState


def _step_state(
    ctx: SteeringContext,
    *,
    step_index: int = 0,
    num_steps: int = 20,
    n_samples: int = 2,
) -> StepState:
    """A StepState shaped the way SampleDiffusion builds one.

    ChemicalSteering reads only `steering_t` and `ctx`, so the rest could be
    anything -- but shaping it like the real call site keeps this from
    quietly diverging from `_sample_rollout`, and means these tests fail
    rather than silently pass if a term starts using a field. Sigmas descend
    (`c_tau < t`) as the noise schedule does, and the coordinate tensors
    carry the sampler's `[B, S, N, 3]` layout against a `[B, 1, N]` mask.
    """
    generator = torch.Generator().manual_seed(step_index)
    shape = (1, n_samples, ctx.n_atoms, 3)
    return StepState(
        xl_noisy=torch.randn(shape, generator=generator),
        noise=torch.randn(shape, generator=generator),
        t=torch.tensor(1.5),
        c_tau=torch.tensor(0.9),
        step_index=step_index,
        num_steps=num_steps,
        start_step=0,
        pass_name="primary",
        atom_mask=torch.ones(1, 1, ctx.n_atoms),
        ctx=ctx,
    )


def test_on_denoised_is_none_when_context_has_no_restraints():
    ctx = SteeringContext(restraints={}, n_atoms=2)
    steering = ChemicalSteering(
        terms={
            "distance_bounds_potential": Term(
                potential=DistanceBoundsPotential(), weight=Constant(0.01)
            )
        },
        num_gd_steps=20,
    )
    x0 = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    update = steering.on_denoised(x0, _step_state(ctx))
    assert update is None


def test_on_denoised_is_none_when_weight_is_zero():
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
        lower=torch.tensor([1.0]),
        upper=torch.tensor([1.0]),
    )
    ctx = SteeringContext(
        restraints={"distance_bounds_potential": restraints}, n_atoms=2
    )
    steering = ChemicalSteering(
        terms={
            "distance_bounds_potential": Term(
                potential=DistanceBoundsPotential(), weight=Constant(0.0)
            )
        },
        num_gd_steps=20,
    )
    x0 = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    update = steering.on_denoised(x0, _step_state(ctx))
    assert update is None


def test_on_denoised_is_none_when_no_term_matches_the_context():
    """A configured term whose restraint family is absent from the context
    (e.g. it produced zero restraints for this query) is skipped, not an
    error."""
    ctx = SteeringContext(restraints={}, n_atoms=2)
    steering = ChemicalSteering(
        terms={
            "distance_bounds_potential": Term(
                potential=DistanceBoundsPotential(), weight=Constant(0.01)
            )
        },
        num_gd_steps=5,
    )
    x0 = torch.randn(1, 1, 2, 3)
    assert steering.on_denoised(x0, _step_state(ctx)) is None


def test_energy_is_zero_on_a_valid_conformer():
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
        lower=torch.tensor([1.0]),
        upper=torch.tensor([1.0]),
    )
    x0 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    energy, _ = DistanceBoundsPotential().energy_and_gradient(x0, restraints, 0.01)
    torch.testing.assert_close(energy, torch.zeros_like(energy))


def test_on_denoised_reduces_bond_violation_by_the_exact_step_size():
    weight = 0.01
    num_gd_steps = 5
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
        lower=torch.tensor([1.0]),
        upper=torch.tensor([1.0]),
    )
    ctx = SteeringContext(
        restraints={"distance_bounds_potential": restraints}, n_atoms=2
    )
    steering = ChemicalSteering(
        terms={
            "distance_bounds_potential": Term(
                potential=DistanceBoundsPotential(), weight=Constant(weight)
            )
        },
        num_gd_steps=num_gd_steps,
    )
    x0 = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    update = steering.on_denoised(x0, _step_state(ctx))
    assert update is not None
    assert update.n_active_terms == 1

    guided = x0 + update.delta
    initial_distance = torch.linalg.norm(x0[..., 0, :] - x0[..., 1, :])
    guided_distance = torch.linalg.norm(guided[..., 0, :] - guided[..., 1, :])

    # The subgradient is exactly +-1, so each of the 5 GD steps moves BOTH
    # atoms by `weight` toward each other, closing the violation by
    # `2 * weight` per step -- independent of the violation's magnitude.
    expected_distance = initial_distance - 2 * num_gd_steps * weight
    torch.testing.assert_close(guided_distance, expected_distance)
    assert guided_distance < initial_distance


def test_on_denoised_strictly_decreases_energy_over_20_steps():
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
        lower=torch.tensor([1.0]),
        upper=torch.tensor([1.0]),
    )
    ctx = SteeringContext(
        restraints={"distance_bounds_potential": restraints}, n_atoms=2
    )
    steering = ChemicalSteering(
        terms={
            "distance_bounds_potential": Term(
                potential=DistanceBoundsPotential(), weight=Constant(0.01)
            )
        },
        num_gd_steps=20,
    )
    x0 = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    potential = DistanceBoundsPotential()

    initial_energy, _ = potential.energy_and_gradient(x0, restraints, 1.0)
    update = steering.on_denoised(x0, _step_state(ctx))
    assert update is not None
    guided = x0 + update.delta
    final_energy, _ = potential.energy_and_gradient(guided, restraints, 1.0)

    assert final_energy.item() < initial_energy.item()


def test_on_denoised_respects_the_interval():
    """A term evaluated only every Nth gradient-descent step contributes
    proportionally less displacement."""
    weight = 0.01
    num_gd_steps = 10
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
        lower=torch.tensor([1.0]),
        upper=torch.tensor([1.0]),
    )
    ctx = SteeringContext(
        restraints={"distance_bounds_potential": restraints}, n_atoms=2
    )
    x0 = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])

    every_step = ChemicalSteering(
        terms={
            "distance_bounds_potential": Term(
                potential=DistanceBoundsPotential(), weight=Constant(weight), interval=1
            )
        },
        num_gd_steps=num_gd_steps,
    )
    every_other_step = ChemicalSteering(
        terms={
            "distance_bounds_potential": Term(
                potential=DistanceBoundsPotential(), weight=Constant(weight), interval=2
            )
        },
        num_gd_steps=num_gd_steps,
    )

    dense_update = every_step.on_denoised(x0, _step_state(ctx))
    sparse_update = every_other_step.on_denoised(x0, _step_state(ctx))
    assert dense_update is not None
    assert sparse_update is not None

    dense_displacement = dense_update.delta.norm()
    sparse_displacement = sparse_update.delta.norm()
    torch.testing.assert_close(dense_displacement, 2 * sparse_displacement)
