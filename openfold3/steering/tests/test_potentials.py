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

"""Analytic-gradient correctness against an independent autograd oracle,
plus a broadcast-regression sweep across batch shapes -- including batch
sizes that collide with the restraint arity, and a multi-sample (S > 1)
rollout.
"""

from __future__ import annotations

import pytest
import torch

from openfold3.steering.potentials import (
    DistanceBoundsPotential,
    register,
)
from openfold3.steering.types import RestraintSet


def _reference_flat_bottom(
    value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Independent, differentiable oracle for the flat-bottom energy."""
    return torch.relu(lower.expand_as(value) - value) + torch.relu(
        value - upper.expand_as(value)
    )


def _autograd_distance_gradient(
    coords: torch.Tensor, restraints: RestraintSet, weight: float
) -> torch.Tensor:
    """Independent autograd oracle: same restraints, built without reusing
    DistanceBoundsPotential.compute_variable."""
    coords = coords.clone().requires_grad_(True)
    i, j = restraints.atom_index[:, 0], restraints.atom_index[:, 1]
    diff = coords.index_select(-2, i) - coords.index_select(-2, j)
    value = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)
    energy = (
        weight * _reference_flat_bottom(value, restraints.lower, restraints.upper)
    ).sum()
    (grad,) = torch.autograd.grad(energy, coords)
    return grad


def test_distance_gradient_matches_autograd():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]]])
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
        lower=torch.tensor([1.0]),
        upper=torch.tensor([1.0]),
    )
    weight = 0.01

    potential = DistanceBoundsPotential()
    _, analytic_grad = potential.energy_and_gradient(coords, restraints, weight)
    autograd_grad = _autograd_distance_gradient(coords, restraints, weight)

    torch.testing.assert_close(analytic_grad, autograd_grad)


def test_distance_gradient_matches_autograd_with_upper_only_bound():
    coords = torch.tensor([[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]])
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
        lower=torch.tensor([float("-inf")]),
        upper=torch.tensor([1.0]),
    )
    weight = 1.0

    potential = DistanceBoundsPotential()
    _, analytic_grad = potential.energy_and_gradient(coords, restraints, weight)
    autograd_grad = _autograd_distance_gradient(coords, restraints, weight)

    torch.testing.assert_close(analytic_grad, autograd_grad)
    # The restraint is violated (distance 2.0 > upper 1.0), so it must be active.
    assert not torch.equal(analytic_grad, torch.zeros_like(analytic_grad))


def test_distance_energy_and_gradient_are_zero_inside_the_window():
    coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
        lower=torch.tensor([0.5]),
        upper=torch.tensor([1.5]),
    )
    potential = DistanceBoundsPotential()
    energy, grad = potential.energy_and_gradient(coords, restraints, 1.0)

    torch.testing.assert_close(energy, torch.zeros_like(energy))
    torch.testing.assert_close(grad, torch.zeros_like(grad))


def test_energy_and_gradient_is_a_noop_with_no_restraints():
    torch.manual_seed(0)
    coords = torch.randn(2, 3, 4, 3)
    restraints = RestraintSet(
        atom_index=torch.empty((0, 2), dtype=torch.int64),
        lower=torch.empty((0,)),
        upper=torch.empty((0,)),
    )
    potential = DistanceBoundsPotential()
    energy, grad = potential.energy_and_gradient(coords, restraints, 0.01)

    assert energy.shape == coords.shape[:-2]
    torch.testing.assert_close(energy, torch.zeros_like(energy))
    torch.testing.assert_close(grad, torch.zeros_like(coords))


# ---------------------------------------------------------------------------
# Broadcast-regression sweep: every leading-axis shape energy_and_gradient
# must handle, including batch sizes that collide with the restraint arity.
# ---------------------------------------------------------------------------

_BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
def test_broadcast_regression_unbatched_n3(batch_size: int):
    """[N, 3] coords (no leading batch/sample axes) must work for every
    restraint-family size, including sizes that collide with arity (2)."""
    torch.manual_seed(batch_size)
    n_atoms = max(batch_size, 2)
    coords = torch.randn(n_atoms, 3, dtype=torch.float64)
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
        lower=torch.tensor([1.0], dtype=torch.float64),
        upper=torch.tensor([1.0], dtype=torch.float64),
    )
    potential = DistanceBoundsPotential()
    energy, grad = potential.energy_and_gradient(coords, restraints, 0.01)

    assert energy.shape == ()
    assert grad.shape == coords.shape
    autograd_grad = _autograd_distance_gradient(coords, restraints, 0.01)
    torch.testing.assert_close(grad, autograd_grad)


@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
def test_broadcast_regression_batched_bsn3(batch_size: int):
    """[B, S, N, 3] coords across a range of batch sizes, including sizes
    that collide with arity (2)."""
    torch.manual_seed(batch_size)
    n_atoms = 5
    n_samples = 3
    coords = torch.randn(batch_size, n_samples, n_atoms, 3, dtype=torch.float64)
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
        lower=torch.tensor([1.0, 0.5], dtype=torch.float64),
        upper=torch.tensor([1.0, 2.0], dtype=torch.float64),
    )
    potential = DistanceBoundsPotential()
    energy, grad = potential.energy_and_gradient(coords, restraints, 0.02)

    assert energy.shape == (batch_size, n_samples)
    assert grad.shape == coords.shape
    autograd_grad = _autograd_distance_gradient(coords, restraints, 0.02)
    torch.testing.assert_close(grad, autograd_grad)


def test_broadcast_regression_sample_axis_s_greater_than_one():
    """S > 1 with B == 1: a dedicated multi-sample rollout, where each
    sample must get its own, independently correct gradient."""
    torch.manual_seed(42)
    coords = torch.randn(1, 5, 4, 3, dtype=torch.float64)
    restraints = RestraintSet(
        atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
        lower=torch.tensor([1.0], dtype=torch.float64),
        upper=torch.tensor([1.0], dtype=torch.float64),
    )
    potential = DistanceBoundsPotential()
    energy, grad = potential.energy_and_gradient(coords, restraints, 0.01)

    assert energy.shape == (1, 5)
    # Each of the 5 samples must get its own, independently correct gradient.
    autograd_grad = _autograd_distance_gradient(coords, restraints, 0.01)
    torch.testing.assert_close(grad, autograd_grad)
    for sample in range(5):
        assert not torch.allclose(grad[0, sample], grad[0, (sample + 1) % 5])


def test_registering_under_a_non_snake_case_name_is_rejected():
    """The registry name is what a user types in the runner yaml, so the
    snake_case convention is enforced at registration rather than left to
    whoever adds the next potential."""
    with pytest.raises(ValueError, match="must be snake_case"):
        register("DistanceBoundsPotential")


def test_registering_two_potentials_under_one_name_is_rejected(register_throwaway):
    """The registry key is also the user-facing config key and the batch
    feature prefix, so a silent overwrite would make one potential
    unreachable and misroute the other's restraints."""
    register_throwaway("colliding_potential")

    with pytest.raises(ValueError, match="already registered"):
        register_throwaway("colliding_potential")


def test_re_registering_the_same_class_is_allowed(
    register_throwaway, isolated_registry
):
    """Idempotent, so a module re-import does not explode."""
    potential = register_throwaway("reimported_potential")
    register("reimported_potential")(potential)

    assert isolated_registry["reimported_potential"] is potential
