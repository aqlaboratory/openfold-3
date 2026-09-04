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

"""The chemical-steering hook inside SampleDiffusion.

Follows the idioms in test_pocket_constraints.py: an identity denoiser and a
directly constructed SampleDiffusion with the stochastic terms zeroed
(`noise_scale=0`, `gamma_0=0`), so a seed is enough to make a rollout
reproducible. The one thing a seed cannot give is what the
`no_random_augmentation` fixture below provides -- see its docstring.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import cast
from unittest.mock import patch

import pytest
import torch

from openfold3.core.model.structure import diffusion_module
from openfold3.core.model.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
)
from openfold3.steering.batch_features import (
    STEERING_ENABLED_KEY,
    context_to_features,
    term_key,
)
from openfold3.steering.config import SteeringSettings
from openfold3.steering.engine import ChemicalSteering
from openfold3.steering.types import RestraintSet, SteeringContext

_TERM = "distance_bounds_potential"
_N_ATOMS = 4
_SEED = 11

# Three sigmas -> two denoising steps, the smallest schedule that still shows
# steering_t advancing between steps.
_NOISE_SCHEDULE = torch.tensor([1.0, 0.5, 0.1])
_NUM_DENOISING_STEPS = len(_NOISE_SCHEDULE) - 1


class _IdentityDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.input_dtypes: list[torch.dtype] = []

    def forward(self, *, xl_noisy, **_kwargs):
        self.calls += 1
        self.input_dtypes.append(xl_noisy.dtype)
        return xl_noisy


def _sampler(denoiser: torch.nn.Module | None = None) -> SampleDiffusion:
    return SampleDiffusion(
        gamma_0=0.0,
        gamma_min=0.0,
        noise_scale=0.0,
        step_scale=1.0,
        # The stand-in denoisers here satisfy the call contract but not the
        # declared DiffusionModule type.
        diffusion_module=cast(
            DiffusionModule, denoiser if denoiser is not None else _IdentityDenoiser()
        ),
    )


def _base_batch(
    batch_dim: int = 1, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    # The rollout takes its coordinate dtype from atom_mask.
    return {
        "atom_mask": torch.ones(batch_dim, _N_ATOMS, dtype=dtype),
        "token_mask": torch.ones(batch_dim, 1),
    }


def _steering_features(
    *, weight: float = 0.25, num_gd_steps: int = 2
) -> dict[str, torch.Tensor]:
    """One badly violated restraint, so guidance visibly moves atoms.

    The window is one-sided (`[0, 1]`, "no longer than 1 Angstrom") on
    purpose. A zero-width window plus a step size large enough to see would
    overshoot the bound and be pushed back the next gradient step, so a pair
    could land back where it started and the correction would cancel -- which
    is exactly what an earlier version of this file did, leaving the "pulls a
    violated restraint together" assertion to pass on float32 rounding noise
    for one of the two samples.
    """
    ctx = SteeringContext(
        restraints={
            _TERM: RestraintSet(
                atom_index=torch.tensor([[0, 1]], dtype=torch.int64),
                lower=torch.tensor([0.0]),
                upper=torch.tensor([1.0]),
            )
        },
        n_atoms=_N_ATOMS,
    )
    settings = SteeringSettings.model_validate(
        {
            "enabled": True,
            "num_gd_steps": num_gd_steps,
            "terms": {_TERM: {"weight": weight}},
        }
    )
    return context_to_features(ctx, settings)


@pytest.fixture
def no_random_augmentation():
    """Hold the coordinate frame fixed across denoising steps.

    Seeding already makes a rollout reproducible: `_sampler` zeroes the
    stochastic terms, and steering draws no randomness of its own (see
    `test_enabled_steering_consumes_no_extra_randomness`), so a steered and
    an unsteered run walk the same RNG stream. What a seed cannot do is keep
    the two comparable atom by atom -- `centre_random_augmentation` recentres
    on the structure's centroid, so moving the two restrained atoms displaces
    every other atom as well, and "only the restrained atoms moved" stops
    being a meaningful assertion.
    """
    with patch.object(
        diffusion_module,
        "centre_random_augmentation",
        new=lambda *, xl, atom_mask: xl,
    ):
        yield


def _recording_step_state(sink: list) -> type:
    """A StepState subclass that appends each instance the rollout builds."""

    class _RecordingStepState(diffusion_module.StepState):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            sink.append(self)

    return _RecordingStepState


def _run(sampler: SampleDiffusion, batch: dict, batch_dim: int = 1) -> torch.Tensor:
    with torch.no_grad():
        return sampler(
            batch=batch,
            si_input=torch.zeros(batch_dim, 1, 1),
            si_trunk=torch.zeros(batch_dim, 1, 1),
            zij_trunk=torch.zeros(batch_dim, 1, 1, 1),
            noise_schedule=_NOISE_SCHEDULE,
            no_rollout_samples=2,
        )


def _separation(coords: torch.Tensor) -> torch.Tensor:
    """Distance between the two restrained atoms, per sample."""
    return torch.linalg.norm(coords[:, :, 0] - coords[:, :, 1], dim=-1)


def test_disabled_steering_is_bit_identical():
    """The property the whole design hangs on: a batch carrying no steering
    features must produce byte-identical output and consume identical RNG.

    Deliberately leaves augmentation in place -- this is the one test that
    has to exercise the real path end to end.
    """
    torch.manual_seed(_SEED)
    unmodified = _run(_sampler(), _base_batch())
    rng_after_unmodified = torch.random.get_rng_state()

    disabled_batch = _base_batch() | {STEERING_ENABLED_KEY: torch.tensor([False])}
    torch.manual_seed(_SEED)
    disabled = _run(_sampler(), disabled_batch)
    rng_after_disabled = torch.random.get_rng_state()

    torch.testing.assert_close(disabled, unmodified, rtol=0.0, atol=0.0)
    assert torch.equal(rng_after_disabled, rng_after_unmodified)


def test_enabled_steering_consumes_no_extra_randomness():
    """Steering must not draw from the RNG stream: if it did, an enabled run
    would desynchronize every subsequent draw rather than merely adding a
    correction -- and every seeded comparison in this file would be comparing
    two different noise trajectories."""
    torch.manual_seed(_SEED)
    _run(_sampler(), _base_batch())
    rng_after_unsteered = torch.random.get_rng_state()

    torch.manual_seed(_SEED)
    _run(_sampler(), _base_batch() | _steering_features())
    rng_after_steered = torch.random.get_rng_state()

    assert torch.equal(rng_after_steered, rng_after_unsteered)


def test_enabled_steering_changes_the_output(no_random_augmentation):
    torch.manual_seed(_SEED)
    unsteered = _run(_sampler(), _base_batch())

    torch.manual_seed(_SEED)
    steered = _run(_sampler(), _base_batch() | _steering_features())

    assert not torch.allclose(steered, unsteered)
    # Only the restrained atoms move; the rest of the structure is untouched.
    torch.testing.assert_close(steered[:, :, 2:], unsteered[:, :, 2:])


def test_steering_pulls_a_violated_restraint_together(no_random_augmentation):
    """Guidance should shorten an over-long restrained distance."""
    torch.manual_seed(_SEED)
    unsteered = _run(_sampler(), _base_batch())
    torch.manual_seed(_SEED)
    steered = _run(_sampler(), _base_batch() | _steering_features())

    # Precondition: the restraint (upper bound 1.0) is violated to begin
    # with, so there is something for guidance to correct.
    assert torch.all(_separation(unsteered) > 1.0)
    assert torch.all(_separation(steered) < _separation(unsteered))


def test_steering_hook_runs_every_denoising_step() -> None:
    """One call per denoising step, with steering_t following Boltz's and
    Protenix's `1 - step_index / num_steps` convention rather than an
    off-by-one variant of it.

    Written out for this schedule: two steps, so steering_t starts at 1.0
    (`1 - 0/2`) and reaches 0.5 (`1 - 1/2`) on the last one. Note it never
    reaches 0 -- the convention counts steps taken, not sigmas visited, which
    is the off-by-one this pins down.
    """
    states: list = []
    denoiser = _IdentityDenoiser()

    with patch.object(diffusion_module, "StepState", new=_recording_step_state(states)):
        _run(_sampler(denoiser), _base_batch() | _steering_features())

    expected_steering_t = [
        1.0 - step / _NUM_DENOISING_STEPS for step in range(_NUM_DENOISING_STEPS)
    ]
    assert denoiser.calls == _NUM_DENOISING_STEPS
    assert expected_steering_t == [1.0, 0.5]  # guards the formula above
    assert [state.steering_t for state in states] == expected_steering_t


def test_steering_does_not_run_in_the_pocket_refinement_pass(
    no_random_augmentation,
) -> None:
    """The refinement rollout is deliberately unsteered, so the hook fires
    only for the primary pass's steps."""
    states: list = []
    batch = _base_batch() | _steering_features()
    batch.update(
        {
            "pocket_sampling_enabled": torch.tensor([True]),
            "pocket_sampling_ligand_atom_mask": torch.tensor([[0, 0, 1, 1]]),
            "pocket_sampling_start_frac": torch.tensor([0.5]),
            "pocket_sampling_ligand_jitter": torch.tensor([0.0]),
        }
    )
    denoiser = _IdentityDenoiser()

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                diffusion_module, "StepState", new=_recording_step_state(states)
            )
        )
        stack.enter_context(
            patch.object(
                diffusion_module,
                "_build_pocket_sampling_seeds",
                new=lambda **_kwargs: torch.zeros(1, 2, _N_ATOMS, 3),
            )
        )
        _run(_sampler(denoiser), batch)

    # Three denoiser calls (2 primary + 1 refinement) but only the two
    # primary steps are steered.
    assert denoiser.calls == 3
    assert [state.step_index for state in states] == [0, 1]


def test_steering_runs_in_float32_under_a_bfloat16_rollout(
    no_random_augmentation,
) -> None:
    """The `.float()` on the way into the engine is load-bearing.

    Potentials accumulate `num_gd_steps` corrections and compare distances
    against bounds; bf16's ~3 decimal digits would quantize both. The engine
    therefore sees float32 no matter what precision the rollout runs at.
    """
    seen: list[torch.dtype] = []
    original = ChemicalSteering.on_denoised

    def _record(self, coords, state):
        seen.append(coords.dtype)
        return original(self, coords, state)

    torch.manual_seed(_SEED)
    with patch.object(ChemicalSteering, "on_denoised", new=_record):
        _run(_sampler(), _base_batch(dtype=torch.bfloat16) | _steering_features())

    assert seen == [torch.float32, torch.float32]


def test_bfloat16_rollout_is_steered_without_changing_any_dtype(no_random_augmentation):
    """Guidance is cast back to the denoised tensor's dtype, so a bf16 rollout
    stays bf16 throughout: same denoiser input dtypes, same output dtype,
    different coordinates."""
    unsteered_denoiser = _IdentityDenoiser()
    torch.manual_seed(_SEED)
    unsteered = _run(_sampler(unsteered_denoiser), _base_batch(dtype=torch.bfloat16))

    steered_denoiser = _IdentityDenoiser()
    torch.manual_seed(_SEED)
    steered = _run(
        _sampler(steered_denoiser),
        _base_batch(dtype=torch.bfloat16) | _steering_features(),
    )

    assert steered.dtype == unsteered.dtype == torch.bfloat16
    assert steered_denoiser.input_dtypes == unsteered_denoiser.input_dtypes
    assert not torch.allclose(steered.float(), unsteered.float())
    assert torch.all(_separation(steered) < _separation(unsteered))


def test_malformed_steering_features_raise_before_the_denoiser_runs():
    denoiser = _IdentityDenoiser()
    batch = _base_batch() | _steering_features()
    del batch[term_key(_TERM, "lower")]

    with pytest.raises(ValueError, match="lower.* is missing"):
        _run(_sampler(denoiser), batch)

    assert denoiser.calls == 0


def test_steering_rejects_multi_query_batches():
    batch = _base_batch(batch_dim=2) | _steering_features()
    with pytest.raises(ValueError, match="one query per model batch"):
        _run(_sampler(), batch, batch_dim=2)


def test_an_unsteered_multi_query_batch_never_reaches_the_steering_guard():
    """The batch-size restriction belongs to steering, not to the sampler.

    `prepare_steering` returns early for a batch with no steering features,
    before its batch_dim check -- so an ordinary multi-query rollout is
    unaffected. That early return is load-bearing and this pins it.
    """
    denoiser = _IdentityDenoiser()

    _run(_sampler(denoiser), _base_batch(batch_dim=2), batch_dim=2)

    assert denoiser.calls == 2


def test_use_steering_without_a_prepared_object_is_rejected():
    """The two rollout arguments must agree: `use_steering=True` with nothing
    prepared would otherwise run an unsteered rollout under a name that says
    the opposite."""
    with pytest.raises(ValueError, match="requires a prepared steering object"):
        _sampler()._sample_rollout(
            batch=_base_batch(),
            xl=torch.zeros(1, 2, _N_ATOMS, 3),
            atom_mask=torch.ones(1, _N_ATOMS),
            si_input=torch.zeros(1, 1, 1),
            si_trunk=torch.zeros(1, 1, 1),
            zij_trunk=torch.zeros(1, 1, 1, 1),
            noise_schedule=torch.tensor([1.0, 0.5]),
            start_step=0,
            use_conditioning=True,
            use_steering=True,
            steering=None,
        )
