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

"""The batch wire format for chemical steering.

Steering restraints are built during featurization (CPU, per query) but
applied in the sampler (GPU, per denoising step), and the only channel
between the two is the feature batch. A ``SteeringContext`` cannot travel
that channel as an object: the collator sends every entry through
``pad_sequence(...).squeeze(-1)`` (``core/data/framework/data_module.py``)
and the model then ``unsqueeze(1)``s every leaf, and only two hardcoded keys
have a non-tensor escape hatch.

So the context is taken apart into plain tensors by ``context_to_features``
in the featurizer, and reassembled by ``prepare_steering`` in the sampler.
Both functions live in this module so that the key names they agree on are
defined once. Splitting them would make a rename on one side silently
disagree with the other, and the symptom would be steering quietly doing
nothing rather than an error.

Reassembly does not trust incoming shapes. Because the collator pads and
squeezes, an emitted ``[n, 2]`` may arrive as ``[1, n, 2]`` or
``[1, 1, n, 2]``, and an emitted ``[n]`` collapses to ``[1]`` when
``n == 1``. Every tensor is therefore reshaped using an explicitly emitted
``_count``, never by inferring which axis is which.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from openfold3.steering.config import SteeringSettings
from openfold3.steering.engine import ChemicalSteering, Term
from openfold3.steering.potentials import CLASS_REGISTRY
from openfold3.steering.schedules import Constant
from openfold3.steering.types import RestraintSet, SteeringContext

STEERING_ENABLED_KEY = "steering_enabled"
STEERING_NUM_GD_STEPS_KEY = "steering_num_gd_steps"
STEERING_N_ATOMS_KEY = "steering_n_atoms"


def term_key(term_name: str, suffix: str) -> str:
    """Batch feature key for one field of one restraint family.

    >>> term_key("distance_bounds_potential", "lower")
    'steering_distance_bounds_potential_lower'
    """
    return f"steering_{term_name}_{suffix}"


@dataclass(frozen=True)
class PreparedSteering:
    """Everything the sampler needs, rebuilt from a collated batch."""

    engine: ChemicalSteering
    ctx: SteeringContext


def context_to_features(
    ctx: SteeringContext,
    settings: SteeringSettings,
) -> dict[str, torch.Tensor]:
    """Flatten a SteeringContext plus its settings into batch tensors.

    Scalars are emitted as 1-element tensors, matching the convention the
    collator and `_batch_scalar` expect.

    Args:
        ctx: the restraints extracted for this query, plus the atom-axis
            length they were built against.
        settings: the run's steering settings. Only terms that are enabled,
            carry a non-zero weight, and have restraints in ``ctx`` are
            emitted; a term configured here but absent from ``ctx`` is
            skipped rather than emitted empty.

    Returns:
        Batch features, or ``{}`` when no term has any restraints -- emitting
        no keys at all is what makes disabled steering a structural no-op.
    """
    active = {
        name: term
        for name, term in settings.active_terms().items()
        if ctx.restraints.get(name) is not None
        and ctx.restraints[name].atom_index.numel() > 0
    }
    if not active:
        return {}

    features: dict[str, torch.Tensor] = {
        STEERING_ENABLED_KEY: torch.tensor([True], dtype=torch.bool),
        STEERING_NUM_GD_STEPS_KEY: torch.tensor(
            [settings.num_gd_steps], dtype=torch.long
        ),
        STEERING_N_ATOMS_KEY: torch.tensor([ctx.n_atoms], dtype=torch.long),
    }
    for name, term in active.items():
        restraints = ctx.restraints[name]
        features[term_key(name, "atom_index")] = restraints.atom_index.to(torch.int64)
        features[term_key(name, "lower")] = restraints.lower.to(torch.float32)
        features[term_key(name, "upper")] = restraints.upper.to(torch.float32)
        features[term_key(name, "count")] = torch.tensor(
            [restraints.atom_index.shape[0]], dtype=torch.long
        )
        features[term_key(name, "weight")] = torch.tensor(
            [term.weight], dtype=torch.float32
        )
        features[term_key(name, "interval")] = torch.tensor(
            [term.interval], dtype=torch.long
        )
    return features


def steering_enabled(batch: Mapping[str, torch.Tensor]) -> bool:
    """Whether this batch carries steering features.

    Probes with ``.get`` because the key is absent entirely when steering is
    off; every other key is then required, and a missing one raises.
    """
    enabled = batch.get(STEERING_ENABLED_KEY)
    return enabled is not None and bool(enabled.flatten()[0].item())


def _required(batch: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in batch:
        raise ValueError(f"steering feature {key!r} is missing from the batch")
    return batch[key]


def _scalar_int(batch: Mapping[str, torch.Tensor], key: str) -> int:
    return int(_required(batch, key).flatten()[0].item())


def _scalar_float(batch: Mapping[str, torch.Tensor], key: str) -> float:
    return float(_required(batch, key).flatten()[0].item())


def _restraints_from_batch(
    batch: Mapping[str, torch.Tensor],
    term_name: str,
    *,
    n_atoms: int,
    device: torch.device,
) -> RestraintSet:
    """Rebuild one term's RestraintSet from its four batch features.

    Reads ``steering_<term>_count`` first and treats it as the shape
    authority: the collator pads and the model unsqueezes a sample axis, so
    the stored shape of ``atom_index`` is not trustworthy and orientation is
    never inferred from it. Every other field is then checked against that
    count, which is what turns a silently mangled feature into an error.

    Args:
        batch: collated batch carrying this term's features.
        term_name: registry key of the potential, e.g.
            ``"distance_bounds_potential"``. Supplies the arity.
        n_atoms: size of the model's atom axis; atom indices must fall inside
            it.
        device: device to place the rebuilt tensors on.

    Returns:
        The term's restraints, indices as int64 and bounds as float32.

    Raises:
        ValueError: if a feature is missing, has the wrong dtype, has an
            element count inconsistent with ``count``, or indexes an atom
            outside ``[0, n_atoms)``.
    """
    arity = CLASS_REGISTRY[term_name].arity
    count = _scalar_int(batch, term_key(term_name, "count"))
    if count < 0:
        raise ValueError(
            f"steering term {term_name!r} has a negative restraint count {count}"
        )

    index_key = term_key(term_name, "atom_index")
    raw_index = _required(batch, index_key)
    if raw_index.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"steering feature {index_key!r} must be an integer tensor; "
            f"got {raw_index.dtype}"
        )
    if raw_index.numel() != count * arity:
        raise ValueError(
            f"steering feature {index_key!r} has {raw_index.numel()} elements, "
            f"expected {count * arity} for {count} restraint(s) of arity {arity}"
        )
    atom_index = raw_index.reshape(count, arity).to(device=device, dtype=torch.int64)
    if count and (int(atom_index.min()) < 0 or int(atom_index.max()) >= n_atoms):
        raise ValueError(
            f"steering feature {index_key!r} indexes atoms outside [0, {n_atoms}): "
            f"[{int(atom_index.min())}, {int(atom_index.max())}]"
        )

    bounds = []
    for suffix in ("lower", "upper"):
        key = term_key(term_name, suffix)
        raw = _required(batch, key)
        if not raw.is_floating_point():
            raise ValueError(
                f"steering feature {key!r} must be a float tensor; got {raw.dtype}"
            )
        if raw.numel() != count:
            raise ValueError(
                f"steering feature {key!r} has {raw.numel()} elements, expected {count}"
            )
        bounds.append(raw.reshape(count).to(device=device, dtype=torch.float32))

    return RestraintSet(atom_index=atom_index, lower=bounds[0], upper=bounds[1])


def prepare_steering(
    batch: Mapping[str, torch.Tensor],
    atom_mask: torch.Tensor,
) -> PreparedSteering | None:
    """Rebuild the steering engine and context from a collated batch.

    Called once per forward, before the first denoiser call, so a malformed
    batch fails before any compute is spent.

    The batch keys consumed, all produced by ``context_to_features`` (which
    emits none of them when steering is off -- hence the ``.get`` probe):

    ==================================  =======  ==========================
    key                                 dtype    meaning
    ==================================  =======  ==========================
    ``steering_enabled``                bool     master switch; absence
                                                 means no steering
    ``steering_num_gd_steps``           int64    gradient steps per
                                                 denoising step
    ``steering_n_atoms``                int64    atom axis the restraints
                                                 were built for
    ``steering_<term>_atom_index``      int64    ``count * arity`` atom
                                                 indices
    ``steering_<term>_lower/_upper``    float32  ``count`` bounds each
    ``steering_<term>_count``           int64    restraint count; the shape
                                                 authority
    ``steering_<term>_weight``          float32  per-step step size (A)
    ``steering_<term>_interval``        int64    apply every Nth GD step
    ==================================  =======  ==========================

    ``<term>`` is a ``CLASS_REGISTRY`` key, so a registered term with no
    ``atom_index`` key in this batch was simply disabled or produced no
    restraints -- not an error.

    Args:
        batch: collated batch, one query per model batch.
        atom_mask: the model's atom mask, ``[batch, ..., n_atoms]``. Supplies
            the batch size, the atom-axis length to validate against, and the
            device the restraints are placed on. Its *values* are not read:
            guidance gradients are not masked, matching upstream.

    Returns:
        A ``PreparedSteering``, or ``None`` when this batch is simply not
        steered. That is an ordinary outcome, not a failure, and covers three
        cases: the run has steering off (`SteeringSettings.enabled` is
        False); the query had nothing to restrain, e.g. no ligand, so
        featurization emitted no keys (see
        ``featurization.maybe_create_steering_features``); or the batch
        carries an enable flag but no registered term has restraints in it.

    Raises:
        ValueError: if the batch holds more than one query, if steering
            features are present but malformed, or if ``steering_n_atoms``
            disagrees with the model's atom axis.
    """
    # The enable probe comes first, and the order matters: everything below
    # applies only to runs that actually steer. In particular the batch-size
    # restriction is a steering restriction, not a sampler-wide one -- an
    # unsteered multi-query rollout is ordinary and must not raise here.
    if not steering_enabled(batch):
        return None

    batch_dim, n_atoms = atom_mask.shape[0], atom_mask.shape[-1]
    if batch_dim != 1:
        raise ValueError(
            "Chemical steering currently supports one query per model batch"
        )

    emitted_n_atoms = _scalar_int(batch, STEERING_N_ATOMS_KEY)
    if emitted_n_atoms != n_atoms:
        raise ValueError(
            f"steering features were built for {emitted_n_atoms} atoms but the "
            f"model's atom axis has {n_atoms}"
        )

    device = atom_mask.device
    restraints: dict[str, RestraintSet] = {}
    terms: dict[str, Term] = {}
    for term_name, potential_cls in CLASS_REGISTRY.items():
        # A registered term with no features in this batch was disabled or
        # produced no restraints for this query, which is not an error.
        if term_key(term_name, "atom_index") not in batch:
            continue
        restraints[term_name] = _restraints_from_batch(
            batch, term_name, n_atoms=n_atoms, device=device
        )
        terms[term_name] = Term(
            potential=potential_cls(),
            weight=Constant(_scalar_float(batch, term_key(term_name, "weight"))),
            interval=_scalar_int(batch, term_key(term_name, "interval")),
        )

    if not terms:
        return None

    engine = ChemicalSteering(
        terms=terms, num_gd_steps=_scalar_int(batch, STEERING_NUM_GD_STEPS_KEY)
    )
    return PreparedSteering(
        engine=engine,
        ctx=SteeringContext(restraints=restraints, n_atoms=n_atoms),
    )
