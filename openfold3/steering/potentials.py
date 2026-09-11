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

"""Flat-bottom chemical potentials with analytic gradients.

Every potential is a flat-bottom restraint: zero energy inside a permitted
interval, linear penalty outside it. The energy is

    U = sum_k w_k * [max(0, l_k - v_k) + max(0, v_k - u_k)]

so dU/dv = -w inside the violated-below region and +w inside violated-above,
and zero in the window. Gradients are analytic, not autograd: the sampler runs
under ``torch.no_grad()`` / inference mode (see
``openfold3/projects/of3_all_atom/model.py``), where autograd on the denoised
tensor would raise, and the guidance loop evaluates each term
``num_gd_steps`` times per denoising step — taping that through autograd would
cost memory proportional to the graph for no benefit, since these are
closed-form derivatives.

Formulation and default parameters are derived from Boltz
(``src/boltz/model/potentials/potentials.py``, MIT). The registry pattern is
derived from Protenix. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import torch
from torch import Tensor

from openfold3.steering.types import RestraintSet

CLASS_REGISTRY: dict[str, type[Potential]] = {}

_TERM_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def register(name: str) -> Callable[[type[Potential]], type[Potential]]:
    """Register a potential under a snake_case name.

    That one string is load-bearing in three places: it is the key users
    write in the runner yaml, the prefix of this term's batch feature keys,
    and how the sampler matches a configured term to its restraints. Since
    users type it, it is snake_case rather than the class name -- and it is
    declared rather than derived from the class, because deriving mangles
    acronyms (`VDWOverlapPotential` would become `v_d_w_overlap_potential`)
    and would silently rename a config key when a class is renamed.

    A silent overwrite would make one potential unreachable and misroute the
    other's restraints, so a collision is rejected outright. Re-registering
    the identical class is a no-op, so a module re-import does not fail.

    Args:
        name: the term's public name, snake_case (`[a-z][a-z0-9_]*`).

    Returns:
        A class decorator that records the name on the class as ``name`` and
        adds it to ``CLASS_REGISTRY``.

    Raises:
        ValueError: if the name is not snake_case, or if a different class is
            already registered under it.
    """
    if not _TERM_NAME_PATTERN.match(name):
        raise ValueError(
            f"potential name {name!r} must be snake_case, e.g. 'distance_bounds'"
        )

    def _register(cls: type[Potential]) -> type[Potential]:
        registered = CLASS_REGISTRY.get(name)
        if registered is not None and registered is not cls:
            raise ValueError(
                f"a different potential is already registered as {name!r}: "
                f"{registered.__module__}.{registered.__qualname__}"
            )
        cls.name = name
        CLASS_REGISTRY[name] = cls
        return cls

    return _register


class Potential(ABC):
    """A flat-bottom restraint family: geometry and energy only.

    Indices and bounds arrive prebuilt in a ``RestraintSet``; a potential
    never touches RDKit or an AtomArray.

    Attributes:
        name: the snake_case registry key, set by ``@register``.
        arity: how many atoms each restraint relates.
    """

    name: ClassVar[str]
    arity: int

    @abstractmethod
    def compute_variable(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """positions [M, R, arity, 3] -> (v [M, R], dv_dx [M, R, arity, 3])"""

    def energy_and_gradient(
        self, coords: Tensor, r: RestraintSet, weight: float
    ) -> tuple[Tensor, Tensor]:
        """Energy and dE/dx for one restraint family at one weight.

        ``coords`` may be ``[N, 3]`` or carry any number of leading batch/
        sample axes, e.g. ``[B, S, N, 3]``. The ``reshape(-1, n_atoms, 3)``
        below is load-bearing: it is what makes the term correct regardless
        of how many leading axes ``coords`` carries, including batch sizes
        that happen to collide with ``arity``.
        """
        leading, n_atoms = coords.shape[:-2], coords.shape[-2]
        flat = coords.reshape(-1, n_atoms, 3)  # [M, N, 3]

        if r.atom_index.numel() == 0:
            return coords.new_zeros(leading), torch.zeros_like(coords)

        pos = flat[:, r.atom_index]  # [M, R, arity, 3]
        v, dv_dx = self.compute_variable(pos)

        below, above = r.lower - v, v - r.upper  # inf bounds never fire
        energy = weight * (below.clamp(min=0) + above.clamp(min=0))
        de_dv = weight * ((above > 0).to(v.dtype) - (below > 0).to(v.dtype))

        contrib = de_dv[..., None, None] * dv_dx  # [M, R, arity, 3]
        grad = torch.zeros_like(flat)
        grad.index_add_(
            1,
            r.atom_index.reshape(-1),
            contrib.reshape(flat.shape[0], -1, 3),
        )
        return energy.sum(-1).reshape(leading), grad.reshape(coords.shape)


@register("distance_bounds_potential")
class DistanceBoundsPotential(Potential):
    """Flat-bottom restraint on interatomic distance, bounds from the RDKit
    distance-geometry bounds matrix.

    Derived from Boltz FlatBottomPotential + DistancePotential
    (src/boltz/model/potentials/potentials.py). See THIRD_PARTY_NOTICES.md.
    """

    arity = 2

    def compute_variable(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        diff = positions[..., 0, :] - positions[..., 1, :]  # [M, R, 3]
        v = diff.norm(dim=-1).clamp_min(1e-6)  # [M, R]
        u = diff / v[..., None]
        return v, torch.stack((u, -u), dim=-2)  # [M, R, 2, 3]
