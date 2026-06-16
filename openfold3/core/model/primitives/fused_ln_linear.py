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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: dispatch, autograd decomposition,
# and fallback plumbing for fused LayerNorm-to-linear operations.

"""LayerNorm followed by Linear, fused into a single Triton kernel.

Drops the ``[..., c_in]`` normalized intermediate. On the pinned LN ->
Linear sites identified by the training audit (DiffusionConditioning,
NoisyPositionEmbedder, OpenFold3 outer trunk), this saves
``sizeof(input.dtype) * input.numel()`` of allocator activity per call.

Forward path is the Triton kernel; backward is autograd-decomposed.
Falls back to ``F.layer_norm + F.linear`` on CPU / when Triton is
missing / when the env flag is off.

Activated by ``OPENFOLD3_FUSED_LN_LINEAR=1``.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold3.core.kernels.triton.fused_ln_linear import (
    fused_ln_linear,
    is_triton_available,
)
from openfold3.core.model.primitives.linear import apply_linear_init_

_FLAG_TRUE = {"1", "true", "True"}


def is_fused_ln_linear_enabled() -> bool:
    """Returns True if OPENFOLD3_FUSED_LN_LINEAR=1 and Triton is available."""
    return (
        os.environ.get("OPENFOLD3_FUSED_LN_LINEAR", "0") in _FLAG_TRUE
        and is_triton_available()
    )


class FusedLNLinear(nn.Module):
    """LayerNorm(c_in) followed by Linear(c_in, c_out), fused.

    Parameter naming matches a side-by-side ``LayerNorm + Linear``:
    ``self.layer_norm.weight``, ``self.layer_norm.bias`` (optional),
    ``self.linear.weight``, ``self.linear.bias`` (optional). This keeps
    state_dict shape compatibility minus the renaming step — see
    ``load_compat_state_dict`` if a pretrained checkpoint needs mapping.

    Args:
        c_in: input feature dim (also LN normalized dim)
        c_out: Linear output dim
        ln_create_scale: whether LN has a learnable scale (default True)
        ln_create_offset: whether LN has a learnable offset (default True)
        linear_bias: whether Linear has a learnable bias (default True)
        linear_init: openfold3 init scheme name (see Linear)
        eps: LN epsilon
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        *,
        ln_create_scale: bool = True,
        ln_create_offset: bool = True,
        linear_bias: bool = True,
        linear_init: str = "default",
        eps: float = 1e-5,
        legacy_ln_name: str | None = None,
        legacy_linear_name: str | None = None,
    ):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.eps = eps

        # LN parameters — reuse openfold3 LayerNorm naming & defaults.
        # Wrapping in a tiny submodule so state-dict keys match
        # `<prefix>.layer_norm.weight` for any future migration.
        self.ln_weight = (
            nn.Parameter(torch.ones(c_in)) if ln_create_scale else None
        )
        self.ln_bias = (
            nn.Parameter(torch.zeros(c_in)) if ln_create_offset else None
        )

        # Linear parameters — match openfold3 Linear conventions.
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        self.bias = nn.Parameter(torch.zeros(c_out)) if linear_bias else None
        apply_linear_init_(self.weight, self.bias, linear_init)

        # Legacy state-dict compat: pretrained checkpoints store params
        # under separate `<legacy_ln_name>.{weight,bias}` and
        # `<legacy_linear_name>.{weight,bias}` keys. _load_from_state_dict
        # rewrites those to our fused-module key layout when present.
        self._legacy_ln_name = legacy_ln_name
        self._legacy_linear_name = legacy_linear_name

    # Note: legacy state-dict remap can't happen in this module's own
    # `_load_from_state_dict` because PyTorch's loader filters keys by
    # the current module's prefix before descending. The parent module
    # is therefore responsible for installing a load-state-dict pre-hook
    # that renames legacy `<parent>.<legacy_ln>.{weight,bias}` and
    # `<parent>.<legacy_lin>.{weight,bias}` to
    # `<parent>.<fused_attr>.{ln_weight,ln_bias,weight,bias}`. See
    # `register_legacy_remap_hook` below.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The fused kernel requires a learnable LN scale; if the caller
        # built FusedLNLinear with create_scale=False (rare) we synthesize
        # a 1.0 vector.
        gamma = (
            self.ln_weight
            if self.ln_weight is not None
            else torch.ones(self.c_in, dtype=x.dtype, device=x.device)
        )
        return fused_ln_linear(
            x,
            gamma,
            self.ln_bias,
            self.weight,
            self.bias,
            self.eps,
        )

    def reference_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Always uses F.layer_norm + F.linear — used by tests."""
        gamma = self.ln_weight
        if gamma is None:
            gamma = torch.ones(self.c_in, dtype=x.dtype, device=x.device)
        x_norm = F.layer_norm(x, (self.c_in,), gamma, self.ln_bias, self.eps)
        return F.linear(x_norm, self.weight, self.bias)


def register_legacy_remap_hook(
    parent: nn.Module,
    triples: list[tuple[str, str, str]],
) -> None:
    """Install a load-state-dict pre-hook on ``parent`` that renames
    legacy LN+Linear keys to the fused-module layout. No-op when the
    fused flag is off.

    Args:
        parent: module that owns the FusedLNLinear attribute(s).
        triples: list of ``(legacy_ln_name, legacy_linear_name,
            fused_attr_name)`` per pair to remap.

    PyTorch's ``load_state_dict`` filters keys by each module's prefix
    before recursing, so a child cannot rewrite legacy sibling keys
    itself; the parent must do it via a pre-hook that fires before
    recursion.
    """
    if not is_fused_ln_linear_enabled():
        return

    def _hook(state_dict, prefix, *_args, **_kwargs):
        for legacy_ln, legacy_lin, fused_attr in triples:
            for legacy_key, new_key in (
                (prefix + legacy_ln + ".weight", prefix + fused_attr + ".ln_weight"),
                (prefix + legacy_ln + ".bias", prefix + fused_attr + ".ln_bias"),
                (prefix + legacy_lin + ".weight", prefix + fused_attr + ".weight"),
                (prefix + legacy_lin + ".bias", prefix + fused_attr + ".bias"),
            ):
                if legacy_key in state_dict:
                    state_dict[new_key] = state_dict.pop(legacy_key)

    parent._register_load_state_dict_pre_hook(_hook)
