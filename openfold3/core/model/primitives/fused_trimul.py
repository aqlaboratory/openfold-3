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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: inference dispatch adapter for the
# standalone fused triangle multiplication tensor operator.

"""Model-policy adapter for the standalone fused Triton trimul operator."""

from __future__ import annotations

import os

import torch

from openfold3.core.kernels.triton.fused_trimul import (
    FusedTrimulTensorParams,
    fused_trimul_tensor,
    is_triton_available,
)
from openfold3.core.utils.chunk_utils import trimul_chunk_cap

_FLAG_TRUE = {"1", "true", "True"}


def is_fused_trimul_enabled() -> bool:
    """True if OPENFOLD3_FUSED_TRIMUL=1 and Triton is available."""
    return (
        os.environ.get("OPENFOLD3_FUSED_TRIMUL", "0") in _FLAG_TRUE
        and is_triton_available()
    )


def _eligible(z: torch.Tensor) -> bool:
    return (
        is_triton_available()
        and z.is_cuda
        and not torch.is_grad_enabled()
        and z.dim() == 4
    )


def _has_unsupported_bias(module) -> bool:
    return any(
        linear.bias is not None
        for linear in (
            module.linear_a_p,
            module.linear_a_g,
            module.linear_b_p,
            module.linear_b_g,
            module.linear_z,
            module.linear_g,
        )
    )


def _tensor_params(module) -> FusedTrimulTensorParams:
    return FusedTrimulTensorParams(
        linear_a_p_weight=module.linear_a_p.weight,
        linear_a_g_weight=module.linear_a_g.weight,
        linear_b_p_weight=module.linear_b_p.weight,
        linear_b_g_weight=module.linear_b_g.weight,
        linear_z_weight=module.linear_z.weight,
        linear_g_weight=module.linear_g.weight,
        ln_in_weight=module.layer_norm_in.weight,
        ln_in_bias=module.layer_norm_in.bias,
        ln_in_eps=module.layer_norm_in.eps,
        ln_out_weight=module.layer_norm_out.weight,
        ln_out_bias=module.layer_norm_out.bias,
        ln_out_eps=module.layer_norm_out.eps,
    )


def fused_trimul_update(
    module,
    z: torch.Tensor,
    mask: torch.Tensor | None,
    with_add: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Adapt a trimul module to the tensor-only fused implementation."""
    if not _eligible(z):
        return None

    if out is not None:
        if not with_add or out.shape != z.shape or out.data_ptr() != z.data_ptr():
            return None
        if z.shape[-1] > 128:
            return None

    if _has_unsupported_bias(module):
        return None
    if mask is None:
        mask = z.new_ones(z.shape[:-1])

    return fused_trimul_tensor(
        z,
        mask,
        _tensor_params(module),
        outgoing=module._outgoing,
        with_add=with_add,
        out=out,
        chunk_cap=trimul_chunk_cap(),
    )
