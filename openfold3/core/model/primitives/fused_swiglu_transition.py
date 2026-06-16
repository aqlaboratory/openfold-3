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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: eligibility and fallback dispatch
# for the fused inference-only SwiGLU pair transition.

"""Dispatch + env-flag gating for the fused SwiGLU pair transition.

Public entry point ``fused_swiglu_transition`` runs the fused Triton
kernel when eligible (CUDA, inference, supported dims) and otherwise
falls back to an eager path numerically identical to
``SwiGLUTransition._transition``. Optional ``residual`` lets the caller
request an in-place ``residual + transition(x)`` write (the zero-extra-
allocation pair-transition case).

Activated by ``OPENFOLD3_FUSED_SWIGLU_TRANSITION=1``.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from openfold3.core.kernels.triton.fused_swiglu_transition import (
    _launch_fused_swiglu_transition,
    is_triton_available,
)

_FLAG_TRUE = {"1", "true", "True"}

# The one-pass LayerNorm holds x in registers, so c_in (= K) is bounded by
# register budget; the hidden tile is streamed in BLOCK_H chunks so hidden
# is bounded only loosely. These match the pair transition (c_z=128, n=4 ->
# hidden=512) with headroom; wider sites fall back to eager.
_MAX_C_IN = 256
_MAX_HIDDEN = 512
# Below this row count, Triton launch overhead exceeds cuBLAS + F.layer_norm.
_MIN_M = 4096


def is_fused_swiglu_transition_enabled() -> bool:
    """True if OPENFOLD3_FUSED_SWIGLU_TRANSITION=1 and Triton is available."""
    return (
        os.environ.get("OPENFOLD3_FUSED_SWIGLU_TRANSITION", "0") in _FLAG_TRUE
        and is_triton_available()
    )


def _eligible(x: torch.Tensor, c_in: int, hidden: int) -> bool:
    return (
        is_triton_available()
        and x.is_cuda
        and not torch.is_grad_enabled()
        and c_in <= _MAX_C_IN
        and hidden <= _MAX_HIDDEN
        and (x.numel() // c_in) >= _MIN_M
    )


def fused_swiglu_transition(
    x: torch.Tensor,  # [*, c_in]
    gamma: torch.Tensor,  # [c_in]
    beta: torch.Tensor | None,  # [c_in] or None
    w_a: torch.Tensor,  # [hidden, c_in]
    w_b: torch.Tensor,  # [hidden, c_in]
    w_out: torch.Tensor,  # [c_in, hidden]
    mask: torch.Tensor | None = None,  # broadcastable to [*, 1]
    eps: float = 1e-5,
    residual: torch.Tensor | None = None,  # [*, c_in]; pass x for in-place
) -> torch.Tensor:
    """Fused LN -> SwiGLU -> Linear -> (*mask) [-> +residual].

    Returns the transition update when ``residual is None``; otherwise
    returns ``residual + update``. When ``residual is x`` and shapes match,
    the fused path writes back in place (no extra allocation).
    """
    c_in = gamma.shape[0]
    hidden = w_a.shape[0]

    if _eligible(x, c_in, hidden):
        # Match input dtype (autocast leaves params in fp32); kernel upcasts
        # to fp32 internally for LN + SiLU + accumulation.
        gamma_d = gamma.to(x.dtype) if gamma.dtype != x.dtype else gamma
        beta_d = (
            beta.to(x.dtype) if (beta is not None and beta.dtype != x.dtype) else beta
        )
        wa_d = w_a.to(x.dtype) if w_a.dtype != x.dtype else w_a
        wb_d = w_b.to(x.dtype) if w_b.dtype != x.dtype else w_b
        wo_d = w_out.to(x.dtype) if w_out.dtype != x.dtype else w_out

        x_2d = x.contiguous().view(-1, c_in)
        mask_1d = mask.reshape(-1) if mask is not None else None
        res_2d = residual.view(-1, c_in) if residual is not None else None

        y_2d = _launch_fused_swiglu_transition(
            x_2d, gamma_d, beta_d, wa_d, wb_d, wo_d, mask_1d, res_2d, eps
        )
        return y_2d.view_as(x)

    # Eager fallback — identical math to SwiGLUTransition._transition.
    x_norm = F.layer_norm(x, (c_in,), gamma, beta, eps)
    h = F.silu(F.linear(x_norm, w_a)) * F.linear(x_norm, w_b)
    out = F.linear(h, w_out)
    if mask is not None:
        out = out * mask
    if residual is not None:
        out = residual + out
    return out
