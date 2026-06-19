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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: inference dispatch and parameter
# packing for fused low-memory triangle multiplicative updates.

"""Dispatch + env-flag gating for the fused triangle multiplicative update.

``fused_trimul_update`` runs the fused Triton kernels (``fused_trimul.py``)
when eligible (CUDA, inference, supported dims) and otherwise returns ``None``
so the caller takes its existing eager / cuEq path. It reads the existing
``TriangleMultiplicativeUpdate`` parameters in place — no weight-layout change,
no retraining — concatenating the separate a/b projections into the dual-GEMM
weights the kernel expects.

LayerNorm is computed in-register inside both the dual-GEMM and the output
GEMM kernels, so the full-pair LN intermediate is never materialized in HBM.
Peak is ~3U (z + ab during dual-GEMM, or z + x + out during output GEMM).

Activated by ``OPENFOLD3_FUSED_TRIMUL=1``.
"""

from __future__ import annotations

import os

import torch

from openfold3.core.kernels.triton.fused_trimul import (
    gated_dual_gemm_fp32,
    gated_out_gemm_residual_fp32,
    is_triton_available,
    ln_transpose_fp32,
)

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
        and z.dim() == 4  # [B, N, N, c_z]; batched/template 5D -> eager
    )


def fused_trimul_update(
    module,
    z: torch.Tensor,
    mask: torch.Tensor | None,
    with_add: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Fused trimul for non-fused TriangleMultiplication{Outgoing,Incoming}.

    Returns the updated pair tensor (``z + update`` if ``with_add`` else
    ``update``), or ``None`` if not eligible (caller falls back).

    Mirrors the eager forward exactly:
        z_n   = LN_in(z)
        a     = sigmoid(linear_a_g(z_n)) * linear_a_p(z_n) * mask
        b     = sigmoid(linear_b_g(z_n)) * linear_b_p(z_n) * mask
        x     = einsum(a, b)               # outgoing or incoming
        x     = LN_out(x)
        out   = sigmoid(linear_g(z_n)) * linear_z(x)   [+ z]

    LN_in is computed in-register inside both kernels (z_n is never in HBM).
    """
    if not _eligible(z):
        return None

    B, N, _, c_z = z.shape
    c_hidden = module.linear_a_p.weight.shape[0]
    outgoing = module._outgoing

    if out is not None:
        if not with_add or out.shape != z.shape or out.data_ptr() != z.data_ptr():
            return None
        if c_z > 128:
            return None

    if mask is None:
        mask = z.new_ones(z.shape[:-1])

    # bias: linear_*_p/_g may carry bias; fold via eager fallback if present.
    if (
        module.linear_a_p.bias is not None
        or module.linear_a_g.bias is not None
        or module.linear_b_p.bias is not None
        or module.linear_b_g.bias is not None
    ):
        return None
    if module.linear_z.bias is not None or module.linear_g.bias is not None:
        return None

    z_2d = z.reshape(-1, c_z)
    mask_flat = mask.reshape(-1)
    ln_in = module.layer_norm_in

    # Stage 2: gated dual GEMM for a and b with fused LN_in.
    # LN is computed in-register per tile — z_n is never materialized.
    wp = torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], dim=0)
    wg = torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], dim=0)
    ab = gated_dual_gemm_fp32(
        z_2d, wp, wg, mask_flat,
        ln_weight=ln_in.weight, ln_bias=ln_in.bias, eps=ln_in.eps,
        output_dtype=(
            torch.bfloat16
            if out is not None and z.dtype == torch.float32
            else None
        ),
    )  # [2*c_hidden, M]
    del wp, wg
    a = ab[:c_hidden].view(c_hidden, B, N, N)
    b = ab[c_hidden:].view(c_hidden, B, N, N)

    # Stage 3: triangular einsum via cuBLAS, keeping (D, B, N, N) layout.
    # Auxiliary confidence heads run under a CUDA autocast(fp32) context.  Keep
    # the fused path's explicit intermediate dtype there; otherwise einsum and
    # the following LN-transpose are promoted back to fp32 and the confidence
    # pairformer loses the trunk/MSA memory win.
    with torch.amp.autocast(device_type="cuda", enabled=False):
        if outgoing:
            x = torch.einsum("cbik,cbjk->cbij", a, b)
        else:
            x = torch.einsum("cbki,cbkj->cbij", a, b)
    del a, b, ab

    # Stage 4: output LayerNorm + transpose (c_hidden, M) → (M, c_hidden).
    x_dm = x.reshape(c_hidden, B * N * N)  # zero-copy view
    with torch.amp.autocast(device_type="cuda", enabled=False):
        x_out_2d = ln_transpose_fp32(
            x_dm, module.layer_norm_out.weight, module.layer_norm_out.bias,
            eps=module.layer_norm_out.eps,
        )  # [M, c_hidden]
    del x, x_dm

    # Stage 5: gated output GEMM + residual, with fused LN_in for the gate.
    # Recomputes LN_in(z) in-register inside the kernel — z_n never in HBM.
    residual_2d = z.reshape(-1, c_z) if with_add else None
    linear_z_weight = module.linear_z.weight
    if linear_z_weight.dtype != x_out_2d.dtype:
        linear_z_weight = linear_z_weight.to(dtype=x_out_2d.dtype)
    out_2d = gated_out_gemm_residual_fp32(
        z_2d, x_out_2d, module.linear_g.weight, linear_z_weight,
        residual_2d,
        ln_weight=ln_in.weight, ln_bias=ln_in.bias, eps=ln_in.eps,
        out=out.reshape(-1, c_z) if out is not None else None,
    )
    return out_2d.view(B, N, N, c_z)
