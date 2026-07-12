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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: inference dispatch and chunked
# outgoing/incoming schedules for the fused triangle multiplication kernels.

"""Dispatch + env-flag gating for the fused triangle multiplicative update.

``fused_trimul_update`` runs the fused Triton kernels (``fused_trimul.py``)
when eligible (CUDA, inference, supported dims) and otherwise returns ``None``
so the caller takes its existing eager / cuEq path. It reads the existing
``TriangleMultiplicativeUpdate`` parameters in place — no weight-layout change,
no retraining — concatenating the separate a/b projections into the dual-GEMM
weights the kernel expects.

LayerNorm mean/rstd is computed once as compact ``[2, M]`` fp32 stats and reused
by both gated GEMMs. The full-pair LN intermediate is never materialized in HBM.
Peak stays near the low-memory fused schedule without cuEq's materialized
``LN_in(z)`` tensor.

Activated by ``OPENFOLD3_FUSED_TRIMUL=1``.
"""

from __future__ import annotations

import os

import torch

from openfold3.core.kernels.triton.fused_trimul import (
    gated_dual_gemm_fp32,
    gated_out_gemm_residual_fp32,
    is_triton_available,
    ln_stats_fp32,
    ln_transpose_fp32,
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
        and z.dim() == 4  # [B, N, N, c_z]; batched/template 5D -> eager
    )


def _chunked_outgoing(
    z_2d: torch.Tensor,
    mask_flat: torch.Tensor,
    ln_stats: torch.Tensor,
    module,
    N: int,
    c_z: int,
    c_hidden: int,
    with_add: bool,
    out: torch.Tensor | None,
    chunk_cap: int,
) -> torch.Tensor:
    """I-row chunked outgoing: compute full b (1U), chunk a per I-rows.

    Outgoing ``"cbik,cbjk->cbij"``: output row i depends on ``a[:,:,i,:]``
    (I-row of a) and ALL of b.  I-rows map to contiguous M-slices
    ``[i*N : (i+1)*N]``, so we can produce a per-chunk and immediately
    einsum + LN + output-GEMM.  Peak ~1.2U above z: b(1U) + tiny chunks.
    """
    ln_in = module.layer_norm_in
    ln_out = module.layer_norm_out
    M = N * N

    # Full b (1U)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        b_full = gated_dual_gemm_fp32(
            z_2d, module.linear_b_p.weight, module.linear_b_g.weight,
            mask_flat,
            ln_weight=ln_in.weight, ln_bias=ln_in.bias,
            ln_stats=ln_stats, eps=ln_in.eps, output_dtype=None,
        )
    b_4d = b_full.view(c_hidden, 1, N, N)

    if out is not None:
        out_2d = out.reshape(-1, c_z)
    else:
        out_2d = torch.empty(
            M, c_z, device=z_2d.device, dtype=z_2d.dtype,
        )

    linear_z_w = module.linear_z.weight
    if linear_z_w.dtype != z_2d.dtype:
        linear_z_w = linear_z_w.to(dtype=z_2d.dtype)

    for i_start in range(0, N, chunk_cap):
        i_end = min(N, i_start + chunk_cap)
        rows = i_end - i_start
        m0, m1 = i_start * N, i_end * N

        z_c = z_2d[m0:m1]
        mask_c = mask_flat[m0:m1]
        ls_c = ln_stats[:, m0:m1].contiguous()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            a_c = gated_dual_gemm_fp32(
                z_c, module.linear_a_p.weight, module.linear_a_g.weight,
                mask_c,
                ln_weight=ln_in.weight, ln_bias=ln_in.bias,
                ln_stats=ls_c, eps=ln_in.eps, output_dtype=None,
            )
        a_4d = a_c.view(c_hidden, 1, rows, N)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_c = torch.einsum("cbik,cbjk->cbij", a_4d, b_4d)
        del a_c, a_4d

        x_dm = x_c.reshape(c_hidden, rows * N)
        del x_c
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_out = ln_transpose_fp32(
                x_dm, ln_out.weight, ln_out.bias, eps=ln_out.eps,
            )
        del x_dm

        res_c = z_c if with_add else None
        gated_out_gemm_residual_fp32(
            z_c, x_out, module.linear_g.weight, linear_z_w,
            res_c,
            ln_weight=ln_in.weight, ln_bias=ln_in.bias,
            ln_stats=ls_c, eps=ln_in.eps, out=out_2d[m0:m1],
        )
        del x_out, ls_c, z_c, mask_c, res_c

    del b_full, b_4d
    return out_2d.view(1, N, N, c_z)


def _chunked_incoming(
    z_2d: torch.Tensor,
    mask_flat: torch.Tensor,
    ln_stats: torch.Tensor,
    module,
    N: int,
    c_z: int,
    c_hidden: int,
    with_add: bool,
    out: torch.Tensor | None,
    chunk_cap: int,
) -> torch.Tensor:
    """K-chunked incoming: accumulate partial einsums over K-chunks.

    Incoming ``"cbki,cbkj->cbij"``: the contraction is over K.  In the
    ``[c_hidden, M]`` layout (M = B*N*N, row-major ``(b,k,j)``-order),
    K-rows ``[k_start*N : k_end*N]`` are contiguous M-slices.  We
    produce a_k and b_k from the same K-chunk of z and accumulate the
    partial ``a_k^T @ b_k`` into ``x_accum`` via in-place ``baddbmm``.
    The first K-chunk uses ``bmm`` (skip reading zeros).

    After the K-loop, ``x_accum`` holds the full einsum result (1U).
    The I-loop chunks LN-transpose + output-GEMM so ``x_out`` is never
    a full 1U buffer.
    Peak ~1.2U above z: x_accum(1U) + ab_k/contig transient(~0.2U).
    """
    ln_in = module.layer_norm_in
    ln_out = module.layer_norm_out
    M = N * N

    # Accumulator for the full einsum result (1U), stored as 3-D for baddbmm
    x_accum = torch.empty(
        c_hidden, N, N, device=z_2d.device, dtype=z_2d.dtype,
    )

    # Pre-concatenate weights (reused across K-chunks)
    wp_ab = torch.cat(
        [module.linear_a_p.weight, module.linear_b_p.weight], dim=0,
    )
    wg_ab = torch.cat(
        [module.linear_a_g.weight, module.linear_b_g.weight], dim=0,
    )

    # K-chunk loop: accumulate partial einsums via in-place baddbmm.
    # First chunk uses bmm (skip reading zeros from x_accum).
    first_chunk = True
    for k_start in range(0, N, chunk_cap):
        k_end = min(N, k_start + chunk_cap)
        k_rows = k_end - k_start
        m0, m1 = k_start * N, k_end * N

        z_c = z_2d[m0:m1]
        mask_c = mask_flat[m0:m1]
        ls_c = ln_stats[:, m0:m1].contiguous()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            ab_k = gated_dual_gemm_fp32(
                z_c, wp_ab, wg_ab, mask_c,
                ln_weight=ln_in.weight, ln_bias=ln_in.bias,
                ln_stats=ls_c, eps=ln_in.eps, output_dtype=None,
            )  # [2*c_hidden, chunk_M]
        del ls_c, z_c, mask_c

        a_k = ab_k[:c_hidden].view(c_hidden, k_rows, N)
        b_k = ab_k[c_hidden:].view(c_hidden, k_rows, N)

        with torch.amp.autocast(device_type="cuda", enabled=False):
            if first_chunk:
                torch.bmm(a_k.transpose(1, 2), b_k, out=x_accum)
                first_chunk = False
            else:
                torch.baddbmm(
                    x_accum, a_k.transpose(1, 2), b_k,
                    beta=1.0, alpha=1.0, out=x_accum,
                )
        del ab_k, a_k, b_k

    del wp_ab, wg_ab

    # x_accum holds the full einsum result (1U).
    # I-chunk both LN-transpose and output-GEMM so x_out is never a
    # full 1U buffer — only chunk-sized pieces exist at once.
    if out is not None:
        out_2d = out.reshape(-1, c_z)
    else:
        out_2d = torch.empty(
            M, c_z, device=z_2d.device, dtype=z_2d.dtype,
        )

    linear_z_w = module.linear_z.weight
    if linear_z_w.dtype != z_2d.dtype:
        linear_z_w = linear_z_w.to(dtype=z_2d.dtype)

    for i_start in range(0, N, chunk_cap):
        i_end = min(N, i_start + chunk_cap)
        rows = i_end - i_start
        m0, m1 = i_start * N, i_end * N

        # I-slice of x_accum → contiguous copy (rows/N fraction of 1U)
        x_dm = x_accum[:, i_start:i_end, :].reshape(
            c_hidden, rows * N,
        ).contiguous()
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_out = ln_transpose_fp32(
                x_dm, ln_out.weight, ln_out.bias, eps=ln_out.eps,
            )
        del x_dm

        z_c = z_2d[m0:m1]
        ls_c = ln_stats[:, m0:m1].contiguous()
        res_c = z_c if with_add else None
        gated_out_gemm_residual_fp32(
            z_c, x_out, module.linear_g.weight, linear_z_w,
            res_c,
            ln_weight=ln_in.weight, ln_bias=ln_in.bias,
            ln_stats=ls_c, eps=ln_in.eps, out=out_2d[m0:m1],
        )
        del x_out, ls_c, z_c, res_c

    del x_accum
    return out_2d.view(1, N, N, c_z)


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

    LN_in stats are materialized once; z_n itself is never materialized in HBM.
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
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ln_stats = ln_stats_fp32(z_2d, eps=ln_in.eps)

    # Dispatch to chunked path if cap is set and beneficial
    chunk_cap = trimul_chunk_cap()
    if chunk_cap is not None and chunk_cap < N and B == 1:
        fn = _chunked_outgoing if outgoing else _chunked_incoming
        return fn(
            z_2d, mask_flat, ln_stats, module,
            N, c_z, c_hidden, with_add, out, chunk_cap,
        )

    # Stage 2: gated dual GEMM for a and b with fused LN_in.
    # LN stats are reused; z_n is still normalized in-register per tile.
    wp = torch.cat([module.linear_a_p.weight, module.linear_b_p.weight], dim=0)
    wg = torch.cat([module.linear_a_g.weight, module.linear_b_g.weight], dim=0)
    ab = gated_dual_gemm_fp32(
        z_2d, wp, wg, mask_flat,
        ln_weight=ln_in.weight, ln_bias=ln_in.bias, ln_stats=ln_stats,
        eps=ln_in.eps,
        output_dtype=None,
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

    # Stage 5: gated output GEMM + residual, reusing Stage-1 LN_in stats.
    residual_2d = z.reshape(-1, c_z) if with_add else None
    linear_z_weight = module.linear_z.weight
    if linear_z_weight.dtype != x_out_2d.dtype:
        linear_z_weight = linear_z_weight.to(dtype=x_out_2d.dtype)
    out_2d = gated_out_gemm_residual_fp32(
        z_2d, x_out_2d, module.linear_g.weight, linear_z_weight,
        residual_2d,
        ln_weight=ln_in.weight, ln_bias=ln_in.bias, ln_stats=ln_stats,
        eps=ln_in.eps,
        out=out.reshape(-1, c_z) if out is not None else None,
    )
    del ln_stats
    return out_2d.view(B, N, N, c_z)
