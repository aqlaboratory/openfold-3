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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: Triton projection, normalization,
# and gated-output kernels for low-memory triangle multiplication inference.

"""Fused Triton kernels for the triangle multiplicative update (inference).

Adapted from the ``esmfold2-trimul-kernel`` project's fused dual-GEMM and
fused output-GEMM-with-residual kernels, which are themselves "inspired by
cuequivariance's ``fused_sigmoid_gated_dual_gemm`` /
``triangle_multiplicative_update`` (https://docs.nvidia.com/cuda/cuequivariance/)
and independently re-implemented in Triton". This adaptation:

  * generalizes the IO/weight dtype to **fp32** (the upstream kernel is
    bf16-only). fp32 accumulation is unchanged; fp32 matmuls use IEEE by
    default so correctness can be checked against eager full-fp32.
  * is **inference / forward-only** (no autograd). The OF3 trimul inference
    path runs under ``torch.is_grad_enabled() == False``; the dispatch falls
    back to eager when grad is needed.

The low-level kernels cover the memory-heavy stages of AF3 trimul:
  Stage 1 — optional compact LN stats: compute mean/rstd once as ``[2, M]``.
    This is ~0.016U for ``c_z=128`` and avoids repeating the row reduction in
    both gated GEMMs without materializing full ``LN_in(z)``.
  Stage 2 — gated dual GEMM: ``a = sigmoid(x@Wga)*(x@Wpa)`` and ``b`` likewise,
    with the row-shared mask folded in, emitted directly in the
    ``[c_hidden, B, N, N]`` layout the stage-3 einsum consumes. The separate
    gate tensors are never written to HBM.
  Stage 5 — gated output GEMM + residual: ``out = residual + g_drop *
    sigmoid(x_in@Wg) * (x_out@Wp)`` in one pass; the ``delta = trimul(z)``
    intermediate is never materialized.

This module also owns the tensor-only whole/chunked schedules. Chunked incoming
uses a full B projection plus strided A-column tiles so each output-row tile is
written once, avoiding repeated read/modify/write of a full contraction
accumulator. True in-place residual output uses alias-safe channel-grouped
accumulation with proportionally larger K chunks, reducing accumulator traffic
without increasing the projection-memory bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE


@dataclass(frozen=True)
class FusedTrimulTensorParams:
    """Tensor-only parameters consumed by the fused trimul schedules."""

    linear_a_p_weight: torch.Tensor
    linear_a_g_weight: torch.Tensor
    linear_b_p_weight: torch.Tensor
    linear_b_g_weight: torch.Tensor
    linear_z_weight: torch.Tensor
    linear_g_weight: torch.Tensor
    ln_in_weight: torch.Tensor
    ln_in_bias: torch.Tensor | None
    ln_in_eps: float
    ln_out_weight: torch.Tensor
    ln_out_bias: torch.Tensor | None
    ln_out_eps: float


if _TRITON_AVAILABLE:

    # Static configs keep one JIT compile per dimension across all lengths.
    # Two pipeline stages help full/chunk-sized rows; c_z=128 in-place writes
    # use K16 while c_z=64 retains K32 for template-path numerical stability.
    _DUAL_GEMM_CFG = dict(
        TILE_M=64, TILE_N=128, TILE_K=16, GROUP_M=8, num_warps=4, num_stages=2,
    )
    _DUAL_GEMM_CFG_N64 = dict(
        TILE_M=64, TILE_N=64, TILE_K=16, GROUP_M=8, num_warps=4, num_stages=2,
    )
    _OUT_GEMM_CFG = dict(
        TILE_M=64, TILE_N=128, TILE_K=16, GROUP_M=8, num_warps=4, num_stages=2,
    )
    _OUT_GEMM_INPLACE_CFG = dict(
        TILE_M=64, TILE_N=128, TILE_K=32, GROUP_M=8, num_warps=4, num_stages=2,
    )
    _OUT_GEMM_INPLACE_CFG_C128 = dict(
        TILE_M=64, TILE_N=128, TILE_K=16, GROUP_M=8, num_warps=4, num_stages=2,
    )
    _OUT_DM_CFG = dict(
        TILE_M=64, TILE_N=128, TILE_K=16, GROUP_M=8, num_warps=4, num_stages=2,
    )
    _LN_STATS_TILE_M = 64

    @triton.jit(
        do_not_specialize=["M", "eps"],
        do_not_specialize_on_alignment=[
            "x_ptr",
            "wp_ptr",
            "wg_ptr",
            "mask_ptr",
            "gamma_ptr",
            "beta_ptr",
            "ln_stats_ptr",
            "out_ptr",
        ],
    )
    def _gated_dual_gemm_kernel(
        x_ptr,  # [M, K] — pair (raw if FUSED_LN, LN'd otherwise), row-major
        wp_ptr,  # [Nproj, K] — concatenated p-projection weight (a then b)
        wg_ptr,  # [Nproj, K] — concatenated g-projection weight (a then b)
        mask_ptr,  # [M] — row-shared mask (1 keep / 0 drop)
        gamma_ptr,  # [K] — LN scale (only used when FUSED_LN)
        beta_ptr,  # [K] — LN bias (only used when FUSED_LN and HAS_LN_BIAS)
        ln_stats_ptr,  # [2, M] mean/rstd (only used when HAS_LN_STATS)
        out_ptr,  # [Nproj, M] — transposed output: sigmoid(x@wg)*(x@wp)*mask
        M,
        Nproj,
        K,
        eps,
        HAS_MASK: tl.constexpr,
        PRECISION: tl.constexpr,  # 0 = default, 1 = ieee, 2 = tf32
        FUSED_LN: tl.constexpr,  # 1 = x is raw, compute LN in-register
        HAS_LN_STATS: tl.constexpr,
        HAS_LN_BIAS: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        BLOCK_CZ: tl.constexpr,  # next_pow2(K), for full-row LN reduction
    ):
        pid_m_raw = tl.program_id(0)
        pid_n_raw = tl.program_id(1)
        num_pid_m = tl.cdiv(M, TILE_M)
        num_pid_n = tl.cdiv(Nproj, TILE_N)
        pid = pid_n_raw * num_pid_m + pid_m_raw
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        pid_m = tl.cast(pid_m, tl.int64)
        pid_n = tl.cast(pid_n, tl.int64)
        M = tl.cast(M, tl.int64)
        Nproj = tl.cast(Nproj, tl.int64)
        K = tl.cast(K, tl.int64)

        offs_m = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
        offs_k = tl.arange(0, TILE_K).to(tl.int64)
        mask_m = offs_m < M

        if FUSED_LN:
            if HAS_LN_STATS:
                mean = tl.load(ln_stats_ptr + offs_m, mask=mask_m, other=0.0).to(
                    tl.float32
                )
                rstd = tl.load(
                    ln_stats_ptr + M + offs_m, mask=mask_m, other=0.0,
                ).to(tl.float32)
            else:
                offs_cz = tl.arange(0, BLOCK_CZ).to(tl.int64)
                cz_mask = offs_cz < K
                z_ptrs = x_ptr + (offs_m[:, None] * K + offs_cz[None, :])
                z_raw = tl.load(
                    z_ptrs, mask=mask_m[:, None] & cz_mask[None, :], other=0.0,
                ).to(tl.float32)
                z_sum = tl.sum(z_raw, axis=1)
                mean = z_sum / K
                z_centered = tl.where(cz_mask[None, :], z_raw - mean[:, None], 0.0)
                var = tl.sum(z_centered * z_centered, axis=1) / K
                rstd = 1.0 / tl.sqrt(var + eps)

        gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        val_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        if FUSED_LN:
            for k_off in range(0, K, TILE_K):
                k_range = k_off + offs_k
                k_mask = k_range < K
                z_k = tl.load(
                    x_ptr + (offs_m[:, None] * K + k_range[None, :]),
                    mask=mask_m[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                z_k_centered = z_k - mean[:, None]
                gamma_k = tl.load(
                    gamma_ptr + k_range, mask=k_mask, other=0.0,
                ).to(tl.float32)
                x_tile = z_k_centered * rstd[:, None] * gamma_k[None, :]
                if HAS_LN_BIAS:
                    beta_k = tl.load(
                        beta_ptr + k_range, mask=k_mask, other=0.0,
                    ).to(tl.float32)
                    x_tile = x_tile + tl.where(
                        k_mask[None, :], beta_k[None, :], 0.0,
                    )
                wp = tl.load(wp_ptr + (offs_n[None, :] * K + k_range[:, None]))
                wg = tl.load(wg_ptr + (offs_n[None, :] * K + k_range[:, None]))
                if PRECISION == 1:
                    val_acc = tl.dot(x_tile, wp, val_acc, input_precision="ieee")
                    gate_acc = tl.dot(x_tile, wg, gate_acc, input_precision="ieee")
                elif PRECISION == 2:
                    val_acc = tl.dot(x_tile, wp, val_acc, input_precision="tf32")
                    gate_acc = tl.dot(x_tile, wg, gate_acc, input_precision="tf32")
                else:
                    x_t = x_tile.to(wp.dtype)
                    val_acc = tl.dot(x_t, wp, val_acc)
                    gate_acc = tl.dot(x_t, wg, gate_acc)
        else:
            x_ptrs = x_ptr + (offs_m[:, None] * K + offs_k[None, :])
            wp_base = wp_ptr + (offs_n[None, :] * K + offs_k[:, None])
            wg_base = wg_ptr + (offs_n[None, :] * K + offs_k[:, None])
            for _ in range(0, tl.cdiv(K, TILE_K)):
                x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
                wp = tl.load(wp_base)
                wg = tl.load(wg_base)
                if PRECISION == 1:
                    val_acc = tl.dot(x, wp, val_acc, input_precision="ieee")
                    gate_acc = tl.dot(x, wg, gate_acc, input_precision="ieee")
                elif PRECISION == 2:
                    val_acc = tl.dot(x, wp, val_acc, input_precision="tf32")
                    gate_acc = tl.dot(x, wg, gate_acc, input_precision="tf32")
                else:
                    val_acc = tl.dot(x, wp, val_acc)
                    gate_acc = tl.dot(x, wg, gate_acc)
                x_ptrs += TILE_K
                wp_base += TILE_K
                wg_base += TILE_K

        delta = tl.sigmoid(gate_acc) * val_acc
        if HAS_MASK:
            mtile = tl.load(mask_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            delta = delta * mtile[:, None]

        # store transposed -> [Nproj, M]
        out_ptrs = out_ptr + (offs_n[:, None] * M + offs_m[None, :])
        tl.store(
            out_ptrs,
            tl.trans(delta).to(out_ptr.type.element_ty),
            mask=mask_m[None, :],
        )

    @triton.jit(
        do_not_specialize=["N", "I_START", "I_ROWS", "M_OUT", "eps"],
        do_not_specialize_on_alignment=[
            "x_ptr",
            "wp_ptr",
            "wg_ptr",
            "mask_ptr",
            "gamma_ptr",
            "beta_ptr",
            "ln_stats_ptr",
            "out_ptr",
        ],
    )
    def _gated_column_gemm_kernel(
        x_ptr,  # [N*N, K] row-major source z[k, i]
        wp_ptr,  # [Nproj, K]
        wg_ptr,  # [Nproj, K]
        mask_ptr,  # [N*N]
        gamma_ptr,  # [K]
        beta_ptr,  # [K]
        ln_stats_ptr,  # [2, N*N]
        out_ptr,  # [Nproj, N*I_ROWS], logical [Nproj, k, i]
        N,
        I_START,
        I_ROWS,
        M_OUT,
        Nproj,
        K,
        eps,
        HAS_MASK: tl.constexpr,
        PRECISION: tl.constexpr,
        HAS_LN_BIAS: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Project the strided source columns z[k, i] into contiguous [c, k, i]."""
        pid_m_raw = tl.program_id(0)
        pid_n_raw = tl.program_id(1)
        num_pid_m = tl.cdiv(M_OUT, TILE_M)
        num_pid_n = tl.cdiv(Nproj, TILE_N)
        pid = pid_n_raw * num_pid_m + pid_m_raw
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        pid_m = tl.cast(pid_m, tl.int64)
        pid_n = tl.cast(pid_n, tl.int64)
        N64 = tl.cast(N, tl.int64)
        i_start64 = tl.cast(I_START, tl.int64)
        i_rows64 = tl.cast(I_ROWS, tl.int64)
        m_out64 = tl.cast(M_OUT, tl.int64)
        nproj64 = tl.cast(Nproj, tl.int64)
        K64 = tl.cast(K, tl.int64)

        # Source work is K-major so adjacent rows read adjacent z[k, i] pairs.
        offs_q = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
        offs_k = tl.arange(0, TILE_K).to(tl.int64)
        mask_q = offs_q < m_out64
        mask_n = offs_n < nproj64
        k_idx = offs_q // i_rows64
        i_local = offs_q - k_idx * i_rows64
        src_m = k_idx * N64 + i_start64 + i_local
        full_m = N64 * N64

        mean = tl.load(
            ln_stats_ptr + src_m, mask=mask_q, other=0.0,
        ).to(tl.float32)
        rstd = tl.load(
            ln_stats_ptr + full_m + src_m, mask=mask_q, other=0.0,
        ).to(tl.float32)

        gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        val_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_off in range(0, K, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < K64
            z_k = tl.load(
                x_ptr + src_m[:, None] * K64 + k_range[None, :],
                mask=mask_q[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gamma_k = tl.load(
                gamma_ptr + k_range, mask=k_mask, other=0.0,
            ).to(tl.float32)
            x_tile = (z_k - mean[:, None]) * rstd[:, None] * gamma_k[None, :]
            if HAS_LN_BIAS:
                beta_k = tl.load(
                    beta_ptr + k_range, mask=k_mask, other=0.0,
                ).to(tl.float32)
                x_tile += tl.where(k_mask[None, :], beta_k[None, :], 0.0)
            wp = tl.load(
                wp_ptr + offs_n[None, :] * K64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            )
            wg = tl.load(
                wg_ptr + offs_n[None, :] * K64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            )
            if PRECISION == 1:
                val_acc = tl.dot(x_tile, wp, val_acc, input_precision="ieee")
                gate_acc = tl.dot(x_tile, wg, gate_acc, input_precision="ieee")
            elif PRECISION == 2:
                val_acc = tl.dot(x_tile, wp, val_acc, input_precision="tf32")
                gate_acc = tl.dot(x_tile, wg, gate_acc, input_precision="tf32")
            else:
                x_t = x_tile.to(wp.dtype)
                val_acc = tl.dot(x_t, wp, val_acc)
                gate_acc = tl.dot(x_t, wg, gate_acc)

        delta = tl.sigmoid(gate_acc) * val_acc
        if HAS_MASK:
            mtile = tl.load(mask_ptr + src_m, mask=mask_q, other=0.0)
            delta *= mtile[:, None].to(tl.float32)

        out_ptrs = out_ptr + offs_n[:, None] * m_out64 + offs_q[None, :]
        tl.store(
            out_ptrs,
            tl.trans(delta).to(out_ptr.type.element_ty),
            mask=mask_n[:, None] & mask_q[None, :],
        )

    @triton.jit(
        do_not_specialize=["M", "eps"],
        do_not_specialize_on_alignment=[
            "x_in_ptr",
            "x_out_ptr",
            "wg_ptr",
            "wp_ptr",
            "residual_ptr",
            "gamma_ptr",
            "beta_ptr",
            "ln_stats_ptr",
            "out_ptr",
        ],
    )
    def _gated_out_gemm_residual_kernel(
        x_in_ptr,  # [M, Cz] — raw pair (if FUSED_LN) or LN_in(pair) (gate input)
        x_out_ptr,  # [M, Ch] — LN_out(einsum) (value input)
        wg_ptr,  # [Cz, Cz] — linear_g weight
        wp_ptr,  # [Cz, Ch] — linear_z weight
        residual_ptr,  # [M, Cz] — pair residual (added in-kernel)
        gamma_ptr,  # [Cz] — LN scale (only used when FUSED_LN)
        beta_ptr,  # [Cz] — LN bias (only used when FUSED_LN and HAS_LN_BIAS)
        ln_stats_ptr,  # [2, M] mean/rstd (only used when HAS_LN_STATS)
        out_ptr,  # [M, Cz]
        M,
        CZ,
        CH,
        eps,
        WITH_ADD: tl.constexpr,
        PRECISION: tl.constexpr,
        FUSED_LN: tl.constexpr,
        HAS_LN_STATS: tl.constexpr,
        HAS_LN_BIAS: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        BLOCK_CZ: tl.constexpr,
    ):
        pid_m_raw = tl.program_id(0)
        pid_n_raw = tl.program_id(1)
        num_pid_m = tl.cdiv(M, TILE_M)
        num_pid_n = tl.cdiv(CZ, TILE_N)
        pid = pid_n_raw * num_pid_m + pid_m_raw
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        pid_m = tl.cast(pid_m, tl.int64)
        pid_n = tl.cast(pid_n, tl.int64)
        M = tl.cast(M, tl.int64)
        CZ = tl.cast(CZ, tl.int64)
        CH = tl.cast(CH, tl.int64)

        offs_m = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
        offs_k = tl.arange(0, TILE_K).to(tl.int64)
        mask_m = offs_m < M
        mask_n = offs_n < CZ

        # value gemm: x_out [M,CH] @ wp[CZ,CH]^T  -> [TILE_M, TILE_N]
        xo_ptrs = x_out_ptr + (offs_m[:, None] * CH + offs_k[None, :])
        wp_base = wp_ptr + (offs_n[None, :] * CH + offs_k[:, None])
        val_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(CH, TILE_K)):
            xo = tl.load(xo_ptrs, mask=mask_m[:, None], other=0.0)
            wp = tl.load(wp_base, mask=mask_n[None, :], other=0.0)
            if PRECISION == 1:
                val_acc = tl.dot(xo, wp, val_acc, input_precision="ieee")
            elif PRECISION == 2:
                val_acc = tl.dot(xo, wp, val_acc, input_precision="tf32")
            else:
                val_acc = tl.dot(xo, wp, val_acc)
            xo_ptrs += TILE_K
            wp_base += TILE_K

        # gate gemm: LN(x_in) [M,CZ] @ wg[CZ,CZ]^T -> [TILE_M, TILE_N]
        if FUSED_LN:
            if HAS_LN_STATS:
                mean = tl.load(ln_stats_ptr + offs_m, mask=mask_m, other=0.0).to(
                    tl.float32
                )
                rstd = tl.load(
                    ln_stats_ptr + M + offs_m, mask=mask_m, other=0.0,
                ).to(tl.float32)
            else:
                offs_cz = tl.arange(0, BLOCK_CZ).to(tl.int64)
                cz_mask = offs_cz < CZ
                z_ptrs = x_in_ptr + (offs_m[:, None] * CZ + offs_cz[None, :])
                z_raw = tl.load(
                    z_ptrs, mask=mask_m[:, None] & cz_mask[None, :], other=0.0,
                ).to(tl.float32)
                z_sum = tl.sum(z_raw, axis=1)
                mean = z_sum / CZ
                z_centered = tl.where(cz_mask[None, :], z_raw - mean[:, None], 0.0)
                var = tl.sum(z_centered * z_centered, axis=1) / CZ
                rstd = 1.0 / tl.sqrt(var + eps)

        gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        if FUSED_LN:
            for k_off in range(0, CZ, TILE_K):
                k_range = k_off + offs_k
                k_mask = k_range < CZ
                z_k = tl.load(
                    x_in_ptr + (offs_m[:, None] * CZ + k_range[None, :]),
                    mask=mask_m[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                z_k_centered = z_k - mean[:, None]
                gamma_k = tl.load(
                    gamma_ptr + k_range, mask=k_mask, other=0.0,
                ).to(tl.float32)
                xi = z_k_centered * rstd[:, None] * gamma_k[None, :]
                if HAS_LN_BIAS:
                    beta_k = tl.load(
                        beta_ptr + k_range, mask=k_mask, other=0.0,
                    ).to(tl.float32)
                    xi = xi + tl.where(k_mask[None, :], beta_k[None, :], 0.0)
                wg = tl.load(
                    wg_ptr + (offs_n[None, :] * CZ + k_range[:, None]),
                    mask=mask_n[None, :] & k_mask[:, None],
                    other=0.0,
                )
                if PRECISION == 1:
                    gate_acc = tl.dot(xi, wg, gate_acc, input_precision="ieee")
                elif PRECISION == 2:
                    gate_acc = tl.dot(xi, wg, gate_acc, input_precision="tf32")
                else:
                    xi_t = xi.to(wg.dtype)
                    gate_acc = tl.dot(xi_t, wg, gate_acc)
        else:
            xi_ptrs = x_in_ptr + (offs_m[:, None] * CZ + offs_k[None, :])
            wg_base = wg_ptr + (offs_n[None, :] * CZ + offs_k[:, None])
            for _ in range(0, tl.cdiv(CZ, TILE_K)):
                xi = tl.load(xi_ptrs, mask=mask_m[:, None], other=0.0)
                wg = tl.load(wg_base, mask=mask_n[None, :], other=0.0)
                if PRECISION == 1:
                    gate_acc = tl.dot(xi, wg, gate_acc, input_precision="ieee")
                elif PRECISION == 2:
                    gate_acc = tl.dot(xi, wg, gate_acc, input_precision="tf32")
                else:
                    gate_acc = tl.dot(xi, wg, gate_acc)
                xi_ptrs += TILE_K
                wg_base += TILE_K

        out_val = tl.sigmoid(gate_acc) * val_acc
        o_ptrs = out_ptr + (offs_m[:, None] * CZ + offs_n[None, :])
        if WITH_ADD:
            r_ptrs = residual_ptr + (offs_m[:, None] * CZ + offs_n[None, :])
            resid = tl.load(
                r_ptrs,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            out_val = out_val + resid
        tl.store(
            o_ptrs,
            out_val.to(out_ptr.type.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit(
        do_not_specialize=["M", "eps_out"],
        do_not_specialize_on_alignment=[
            "x_in_ptr",
            "x_dm_ptr",
            "wg_ptr",
            "wp_ptr",
            "residual_ptr",
            "gamma_in_ptr",
            "beta_in_ptr",
            "ln_stats_ptr",
            "gamma_out_ptr",
            "beta_out_ptr",
            "out_ptr",
        ],
    )
    def _gated_out_from_dm_kernel(
        x_in_ptr,  # [M, CZ] raw pair input for gate/residual
        x_dm_ptr,  # [CH, M] contraction output in channel-major layout
        wg_ptr,  # [CZ, CZ]
        wp_ptr,  # [CZ, CH]
        residual_ptr,  # [M, CZ]
        gamma_in_ptr,  # [CZ]
        beta_in_ptr,  # [CZ]
        ln_stats_ptr,  # [2, M]
        gamma_out_ptr,  # [CH]
        beta_out_ptr,  # [CH]
        out_ptr,  # [M, CZ]
        M,
        CZ,
        CH,
        eps_out,
        WITH_ADD: tl.constexpr,
        PRECISION: tl.constexpr,
        HAS_LN_IN_BIAS: tl.constexpr,
        HAS_LN_OUT_BIAS: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Fuse transposing LN_out, value/gate GEMMs, residual, and store."""
        pid_m_raw = tl.program_id(0)
        pid_n_raw = tl.program_id(1)
        num_pid_m = tl.cdiv(M, TILE_M)
        num_pid_n = tl.cdiv(CZ, TILE_N)
        pid = pid_n_raw * num_pid_m + pid_m_raw
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        pid_m = tl.cast(pid_m, tl.int64)
        pid_n = tl.cast(pid_n, tl.int64)
        M64 = tl.cast(M, tl.int64)
        CZ64 = tl.cast(CZ, tl.int64)
        CH64 = tl.cast(CH, tl.int64)
        offs_m = pid_m * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N).to(tl.int64)
        offs_k = tl.arange(0, TILE_K).to(tl.int64)
        mask_m = offs_m < M64
        mask_n = offs_n < CZ64

        # Two-pass LN_out: collect sum/sumsq, then normalize tiles into value GEMM.
        out_sum = tl.zeros((TILE_M,), dtype=tl.float32)
        out_sumsq = tl.zeros((TILE_M,), dtype=tl.float32)
        for k_off in range(0, CH, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < CH64
            x_k = tl.load(
                x_dm_ptr + k_range[None, :] * M64 + offs_m[:, None],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            x_k = tl.where(k_mask[None, :], x_k, 0.0)
            out_sum += tl.sum(x_k, axis=1)
            out_sumsq += tl.sum(x_k * x_k, axis=1)
        out_mean = out_sum / CH64
        out_var = tl.maximum(out_sumsq / CH64 - out_mean * out_mean, 0.0)
        out_rstd = 1.0 / tl.sqrt(out_var + eps_out)
        val_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_off in range(0, CH, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < CH64
            x_k = tl.load(
                x_dm_ptr + k_range[None, :] * M64 + offs_m[:, None],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gamma_out = tl.load(
                gamma_out_ptr + k_range, mask=k_mask, other=0.0
            ).to(tl.float32)
            x_value = (
                (x_k - out_mean[:, None])
                * out_rstd[:, None]
                * gamma_out[None, :]
            )
            if HAS_LN_OUT_BIAS:
                beta_out = tl.load(
                    beta_out_ptr + k_range, mask=k_mask, other=0.0
                ).to(tl.float32)
                x_value += tl.where(k_mask[None, :], beta_out[None, :], 0.0)
            wp = tl.load(
                wp_ptr + offs_n[None, :] * CH64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            )
            if PRECISION == 1:
                val_acc = tl.dot(x_value, wp, val_acc, input_precision="ieee")
            elif PRECISION == 2:
                val_acc = tl.dot(x_value, wp, val_acc, input_precision="tf32")
            else:
                val_acc = tl.dot(x_value.to(wp.dtype), wp, val_acc)

        in_mean = tl.load(
            ln_stats_ptr + offs_m, mask=mask_m, other=0.0
        ).to(tl.float32)
        in_rstd = tl.load(
            ln_stats_ptr + M64 + offs_m, mask=mask_m, other=0.0
        ).to(tl.float32)
        gate_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
        for k_off in range(0, CZ, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < CZ64
            z_k = tl.load(
                x_in_ptr + offs_m[:, None] * CZ64 + k_range[None, :],
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gamma_in = tl.load(
                gamma_in_ptr + k_range, mask=k_mask, other=0.0
            ).to(tl.float32)
            x_gate = (
                (z_k - in_mean[:, None])
                * in_rstd[:, None]
                * gamma_in[None, :]
            )
            if HAS_LN_IN_BIAS:
                beta_in = tl.load(
                    beta_in_ptr + k_range, mask=k_mask, other=0.0
                ).to(tl.float32)
                x_gate += tl.where(k_mask[None, :], beta_in[None, :], 0.0)
            wg = tl.load(
                wg_ptr + offs_n[None, :] * CZ64 + k_range[:, None],
                mask=mask_n[None, :] & k_mask[:, None],
                other=0.0,
            )
            if PRECISION == 1:
                gate_acc = tl.dot(x_gate, wg, gate_acc, input_precision="ieee")
            elif PRECISION == 2:
                gate_acc = tl.dot(x_gate, wg, gate_acc, input_precision="tf32")
            else:
                gate_acc = tl.dot(x_gate.to(wg.dtype), wg, gate_acc)

        out_val = tl.sigmoid(gate_acc) * val_acc
        if WITH_ADD:
            residual = tl.load(
                residual_ptr + offs_m[:, None] * CZ64 + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            out_val += residual
        tl.store(
            out_ptr + offs_m[:, None] * CZ64 + offs_n[None, :],
            out_val.to(out_ptr.type.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit(
        do_not_specialize=["M", "eps"],
        do_not_specialize_on_alignment=["x_ptr", "out_ptr"],
    )
    def _ln_stats_kernel(
        x_ptr,  # [M, D]
        out_ptr,  # [2, M] float32: mean then rstd
        M,
        D: tl.constexpr,
        eps,
        TILE_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        M64 = M.to(tl.int64)
        offs_m = pid * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
        mask_m = offs_m < M64
        mask_d = offs_d < D

        x = tl.load(
            x_ptr + offs_m[:, None] * D + offs_d[None, :],
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        x = tl.where(mask_d[None, :], x, 0.0)
        mean = tl.sum(x, axis=1) / D
        x_c = tl.where(mask_d[None, :], x - mean[:, None], 0.0)
        var = tl.sum(x_c * x_c, axis=1) / D
        rstd = 1.0 / tl.sqrt(var + eps)

        tl.store(out_ptr + offs_m, mean, mask=mask_m)
        tl.store(out_ptr + M64 + offs_m, rstd, mask=mask_m)

    # ------------------------------------------------------------------ #
    # Transposing LayerNorm: (D, M) → (M, D)                             #
    # ------------------------------------------------------------------ #

    _LN_TRANSPOSE_TILE_M = 64

    @triton.jit(
        do_not_specialize=["M"],
        do_not_specialize_on_alignment=["x_ptr", "w_ptr", "b_ptr", "out_ptr"],
    )
    def _ln_transpose_kernel(
        x_ptr,      # (D, M) D-major: x[d, m] at x_ptr + d * M + m
        w_ptr,      # (D,) — LN scale
        b_ptr,      # (D,) — LN bias (only if HAS_BIAS)
        out_ptr,    # (M, D) M-major: out[m, d] at out_ptr + m * D + d
        M,
        D: tl.constexpr,
        EPS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        TILE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0).to(tl.int64)
        M64 = M.to(tl.int64)

        offs_m = pid * TILE_M + tl.arange(0, TILE_M).to(tl.int64)
        offs_d = tl.arange(0, D).to(tl.int64)
        mask_m = offs_m < M64

        # Read D-major: x_ptr[d, m] = x_ptr + d * M + m
        x_ptrs = x_ptr + offs_d[None, :] * M64 + offs_m[:, None]
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=1) / D
        x_c = x - mean[:, None]
        var = tl.sum(x_c * x_c, axis=1) / D
        rstd = 1.0 / tl.sqrt(var + EPS)
        x_hat = x_c * rstd[:, None]

        w = tl.load(w_ptr + offs_d).to(tl.float32)
        y = x_hat * w[None, :]
        if HAS_BIAS:
            b = tl.load(b_ptr + offs_d).to(tl.float32)
            y = y + b[None, :]

        # Write M-major: out_ptr[m, d] = out_ptr + m * D + d
        out_ptrs = out_ptr + offs_m[:, None] * D + offs_d[None, :]
        tl.store(out_ptrs, y.to(out_ptr.type.element_ty), mask=mask_m[:, None])


def _precision_flag(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 2 if torch.backends.cuda.matmul.allow_tf32 else 1
    return 0


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def gated_dual_gemm_fp32(
    x: torch.Tensor,  # [M, K] — raw pair (if fused LN) or LN'd pair
    wp: torch.Tensor,  # [Nproj, K]
    wg: torch.Tensor,  # [Nproj, K]
    mask: torch.Tensor | None,  # [M] or None
    ln_weight: torch.Tensor | None = None,  # [K] — LN gamma (fused LN)
    ln_bias: torch.Tensor | None = None,  # [K] — LN beta (fused LN)
    ln_stats: torch.Tensor | None = None,  # [2, M] — precomputed mean/rstd
    eps: float = 1e-5,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return ``sigmoid(LN?(x)@wg^T) * (LN?(x)@wp^T) * mask`` as ``[Nproj, M]``.

    When ``ln_weight`` is provided, LayerNorm is computed in-register per tile
    (no HBM intermediate). When ``None``, ``x`` is assumed already LN'd.
    """
    M, K = x.shape
    Nproj = wp.shape[0]
    x = x.contiguous()
    wp = wp.contiguous()
    wg = wg.contiguous()
    out = torch.empty(
        (Nproj, M),
        device=x.device,
        dtype=output_dtype if output_dtype is not None else x.dtype,
    )
    mask_flat = mask.contiguous().view(-1) if mask is not None else None
    _dummy = x
    fused_ln = ln_weight is not None
    if ln_stats is not None and not fused_ln:
        raise ValueError("ln_stats requires ln_weight / fused LN")
    BLOCK_CZ = _next_power_of_two(K) if fused_ln else 1
    gamma = ln_weight.contiguous() if fused_ln else x.new_zeros(1)
    beta = ln_bias.contiguous() if ln_bias is not None else x.new_zeros(1)
    stats = ln_stats.contiguous() if ln_stats is not None else x.new_zeros(1)
    cfg = _DUAL_GEMM_CFG_N64 if Nproj < 128 else _DUAL_GEMM_CFG

    def grid(meta):
        return (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(Nproj, meta["TILE_N"]))

    _gated_dual_gemm_kernel[grid](
        x,
        wp,
        wg,
        mask_flat if mask_flat is not None else _dummy,
        gamma,
        beta,
        stats,
        out,
        M,
        Nproj,
        K,
        eps,
        HAS_MASK=mask is not None,
        PRECISION=_precision_flag(x.dtype),
        FUSED_LN=fused_ln,
        HAS_LN_STATS=ln_stats is not None,
        HAS_LN_BIAS=ln_bias is not None,
        BLOCK_CZ=BLOCK_CZ,
        **cfg,
    )
    return out


def gated_column_gemm_fp32(
    x: torch.Tensor,  # [N*N, K]
    wp: torch.Tensor,  # [Nproj, K]
    wg: torch.Tensor,  # [Nproj, K]
    mask: torch.Tensor | None,  # [N*N]
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor | None,
    ln_stats: torch.Tensor,  # [2, N*N]
    *,
    n: int,
    i_start: int,
    i_rows: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Project ``x[k, i]`` into contiguous ``[Nproj, N, i_rows]`` storage."""
    full_m, K = x.shape
    if full_m != n * n:
        raise ValueError(f"x has {full_m} rows, expected n*n={n * n}")
    if i_start < 0 or i_rows <= 0 or i_start + i_rows > n:
        raise ValueError(
            f"invalid column range [{i_start}, {i_start + i_rows}) for n={n}"
        )
    Nproj = wp.shape[0]
    M_OUT = i_rows * n
    x = x.contiguous()
    wp = wp.contiguous()
    wg = wg.contiguous()
    stats = ln_stats.contiguous()
    mask_flat = mask.contiguous().view(-1) if mask is not None else None
    gamma = ln_weight.contiguous()
    beta = ln_bias.contiguous() if ln_bias is not None else x.new_zeros(1)
    out = torch.empty((Nproj, M_OUT), device=x.device, dtype=x.dtype)
    cfg = _DUAL_GEMM_CFG_N64 if Nproj < 128 else _DUAL_GEMM_CFG

    def grid(meta):
        return (
            triton.cdiv(M_OUT, meta["TILE_M"]),
            triton.cdiv(Nproj, meta["TILE_N"]),
        )

    _gated_column_gemm_kernel[grid](
        x,
        wp,
        wg,
        mask_flat if mask_flat is not None else x,
        gamma,
        beta,
        stats,
        out,
        n,
        i_start,
        i_rows,
        M_OUT,
        Nproj,
        K,
        eps,
        HAS_MASK=mask is not None,
        PRECISION=_precision_flag(x.dtype),
        HAS_LN_BIAS=ln_bias is not None,
        **cfg,
    )
    return out.view(Nproj, n, i_rows)


def ln_stats_fp32(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Return per-row LayerNorm mean/rstd as float32 ``[2, M]``."""
    assert x.ndim == 2
    M, D = x.shape
    x = x.contiguous()
    out = torch.empty((2, M), device=x.device, dtype=torch.float32)
    block_d = _next_power_of_two(D)

    grid = (triton.cdiv(M, _LN_STATS_TILE_M),)
    _ln_stats_kernel[grid](
        x,
        out,
        M,
        D=D,
        eps=eps,
        TILE_M=_LN_STATS_TILE_M,
        BLOCK_D=block_d,
        num_warps=8,
        num_stages=2,
    )
    return out


def gated_out_gemm_residual_fp32(
    x_in: torch.Tensor,  # [M, Cz] — raw pair (if fused LN) or LN'd pair
    x_out: torch.Tensor,  # [M, Ch]
    wg: torch.Tensor,  # [Cz, Cz]  (linear_g)
    wp: torch.Tensor,  # [Cz, Ch]  (linear_z)
    residual: torch.Tensor | None,  # [M, Cz] or None
    ln_weight: torch.Tensor | None = None,  # [Cz] — LN gamma (fused LN)
    ln_bias: torch.Tensor | None = None,  # [Cz] — LN beta (fused LN)
    ln_stats: torch.Tensor | None = None,  # [2, M] — precomputed mean/rstd
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``[residual +] sigmoid(LN?(x_in)@wg^T) * (x_out@wp^T)`` as ``[M, Cz]``.

    When ``ln_weight`` is provided, LayerNorm on ``x_in`` is computed in-register
    (no HBM intermediate). When ``None``, ``x_in`` is assumed already LN'd.
    """
    M, CZ = x_in.shape
    CH = x_out.shape[1]
    x_in = x_in.contiguous()
    x_out = x_out.contiguous()
    wg = wg.contiguous()
    wp = wp.contiguous()
    if out is None:
        cfg = _OUT_GEMM_CFG
    else:
        cfg = _OUT_GEMM_INPLACE_CFG_C128 if CZ == 128 else _OUT_GEMM_INPLACE_CFG
    if out is None:
        out = torch.empty((M, CZ), device=x_in.device, dtype=x_in.dtype)
    else:
        if out.shape != (M, CZ):
            raise ValueError(f"out shape {out.shape} does not match {(M, CZ)}")
        if out.device != x_in.device or out.dtype != x_in.dtype:
            raise ValueError("out must match x_in device and dtype")
        out = out.contiguous()
        # In-place residual writeback aliases x_in/residual.  It is safe only
        # when a single program owns the full channel row: the kernel reads the
        # complete row for LN/gating before storing that same row back.  With
        # multiple column programs, one program could overwrite columns another
        # still needs for LayerNorm.
        assert cfg["TILE_N"] >= CZ
    with_add = residual is not None
    resid = residual.contiguous().view(M, CZ) if with_add else x_in
    fused_ln = ln_weight is not None
    if ln_stats is not None and not fused_ln:
        raise ValueError("ln_stats requires ln_weight / fused LN")
    BLOCK_CZ = _next_power_of_two(CZ) if fused_ln else 1
    gamma = ln_weight.contiguous() if fused_ln else x_in.new_zeros(1)
    beta = ln_bias.contiguous() if ln_bias is not None else x_in.new_zeros(1)
    stats = ln_stats.contiguous() if ln_stats is not None else x_in.new_zeros(1)

    def grid(meta):
        return (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(CZ, meta["TILE_N"]))

    _gated_out_gemm_residual_kernel[grid](
        x_in,
        x_out,
        wg,
        wp,
        resid,
        gamma,
        beta,
        stats,
        out,
        M,
        CZ,
        CH,
        eps,
        WITH_ADD=with_add,
        PRECISION=_precision_flag(x_in.dtype),
        FUSED_LN=fused_ln,
        HAS_LN_STATS=ln_stats is not None,
        HAS_LN_BIAS=ln_bias is not None,
        BLOCK_CZ=BLOCK_CZ,
        **cfg,
    )
    return out


def gated_out_from_dm_residual_fp32(
    x_in: torch.Tensor,  # [M, CZ]
    x_dm: torch.Tensor,  # [CH, M]
    wg: torch.Tensor,  # [CZ, CZ]
    wp: torch.Tensor,  # [CZ, CH]
    residual: torch.Tensor | None,
    ln_in_weight: torch.Tensor,
    ln_in_bias: torch.Tensor | None,
    ln_stats: torch.Tensor,
    ln_out_weight: torch.Tensor,
    ln_out_bias: torch.Tensor | None,
    *,
    ln_out_eps: float = 1e-5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse LN_out on D-major contraction output with gated output projection."""
    M, CZ = x_in.shape
    CH, x_m = x_dm.shape
    if x_m != M:
        raise ValueError(f"x_dm has M={x_m}, expected {M}")
    if CH < 128:
        x_out = ln_transpose_fp32(
            x_dm, ln_out_weight, ln_out_bias, eps=ln_out_eps
        )
        return gated_out_gemm_residual_fp32(
            x_in,
            x_out,
            wg,
            wp,
            residual,
            ln_weight=ln_in_weight,
            ln_bias=ln_in_bias,
            ln_stats=ln_stats,
            out=out,
        )
    if out is None:
        out = torch.empty((M, CZ), device=x_in.device, dtype=x_in.dtype)
    elif out.shape != (M, CZ):
        raise ValueError(f"out shape {out.shape} does not match {(M, CZ)}")

    x_in = x_in.contiguous()
    x_dm = x_dm.contiguous()
    wg = wg.contiguous()
    wp = wp.contiguous()
    stats = ln_stats.contiguous()
    gamma_in = ln_in_weight.contiguous()
    beta_in = (
        ln_in_bias.contiguous() if ln_in_bias is not None else x_in.new_zeros(1)
    )
    gamma_out = ln_out_weight.contiguous()
    beta_out = (
        ln_out_bias.contiguous() if ln_out_bias is not None else x_in.new_zeros(1)
    )
    resid = residual.contiguous().view(M, CZ) if residual is not None else x_in
    if CZ > 128 or CH > 128:
        raise ValueError("fused D-major output supports CZ/CH <= 128")
    if out.data_ptr() == x_in.data_ptr() and _OUT_DM_CFG["TILE_N"] < CZ:
        raise ValueError("in-place D-major output requires one channel tile")

    grid = (
        triton.cdiv(M, _OUT_DM_CFG["TILE_M"]),
        triton.cdiv(CZ, _OUT_DM_CFG["TILE_N"]),
    )
    _gated_out_from_dm_kernel[grid](
        x_in,
        x_dm,
        wg,
        wp,
        resid,
        gamma_in,
        beta_in,
        stats,
        gamma_out,
        beta_out,
        out,
        M,
        CZ,
        CH,
        ln_out_eps,
        WITH_ADD=residual is not None,
        PRECISION=_precision_flag(x_in.dtype),
        HAS_LN_IN_BIAS=ln_in_bias is not None,
        HAS_LN_OUT_BIAS=ln_out_bias is not None,
        **_OUT_DM_CFG,
    )
    return out


def ln_transpose_fp32(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm with layout transpose: ``(D, M)`` D-major → ``(M, D)`` M-major.

    Reads ``x[d, m]`` from ``(D, M)`` contiguous input, normalises over D per
    row m, and writes ``out[m, d]`` in ``(M, D)`` contiguous output. One kernel,
    no intermediate.
    """
    assert x.ndim == 2 and x.is_contiguous()
    D, M = x.shape
    assert weight.shape[0] == D

    out = torch.empty((M, D), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(M, _LN_TRANSPOSE_TILE_M),)
    _dummy_bias = x.new_zeros(1)

    _ln_transpose_kernel[grid](
        x,
        weight.contiguous(),
        bias.contiguous() if bias is not None else _dummy_bias,
        out,
        M,
        D=D,
        EPS=eps,
        HAS_BIAS=bias is not None,
        TILE_M=_LN_TRANSPOSE_TILE_M,
        num_warps=8,
        num_stages=2,
    )
    return out


def _linear_z_weight(
    params: FusedTrimulTensorParams,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = params.linear_z_weight
    return weight if weight.dtype == dtype else weight.to(dtype=dtype)


def _trimul_chunked_outgoing_tensor(
    z_2d: torch.Tensor,
    mask_flat: torch.Tensor,
    ln_stats: torch.Tensor,
    params: FusedTrimulTensorParams,
    *,
    n: int,
    c_z: int,
    c_hidden: int,
    with_add: bool,
    out: torch.Tensor | None,
    chunk_cap: int,
) -> torch.Tensor:
    """Preserved I-row schedule for outgoing triangle multiplication."""
    m = n * n
    with torch.amp.autocast(device_type="cuda", enabled=False):
        b_full = gated_dual_gemm_fp32(
            z_2d,
            params.linear_b_p_weight,
            params.linear_b_g_weight,
            mask_flat,
            ln_weight=params.ln_in_weight,
            ln_bias=params.ln_in_bias,
            ln_stats=ln_stats,
            eps=params.ln_in_eps,
        )
    b_4d = b_full.view(c_hidden, 1, n, n)
    out_2d = (
        out.reshape(m, c_z)
        if out is not None
        else torch.empty((m, c_z), device=z_2d.device, dtype=z_2d.dtype)
    )
    linear_z_w = _linear_z_weight(params, z_2d.dtype)

    for i_start in range(0, n, chunk_cap):
        i_end = min(n, i_start + chunk_cap)
        rows = i_end - i_start
        m0, m1 = i_start * n, i_end * n
        z_c = z_2d[m0:m1]
        mask_c = mask_flat[m0:m1]
        stats_c = ln_stats[:, m0:m1].contiguous()
        with torch.amp.autocast(device_type="cuda", enabled=False):
            a_c = gated_dual_gemm_fp32(
                z_c,
                params.linear_a_p_weight,
                params.linear_a_g_weight,
                mask_c,
                ln_weight=params.ln_in_weight,
                ln_bias=params.ln_in_bias,
                ln_stats=stats_c,
                eps=params.ln_in_eps,
            )
            x_c = torch.einsum(
                "cbik,cbjk->cbij",
                a_c.view(c_hidden, 1, rows, n),
                b_4d,
            )
        del a_c
        x_dm = x_c.reshape(c_hidden, rows * n)
        del x_c
        gated_out_from_dm_residual_fp32(
            z_c,
            x_dm,
            params.linear_g_weight,
            linear_z_w,
            z_c if with_add else None,
            params.ln_in_weight,
            params.ln_in_bias,
            stats_c,
            params.ln_out_weight,
            params.ln_out_bias,
            ln_out_eps=params.ln_out_eps,
            out=out_2d[m0:m1],
        )
        del x_dm, stats_c

    return out_2d.view(1, n, n, c_z)


def _trimul_chunked_incoming_grouped_tensor(
    z_2d: torch.Tensor,
    mask_flat: torch.Tensor,
    ln_stats: torch.Tensor,
    params: FusedTrimulTensorParams,
    *,
    n: int,
    c_z: int,
    c_hidden: int,
    with_add: bool,
    out: torch.Tensor | None,
    chunk_cap: int,
) -> torch.Tensor:
    """Alias-safe incoming schedule with larger K chunks per channel group.

    Grouping hidden channels keeps ``group_size * grouped_k_cap`` no larger
    than ``c_hidden * chunk_cap``. The projection transient therefore retains
    the original memory bound while each accumulator channel is read/written
    fewer times.
    """
    m = n * n
    group_size = min(64, c_hidden)
    grouped_k_cap = min(n, chunk_cap * c_hidden // group_size)
    x_accum = torch.empty(
        (c_hidden, n, n), device=z_2d.device, dtype=z_2d.dtype
    )

    for c_start in range(0, c_hidden, group_size):
        c_end = min(c_hidden, c_start + group_size)
        channels = c_end - c_start
        wp_ab = torch.cat(
            [
                params.linear_a_p_weight[c_start:c_end],
                params.linear_b_p_weight[c_start:c_end],
            ],
            dim=0,
        )
        wg_ab = torch.cat(
            [
                params.linear_a_g_weight[c_start:c_end],
                params.linear_b_g_weight[c_start:c_end],
            ],
            dim=0,
        )
        x_group = x_accum[c_start:c_end]
        first_chunk = True
        for k_start in range(0, n, grouped_k_cap):
            k_end = min(n, k_start + grouped_k_cap)
            k_rows = k_end - k_start
            m0, m1 = k_start * n, k_end * n
            with torch.amp.autocast(device_type="cuda", enabled=False):
                ab_k = gated_dual_gemm_fp32(
                    z_2d[m0:m1],
                    wp_ab,
                    wg_ab,
                    mask_flat[m0:m1],
                    ln_weight=params.ln_in_weight,
                    ln_bias=params.ln_in_bias,
                    ln_stats=ln_stats[:, m0:m1].contiguous(),
                    eps=params.ln_in_eps,
                )
            a_k = ab_k[:channels].view(channels, k_rows, n)
            b_k = ab_k[channels:].view(channels, k_rows, n)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                if first_chunk:
                    torch.bmm(a_k.transpose(1, 2), b_k, out=x_group)
                    first_chunk = False
                else:
                    torch.baddbmm(
                        x_group,
                        a_k.transpose(1, 2),
                        b_k,
                        beta=1.0,
                        alpha=1.0,
                        out=x_group,
                    )
            del ab_k, a_k, b_k

    out_2d = (
        out.reshape(m, c_z)
        if out is not None
        else torch.empty((m, c_z), device=z_2d.device, dtype=z_2d.dtype)
    )
    linear_z_w = _linear_z_weight(params, z_2d.dtype)
    for i_start in range(0, n, chunk_cap):
        i_end = min(n, i_start + chunk_cap)
        rows = i_end - i_start
        m0, m1 = i_start * n, i_end * n
        x_dm = (
            x_accum[:, i_start:i_end, :]
            .reshape(c_hidden, rows * n)
            .contiguous()
        )
        z_c = z_2d[m0:m1]
        stats_c = ln_stats[:, m0:m1].contiguous()
        gated_out_from_dm_residual_fp32(
            z_c,
            x_dm,
            params.linear_g_weight,
            linear_z_w,
            z_c if with_add else None,
            params.ln_in_weight,
            params.ln_in_bias,
            stats_c,
            params.ln_out_weight,
            params.ln_out_bias,
            ln_out_eps=params.ln_out_eps,
            out=out_2d[m0:m1],
        )
        del x_dm, stats_c

    return out_2d.view(1, n, n, c_z)


def _trimul_chunked_incoming_column_tensor(
    z_2d: torch.Tensor,
    mask_flat: torch.Tensor,
    ln_stats: torch.Tensor,
    params: FusedTrimulTensorParams,
    *,
    n: int,
    c_z: int,
    c_hidden: int,
    with_add: bool,
    out: torch.Tensor | None,
    chunk_cap: int,
) -> torch.Tensor:
    """Incoming I-row schedule with one full B and strided A-column tiles."""
    m = n * n
    with torch.amp.autocast(device_type="cuda", enabled=False):
        b_full = gated_dual_gemm_fp32(
            z_2d,
            params.linear_b_p_weight,
            params.linear_b_g_weight,
            mask_flat,
            ln_weight=params.ln_in_weight,
            ln_bias=params.ln_in_bias,
            ln_stats=ln_stats,
            eps=params.ln_in_eps,
        )
    b_3d = b_full.view(c_hidden, n, n)
    out_2d = (
        out.reshape(m, c_z)
        if out is not None
        else torch.empty((m, c_z), device=z_2d.device, dtype=z_2d.dtype)
    )
    linear_z_w = _linear_z_weight(params, z_2d.dtype)

    for i_start in range(0, n, chunk_cap):
        rows = min(chunk_cap, n - i_start)
        m0, m1 = i_start * n, (i_start + rows) * n
        with torch.amp.autocast(device_type="cuda", enabled=False):
            a_ki = gated_column_gemm_fp32(
                z_2d,
                params.linear_a_p_weight,
                params.linear_a_g_weight,
                mask_flat,
                params.ln_in_weight,
                params.ln_in_bias,
                ln_stats,
                n=n,
                i_start=i_start,
                i_rows=rows,
                eps=params.ln_in_eps,
            )
            x_c = torch.bmm(a_ki.transpose(1, 2), b_3d)
        del a_ki
        x_dm = x_c.reshape(c_hidden, rows * n)
        del x_c
        z_c = z_2d[m0:m1]
        stats_c = ln_stats[:, m0:m1].contiguous()
        gated_out_from_dm_residual_fp32(
            z_c,
            x_dm,
            params.linear_g_weight,
            linear_z_w,
            z_c if with_add else None,
            params.ln_in_weight,
            params.ln_in_bias,
            stats_c,
            params.ln_out_weight,
            params.ln_out_bias,
            ln_out_eps=params.ln_out_eps,
            out=out_2d[m0:m1],
        )
        del x_dm, stats_c

    return out_2d.view(1, n, n, c_z)


def _trimul_whole_tensor(
    z: torch.Tensor,
    mask_flat: torch.Tensor,
    ln_stats: torch.Tensor,
    params: FusedTrimulTensorParams,
    *,
    outgoing: bool,
    with_add: bool,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Preserved full-pair fused schedule for either direction."""
    batch, n, _, c_z = z.shape
    c_hidden = params.linear_a_p_weight.shape[0]
    z_2d = z.reshape(-1, c_z)
    wp_ab = torch.cat(
        [params.linear_a_p_weight, params.linear_b_p_weight], dim=0
    )
    wg_ab = torch.cat(
        [params.linear_a_g_weight, params.linear_b_g_weight], dim=0
    )
    ab = gated_dual_gemm_fp32(
        z_2d,
        wp_ab,
        wg_ab,
        mask_flat,
        ln_weight=params.ln_in_weight,
        ln_bias=params.ln_in_bias,
        ln_stats=ln_stats,
        eps=params.ln_in_eps,
    )
    a = ab[:c_hidden].view(c_hidden, batch, n, n)
    b = ab[c_hidden:].view(c_hidden, batch, n, n)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        if outgoing:
            x = torch.einsum("cbik,cbjk->cbij", a, b)
        else:
            x = torch.einsum("cbki,cbkj->cbij", a, b)
    del a, b, ab
    x_dm = x.reshape(c_hidden, batch * n * n)
    linear_z_w = _linear_z_weight(params, x_dm.dtype)
    out_2d = gated_out_from_dm_residual_fp32(
        z_2d,
        x_dm,
        params.linear_g_weight,
        linear_z_w,
        z_2d if with_add else None,
        params.ln_in_weight,
        params.ln_in_bias,
        ln_stats,
        params.ln_out_weight,
        params.ln_out_bias,
        ln_out_eps=params.ln_out_eps,
        out=out.reshape(-1, c_z) if out is not None else None,
    )
    del x, x_dm
    return out_2d.view(batch, n, n, c_z)


def fused_trimul_tensor(
    z: torch.Tensor,
    mask: torch.Tensor,
    params: FusedTrimulTensorParams,
    *,
    outgoing: bool,
    with_add: bool,
    out: torch.Tensor | None,
    chunk_cap: int | None,
) -> torch.Tensor:
    """Run fused trimul from tensors only; model policy remains in the adapter."""
    batch, n, _, c_z = z.shape
    c_hidden = params.linear_a_p_weight.shape[0]
    z_2d = z.reshape(-1, c_z)
    mask_flat = mask.reshape(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        stats = ln_stats_fp32(z_2d, params.ln_in_eps)

    if chunk_cap is not None and chunk_cap < n and batch == 1:
        kwargs = {
            "n": n,
            "c_z": c_z,
            "c_hidden": c_hidden,
            "with_add": with_add,
            "out": out,
            "chunk_cap": chunk_cap,
        }
        if outgoing:
            return _trimul_chunked_outgoing_tensor(
                z_2d, mask_flat, stats, params, **kwargs
            )
        # The column schedule is fastest for c_hidden=128 update-only calls but
        # cannot overwrite z. In-place output and c64 use grouped accumulation,
        # which is alias-safe and retains the projection-memory bound.
        if c_hidden >= 128 and out is None:
            return _trimul_chunked_incoming_column_tensor(
                z_2d, mask_flat, stats, params, **kwargs
            )
        return _trimul_chunked_incoming_grouped_tensor(
            z_2d, mask_flat, stats, params, **kwargs
        )

    return _trimul_whole_tensor(
        z,
        mask_flat,
        stats,
        params,
        outgoing=outgoing,
        with_add=with_add,
        out=out,
    )
