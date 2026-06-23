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

Two kernels cover the memory-heavy stages of the AF3 (non-fused) trimul:
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
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    # Single static config (no runtime autotune): one JIT compile, all N.
    # M,N,K are runtime args so a new sequence length never recompiles.
    _DUAL_GEMM_CFG = dict(TILE_M=64, TILE_N=128, TILE_K=16, GROUP_M=8)
    _OUT_GEMM_CFG = dict(TILE_M=64, TILE_N=128, TILE_K=16, GROUP_M=8)
    _OUT_GEMM_INPLACE_CFG = dict(TILE_M=64, TILE_N=128, TILE_K=32, GROUP_M=8)
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
    cfg = _DUAL_GEMM_CFG

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
    cfg = _OUT_GEMM_INPLACE_CFG if out is not None else _OUT_GEMM_CFG
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
