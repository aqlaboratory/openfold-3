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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: fused gated output projection for
# the original low-memory triangle-attention inference path.

"""Fused Triton kernel for the triangle attention gated output (inference).

Replaces the separate ``_wrap_up`` path::

    gate = sigmoid(linear_g(LN(z)))   # [M, H*c_h]
    o    = gate * attn_out            # [M, H*c_h]
    o    = linear_o(o)                # [M, c_z]

with a single tiled kernel that:
  1. Loads raw z and recomputes LN in-register (c_z=128 fits one pass)
  2. Gate GEMM: LN(z) @ W_g^T  (K-loop over c_z)
  3. Sigmoid + element-wise multiply with attn_out
  4. Output GEMM: gated @ W_o^T  (K-loop over H*c_h)

The gate and gated intermediates never leave registers → no HBM allocation.

Forward / inference only. Requires CZ == CH (both 128 in the AF3 triangle
attention config).
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

    _GATED_OUT_CFG = dict(TILE_M=64, TILE_N=64, TILE_K=32, GROUP_M=8)

    @triton.jit
    def _fused_gated_attn_out_kernel(
        z_ptr,         # [M, CZ] — raw input (pre-LN), row-major
        attn_ptr,      # [M, CH] — attention output, row-major
        gamma_ptr,     # [CZ]    — LN scale
        beta_ptr,      # [CZ]    — LN offset (guarded by HAS_LN_BIAS)
        wg_ptr,        # [CH, CZ] — linear_g weight, row-major
        wo_ptr,        # [CZ, CH] — linear_o weight, row-major
        out_ptr,       # [M, CZ] — output, row-major
        M,
        CZ,
        CH,
        eps,
        HAS_LN_BIAS: tl.constexpr,
        PRECISION: tl.constexpr,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        BLOCK_CZ: tl.constexpr,  # next_pow2(CZ), covers full reduction
    ):
        """Each program computes a [TILE_M, TILE_N] output tile.

        Grid: (cdiv(M, TILE_M), cdiv(CZ, TILE_N)).
        """
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
        offs_cz = tl.arange(0, BLOCK_CZ).to(tl.int64)
        mask_m = offs_m < M
        cz_mask = offs_cz < CZ

        # ── Step 1: Load z and compute LN in registers ──────────────
        z_ptrs = z_ptr + (offs_m[:, None] * CZ + offs_cz[None, :])
        z_raw = tl.load(
            z_ptrs, mask=mask_m[:, None] & cz_mask[None, :], other=0.0,
        ).to(tl.float32)

        z_sum = tl.sum(z_raw, axis=1)
        mean = z_sum / CZ
        z_centered = tl.where(cz_mask[None, :], z_raw - mean[:, None], 0.0)
        var = tl.sum(z_centered * z_centered, axis=1) / CZ
        rstd = 1.0 / tl.sqrt(var + eps)

        gamma = tl.load(gamma_ptr + offs_cz, mask=cz_mask, other=0.0).to(tl.float32)
        z_hat = z_centered * rstd[:, None] * gamma[None, :]
        if HAS_LN_BIAS:
            beta = tl.load(beta_ptr + offs_cz, mask=cz_mask, other=0.0).to(tl.float32)
            z_hat = z_hat + tl.where(cz_mask[None, :], beta[None, :], 0.0)
        # z_hat: [TILE_M, BLOCK_CZ] in registers — LN'd input for gate GEMM

        # ── Step 2: Gate GEMM — z_hat @ W_g^T → [TILE_M, CH] ───────
        # W_g is [CH, CZ]. We compute z_hat [TILE_M, CZ] @ W_g^T [CZ, CH].
        # K-loop over CZ with TILE_K. For each K-tile, reload the matching
        # z_hat columns from z and reapply LN (cheap: just scale+shift with
        # saved mean/rstd). This avoids register slicing.
        #
        # Actually, we accumulate the full [TILE_M, CH] gate. But CH must be
        # compile-time for the accumulator. Since CH == CZ in our use case
        # (both 128), we use BLOCK_CZ as the accumulator width.
        offs_k = tl.arange(0, TILE_K).to(tl.int64)
        gate_acc = tl.zeros((TILE_M, BLOCK_CZ), dtype=tl.float32)

        for k_off in range(0, CZ, TILE_K):
            k_range = k_off + offs_k
            k_mask = k_range < CZ

            # Reload z columns for this K-tile and apply saved LN.
            z_k = tl.load(
                z_ptr + (offs_m[:, None] * CZ + k_range[None, :]),
                mask=mask_m[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            # Apply LN: (z - mean) * rstd * gamma [+ beta]
            z_k_centered = z_k - mean[:, None]
            gamma_k = tl.load(
                gamma_ptr + k_range, mask=k_mask, other=0.0,
            ).to(tl.float32)
            z_k_hat = z_k_centered * rstd[:, None] * gamma_k[None, :]
            if HAS_LN_BIAS:
                beta_k = tl.load(
                    beta_ptr + k_range, mask=k_mask, other=0.0,
                ).to(tl.float32)
                z_k_hat = z_k_hat + tl.where(k_mask[None, :], beta_k[None, :], 0.0)
            # z_k_hat: [TILE_M, TILE_K]

            # W_g slice: [BLOCK_CZ(=CH), TILE_K] → transpose to [TILE_K, BLOCK_CZ]
            wg_slice = tl.load(
                wg_ptr + (offs_cz[:, None] * CZ + k_range[None, :]),
                mask=(offs_cz[:, None] < CH) & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            wg_t = tl.trans(wg_slice)  # [TILE_K, BLOCK_CZ]

            if PRECISION == 1:
                gate_acc = tl.dot(z_k_hat, wg_t, gate_acc, input_precision="ieee")
            else:
                gate_acc = tl.dot(z_k_hat, wg_t, gate_acc)

        gate = tl.sigmoid(gate_acc)  # [TILE_M, BLOCK_CZ(=CH)]

        # ── Step 3: Load attn_out and element-wise multiply ─────────
        attn_tile = tl.load(
            attn_ptr + (offs_m[:, None] * CH + offs_cz[None, :]),
            mask=mask_m[:, None] & (offs_cz[None, :] < CH),
            other=0.0,
        ).to(tl.float32)
        gated = gate * attn_tile  # [TILE_M, BLOCK_CZ(=CH)] in registers

        # ── Step 4: Output GEMM — gated @ W_o^T → [TILE_M, TILE_N] ─
        # W_o is [CZ, CH]. We compute gated [TILE_M, CH] @ W_o^T [CH, CZ].
        # Only TILE_N output columns. K-loop over CH with TILE_K.
        # For each K-tile, slice gated[:, k:k+TILE_K] — but gated is in regs.
        # Same issue as z_hat. We re-derive: reload z, recompute LN, gate,
        # multiply attn_out per K-tile.
        #
        # Actually, gated is [TILE_M, BLOCK_CZ] already fully computed.
        # We can avoid re-deriving by iterating over the K dimension of gated.
        # Triton register tensors support compile-time slicing via offsets.
        # We use a different strategy: since BLOCK_CZ is compile-time and
        # equals the full CH, we can load W_o for the output TILE_N columns
        # in one shot: W_o[offs_n, :] = [TILE_N, CH].
        # Then dot: gated [TILE_M, BLOCK_CZ] @ W_o_slice^T [BLOCK_CZ, TILE_N].

        wo_slice = tl.load(
            wo_ptr + (offs_n[:, None] * CH + offs_cz[None, :]),
            mask=(offs_n[:, None] < CZ) & (offs_cz[None, :] < CH),
            other=0.0,
        ).to(tl.float32)
        wo_t = tl.trans(wo_slice)  # [BLOCK_CZ, TILE_N]

        if PRECISION == 1:
            out_tile = tl.dot(gated, wo_t, input_precision="ieee")
        else:
            out_tile = tl.dot(gated, wo_t)

        # ── Store ───────────────────────────────────────────────────
        o_ptrs = out_ptr + (offs_m[:, None] * CZ + offs_n[None, :])
        tl.store(
            o_ptrs,
            out_tile.to(out_ptr.type.element_ty),
            mask=mask_m[:, None] & (offs_n[None, :] < CZ),
        )


def _precision_flag(dtype: torch.dtype) -> int:
    return 1 if dtype == torch.float32 else 0


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fused_gated_attn_out(
    z_2d: torch.Tensor,
    attn_out_2d: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor | None,
    w_gate: torch.Tensor,
    w_out: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused gate-multiply-output for triangle attention.

    Computes ``linear_o(sigmoid(LN(z) @ W_g^T) * attn_out)`` with the LN,
    gate, and gated intermediates in registers only.

    Args:
        z_2d:        [M, c_z]     raw input (pre-LN)
        attn_out_2d: [M, H*c_h]   attention output (c_h*H == c_z required)
        ln_weight:   [c_z]        LayerNorm gamma
        ln_bias:     [c_z] | None LayerNorm beta
        w_gate:      [H*c_h, c_z] linear_g.weight
        w_out:       [c_z, H*c_h] linear_o.weight
        eps:         LN epsilon
    """
    M, CZ = z_2d.shape
    CH = attn_out_2d.shape[1]
    assert CZ == CH, (
        f"fused_gated_attn_out requires c_z == H*c_h, got {CZ} vs {CH}"
    )

    z_2d = z_2d.contiguous()
    attn_out_2d = attn_out_2d.contiguous()
    ln_weight = ln_weight.contiguous()
    w_gate = w_gate.contiguous()
    w_out = w_out.contiguous()

    BLOCK_CZ = _next_power_of_two(CZ)
    assert BLOCK_CZ <= 256, f"CZ={CZ} too large for one-pass in-register LN"

    out = torch.empty((M, CZ), device=z_2d.device, dtype=z_2d.dtype)
    beta_ptr = ln_bias.contiguous() if ln_bias is not None else z_2d.new_zeros(1)
    cfg = _GATED_OUT_CFG

    def grid(meta):
        return (triton.cdiv(M, meta["TILE_M"]), triton.cdiv(CZ, meta["TILE_N"]))

    _fused_gated_attn_out_kernel[grid](
        z_2d,
        attn_out_2d,
        ln_weight,
        beta_ptr,
        w_gate,
        w_out,
        out,
        M,
        CZ,
        CH,
        eps,
        HAS_LN_BIAS=ln_bias is not None,
        PRECISION=_precision_flag(z_2d.dtype),
        BLOCK_CZ=BLOCK_CZ,
        **cfg,
    )
    return out
