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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: fused Triton LayerNorm, SwiGLU,
# output projection, mask, and residual kernel for pair transitions.

"""Fused Triton kernel for the SwiGLU pair transition.

Fuses the whole transition

    x -> LayerNorm -> SwiGLU(SiLU(linear_a(x)) * linear_b(x)) -> linear_out
      -> * mask  [-> + residual]

into a single row-tiled kernel. The [M, hidden] SwiGLU expansion (the
dominant inference memory peak for the pair representation) never
touches HBM: each program keeps only a [BLOCK_M, BLOCK_H] slice of it in
registers, accumulating the down-projection on the fly. The LayerNorm
normalized copy is likewise computed in-registers and never materialized.

With ``residual`` aliasing the input (the in-place pair-transition case),
the kernel reads ``x`` for LayerNorm and writes ``x + transition(x)`` back
into the same storage, so the op runs at ~zero extra allocation.

Forward / inference only: gated on ``not torch.is_grad_enabled()`` by the
caller; there is no autograd.Function here. Numerics mirror the eager
path -- SiLU is computed in fp32 (as in ``swiglu.py``) and every ``tl.dot``
accumulates in fp32 (as in ``fused_ln_linear.py``).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    @triton.jit
    def _silu_fp32(x):
        # x assumed fp32; SiLU(x) = x * sigmoid(x). Matches swiglu.py.
        return x * tl.sigmoid(x)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 128, "BLOCK_H": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_H": 64}, num_warps=4, num_stages=1),
            triton.Config({"BLOCK_M": 32, "BLOCK_H": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 32, "BLOCK_H": 64}, num_warps=4, num_stages=1),
        ],
        key=[],
        restore_value=["X_ptr", "Res_ptr", "Y_ptr"],
    )
    @triton.jit(
        do_not_specialize=[
            "stride_x_m",
            "stride_x_k",
            "stride_wa_h",
            "stride_wa_k",
            "stride_wb_h",
            "stride_wb_k",
            "stride_wo_n",
            "stride_wo_h",
            "stride_res_m",
            "stride_res_n",
            "stride_y_m",
            "stride_y_n",
            "stride_mask_m",
            "M",
            "eps",
        ],
        do_not_specialize_on_alignment=[
            "X_ptr",
            "WA_ptr",
            "WB_ptr",
            "WOUT_ptr",
            "Gamma_ptr",
            "Beta_ptr",
            "Mask_ptr",
            "Res_ptr",
            "Y_ptr",
        ],
    )
    def _fused_swiglu_transition_fwd_kernel(
        X_ptr,  # [M, K]      input
        WA_ptr,  # [H, K]     linear_a.weight  (nn.Linear stores [out, in])
        WB_ptr,  # [H, K]     linear_b.weight
        WOUT_ptr,  # [C_OUT, H] linear_out.weight
        Gamma_ptr,  # [K]      LN scale
        Beta_ptr,  # [K]       LN offset (optional)
        Mask_ptr,  # [M]       per-row mask (optional)
        Res_ptr,  # [M, C_OUT] residual to add (optional; may alias X)
        Y_ptr,  # [M, C_OUT]  output (may alias X / Res)
        stride_x_m,
        stride_x_k,
        stride_wa_h,
        stride_wa_k,
        stride_wb_h,
        stride_wb_k,
        stride_wo_n,
        stride_wo_h,
        stride_res_m,
        stride_res_n,
        stride_y_m,
        stride_y_n,
        stride_mask_m,
        M,
        K: tl.constexpr,
        H: tl.constexpr,
        C_OUT: tl.constexpr,
        eps,
        HAS_LN_BIAS: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program handles BLOCK_M rows; loops over hidden in BLOCK_H chunks."""
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_64 = offs_m.to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        k_mask = offs_k < K
        n_mask = offs_n < C_OUT

        # --- LayerNorm of x[BLOCK_M, K] in fp32 registers (one pass). ---
        x_ptrs = X_ptr + offs_m_64[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(
            tl.float32
        )
        mean = tl.sum(x, axis=1) / K
        x_centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(x_centered * x_centered, axis=1) / K
        rstd = 1.0 / tl.sqrt(var + eps)
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_hat = x_centered * rstd[:, None] * gamma[None, :]
        if HAS_LN_BIAS:
            beta = tl.load(Beta_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x_hat + tl.where(k_mask[None, :], beta[None, :], 0.0)
        # x_hat: [BLOCK_M, BLOCK_K], fp32, held in registers across the H-loop.

        # --- H-tiled up-projection -> SiLU*mul -> down-projection. ---
        acc_out = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            h_mask = offs_h < H

            # linear_a chunk: load W_a[h_chunk, :] = [BLOCK_H, K], dot x_hat @ W_a^T.
            wa_ptrs = (
                WA_ptr + offs_h[:, None] * stride_wa_h + offs_k[None, :] * stride_wa_k
            )
            wa = tl.load(wa_ptrs, mask=h_mask[:, None] & k_mask[None, :], other=0.0).to(
                tl.float32
            )
            acc_a = tl.dot(
                x_hat, tl.trans(wa), out_dtype=tl.float32, allow_tf32=ALLOW_TF32
            )  # [BLOCK_M, BLOCK_H]

            # linear_b chunk.
            wb_ptrs = (
                WB_ptr + offs_h[:, None] * stride_wb_h + offs_k[None, :] * stride_wb_k
            )
            wb = tl.load(wb_ptrs, mask=h_mask[:, None] & k_mask[None, :], other=0.0).to(
                tl.float32
            )
            acc_b = tl.dot(
                x_hat, tl.trans(wb), out_dtype=tl.float32, allow_tf32=ALLOW_TF32
            )

            # SiLU*mul in fp32 (out-of-range hidden lanes -> silu(0)=0 -> 0).
            silu = _silu_fp32(acc_a) * acc_b  # [BLOCK_M, BLOCK_H]

            # linear_out chunk: load W_out[:, h_chunk] = [C_OUT, BLOCK_H];
            # accumulate silu @ W_out_chunk^T -> [BLOCK_M, C_OUT].
            wo_ptrs = (
                WOUT_ptr + offs_n[:, None] * stride_wo_n + offs_h[None, :] * stride_wo_h
            )
            wo = tl.load(wo_ptrs, mask=n_mask[:, None] & h_mask[None, :], other=0.0).to(
                tl.float32
            )
            acc_out += tl.dot(
                silu, tl.trans(wo), out_dtype=tl.float32, allow_tf32=ALLOW_TF32
            )

        # --- mask, residual, store. ---
        if HAS_MASK:
            mask_val = tl.load(
                Mask_ptr + offs_m * stride_mask_m, mask=m_mask, other=0.0
            )
            acc_out = acc_out * mask_val[:, None].to(tl.float32)

        if HAS_RESIDUAL:
            res_ptrs = (
                Res_ptr
                + offs_m_64[:, None] * stride_res_m
                + offs_n[None, :] * stride_res_n
            )
            res = tl.load(
                res_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0
            ).to(tl.float32)
            acc_out = acc_out + res

        y_ptrs = Y_ptr + offs_m_64[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
        tl.store(
            y_ptrs,
            acc_out.to(Y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    def _next_power_of_two(n: int) -> int:
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def _launch_fused_swiglu_transition(
        x_2d: torch.Tensor,  # [M, K] contiguous
        gamma: torch.Tensor,
        beta: torch.Tensor | None,
        w_a: torch.Tensor,
        w_b: torch.Tensor,
        w_out: torch.Tensor,
        mask_1d: torch.Tensor | None,  # [M] or None
        residual_2d: torch.Tensor | None,  # [M, C_OUT] or None (may alias x_2d)
        eps: float,
    ) -> torch.Tensor:
        M, K = x_2d.shape
        H = w_a.shape[0]
        C_OUT = w_out.shape[0]

        # In-place when the residual aliases the input and C_OUT == K: write
        # back into x_2d. Otherwise allocate a fresh output.
        in_place = residual_2d is not None and residual_2d.data_ptr() == x_2d.data_ptr()
        if in_place:
            y = x_2d
        else:
            y = torch.empty((M, C_OUT), dtype=x_2d.dtype, device=x_2d.device)

        BLOCK_K = max(_next_power_of_two(K), 16)
        BLOCK_N = max(_next_power_of_two(C_OUT), 16)
        allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)

        # Honor the same global flag cuBLAS honors: PyTorch defaults
        # matmul.allow_tf32 to False, so eager F.linear on fp32 uses true
        # fp32. Matching it keeps the fused path numerically aligned with the
        # eager trunk (TF32 drift through the diffusion rollout otherwise
        # reaches ~1 A RMSD). bf16 inputs are upcast to fp32 in-kernel, so
        # this flag only bites the fp32 path, exactly as for cuBLAS.
        # BLOCK_M / BLOCK_H / warps / stages come from @triton.autotune.
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)

        beta_ptr = beta if beta is not None else x_2d.new_zeros(1)
        mask_ptr = mask_1d if mask_1d is not None else x_2d.new_zeros(1)
        stride_mask_m = mask_1d.stride(0) if mask_1d is not None else 0
        res = residual_2d if residual_2d is not None else x_2d
        stride_res_m = res.stride(0)
        stride_res_n = res.stride(1)

        _fused_swiglu_transition_fwd_kernel[grid](
            x_2d,
            w_a,
            w_b,
            w_out,
            gamma,
            beta_ptr,
            mask_ptr,
            res,
            y,
            x_2d.stride(0),
            x_2d.stride(1),
            w_a.stride(0),
            w_a.stride(1),
            w_b.stride(0),
            w_b.stride(1),
            w_out.stride(0),
            w_out.stride(1),
            stride_res_m,
            stride_res_n,
            y.stride(0),
            y.stride(1),
            stride_mask_m,
            M,
            K,
            H,
            C_OUT,
            eps,
            HAS_LN_BIAS=beta is not None,
            HAS_MASK=mask_1d is not None,
            HAS_RESIDUAL=residual_2d is not None,
            ALLOW_TF32=allow_tf32,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
        )
        return y
