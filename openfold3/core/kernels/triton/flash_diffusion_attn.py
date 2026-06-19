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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: forward-only Triton diffusion
# attention that streams K/V while applying mask and pair bias.

"""Flash-attention-style diffusion self-attention with pair bias.

This is a forward-only inference kernel for the diffusion transformer token
attention path:

    q/k/v:      [B, S, H, N, C]
    mask_bias:  [B, S|1, 1, 1, N]
    pair_bias:  [B, S|1, H, N, N]

``S`` is the diffusion sample dimension.  The pair bias is usually shared
across samples (shape [B, 1, H, N, N]), while the mask may be expanded across
samples with stride 0.  The kernel streams K/V tiles and applies online
softmax, avoiding materialization of the [B, S, H, N, N] score tensor.
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


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


if _TRITON_AVAILABLE:

    @triton.jit(
        do_not_specialize=[
            "N_CTX",
            "S_dim",
            "stride_q_b",
            "stride_q_s",
            "stride_q_h",
            "stride_q_n",
            "stride_q_c",
            "stride_k_b",
            "stride_k_s",
            "stride_k_h",
            "stride_k_n",
            "stride_k_c",
            "stride_v_b",
            "stride_v_s",
            "stride_v_h",
            "stride_v_n",
            "stride_v_c",
            "stride_pb_b",
            "stride_pb_s",
            "stride_pb_h",
            "stride_pb_q",
            "stride_pb_k",
            "stride_mb_b",
            "stride_mb_s",
            "stride_mb_n",
            "stride_o_b",
            "stride_o_s",
            "stride_o_h",
            "stride_o_n",
            "stride_o_c",
            "softmax_scale",
        ],
        do_not_specialize_on_alignment=[
            "Q_ptr",
            "K_ptr",
            "V_ptr",
            "PAIR_BIAS_ptr",
            "MASK_BIAS_ptr",
            "OUT_ptr",
        ],
    )
    def _flash_diffusion_attn_kernel(
        Q_ptr,          # [B, S, H, N, C]
        K_ptr,          # [B, S, H, N, C]
        V_ptr,          # [B, S, H, N, C]
        PAIR_BIAS_ptr,  # [B, S|1, H, N, N]
        MASK_BIAS_ptr,  # [B, S|1, 1, 1, N] squeezed logically to [B, S|1, N]
        OUT_ptr,        # [B, S, H, N, C]
        N_CTX,
        S_dim,
        stride_q_b, stride_q_s, stride_q_h, stride_q_n, stride_q_c,
        stride_k_b, stride_k_s, stride_k_h, stride_k_n, stride_k_c,
        stride_v_b, stride_v_s, stride_v_h, stride_v_n, stride_v_c,
        stride_pb_b, stride_pb_s, stride_pb_h, stride_pb_q, stride_pb_k,
        stride_mb_b, stride_mb_s, stride_mb_n,
        stride_o_b, stride_o_s, stride_o_h, stride_o_n, stride_o_c,
        softmax_scale,
        H: tl.constexpr,
        BLOCK_C: tl.constexpr,
        CH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bsh = tl.program_id(1)

        h = pid_bsh % H
        bs = pid_bsh // H
        s = bs % S_dim
        b = bs // S_dim

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n_base = tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, BLOCK_C)
        mask_m = offs_m < N_CTX
        mask_c = offs_c < CH

        q_base = b * stride_q_b + s * stride_q_s + h * stride_q_h
        q = tl.load(
            Q_ptr
            + q_base
            + offs_m[:, None] * stride_q_n
            + offs_c[None, :] * stride_q_c,
            mask=mask_m[:, None] & mask_c[None, :],
            other=0.0,
        )

        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

        k_base = b * stride_k_b + s * stride_k_s + h * stride_k_h
        v_base = b * stride_v_b + s * stride_v_s + h * stride_v_h
        pb_row_base = (
            b * stride_pb_b
            + s * stride_pb_s
            + h * stride_pb_h
            + offs_m * stride_pb_q
        )
        mb_base = b * stride_mb_b + s * stride_mb_s

        for n_start in range(0, N_CTX, BLOCK_N):
            offs_n = n_start + offs_n_base
            mask_n = offs_n < N_CTX

            k = tl.load(
                K_ptr
                + k_base
                + offs_n[:, None] * stride_k_n
                + offs_c[None, :] * stride_k_c,
                mask=mask_n[:, None] & mask_c[None, :],
                other=0.0,
            )

            if ALLOW_TF32:
                qk = tl.dot(q, tl.trans(k), allow_tf32=True)
            else:
                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
            qk = qk * softmax_scale

            pair_bias = tl.load(
                PAIR_BIAS_ptr
                + pb_row_base[:, None]
                + offs_n[None, :] * stride_pb_k,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            mask_bias = tl.load(
                MASK_BIAS_ptr + mb_base + offs_n * stride_mb_n,
                mask=mask_n,
                other=float("-inf"),
            ).to(tl.float32)

            qk = qk + pair_bias + mask_bias[None, :]
            qk = tl.where(mask_n[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp(qk - m_ij[:, None])
            alpha = tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            m_i = m_ij

            v = tl.load(
                V_ptr
                + v_base
                + offs_n[:, None] * stride_v_n
                + offs_c[None, :] * stride_v_c,
                mask=mask_n[:, None] & mask_c[None, :],
                other=0.0,
            )
            if ALLOW_TF32:
                acc = tl.dot(p.to(V_ptr.dtype.element_ty), v, acc, allow_tf32=True)
            else:
                acc = tl.dot(
                    p.to(V_ptr.dtype.element_ty), v, acc, input_precision="ieee"
                )

        acc = acc / l_i[:, None]

        out_base = b * stride_o_b + s * stride_o_s + h * stride_o_h
        tl.store(
            OUT_ptr
            + out_base
            + offs_m[:, None] * stride_o_n
            + offs_c[None, :] * stride_o_c,
            acc.to(OUT_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_c[None, :],
        )


def flash_diffusion_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_bias: torch.Tensor,
    pair_bias: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Run diffusion self-attention without materializing attention scores.

    Args:
        q, k, v: tensors with shape ``[B, S, H, N, C]``.
        mask_bias: additive mask bias with shape ``[B, S|1, 1, 1, N]``.
        pair_bias: additive pair bias with shape ``[B, S|1, H, N, N]``.
        softmax_scale: scale applied to Q before the QK matmul.

    Returns:
        Tensor with shape ``[B, S, H, N, C]``.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("flash_diffusion_attn requires CUDA tensors")
    if q.dim() != 5:
        raise ValueError(f"expected q shape [B,S,H,N,C], got {q.shape}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(f"k/v must match q shape, got {k.shape}, {v.shape}, {q.shape}")

    B, S, H, N, CH = q.shape
    if pair_bias.shape[0] != B or pair_bias.shape[2:] != (H, N, N):
        raise ValueError(
            f"expected pair_bias [B,S|1,H,N,N] compatible with "
            f"{(B, S, H, N, N)}, got {pair_bias.shape}"
        )
    if pair_bias.shape[1] not in (1, S):
        raise ValueError(
            f"pair_bias sample dim must be 1 or {S}, got {pair_bias.shape}"
        )
    if mask_bias.shape[0] != B or mask_bias.shape[-1] != N:
        raise ValueError(
            f"expected mask_bias [B,S|1,1,1,N] compatible with "
            f"{(B, S, 1, 1, N)}, got {mask_bias.shape}"
        )
    if mask_bias.shape[1] not in (1, S):
        raise ValueError(
            f"mask_bias sample dim must be 1 or {S}, got {mask_bias.shape}"
        )
    if mask_bias.shape[-3:-1] != (1, 1):
        raise ValueError(f"expected singleton mask dims, got {mask_bias.shape}")

    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()

    out = torch.empty_like(q)

    # Keep sequence length out of constexprs.  CH/BLOCK_C specialize by model
    # signature, but not by target length.
    block_c = _next_power_of_two(CH)
    if block_c > 128:
        raise ValueError(f"head dim {CH} is unsupported")
    block_m = 128
    # Keep tile sizes independent of the target length.  For the OF3 diffusion
    # signature (CH=48), a 128-row query tile with a wider K/V tile gives the
    # best warmed throughput on both single-sample long-N and 5-sample rollout
    # shapes while compiling once per operation signature.
    block_n = 64
    num_warps = 4 if block_c <= 64 else 8

    pair_s_stride = pair_bias.stride(1) if pair_bias.shape[1] == S else 0
    mask_s_stride = mask_bias.stride(1) if mask_bias.shape[1] == S else 0

    grid = (triton.cdiv(N, block_m), B * S * H)
    allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)

    _flash_diffusion_attn_kernel[grid](
        q, k, v, pair_bias, mask_bias, out,
        N, S,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3), v.stride(4),
        pair_bias.stride(0), pair_s_stride, pair_bias.stride(2),
        pair_bias.stride(3), pair_bias.stride(4),
        mask_bias.stride(0), mask_s_stride, mask_bias.stride(-1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        float(softmax_scale),
        H=H,
        BLOCK_C=block_c,
        CH=CH,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        ALLOW_TF32=allow_tf32,
        num_warps=num_warps,
        num_stages=2,
    )

    return out
