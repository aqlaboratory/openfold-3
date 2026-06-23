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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: cap-blocked Triton triangle
# attention kernels for packed and generic V1 inference paths.

"""Flash-attention-style triangle-attention kernels for the optimized path.

Only the cap-blocked V1 triangle-attention path is kept. Its internal generic
kernel handles non-packed blocks, while the grouped-I2 kernel handles the
production packed Q/K/V/G block layout.

Forward / inference only — no backward implementation.
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

    @triton.jit(
        do_not_specialize=[
            "I_dim",
            "J_dim",
            "stride_q_b",
            "stride_q_i",
            "stride_q_h",
            "stride_q_j",
            "stride_q_c",
            "stride_k_b",
            "stride_k_i",
            "stride_k_h",
            "stride_k_j",
            "stride_k_c",
            "stride_v_b",
            "stride_v_i",
            "stride_v_h",
            "stride_v_j",
            "stride_v_c",
            "stride_g_b",
            "stride_g_i",
            "stride_g_j",
            "stride_g_h",
            "stride_g_c",
            "stride_tb_b",
            "stride_tb_h",
            "stride_tb_i",
            "stride_tb_j",
            "stride_mb_b",
            "stride_mb_i",
            "stride_mb_j",
            "mask_i_offset",
            "stride_o_b",
            "stride_o_i",
            "stride_o_j",
            "stride_o_h",
            "stride_o_c",
            "softmax_scale",
            "mask_inf",
        ],
        do_not_specialize_on_alignment=[
            "Q_ptr",
            "K_ptr",
            "V_ptr",
            "G_ptr",
            "TRI_BIAS_ptr",
            "MASK_BIAS_ptr",
            "OUT_ptr",
        ],
    )
    def _flash_tri_attn_kernel(
        Q_ptr,         # [B, I, H, J, CH]   query, scaled outside
        K_ptr,         # [B, I, H, J, CH]   key
        V_ptr,         # [B, I, H, J, CH]   value
        G_ptr,         # [B, I, J, H, CH]   sigmoid-gate (optional)
        TRI_BIAS_ptr,  # [B, H, I, J]       triangle bias (after squeeze)
        MASK_BIAS_ptr, # [B, I, J]          mask bias (after squeeze)
        OUT_ptr,       # [B, I, J, H, CH]   gated attn output, layout (B,I,J,H,CH)
        I_dim,
        J_dim,
        # strides
        stride_q_b, stride_q_i, stride_q_h, stride_q_j, stride_q_c,
        stride_k_b, stride_k_i, stride_k_h, stride_k_j, stride_k_c,
        stride_v_b, stride_v_i, stride_v_h, stride_v_j, stride_v_c,
        stride_g_b, stride_g_i, stride_g_j, stride_g_h, stride_g_c,
        stride_tb_b, stride_tb_h, stride_tb_i, stride_tb_j,
        stride_mb_b, stride_mb_i, stride_mb_j,
        mask_i_offset,
        stride_o_b, stride_o_i, stride_o_j, stride_o_h, stride_o_c,
        softmax_scale,
        mask_inf,
        H: tl.constexpr,
        HAS_MASK: tl.constexpr,
        MASK_IS_RAW: tl.constexpr,
        HAS_GATE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        CH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Flash-attn-style kernel — one program per (B*I*H, m-tile).

        Each program reads its row of Q (``[BLOCK_M, CH]``), then loops over
        K/V tiles of size ``BLOCK_N`` along J, updating ``o_acc, m_i, l_i``
        with online softmax.
        """
        # Decode grid indices.
        pid_m = tl.program_id(0)
        pid_bih = tl.program_id(1)
        # pid_bih encodes (b, i, h):  pid_bih = ((b * I_dim) + i) * H + h
        h = pid_bih % H
        bi = pid_bih // H
        i = bi % I_dim
        b = bi // I_dim

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_c = tl.arange(0, CH)
        mask_m = offs_m < J_dim

        # ── Load Q tile ────────────────────────────────────────────────
        q_base = (
            b * stride_q_b + i * stride_q_i + h * stride_q_h
        )
        q_ptrs = (
            Q_ptr + q_base
            + offs_m[:, None] * stride_q_j
            + offs_c[None, :] * stride_q_c
        )
        q = tl.load(
            q_ptrs, mask=mask_m[:, None], other=0.0,
        )

        # ── Flash-attn online-softmax state ───────────────────────────
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        o_acc = tl.zeros((BLOCK_M, CH), dtype=tl.float32)

        # Pre-compute base addresses for K, V, biases (constant inside the loop).
        k_base = b * stride_k_b + i * stride_k_i + h * stride_k_h
        v_base = b * stride_v_b + i * stride_v_i + h * stride_v_h
        # triangle_bias[b, h, q, k] — depends on Q row position, NOT i_batch.
        # We load a [BLOCK_M, BLOCK_N] tile per program; precompute the
        # per-row base for the m-tile here.
        tb_row_base = (
            b * stride_tb_b + h * stride_tb_h
            + offs_m * stride_tb_i  # [BLOCK_M] row offsets
        )
        mb_base = b * stride_mb_b + (i + mask_i_offset) * stride_mb_i

        # ── KV loop ───────────────────────────────────────────────────
        for n_start in range(0, J_dim, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < J_dim

            # Load K tile  [BLOCK_N, CH]
            k = tl.load(
                K_ptr + k_base
                + offs_n[:, None] * stride_k_j
                + offs_c[None, :] * stride_k_c,
                mask=mask_n[:, None], other=0.0,
            )

            # qk: [BLOCK_M, BLOCK_N]
            if ALLOW_TF32:
                qk = tl.dot(q, tl.trans(k), allow_tf32=True)
            else:
                qk = tl.dot(q, tl.trans(k), input_precision="ieee")
            rcp_ln2 = 1.4426950408889634
            qk = qk * (softmax_scale * rcp_ln2)

            # Triangle bias [BLOCK_M, BLOCK_N] — bias[b, h, q_row, k_col]
            tb = tl.load(
                TRI_BIAS_ptr
                + tb_row_base[:, None]
                + offs_n[None, :] * stride_tb_j,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            qk = qk + tb * rcp_ln2

            if HAS_MASK:
                mb = tl.load(
                    MASK_BIAS_ptr + mb_base + offs_n * stride_mb_j,
                    mask=mask_n, other=0.0,
                ).to(tl.float32)
                if MASK_IS_RAW:
                    mb = mask_inf * (mb - 1.0)
                qk = qk + mb[None, :] * rcp_ln2

            # Mask out-of-range Ks.
            qk = tl.where(mask_n[None, :], qk, float("-inf"))

            # Online softmax update.
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, 1)
            o_acc = o_acc * alpha[:, None]
            m_i = m_ij

            # Load V tile [BLOCK_N, CH]
            v = tl.load(
                V_ptr + v_base
                + offs_n[:, None] * stride_v_j
                + offs_c[None, :] * stride_v_c,
                mask=mask_n[:, None], other=0.0,
            )

            if ALLOW_TF32:
                o_acc = tl.dot(
                    p.to(V_ptr.dtype.element_ty), v, o_acc, allow_tf32=True
                )
            else:
                o_acc = tl.dot(
                    p.to(V_ptr.dtype.element_ty), v, o_acc, input_precision="ieee"
                )

        # Finalize.
        o_acc = o_acc / l_i[:, None]

        # ── Apply gate (loaded per-tile, applied in registers) ─────────
        if HAS_GATE:
            g_base = (
                b * stride_g_b + i * stride_g_i + h * stride_g_h
            )
            g_ptrs = (
                G_ptr + g_base
                + offs_m[:, None] * stride_g_j
                + offs_c[None, :] * stride_g_c
            )
            g = tl.load(
                g_ptrs, mask=mask_m[:, None], other=0.0,
            ).to(tl.float32)
            o_acc = o_acc * tl.sigmoid(g)

        # ── Store output ──────────────────────────────────────────────
        out_base = (
            b * stride_o_b + i * stride_o_i + h * stride_o_h
        )
        out_ptrs = (
            OUT_ptr + out_base
            + offs_m[:, None] * stride_o_j
            + offs_c[None, :] * stride_o_c
        )
        tl.store(
            out_ptrs,
            o_acc.to(OUT_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )



    @triton.jit(
        do_not_specialize=[
            "I_dim",
            "J_dim",
            "mask_i_offset",
            "softmax_scale",
            "mask_inf",
        ],
    )
    def _flash_tri_attn_v1_group_i2_kernel(
        Q_ptr,         # [1, rows, J, H, CH], packed-QKVG view
        K_ptr,         # [1, rows, J, H, CH], packed-QKVG view
        V_ptr,         # [1, rows, J, H, CH], packed-QKVG view
        G_ptr,         # [1, rows, J, H, CH], packed-QKVG view
        TRI_BIAS_ptr,  # [1, H, J, J], contiguous
        MASK_BIAS_ptr, # [1, J, J], contiguous raw mask
        OUT_ptr,       # [1, rows, J, H, CH], aliases Q
        I_dim,
        J_dim,
        mask_i_offset,
        softmax_scale,
        mask_inf,
        H: tl.constexpr,
        HAS_MASK: tl.constexpr,
        CH: tl.constexpr,
        QKVG_ROW_STRIDE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Grouped-I2 V1 kernel for block-local packed Q/K/V/G rows."""
        pid_m = tl.program_id(0)
        pid_group_h = tl.program_id(1)
        h = pid_group_h % H
        i0 = (pid_group_h // H) * 2
        i1 = i0 + 1
        has_i1 = i1 < I_dim

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n_base = tl.arange(0, BLOCK_N)
        offs_c = tl.arange(0, CH)
        mask_m = offs_m < J_dim
        rcp_ln2 = 1.4426950408889634
        q0_base = i0 * J_dim * QKVG_ROW_STRIDE + h * CH
        q1_base = i1 * J_dim * QKVG_ROW_STRIDE + h * CH
        q0 = tl.load(
            Q_ptr
            + q0_base
            + offs_m[:, None] * QKVG_ROW_STRIDE
            + offs_c[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        q1 = tl.load(
            Q_ptr
            + q1_base
            + offs_m[:, None] * QKVG_ROW_STRIDE
            + offs_c[None, :],
            mask=mask_m[:, None] & has_i1,
            other=0.0,
        )

        m0 = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        m1 = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        l1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        o0 = tl.zeros((BLOCK_M, CH), dtype=tl.float32)
        o1 = tl.zeros((BLOCK_M, CH), dtype=tl.float32)

        tb_row = h * J_dim * J_dim + offs_m[:, None] * J_dim
        k0_base = i0 * J_dim * QKVG_ROW_STRIDE + h * CH
        k1_base = i1 * J_dim * QKVG_ROW_STRIDE + h * CH
        v0_base = i0 * J_dim * QKVG_ROW_STRIDE + h * CH
        v1_base = i1 * J_dim * QKVG_ROW_STRIDE + h * CH
        mb0_base = (i0 + mask_i_offset) * J_dim
        mb1_base = (i1 + mask_i_offset) * J_dim

        for n_start in range(0, J_dim, BLOCK_N):
            offs_n = n_start + offs_n_base
            mask_n = offs_n < J_dim
            tb = tl.load(
                TRI_BIAS_ptr + tb_row + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            tb = tb * rcp_ln2

            k0 = tl.load(
                K_ptr
                + k0_base
                + offs_n[:, None] * QKVG_ROW_STRIDE
                + offs_c[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            qk0 = tl.dot(
                q0,
                tl.trans(k0),
                out_dtype=tl.float32,
                allow_tf32=True,
            )
            qk0 = qk0 * (softmax_scale * rcp_ln2) + tb
            if HAS_MASK:
                mb0 = tl.load(
                    MASK_BIAS_ptr + mb0_base + offs_n,
                    mask=mask_n,
                    other=0.0,
                ).to(tl.float32)
                mb0 = mask_inf * (mb0 - 1.0)
                qk0 = qk0 + mb0[None, :] * rcp_ln2
            qk0 = tl.where(mask_n[None, :], qk0, float("-inf"))
            m0_new = tl.maximum(m0, tl.max(qk0, 1))
            p0 = tl.math.exp2(qk0 - m0_new[:, None])
            a0 = tl.math.exp2(m0 - m0_new)
            l0 = l0 * a0 + tl.sum(p0, 1)
            o0 = o0 * a0[:, None]
            m0 = m0_new
            v0 = tl.load(
                V_ptr
                + v0_base
                + offs_n[:, None] * QKVG_ROW_STRIDE
                + offs_c[None, :],
                mask=mask_n[:, None],
                other=0.0,
            )
            o0 = tl.dot(
                p0.to(V_ptr.dtype.element_ty),
                v0,
                o0,
                out_dtype=tl.float32,
                allow_tf32=True,
            )

            if has_i1:
                k1 = tl.load(
                    K_ptr
                    + k1_base
                    + offs_n[:, None] * QKVG_ROW_STRIDE
                    + offs_c[None, :],
                    mask=mask_n[:, None],
                    other=0.0,
                )
                qk1 = tl.dot(
                    q1,
                    tl.trans(k1),
                    out_dtype=tl.float32,
                    allow_tf32=True,
                )
                qk1 = qk1 * (softmax_scale * rcp_ln2) + tb
                if HAS_MASK:
                    mb1 = tl.load(
                        MASK_BIAS_ptr + mb1_base + offs_n,
                        mask=mask_n,
                        other=0.0,
                    ).to(tl.float32)
                    mb1 = mask_inf * (mb1 - 1.0)
                    qk1 = qk1 + mb1[None, :] * rcp_ln2
                qk1 = tl.where(mask_n[None, :], qk1, float("-inf"))
                m1_new = tl.maximum(m1, tl.max(qk1, 1))
                p1 = tl.math.exp2(qk1 - m1_new[:, None])
                a1 = tl.math.exp2(m1 - m1_new)
                l1 = l1 * a1 + tl.sum(p1, 1)
                o1 = o1 * a1[:, None]
                m1 = m1_new
                v1 = tl.load(
                    V_ptr
                    + v1_base
                    + offs_n[:, None] * QKVG_ROW_STRIDE
                    + offs_c[None, :],
                    mask=mask_n[:, None],
                    other=0.0,
                )
                o1 = tl.dot(
                    p1.to(V_ptr.dtype.element_ty),
                    v1,
                    o1,
                    out_dtype=tl.float32,
                    allow_tf32=True,
                )

        g0_base = i0 * J_dim * QKVG_ROW_STRIDE + h * CH
        g0 = tl.load(
            G_ptr
            + g0_base
            + offs_m[:, None] * QKVG_ROW_STRIDE
            + offs_c[None, :],
            mask=mask_m[:, None],
            other=0.0,
        ).to(tl.float32)
        out0 = (o0 / l0[:, None]) * tl.sigmoid(g0)
        o0_base = i0 * J_dim * QKVG_ROW_STRIDE + h * CH
        tl.store(
            OUT_ptr
            + o0_base
            + offs_m[:, None] * QKVG_ROW_STRIDE
            + offs_c[None, :],
            out0.to(OUT_ptr.dtype.element_ty),
            mask=mask_m[:, None],
        )

        if has_i1:
            g1_base = i1 * J_dim * QKVG_ROW_STRIDE + h * CH
            g1 = tl.load(
                G_ptr
                + g1_base
                + offs_m[:, None] * QKVG_ROW_STRIDE
                + offs_c[None, :],
                mask=mask_m[:, None],
                other=0.0,
            ).to(tl.float32)
            out1 = (o1 / l1[:, None]) * tl.sigmoid(g1)
            o1_base = i1 * J_dim * QKVG_ROW_STRIDE + h * CH
            tl.store(
                OUT_ptr
                + o1_base
                + offs_m[:, None] * QKVG_ROW_STRIDE
                + offs_c[None, :],
                out1.to(OUT_ptr.dtype.element_ty),
                mask=mask_m[:, None],
            )


def flash_tri_attn_v1_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    triangle_bias: torch.Tensor,
    mask: torch.Tensor | None,
    row_start: int,
    softmax_scale: float,
    out: torch.Tensor,
    mask_inf: float,
) -> torch.Tensor:
    """Cap-blocked custom V1 triangle-attention core.

    ``q/k/v/gate/out`` are block-local ``[1, rows, J, H, CH]`` tensors. The
    triangle bias remains full-length ``[1, 1, H, J, J]`` and is shared across
    row blocks; ``row_start`` maps the local row index to the raw full mask.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    assert q.dim() == 5, q.shape
    B, rows, J_dim, H, CH = q.shape
    assert B == 1, q.shape
    assert k.shape == q.shape and v.shape == q.shape and gate.shape == q.shape
    assert out.shape == q.shape, out.shape
    assert triangle_bias.shape == (B, 1, H, J_dim, J_dim), triangle_bias.shape
    if mask is not None:
        assert mask.shape == (B, J_dim, J_dim), mask.shape

    tri = triangle_bias.squeeze(1)
    if not tri.is_contiguous():
        tri = tri.contiguous()

    if mask is None:
        mb = q.new_zeros(1)
        has_mask = False
        s_mb_b = s_mb_i = s_mb_j = 0
    else:
        mb = mask if mask.is_contiguous() else mask.contiguous()
        has_mask = True
        s_mb_b, s_mb_i, s_mb_j = mb.stride()

    assert CH in (16, 32, 64, 128), f"CH={CH} unsupported"

    qkvg_row_stride = H * CH * 4
    packed_stride = (
        rows * J_dim * qkvg_row_stride,
        J_dim * qkvg_row_stride,
        qkvg_row_stride,
        CH,
        1,
    )
    elem_size = q.element_size()
    allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    if (
        allow_tf32
        and
        q.stride() == packed_stride
        and k.stride() == packed_stride
        and v.stride() == packed_stride
        and gate.stride() == packed_stride
        and out.stride() == packed_stride
        and k.data_ptr() - q.data_ptr() == H * CH * elem_size
        and v.data_ptr() - q.data_ptr() == 2 * H * CH * elem_size
        and gate.data_ptr() - q.data_ptr() == 3 * H * CH * elem_size
        and out.data_ptr() == q.data_ptr()
    ):
        BLOCK_M = 128
        BLOCK_N = 16
        grid = (triton.cdiv(J_dim, BLOCK_M), triton.cdiv(rows, 2) * H)
        _flash_tri_attn_v1_group_i2_kernel[grid](
            q,
            k,
            v,
            gate,
            tri,
            mb,
            out,
            rows,
            J_dim,
            row_start,
            softmax_scale,
            float(mask_inf),
            H=H,
            HAS_MASK=has_mask,
            CH=CH,
            QKVG_ROW_STRIDE=qkvg_row_stride,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return out

    BLOCK_M = 128
    BLOCK_N = 16
    num_stages = 2 if q.dtype in (torch.float16, torch.bfloat16) else 1
    grid = (triton.cdiv(J_dim, BLOCK_M), B * rows * H)

    s_q_b, s_q_i, s_q_j, s_q_h, s_q_c = q.stride()
    s_k_b, s_k_i, s_k_j, s_k_h, s_k_c = k.stride()
    s_v_b, s_v_i, s_v_j, s_v_h, s_v_c = v.stride()
    s_g_b, s_g_i, s_g_j, s_g_h, s_g_c = gate.stride()
    s_o_b, s_o_i, s_o_j, s_o_h, s_o_c = out.stride()

    _flash_tri_attn_kernel[grid](
        q,
        k,
        v,
        gate,
        tri,
        mb,
        out,
        rows,
        J_dim,
        s_q_b,
        s_q_i,
        s_q_h,
        s_q_j,
        s_q_c,
        s_k_b,
        s_k_i,
        s_k_h,
        s_k_j,
        s_k_c,
        s_v_b,
        s_v_i,
        s_v_h,
        s_v_j,
        s_v_c,
        s_g_b,
        s_g_i,
        s_g_j,
        s_g_h,
        s_g_c,
        tri.stride(0),
        tri.stride(1),
        tri.stride(2),
        tri.stride(3),
        s_mb_b if has_mask else 0,
        s_mb_i if has_mask else 0,
        s_mb_j if has_mask else 0,
        row_start,
        s_o_b,
        s_o_i,
        s_o_j,
        s_o_h,
        s_o_c,
        softmax_scale,
        float(mask_inf),
        H=H,
        HAS_MASK=has_mask,
        MASK_IS_RAW=True,
        HAS_GATE=True,
        ALLOW_TF32=allow_tf32,
        CH=CH,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=num_stages,
    )
    return out
