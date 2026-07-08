# Copyright 2026 Advanced Micro Devices, Inc.
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

"""Triton Evoformer (triangle) attention kernel.

Fused forward with an optimized backward:
  - ``d_pair_bias`` accumulated in registers and folded into the dQ kernel,
    eliminating the per-element ``tl.atomic_add`` (the dominant backward cost);
  - dK/dV written directly in BLLHS layout (kernel does the reorder) with a
    folded dO preprocess that removes two transpose+contiguous copies;
  - per-dtype forward tiles;
  - MFMA-tile (``matrix_instr_nonkdim``) and ``waves_per_eu`` tuning gated to HIP.

Acknowledges prior LLNL evoformer training optimization work.
"""

import os
import warnings

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Triton is not available; the OpenFold3 Triton evoformer kernel is "
        "disabled and a non-kernel attention path will be used.",
        stacklevel=2,
    )
    _TRITON_AVAILABLE = False

# Sentinel: replaced with EvoformerAttention.apply when Triton is available.
TritonEvoformer = None
# Opt-in variable-shape entry point (set when Triton is available).
TritonEvoformerDynamic = None

if _TRITON_AVAILABLE:

    def is_hip():
        """Check if the current backend is HIP."""
        return triton.runtime.driver.active.get_current_target().backend == "hip"

    @triton.jit
    def _attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        V_block_ptr,
        res_mask_block_ptr,
        pair_bias_block_ptr,
        block_index_q,
        DIM,
        stride_K_seq,
        stride_V_seq,
        stride_mask_seq,
        stride_pair_bias_seq2,
        softmax_scale,
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        HAS_PAIR_BIAS: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        offs_q: tl.constexpr,
        offs_kv: tl.constexpr,
        offs_d: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        USE_EXP2: tl.constexpr = False,
    ):
        """Run the inner loop of the forward pass of the attention mechanism."""
        lo, hi = 0, SEQ_LEN
        Q_block = Q_block * tl.full((1,), softmax_scale, dtype=Q_block.dtype)

        for start_kv in range(lo, hi, BLOCK_SIZE_KV):
            start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
            if EVEN_Q & EVEN_KV:
                if HAS_PAIR_BIAS:
                    pair_bias_block = tl.load(pair_bias_block_ptr)
                res_mask_block = tl.load(res_mask_block_ptr).broadcast_to(
                    (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
                )
                if EVEN_DIM:
                    K_block = tl.load(K_block_ptr)
                    V_block = tl.load(V_block_ptr)
                else:
                    K_block = tl.load(
                        K_block_ptr, mask=offs_d[:, None] < DIM, other=0.0
                    )
                    V_block = tl.load(
                        V_block_ptr, mask=offs_d[None, :] < DIM, other=0.0
                    )
            else:
                if HAS_PAIR_BIAS:
                    pair_bias_block = tl.load(
                        pair_bias_block_ptr,
                        mask=(offs_q[:, None] < SEQ_LEN)
                        & ((start_kv + offs_kv)[None, :] < SEQ_LEN),
                        other=float("-inf"),
                    )
                res_mask_block = tl.load(
                    res_mask_block_ptr,
                    mask=(start_kv + offs_kv)[None, :] < SEQ_LEN,
                    other=float("-inf"),
                ).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
                if EVEN_DIM:
                    K_block = tl.load(
                        K_block_ptr,
                        mask=(start_kv + offs_kv)[None, :] < SEQ_LEN,
                        other=0.0,
                    )
                    V_block = tl.load(
                        V_block_ptr,
                        mask=(start_kv + offs_kv)[:, None] < SEQ_LEN,
                        other=0.0,
                    )
                else:
                    K_block = tl.load(
                        K_block_ptr,
                        mask=((start_kv + offs_kv)[None, :] < SEQ_LEN)
                        & (offs_d[:, None] < DIM),
                        other=0.0,
                    )
                    V_block = tl.load(
                        V_block_ptr,
                        mask=((start_kv + offs_kv)[:, None] < SEQ_LEN)
                        & (offs_d[None, :] < DIM),
                        other=0.0,
                    )

            QK_block = tl.dot(Q_block, K_block) + res_mask_block
            if HAS_PAIR_BIAS:
                QK_block += pair_bias_block

            if not EVEN_KV:
                QK_block += tl.where(
                    (start_kv + offs_kv)[None, :] < SEQ_LEN, 0, float("-inf")
                )

            # base-2 softmax: scale the full logit tile (dot + biases) by log2(e)
            # so exp2 == exp but is faster. Probabilities (hence all gradients) are
            # unchanged; M is stored in base-2 for the backward. Opt-in via USE_EXP2
            # (default off); works for both fp32 and bf16.
            log2e = 1.4426950408889634  # log2(e): exp(x) == exp2(x * log2e)
            if USE_EXP2:
                QK_block = QK_block * log2e
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block = QK_block - m_ij[:, None]

            P_block = tl.math.exp2(QK_block) if USE_EXP2 else tl.math.exp(QK_block)
            l_ij = tl.sum(P_block, 1)

            alpha = tl.math.exp2(m_i - m_ij) if USE_EXP2 else tl.math.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij

            P_block = P_block.to(V_block.dtype)
            O_block = O_block * alpha[:, None]
            O_block = tl.dot(P_block, V_block, O_block)

            m_i = m_ij

            V_block_ptr += BLOCK_SIZE_KV * stride_V_seq
            K_block_ptr += BLOCK_SIZE_KV * stride_K_seq
            if HAS_PAIR_BIAS:
                pair_bias_block_ptr += BLOCK_SIZE_KV * stride_pair_bias_seq2
            res_mask_block_ptr += BLOCK_SIZE_KV * stride_mask_seq

        return O_block, l_i, m_i

    @triton.heuristics(
        {
            "EVEN_Q": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_Q"] == 0,
            "EVEN_KV": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_KV"] == 0,
            "EVEN_DIM": lambda args: args["DIM"] == args["BLOCK_DIM"],
        }
    )
    @triton.jit
    def _attn_fwd(
        Q,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
        K,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
        V,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
        res_mask,  # BATCH_SIZE, N_SEQ, 1, SEQ_LEN, 1
        pair_bias,  # BATCH_SIZE, 1, HEAD, SEQ_LEN, SEQ_LEN
        softmax_scale,
        M,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN
        O,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
        stride_Q_batch,
        stride_Q_msa,
        stride_Q_head,
        stride_Q_seq,
        stride_Q_dim,
        stride_K_batch,
        stride_K_msa,
        stride_K_head,
        stride_K_seq,
        stride_K_dim,
        stride_V_batch,
        stride_V_msa,
        stride_V_head,
        stride_V_seq,
        stride_V_dim,
        stride_O_batch,
        stride_O_msa,
        stride_O_head,
        stride_O_seq,
        stride_O_dim,
        stride_pair_bias_batch,
        stride_pair_bias_head,
        stride_pair_bias_seq1,
        stride_pair_bias_seq2,
        stride_mask_batch,
        stride_mask_msa,
        stride_mask_seq,
        BATCH_SIZE,
        HEAD: tl.constexpr,
        N_SEQ: tl.constexpr,
        SEQ_LEN: tl.constexpr,
        DIM: tl.constexpr,
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        HAS_PAIR_BIAS: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        USE_EXP2: tl.constexpr = False,
    ):
        """Run the forward pass of the attention mechanism."""
        block_index_q = tl.program_id(0)

        index_batch_msa_head = tl.program_id(1)
        index_batch_msa = index_batch_msa_head // HEAD
        index_head = index_batch_msa_head % HEAD
        index_batch = index_batch_msa // N_SEQ
        index_msa = index_batch_msa % N_SEQ

        # Cast to int64 to avoid int32 overflow for large sequences
        qvk_offset = (
            index_batch.to(tl.int64) * stride_Q_batch
            + index_msa.to(tl.int64) * stride_Q_msa
            + index_head * stride_Q_head
        )
        offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
        offs_kv = tl.arange(0, BLOCK_SIZE_KV)
        offs_d = tl.arange(0, BLOCK_DIM)

        Q_block_ptr = (
            Q + qvk_offset + (offs_q[:, None] * stride_Q_seq + offs_d[None, :])
        )
        V_block_ptr = (
            V + qvk_offset + (offs_kv[:, None] * stride_V_seq + offs_d[None, :])
        )
        K_block_ptr = (
            K + qvk_offset + (offs_kv[None, :] * stride_K_seq + offs_d[:, None])
        )
        pair_bias_block_ptr = (
            pair_bias
            + index_batch * stride_pair_bias_batch
            + index_head * stride_pair_bias_head
            + (
                offs_q[:, None] * stride_pair_bias_seq1
                + offs_kv[None, :] * stride_pair_bias_seq2
            )
        )
        O_block_ptr = (
            O + qvk_offset + (offs_q[:, None] * stride_O_seq + offs_d[None, :])
        )

        res_mask_block_ptr = (
            res_mask
            + index_batch * stride_mask_batch
            + index_msa * stride_mask_msa
            + (offs_kv[None, :] * stride_mask_seq)
        )

        m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
        O_block = tl.zeros([BLOCK_SIZE_Q, BLOCK_DIM], dtype=tl.float32)

        # Load Q block; it stays in SRAM for the duration of the inner loop
        if EVEN_Q & EVEN_KV:
            if EVEN_DIM:
                Q_block = tl.load(Q_block_ptr)
            else:
                Q_block = tl.load(Q_block_ptr, mask=offs_d[None, :] < DIM, other=0.0)
        else:
            if EVEN_DIM:
                Q_block = tl.load(
                    Q_block_ptr, mask=offs_q[:, None] < SEQ_LEN, other=0.0
                )
            else:
                Q_block = tl.load(
                    Q_block_ptr,
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < DIM),
                    other=0.0,
                )

        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            res_mask_block_ptr,
            pair_bias_block_ptr,
            block_index_q,
            DIM,
            stride_K_seq,
            stride_V_seq,
            stride_mask_seq,
            stride_pair_bias_seq2,
            softmax_scale,
            EVEN_Q,
            EVEN_KV,
            EVEN_DIM,
            HAS_PAIR_BIAS,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            BLOCK_DIM,
            offs_q,
            offs_kv,
            offs_d,
            SEQ_LEN,
            USE_EXP2,
        )

        m_i += (
            tl.math.log2(l_i) if USE_EXP2 else tl.math.log(l_i)
        )  # base matches fwd-inner exp2 gate
        O_block = O_block / l_i[:, None]
        O_block = O_block.to(O.type.element_ty)
        m_ptrs = M + index_batch_msa_head * SEQ_LEN + offs_q

        if EVEN_Q:
            tl.store(m_ptrs, m_i)
            if EVEN_DIM:
                tl.store(O_block_ptr, O_block)
            else:
                tl.store(O_block_ptr, O_block, mask=offs_d[None, :] < DIM)
        else:
            tl.store(m_ptrs, m_i, mask=offs_q < SEQ_LEN)
            if EVEN_DIM:
                tl.store(O_block_ptr, O_block, mask=offs_q[:, None] < SEQ_LEN)
            else:
                tl.store(
                    O_block_ptr,
                    O_block,
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < DIM),
                )

    @triton.jit
    def _attn_fwd_dyn(
        Q,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
        K,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
        V,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
        res_mask,  # BATCH_SIZE, N_SEQ, 1, SEQ_LEN, 1
        pair_bias,  # BATCH_SIZE, 1, HEAD, SEQ_LEN, SEQ_LEN
        softmax_scale,
        M,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN
        O,  # BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM
        stride_Q_batch,
        stride_Q_msa,
        stride_Q_head,
        stride_Q_seq,
        stride_Q_dim,
        stride_K_batch,
        stride_K_msa,
        stride_K_head,
        stride_K_seq,
        stride_K_dim,
        stride_V_batch,
        stride_V_msa,
        stride_V_head,
        stride_V_seq,
        stride_V_dim,
        stride_O_batch,
        stride_O_msa,
        stride_O_head,
        stride_O_seq,
        stride_O_dim,
        stride_pair_bias_batch,
        stride_pair_bias_head,
        stride_pair_bias_seq1,
        stride_pair_bias_seq2,
        stride_mask_batch,
        stride_mask_msa,
        stride_mask_seq,
        BATCH_SIZE,
        HEAD: tl.constexpr,
        N_SEQ,
        SEQ_LEN,
        DIM: tl.constexpr,
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        HAS_PAIR_BIAS: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        USE_EXP2: tl.constexpr = False,
    ):
        """Dynamic-shape forward: N_SEQ & SEQ_LEN runtime, EVEN_* off.

        Opt-in second entry point for variable-size callers (one compile for all
        sizes); the specialized _attn_fwd stays the default. Shares _attn_fwd_inner.
        """
        block_index_q = tl.program_id(0)

        index_batch_msa_head = tl.program_id(1)
        index_batch_msa = index_batch_msa_head // HEAD
        index_head = index_batch_msa_head % HEAD
        index_batch = index_batch_msa // N_SEQ
        index_msa = index_batch_msa % N_SEQ

        # Cast to int64 to avoid int32 overflow for large sequences
        qvk_offset = (
            index_batch.to(tl.int64) * stride_Q_batch
            + index_msa.to(tl.int64) * stride_Q_msa
            + index_head * stride_Q_head
        )
        offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
        offs_kv = tl.arange(0, BLOCK_SIZE_KV)
        offs_d = tl.arange(0, BLOCK_DIM)

        Q_block_ptr = (
            Q + qvk_offset + (offs_q[:, None] * stride_Q_seq + offs_d[None, :])
        )
        V_block_ptr = (
            V + qvk_offset + (offs_kv[:, None] * stride_V_seq + offs_d[None, :])
        )
        K_block_ptr = (
            K + qvk_offset + (offs_kv[None, :] * stride_K_seq + offs_d[:, None])
        )
        pair_bias_block_ptr = (
            pair_bias
            + index_batch * stride_pair_bias_batch
            + index_head * stride_pair_bias_head
            + (
                offs_q[:, None] * stride_pair_bias_seq1
                + offs_kv[None, :] * stride_pair_bias_seq2
            )
        )
        O_block_ptr = (
            O + qvk_offset + (offs_q[:, None] * stride_O_seq + offs_d[None, :])
        )

        res_mask_block_ptr = (
            res_mask
            + index_batch * stride_mask_batch
            + index_msa * stride_mask_msa
            + (offs_kv[None, :] * stride_mask_seq)
        )

        m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
        O_block = tl.zeros([BLOCK_SIZE_Q, BLOCK_DIM], dtype=tl.float32)

        # Load Q block; it stays in SRAM for the duration of the inner loop
        if EVEN_Q & EVEN_KV:
            if EVEN_DIM:
                Q_block = tl.load(Q_block_ptr)
            else:
                Q_block = tl.load(Q_block_ptr, mask=offs_d[None, :] < DIM, other=0.0)
        else:
            if EVEN_DIM:
                Q_block = tl.load(
                    Q_block_ptr, mask=offs_q[:, None] < SEQ_LEN, other=0.0
                )
            else:
                Q_block = tl.load(
                    Q_block_ptr,
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < DIM),
                    other=0.0,
                )

        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            res_mask_block_ptr,
            pair_bias_block_ptr,
            block_index_q,
            DIM,
            stride_K_seq,
            stride_V_seq,
            stride_mask_seq,
            stride_pair_bias_seq2,
            softmax_scale,
            EVEN_Q,
            EVEN_KV,
            EVEN_DIM,
            HAS_PAIR_BIAS,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            BLOCK_DIM,
            offs_q,
            offs_kv,
            offs_d,
            SEQ_LEN,
            USE_EXP2,
        )

        m_i += (
            tl.math.log2(l_i) if USE_EXP2 else tl.math.log(l_i)
        )  # base matches fwd-inner exp2 gate
        O_block = O_block / l_i[:, None]
        O_block = O_block.to(O.type.element_ty)
        m_ptrs = M + index_batch_msa_head * SEQ_LEN + offs_q

        if EVEN_Q:
            tl.store(m_ptrs, m_i)
            if EVEN_DIM:
                tl.store(O_block_ptr, O_block)
            else:
                tl.store(O_block_ptr, O_block, mask=offs_d[None, :] < DIM)
        else:
            tl.store(m_ptrs, m_i, mask=offs_q < SEQ_LEN)
            if EVEN_DIM:
                tl.store(O_block_ptr, O_block, mask=offs_q[:, None] < SEQ_LEN)
            else:
                tl.store(
                    O_block_ptr,
                    O_block,
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_d[None, :] < DIM),
                )

    @triton.jit
    def _attn_bwd_preprocess(
        O,
        dO,
        D,
        SEQ_LEN,
        BLOCK_SIZE_Q: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        dO_out=None,
        HEAD: tl.constexpr = 1,
        READ_DO_BLLHS: tl.constexpr = False,
        stride_batch=0,
        stride_msa=0,
        stride_head=0,
        stride_seq=0,
        N_SEQ: tl.constexpr = 1,
    ):
        """Run the preprocessing step of the backward pass of the attention
        mechanism.

        When READ_DO_BLLHS, dO is the raw [B,N,L,H,D] incoming grad: read it
        strided and write the internal [B,N,H,L,D] copy (dO_out) the other bwd
        kernels need, folding the standalone transpose+contiguous into this kernel.
        """
        block_index_q = tl.program_id(0)
        offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
        index_batch_msa_head = tl.program_id(1)
        offs_dim = tl.arange(0, BLOCK_DIM)

        # Cast to int64 to avoid int32 overflow for large sequences
        # Internal-layout slab from explicit strides (contiguous =>
        # stride_seq==DIM and slab==idx*SEQ_LEN*DIM, identical to before).
        ibm = index_batch_msa_head // HEAD
        ih = index_batch_msa_head % HEAD
        ib = ibm // N_SEQ
        im = ibm % N_SEQ
        slab = (
            ib.to(tl.int64) * stride_batch
            + im.to(tl.int64) * stride_msa
            + ih.to(tl.int64) * stride_head
        )
        q_mask = (offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM)

        # Load a single block of BLOCK_SIZE_Q rows of O
        O_block = tl.load(
            O + slab + offs_q[:, None] * stride_seq + offs_dim[None, :],
            mask=q_mask,
            other=0.0,
        )
        # Load a single block of BLOCK_SIZE_Q rows of dO
        if READ_DO_BLLHS:
            # BLLHS [B,N,L,H,D]: head stride DIM, residue stride HEAD*DIM.
            bn = index_batch_msa_head.to(tl.int64) // HEAD
            h = index_batch_msa_head.to(tl.int64) % HEAD
            do_base = bn * SEQ_LEN * HEAD * DIM + h * DIM
            dO_raw = tl.load(
                dO + do_base + offs_q[:, None] * (HEAD * DIM) + offs_dim[None, :],
                mask=q_mask,
                other=0.0,
            )
            tl.store(
                dO_out + slab + offs_q[:, None] * stride_seq + offs_dim[None, :],
                dO_raw,
                mask=q_mask,
            )
            dO_block = dO_raw.to(tl.float32)
        else:
            dO_block = tl.load(
                dO + slab + offs_q[:, None] * stride_seq + offs_dim[None, :],
                mask=q_mask,
                other=0.0,
            ).to(tl.float32)
        # Compute the D block
        D_block = tl.sum(dO_block * O_block, axis=1)  # Shape: (BLOCK_SIZE_Q,)
        # Store the D block
        D_block_ptrs = D + index_batch_msa_head.to(tl.int64) * SEQ_LEN + offs_q
        tl.store(D_block_ptrs, D_block, mask=offs_q < SEQ_LEN)

    @triton.jit
    def _attn_bwd_dbias_dq(
        Q,
        K,
        V,
        res_mask,
        pair_bias,
        softmax_scale,
        dO,
        dQ,
        d_pair_bias,
        M,
        D,
        stride_batch,
        stride_head,
        stride_msa,
        stride_seq,
        stride_pair_bias_batch,
        stride_pair_bias_head,
        stride_pair_bias_seq1,
        stride_pair_bias_seq2,
        stride_mask_batch,
        stride_mask_msa,
        stride_mask_seq,
        stride_d_pair_bias_batch,
        stride_d_pair_bias_head,
        stride_d_pair_bias_seq1,
        stride_d_pair_bias_seq2,
        HEAD,
        N_SEQ,
        SEQ_LEN,
        BLOCK_DIM: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        HAS_PAIR_BIAS: tl.constexpr,
        USE_EXP2: tl.constexpr = False,
    ):
        """dbias accumulated in registers and stored once (no atomic); dQ via
        relaxed atomic. HAS_PAIR_BIAS=False skips all d_pair_bias work."""
        pid_q = tl.program_id(0)
        pid_kv = tl.program_id(1)
        pid_bh = tl.program_id(2)
        index_batch = pid_bh // HEAD
        index_head = pid_bh % HEAD

        offs_q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
        offs_kv = pid_kv * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)
        offs_dim = tl.arange(0, BLOCK_DIM)

        q_in = offs_q < SEQ_LEN
        kv_in = offs_kv < SEQ_LEN
        dim_in = offs_dim < DIM

        bh_off = (
            index_batch.to(tl.int64) * stride_batch
            + index_head.to(tl.int64) * stride_head
        )

        kvT_col = offs_kv[None, :] * stride_seq + offs_dim[:, None]

        qk_valid = q_in[:, None] & kv_in[None, :]
        if HAS_PAIR_BIAS:
            pb_ptrs = (
                pair_bias
                + index_batch.to(tl.int64) * stride_pair_bias_batch
                + index_head.to(tl.int64) * stride_pair_bias_head
                + offs_q[:, None] * stride_pair_bias_seq1
                + offs_kv[None, :] * stride_pair_bias_seq2
            )
            pair_bias_block = tl.load(pb_ptrs, mask=qk_valid, other=0.0)

        dbias_block = tl.zeros([BLOCK_SIZE_Q, BLOCK_SIZE_KV], dtype=tl.float32)

        for index_msa in tl.range(0, N_SEQ, num_stages=PIPE_STAGES):
            msa_off = bh_off + index_msa.to(tl.int64) * stride_msa
            md_off = ((index_batch * N_SEQ + index_msa) * HEAD + index_head).to(
                tl.int64
            ) * SEQ_LEN

            K_T_block = tl.load(
                K + msa_off + kvT_col,
                mask=kv_in[None, :] & dim_in[:, None],
                other=0.0,
            )
            V_T_block = tl.load(
                V + msa_off + kvT_col,
                mask=kv_in[None, :] & dim_in[:, None],
                other=0.0,
            )

            qd = offs_q[:, None] * stride_seq + offs_dim[None, :]
            q_valid = q_in[:, None] & dim_in[None, :]
            Q_block = tl.load(Q + msa_off + qd, mask=q_valid, other=0.0)
            dO_block = tl.load(dO + msa_off + qd, mask=q_valid, other=0.0)

            M_block = tl.load(M + md_off + offs_q, mask=q_in, other=0.0)[:, None]
            Di = tl.load(D + md_off + offs_q, mask=q_in, other=0.0)

            rm_ptrs = (
                res_mask
                + index_batch.to(tl.int64) * stride_mask_batch
                + index_msa.to(tl.int64) * stride_mask_msa
                + offs_kv * stride_mask_seq
            )
            res_mask_block = tl.load(rm_ptrs, mask=kv_in, other=float("-inf"))[
                None, :
            ].broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))

            Q_scaled = Q_block * tl.full((1,), softmax_scale, dtype=Q_block.dtype)
            QK_block = tl.dot(Q_scaled, K_T_block) + res_mask_block
            if HAS_PAIR_BIAS:
                QK_block += pair_bias_block

            if USE_EXP2:  # opt-in exp2 (match fwd); both fp32 and bf16
                log2e = 1.4426950408889634  # log2(e): exp(x) == exp2(x * log2e)
                QK_block = QK_block * log2e
            # Fully-masked rows lose the logsumexp correction (the masking bias is a
            # large finite value), so clamp (qk-M <= 0 by construction) and renormalize
            # the reconstructed probabilities. Threshold -1e3 is decoupled from the
            # masking ``inf`` (fires for any inf >= 1e3, never on real logits ~O(1e2)).
            P_arg = tl.minimum(QK_block - M_block, 0.0)
            P_block = tl.math.exp2(P_arg) if USE_EXP2 else tl.math.exp(P_arg)
            P_block = tl.where(res_mask_block <= -1e3, P_block / SEQ_LEN, P_block)
            dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
            dS_block = P_block * (dP_block - Di[:, None])

            if HAS_PAIR_BIAS:
                dbias_block += dS_block

            dS_cast = dS_block.to(K_T_block.dtype)
            dQ_block = softmax_scale * tl.dot(dS_cast, tl.trans(K_T_block))
            tl.atomic_add(
                dQ + msa_off + qd,
                dQ_block,
                mask=q_valid,
                sem="relaxed",
            )

        if HAS_PAIR_BIAS:
            dbias_ptrs = (
                d_pair_bias
                + index_batch.to(tl.int64) * stride_d_pair_bias_batch
                + index_head.to(tl.int64) * stride_d_pair_bias_head
                + offs_q[:, None] * stride_d_pair_bias_seq1
                + offs_kv[None, :] * stride_d_pair_bias_seq2
            )
            tl.store(dbias_ptrs, dbias_block, mask=qk_valid)

    @triton.heuristics(
        {
            "EVEN_Q": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_Q"] == 0,
            "EVEN_KV": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_KV"] == 0,
            "EVEN_DIM": lambda args: args["DIM"] == args["BLOCK_DIM"],
        }
    )
    @triton.jit
    def _attn_bwd_dk_dv(
        Q,
        K,
        V,
        res_mask,
        pair_bias,
        softmax_scale,
        dO,
        dQ,
        dK,
        dV,
        M,
        D,
        stride_batch,
        stride_head,
        stride_msa,
        stride_seq,
        stride_pair_bias_batch,
        stride_pair_bias_head,
        stride_pair_bias_seq1,
        stride_pair_bias_seq2,
        stride_mask_batch,
        stride_mask_msa,
        stride_mask_seq,
        HEAD,
        N_SEQ,
        SEQ_LEN,
        BLOCK_DIM: tl.constexpr,
        DIM: tl.constexpr,
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
        WRITE_BLLHS: tl.constexpr = False,
        USE_EXP2: tl.constexpr = False,
    ):
        """Run the backward pass of the attention mechanism."""
        index_batch_msa_head = tl.program_id(1)
        index_batch_msa = index_batch_msa_head // HEAD
        index_head = index_batch_msa_head % HEAD
        index_batch = index_batch_msa // N_SEQ
        index_msa = index_batch_msa % N_SEQ

        # Cast indices to int64 to avoid int32 overflow
        offset_batch_msa_head = (
            index_batch.to(tl.int64) * stride_batch
            + index_msa.to(tl.int64) * stride_msa
            + index_head.to(tl.int64) * stride_head
        )
        offset_batch_msa_head_seq = index_batch_msa_head.to(tl.int64) * SEQ_LEN

        Q += offset_batch_msa_head
        K += offset_batch_msa_head
        V += offset_batch_msa_head
        dO += offset_batch_msa_head
        dQ += offset_batch_msa_head
        # dK/dV optionally written directly in BLLHS layout to skip a transpose.
        if WRITE_BLLHS:
            out_off = (
                index_batch.to(tl.int64) * stride_batch
                + index_msa.to(tl.int64) * stride_msa
                + index_head.to(tl.int64) * DIM
            )
            dK += out_off
            dV += out_off
        else:
            dK += offset_batch_msa_head
            dV += offset_batch_msa_head

        M += offset_batch_msa_head_seq
        D += offset_batch_msa_head_seq

        offs_dim = tl.arange(0, BLOCK_DIM)

        index_block_kv = tl.program_id(0)
        offs_kv = index_block_kv * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)
        offs_q = tl.arange(0, BLOCK_SIZE_Q)

        dK_block = tl.zeros([BLOCK_SIZE_KV, BLOCK_DIM], dtype=tl.float32)
        dV_block = tl.zeros([BLOCK_SIZE_KV, BLOCK_DIM], dtype=tl.float32)

        res_mask_block_ptr = (
            res_mask
            + index_batch.to(tl.int64) * stride_mask_batch
            + index_msa.to(tl.int64) * stride_mask_msa
            + offs_kv[None, :] * stride_mask_seq
        )

        # K and V stay in SRAM throughout the inner loop
        if EVEN_Q & EVEN_KV:
            res_mask_T_block = tl.trans(tl.load(res_mask_block_ptr)).broadcast_to(
                (BLOCK_SIZE_KV, BLOCK_SIZE_Q)
            )
            if EVEN_DIM:
                K_block = tl.load(
                    K + offs_kv[:, None] * stride_seq + offs_dim[None, :]
                )  # Shape: (BLOCK_SIZE_KV, DIM)
                V_block = tl.load(
                    V + offs_kv[:, None] * stride_seq + offs_dim[None, :]
                )  # Shape: (BLOCK_SIZE_KV, DIM)
            else:
                K_block = tl.load(
                    K + offs_kv[:, None] * stride_seq + offs_dim[None, :],
                    mask=offs_dim[None, :] < DIM,
                    other=0.0,
                )  # Shape: (BLOCK_SIZE_KV, DIM)
                V_block = tl.load(
                    V + offs_kv[:, None] * stride_seq + offs_dim[None, :],
                    mask=offs_dim[None, :] < DIM,
                    other=0.0,
                )  # Shape: (BLOCK_SIZE_KV, DIM)
        else:
            res_mask_T_block = tl.trans(
                tl.load(
                    res_mask_block_ptr,
                    mask=offs_kv[None, :] < SEQ_LEN,
                    other=float("-inf"),
                )
            ).broadcast_to((BLOCK_SIZE_KV, BLOCK_SIZE_Q))
            if EVEN_DIM:
                K_block = tl.load(
                    K + offs_kv[:, None] * stride_seq + offs_dim[None, :],
                    mask=offs_kv[:, None] < SEQ_LEN,
                    other=0.0,
                )
                V_block = tl.load(
                    V + offs_kv[:, None] * stride_seq + offs_dim[None, :],
                    mask=offs_kv[:, None] < SEQ_LEN,
                    other=0.0,
                )
            else:
                K_block = tl.load(
                    K + offs_kv[:, None] * stride_seq + offs_dim[None, :],
                    mask=(offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                    other=0.0,
                )
                V_block = tl.load(
                    V + offs_kv[:, None] * stride_seq + offs_dim[None, :],
                    mask=(offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                    other=0.0,
                )

        pair_bias_T_block_ptr = (
            pair_bias
            + (
                index_batch.to(tl.int64) * stride_pair_bias_batch
                + index_head.to(tl.int64) * stride_pair_bias_head
            )
            + offs_q[None, :] * stride_pair_bias_seq1
            + offs_kv[:, None] * stride_pair_bias_seq2
        )
        qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None]
        dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :]

        K_block = K_block * tl.full((1,), softmax_scale, dtype=K_block.dtype)

        curr_q = 0
        num_steps = (SEQ_LEN + BLOCK_SIZE_Q - 1) // BLOCK_SIZE_Q

        for _blk_idx in range(num_steps):
            offs_q = curr_q + tl.arange(0, BLOCK_SIZE_Q)

            if EVEN_Q & EVEN_KV:
                m = tl.load(M + offs_q)
                pair_bias_T_block = tl.load(pair_bias_T_block_ptr)
                Di = tl.load(D + offs_q)  # [(BLOCK_SIZE_Q, )]
                if EVEN_DIM:
                    qT_block = tl.load(qT_ptrs)
                    dO_block = tl.load(dO_ptrs)
                else:
                    qT_block = tl.load(qT_ptrs, mask=offs_dim[:, None] < DIM, other=0.0)
                    dO_block = tl.load(dO_ptrs, mask=offs_dim[None, :] < DIM, other=0.0)
            else:
                m = tl.load(M + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
                pair_bias_T_block = tl.load(
                    pair_bias_T_block_ptr,
                    mask=(offs_q[None, :] < SEQ_LEN) & (offs_kv[:, None] < SEQ_LEN),
                    other=float("-inf"),
                )
                Di = tl.load(D + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
                if EVEN_DIM:
                    qT_block = tl.load(
                        qT_ptrs, mask=offs_q[None, :] < SEQ_LEN, other=0.0
                    )
                    dO_block = tl.load(
                        dO_ptrs, mask=offs_q[:, None] < SEQ_LEN, other=0.0
                    )
                else:
                    qT_block = tl.load(
                        qT_ptrs,
                        mask=(offs_q[None, :] < SEQ_LEN) & (offs_dim[:, None] < DIM),
                        other=0.0,
                    )
                    dO_block = tl.load(
                        dO_ptrs,
                        mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                        other=0.0,
                    )

            # Compute P^T = K Q^T (transposed attention scores)
            QK_T_block = (
                tl.dot(K_block, qT_block) + pair_bias_T_block + res_mask_T_block
            )

            if not (EVEN_Q & EVEN_KV):
                QK_T_block += tl.where(
                    (offs_kv[:, None] < SEQ_LEN) & (offs_q[None, :] < SEQ_LEN),
                    0,
                    float("-inf"),
                )

            if USE_EXP2:  # opt-in exp2 (match fwd); both fp32 and bf16
                log2e = 1.4426950408889634  # log2(e): exp(x) == exp2(x * log2e)
                QK_T_block = QK_T_block * log2e
            # Fully-masked rows: clamp + renormalize; threshold -1e3 decoupled from the
            P_T_arg = tl.minimum(QK_T_block - m[None, :], 0.0)
            P_T_block = tl.math.exp2(P_T_arg) if USE_EXP2 else tl.math.exp(P_T_arg)
            P_T_block = tl.where(
                res_mask_T_block <= -1e3, P_T_block / SEQ_LEN, P_T_block
            )

            dV_block += tl.dot(P_T_block.to(K_block.dtype), dO_block)

            dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
            dS_T_block = P_T_block * (dpT_block - Di[None, :])
            dS_T_block = dS_T_block.to(K_block.dtype)

            dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))

            # Increment pointers
            curr_q += BLOCK_SIZE_Q
            qT_ptrs += BLOCK_SIZE_Q * stride_seq
            dO_ptrs += BLOCK_SIZE_Q * stride_seq
            pair_bias_T_block_ptr += BLOCK_SIZE_Q * stride_pair_bias_seq1

        out_seq_stride = (HEAD * DIM) if WRITE_BLLHS else stride_seq
        dV_block_ptrs = dV + offs_kv[:, None] * out_seq_stride + offs_dim[None, :]
        dK_block_ptrs = dK + offs_kv[:, None] * out_seq_stride + offs_dim[None, :]

        if EVEN_Q & EVEN_KV:
            if EVEN_DIM:
                tl.store(dV_block_ptrs, dV_block)
                tl.store(dK_block_ptrs, dK_block)
            else:
                tl.store(dV_block_ptrs, dV_block, mask=offs_dim[None, :] < DIM)
                tl.store(dK_block_ptrs, dK_block, mask=offs_dim[None, :] < DIM)
        else:
            if EVEN_DIM:
                tl.store(dV_block_ptrs, dV_block, mask=offs_kv[:, None] < SEQ_LEN)
                tl.store(dK_block_ptrs, dK_block, mask=offs_kv[:, None] < SEQ_LEN)
            else:
                tl.store(
                    dV_block_ptrs,
                    dV_block,
                    mask=(offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                )
                tl.store(
                    dK_block_ptrs,
                    dK_block,
                    mask=(offs_kv[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                )

    class EvoformerAttention(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            Q,
            K,
            V,
            res_mask,
            pair_bias,
            has_pair_bias=True,
            dynamic=False,
            softmax_scale=None,
        ):
            """Run the forward pass of the attention mechanism.

            softmax_scale: optional caller-supplied softmax scale; defaults to
            DIM**-0.5 (standard scaled dot-product). Runtime arg, no recompile.

            dynamic=True selects the SEQ_LEN/N_SEQ-runtime forward (_attn_fwd_dyn):
            one compile for all sequence lengths (opt-in; see _attn_fwd_dyn).

            has_pair_bias: set False when pair_bias is all-zeros (MSA column attention).
            This eliminates all pair_bias HBM loads in the forward kernel.
            """
            # Q, K, V: [Batch, N_seq, N_res, Head, Dim]
            # res_mask: [Batch, N_seq, 1, 1, N_res]
            # pair_bias: [Batch, 1, Head, N_res, N_res]

            DIM_Q, DIM_K, DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
            assert DIM_Q == DIM_K and DIM_K == DIM_V

            # TRANSPOSE-FOLD (grad-gated): in inference (no grad needed) pass
            # strided views and skip the 3 .contiguous() copies -> faster forward.
            # In training keep contiguous so the backward reads coalesced (strided
            # backward reads are slower than the copies). NOTE: torch disables grad
            # *inside* autograd.Function.forward, so is_grad_enabled() is always
            # False here -- gate on input requires_grad instead.
            if (
                Q.requires_grad
                or K.requires_grad
                or V.requires_grad
                or pair_bias.requires_grad
            ):
                Q = Q.transpose(-2, -3).contiguous()
                K = K.transpose(-2, -3).contiguous()
                V = V.transpose(-2, -3).contiguous()
            else:
                Q = Q.transpose(-2, -3)
                K = K.transpose(-2, -3)
                V = V.transpose(-2, -3)

            BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM = Q.shape

            assert res_mask.shape == (
                BATCH_SIZE,
                N_SEQ,
                1,
                1,
                SEQ_LEN,
            ), f"{tuple(res_mask.shape)} != {(BATCH_SIZE, N_SEQ, 1, 1, SEQ_LEN)}"
            assert pair_bias.shape == (
                BATCH_SIZE,
                1,
                HEAD,
                SEQ_LEN,
                SEQ_LEN,
            ), f"{tuple(pair_bias.shape)} != {(BATCH_SIZE, 1, HEAD, SEQ_LEN, SEQ_LEN)}"

            if softmax_scale is None:
                softmax_scale = DIM**-0.5
            BLOCK_DIM = max(triton.next_power_of_2(DIM), 32)

            O = torch.empty_like(Q)

            extra_kern_args = {}
            if is_hip():
                waves_per_eu = 3 if DIM <= 64 else 2
                extra_kern_args = {
                    "waves_per_eu": waves_per_eu,
                    "allow_flush_denorm": True,
                }
            else:
                # MFMA/waves_per_eu tunings are HIP-only; on other backends the
                # kernel runs correct but untuned. Warn once (warnings dedups).
                warnings.warn(
                    "OpenFold3 Triton evoformer kernel is running on a non-HIP "
                    "backend with default, untuned Triton config (AMD MFMA / "
                    "waves_per_eu tunings disabled). Correctness holds; "
                    "performance is not tuned for this backend.",
                    stacklevel=2,
                )

            # Forward tiles Q=64, KV=16: validated fastest on real (low_mem chunked)
            block_size_q = 64
            _oss_bkv = 16

            grid = lambda args: (  # noqa: E731
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
                BATCH_SIZE * N_SEQ * HEAD,
                1,
            )

            # M is the logsumexp for the backward pass, one for each query
            M = torch.empty(
                (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN), device=Q.device, dtype=torch.float32
            )

            # exp2 base-2 softmax: opt-in (default off), works for both fp32 and bf16.
            # Enable with env OF3_TRITON_EXP2=1.
            use_exp2 = os.environ.get("OF3_TRITON_EXP2") == "1"

            # Default: specialized _attn_fwd (SEQ_LEN/EVEN_* constexpr via heuristics).
            # dynamic: _attn_fwd_dyn (SEQ_LEN/N_SEQ runtime) with EVEN_* passed off.
            _fwd_kernel = _attn_fwd_dyn if dynamic else _attn_fwd
            _extra_even = (
                dict(EVEN_Q=False, EVEN_KV=False, EVEN_DIM=(DIM == BLOCK_DIM))
                if dynamic
                else {}
            )
            _fwd_kernel[grid](
                Q=Q,
                K=K,
                V=V,
                res_mask=res_mask,
                pair_bias=pair_bias,
                softmax_scale=softmax_scale,
                M=M,
                O=O,
                stride_Q_batch=Q.stride(0),
                stride_Q_msa=Q.stride(1),
                stride_Q_head=Q.stride(2),
                stride_Q_seq=Q.stride(3),
                stride_Q_dim=Q.stride(4),
                stride_K_batch=K.stride(0),
                stride_K_msa=K.stride(1),
                stride_K_head=K.stride(2),
                stride_K_seq=K.stride(3),
                stride_K_dim=K.stride(4),
                stride_V_batch=V.stride(0),
                stride_V_msa=V.stride(1),
                stride_V_head=V.stride(2),
                stride_V_seq=V.stride(3),
                stride_V_dim=V.stride(4),
                stride_O_batch=O.stride(0),
                stride_O_msa=O.stride(1),
                stride_O_head=O.stride(2),
                stride_O_seq=O.stride(3),
                stride_O_dim=O.stride(4),
                stride_pair_bias_batch=pair_bias.stride(0),
                stride_pair_bias_head=pair_bias.stride(2),
                stride_pair_bias_seq1=pair_bias.stride(3),
                stride_pair_bias_seq2=pair_bias.stride(4),
                stride_mask_batch=res_mask.stride(0),
                stride_mask_msa=res_mask.stride(1),
                stride_mask_seq=res_mask.stride(4),
                BATCH_SIZE=BATCH_SIZE,
                HEAD=HEAD,
                N_SEQ=N_SEQ,
                SEQ_LEN=SEQ_LEN,
                DIM=DIM,
                BLOCK_DIM=BLOCK_DIM,
                HAS_PAIR_BIAS=has_pair_bias,
                BLOCK_SIZE_Q=block_size_q,
                BLOCK_SIZE_KV=_oss_bkv,
                USE_EXP2=use_exp2,
                num_warps=4,
                num_stages=1,
                **_extra_even,
                **extra_kern_args,
            )

            ctx.save_for_backward(Q, K, V, res_mask, pair_bias, O, M)
            ctx.grid = grid
            ctx.softmax_scale = softmax_scale
            ctx.DIM = DIM
            ctx.has_pair_bias = has_pair_bias
            ctx.use_exp2 = use_exp2

            O = O.transpose(-2, -3).contiguous()

            return O

        @staticmethod
        def backward(ctx, dO):
            """Run the backward pass of the attention mechanism."""

            Q, K, V, res_mask, pair_bias, O, M = ctx.saved_tensors
            # preprocess writes the internal-layout dO, folding away a transpose.
            dO_bllhs = dO.contiguous()
            dO = torch.empty_like(Q)  # internal layout, filled by preprocess

            assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
            dQ = torch.zeros_like(Q)
            # dK/dV written directly in BLLHS layout by the kernel (skips a transpose).
            _b, _n, _h, _l, _d = Q.shape
            dK = torch.empty((_b, _n, _l, _h, _d), dtype=K.dtype, device=K.device)
            dV = torch.empty((_b, _n, _l, _h, _d), dtype=V.dtype, device=V.device)

            BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM = dQ.shape

            if ctx.has_pair_bias:
                d_pair_bias = torch.empty(
                    (BATCH_SIZE, 1, HEAD, SEQ_LEN, SEQ_LEN),
                    device=pair_bias.device,
                    dtype=torch.float32,
                ).zero_()
            else:
                d_pair_bias = torch.empty(
                    (1, 1, 1, 1, 1), device=pair_bias.device, dtype=torch.float32
                )

            BLOCK_DIM = max(triton.next_power_of_2(DIM), 32)

            D = torch.empty_like(M)  # Shape: (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN)

            preprocess_grid = lambda args: (  # noqa: E731
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
                BATCH_SIZE * N_SEQ * HEAD,
                1,
            )
            _attn_bwd_preprocess[preprocess_grid](
                O=O,
                dO=dO_bllhs,
                D=D,
                SEQ_LEN=SEQ_LEN,
                DIM=DIM,
                BLOCK_DIM=BLOCK_DIM,
                BLOCK_SIZE_Q=16,
                dO_out=dO,
                HEAD=HEAD,
                READ_DO_BLLHS=True,
                stride_batch=O.stride(0),
                stride_msa=O.stride(1),
                stride_head=O.stride(2),
                stride_seq=O.stride(3),
                N_SEQ=N_SEQ,
                num_warps=4,
                num_stages=2,
            )

            bwd_dk_dv_grid = lambda args: (  # noqa: E731
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_KV"]),
                BATCH_SIZE * N_SEQ * HEAD,
                1,
            )
            _attn_bwd_dk_dv[bwd_dk_dv_grid](
                Q=Q,
                K=K,
                V=V,
                res_mask=res_mask,
                pair_bias=pair_bias,
                softmax_scale=ctx.softmax_scale,
                dO=dO,
                dQ=dQ,
                dK=dK,
                dV=dV,
                M=M,
                D=D,
                stride_batch=Q.stride(0),
                stride_msa=Q.stride(1),
                stride_head=Q.stride(2),
                stride_seq=Q.stride(3),
                stride_pair_bias_batch=pair_bias.stride(0),
                stride_pair_bias_head=pair_bias.stride(2),
                stride_pair_bias_seq1=pair_bias.stride(3),
                stride_pair_bias_seq2=pair_bias.stride(4),
                stride_mask_batch=res_mask.stride(0),
                stride_mask_msa=res_mask.stride(1),
                stride_mask_seq=res_mask.stride(4),
                HEAD=HEAD,
                N_SEQ=N_SEQ,
                SEQ_LEN=SEQ_LEN,
                BLOCK_DIM=BLOCK_DIM,
                DIM=ctx.DIM,
                BLOCK_SIZE_Q=64,
                BLOCK_SIZE_KV=128,
                num_warps=4,
                num_stages=1,
                # MFMA tile 16 + waves_per_eu=2 (HIP only) for dk_dv.
                **({"waves_per_eu": 2, "matrix_instr_nonkdim": 16} if is_hip() else {}),
                WRITE_BLLHS=True,
                USE_EXP2=ctx.use_exp2,
            )

            dQ_acc = torch.zeros_like(Q, dtype=torch.float32)
            # Shape/dtype-adaptive dbias tile.
            if Q.dtype == torch.float32:
                _BQ, _BKV = 128, 64
            elif HEAD >= 8 or SEQ_LEN >= 512:
                _BQ, _BKV = 128, 128
            else:
                _BQ, _BKV = 128, 64
            bwd_dbias_dq_grid = lambda args: (  # noqa: E731
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_KV"]),
                BATCH_SIZE * HEAD,
            )
            _attn_bwd_dbias_dq[bwd_dbias_dq_grid](
                Q=Q,
                K=K,
                V=V,
                res_mask=res_mask,
                pair_bias=pair_bias,
                softmax_scale=ctx.softmax_scale,
                dO=dO,
                dQ=dQ_acc,
                d_pair_bias=d_pair_bias,
                M=M,
                D=D,
                stride_batch=Q.stride(0),
                stride_head=Q.stride(2),
                stride_msa=Q.stride(1),
                stride_seq=Q.stride(3),
                stride_pair_bias_batch=pair_bias.stride(0),
                stride_pair_bias_head=pair_bias.stride(2),
                stride_pair_bias_seq1=pair_bias.stride(3),
                stride_pair_bias_seq2=pair_bias.stride(4),
                stride_mask_batch=res_mask.stride(0),
                stride_mask_msa=res_mask.stride(1),
                stride_mask_seq=res_mask.stride(4),
                stride_d_pair_bias_batch=d_pair_bias.stride(0),
                stride_d_pair_bias_head=d_pair_bias.stride(2),
                stride_d_pair_bias_seq1=d_pair_bias.stride(3),
                stride_d_pair_bias_seq2=d_pair_bias.stride(4),
                HEAD=HEAD,
                N_SEQ=N_SEQ,
                SEQ_LEN=SEQ_LEN,
                BLOCK_DIM=BLOCK_DIM,
                DIM=ctx.DIM,
                BLOCK_SIZE_Q=_BQ,
                BLOCK_SIZE_KV=_BKV,
                PIPE_STAGES=2,
                num_warps=8,
                num_stages=1,
                **({"waves_per_eu": 2, "matrix_instr_nonkdim": 16} if is_hip() else {}),
                HAS_PAIR_BIAS=ctx.has_pair_bias,
                USE_EXP2=ctx.use_exp2,
            )

            dQ = dQ_acc.to(Q.dtype)
            dQ = dQ.transpose(-2, -3).contiguous()
            d_pb = d_pair_bias.to(dO.dtype) if ctx.has_pair_bias else None
            # grads: Q,K,V, res_mask, pair_bias, has_pair_bias, dynamic, softmax_scale
            return dQ, dK, dV, None, d_pb, None, None, None

    TritonEvoformer = EvoformerAttention.apply

    def TritonEvoformerDynamic(Q, K, V, res_mask, pair_bias, has_pair_bias=True):
        """Variable-shape entry point: one Triton compile across all sequence
        lengths (opt-in). Same output as TritonEvoformer; trades a small per-call
        cost for no per-size recompile. Best for variable-size / cold-cache."""
        return EvoformerAttention.apply(
            Q, K, V, res_mask, pair_bias, has_pair_bias, True
        )
