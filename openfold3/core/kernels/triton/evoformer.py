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

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# Sentinel: replaced with EvoformerAttention.apply when Triton is available.
TritonEvoformer = None

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

            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block = QK_block - m_ij[:, None]

            P_block = tl.math.exp(QK_block)
            l_ij = tl.sum(P_block, 1)

            alpha = tl.math.exp(m_i - m_ij)
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
        )

        m_i += tl.math.log(l_i)
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
    ):
        """Run the preprocessing step of the backward pass of the attention
        mechanism."""
        block_index_q = tl.program_id(0)
        offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
        index_batch_msa_head = tl.program_id(1)
        offs_dim = tl.arange(0, BLOCK_DIM)

        # Cast to int64 to avoid int32 overflow for large sequences
        bwd_offset = index_batch_msa_head.to(tl.int64) * SEQ_LEN * DIM

        # Load a single block of BLOCK_SIZE_Q rows of O
        O_block = tl.load(
            O + bwd_offset + offs_q[:, None] * DIM + offs_dim[None, :],
            mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
            other=0.0,
        )
        # Load a single block of BLOCK_SIZE_Q rows of dO
        dO_block = tl.load(
            dO + bwd_offset + offs_q[:, None] * DIM + offs_dim[None, :],
            mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
            other=0.0,
        ).to(tl.float32)
        # Compute the D block
        D_block = tl.sum(dO_block * O_block, axis=1)  # Shape: (BLOCK_SIZE_Q,)
        # Store the D block
        D_block_ptrs = D + index_batch_msa_head.to(tl.int64) * SEQ_LEN + offs_q
        tl.store(D_block_ptrs, D_block, mask=offs_q < SEQ_LEN)

    @triton.heuristics(
        {
            "EVEN_Q": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_Q"] == 0,
            "EVEN_KV": lambda args: args["SEQ_LEN"] % args["BLOCK_SIZE_KV"] == 0,
            "EVEN_DIM": lambda args: args["DIM"] == args["BLOCK_DIM"],
        }
    )
    @triton.jit
    def _attn_bwd_dq(
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
        EVEN_Q: tl.constexpr,
        EVEN_KV: tl.constexpr,
        EVEN_DIM: tl.constexpr,
        BLOCK_SIZE_Q: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
    ):
        """Run the backward pass of the attention mechanism."""
        index_batch_msa_head = tl.program_id(1)
        index_batch_msa = index_batch_msa_head // HEAD
        index_head = index_batch_msa_head % HEAD
        index_batch = index_batch_msa // N_SEQ
        index_msa = index_batch_msa % N_SEQ

        # Cast indices to int64 to avoid int32 overflow
        offset_batch_head_msa = (
            index_batch.to(tl.int64) * stride_batch
            + index_head.to(tl.int64) * stride_head
            + index_msa.to(tl.int64) * stride_msa
        )
        offset_batch_head_msa_seq = index_batch_msa_head.to(tl.int64) * SEQ_LEN

        Q += offset_batch_head_msa
        K += offset_batch_head_msa
        V += offset_batch_head_msa
        dO += offset_batch_head_msa
        dQ += offset_batch_head_msa
        dK += offset_batch_head_msa
        dV += offset_batch_head_msa

        M += offset_batch_head_msa_seq
        D += offset_batch_head_msa_seq

        offs_dim = tl.arange(0, BLOCK_DIM)

        index_block_kv = tl.program_id(0)
        offs_q = index_block_kv * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

        dQ_block = tl.zeros([BLOCK_SIZE_Q, BLOCK_DIM], dtype=tl.float32)

        if EVEN_Q & EVEN_KV:
            M_block = tl.load(M + offs_q)
            Di = tl.load(D + offs_q)
            if EVEN_DIM:
                Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :])
                dO_block = tl.load(
                    dO + offs_q[:, None] * stride_seq + offs_dim[None, :]
                )
            else:
                Q_block = tl.load(
                    Q + offs_q[:, None] * stride_seq + offs_dim[None, :],
                    mask=offs_dim[None, :] < DIM,
                    other=0.0,
                )
                dO_block = tl.load(
                    dO + offs_q[:, None] * stride_seq + offs_dim[None, :],
                    mask=offs_dim[None, :] < DIM,
                    other=0.0,
                )
        else:
            M_block = tl.load(M + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
            Di = tl.load(D + offs_q, mask=offs_q < SEQ_LEN, other=0.0)
            if EVEN_DIM:
                Q_block = tl.load(
                    Q + offs_q[:, None] * stride_seq + offs_dim[None, :],
                    mask=offs_q[:, None] < SEQ_LEN,
                    other=0.0,
                )
                dO_block = tl.load(
                    dO + offs_q[:, None] * stride_seq + offs_dim[None, :],
                    mask=offs_q[:, None] < SEQ_LEN,
                    other=0.0,
                )
            else:
                Q_block = tl.load(
                    Q + offs_q[:, None] * stride_seq + offs_dim[None, :],
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                    other=0.0,
                )
                dO_block = tl.load(
                    dO + offs_q[:, None] * stride_seq + offs_dim[None, :],
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                    other=0.0,
                )

        M_block = M_block[:, None]

        offs_kv = tl.arange(0, BLOCK_SIZE_KV)
        pair_bias_block_ptr = (
            pair_bias
            + index_batch.to(tl.int64) * stride_pair_bias_batch
            + index_head.to(tl.int64) * stride_pair_bias_head
            + offs_q[:, None] * stride_pair_bias_seq1
            + offs_kv[None, :] * stride_pair_bias_seq2
        )

        d_pair_bias_block_ptr = (
            d_pair_bias
            + index_batch.to(tl.int64) * stride_d_pair_bias_batch
            + index_head.to(tl.int64) * stride_d_pair_bias_head
            + (offs_q[:, None] * stride_d_pair_bias_seq1)
            + (offs_kv[None, :] * stride_d_pair_bias_seq2)
        )
        res_mask_block_ptr = (
            res_mask
            + index_batch.to(tl.int64) * stride_mask_batch
            + index_msa.to(tl.int64) * stride_mask_msa
            + (offs_kv[None, :] * stride_mask_seq)
        )

        kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None]
        vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None]

        Q_block = Q_block * tl.full((1,), softmax_scale, dtype=Q_block.dtype)

        curr_kv = 0
        num_steps = (SEQ_LEN + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV

        for blk_idx in range(num_steps):
            if EVEN_Q & EVEN_KV:
                pair_bias_block = tl.load(pair_bias_block_ptr)
                res_mask_block = tl.load(res_mask_block_ptr).broadcast_to(
                    (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
                )
                if EVEN_DIM:
                    K_T_block = tl.load(kT_ptrs)
                    V_T_block = tl.load(vT_ptrs)
                else:
                    K_T_block = tl.load(
                        kT_ptrs, mask=offs_dim[:, None] < DIM, other=0.0
                    )
                    V_T_block = tl.load(
                        vT_ptrs, mask=offs_dim[:, None] < DIM, other=0.0
                    )
            else:
                pair_bias_block = tl.load(
                    pair_bias_block_ptr,
                    mask=(offs_q[:, None] < SEQ_LEN)
                    & ((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN),
                    other=float("-inf"),
                )
                res_mask_block = tl.load(
                    res_mask_block_ptr,
                    mask=(blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN,
                    other=float("-inf"),
                ).broadcast_to((BLOCK_SIZE_Q, BLOCK_SIZE_KV))
                if EVEN_DIM:
                    K_T_block = tl.load(
                        kT_ptrs,
                        mask=(blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN,
                        other=0.0,
                    )
                    V_T_block = tl.load(
                        vT_ptrs,
                        mask=(blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN,
                        other=0.0,
                    )
                else:
                    K_T_block = tl.load(
                        kT_ptrs,
                        mask=((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN)
                        & (offs_dim[:, None] < DIM),
                        other=0.0,
                    )
                    V_T_block = tl.load(
                        vT_ptrs,
                        mask=((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN)
                        & (offs_dim[:, None] < DIM),
                        other=0.0,
                    )

            QK_block = tl.dot(Q_block, K_T_block) + pair_bias_block + res_mask_block

            if not EVEN_KV:
                QK_block += tl.where(
                    (blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN,
                    0,
                    float("-inf"),
                )

            P_block = tl.math.exp(QK_block - M_block)

            dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
            dS_block = P_block * (dP_block - Di[:, None])

            # Update d_pair_bias atomic add with float32 precision
            tl.atomic_add(
                d_pair_bias_block_ptr,
                dS_block,
                mask=(offs_q[:, None] < SEQ_LEN)
                & ((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN),
            )
            dS_block = dS_block.to(K_T_block.dtype)

            dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

            curr_kv += BLOCK_SIZE_KV
            kT_ptrs += BLOCK_SIZE_KV * stride_seq
            vT_ptrs += BLOCK_SIZE_KV * stride_seq
            pair_bias_block_ptr += BLOCK_SIZE_KV * stride_pair_bias_seq2
            d_pair_bias_block_ptr += BLOCK_SIZE_KV * stride_d_pair_bias_seq2
            res_mask_block_ptr += BLOCK_SIZE_KV * stride_mask_seq

        dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :]
        if EVEN_Q & EVEN_KV:
            if EVEN_DIM:
                tl.store(dQ_block_ptrs, dQ_block)
            else:
                tl.store(dQ_block_ptrs, dQ_block, mask=offs_dim[None, :] < DIM)
        else:
            if EVEN_DIM:
                tl.store(dQ_block_ptrs, dQ_block, mask=offs_q[:, None] < SEQ_LEN)
            else:
                tl.store(
                    dQ_block_ptrs,
                    dQ_block,
                    mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
                )

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

            P_T_block = tl.math.exp(QK_T_block - m[None, :])

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

        dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :]
        dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :]

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
        def forward(ctx, Q, K, V, res_mask, pair_bias, has_pair_bias=True):
            """Run the forward pass of the attention mechanism.

            has_pair_bias: set False when pair_bias is all-zeros (MSA column attention).
            This eliminates all pair_bias HBM loads in the forward kernel.
            """
            # Q, K, V: [Batch, N_seq, N_res, Head, Dim]
            # res_mask: [Batch, N_seq, 1, 1, N_res]
            # pair_bias: [Batch, 1, Head, N_res, N_res]

            DIM_Q, DIM_K, DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
            assert DIM_Q == DIM_K and DIM_K == DIM_V

            Q = Q.transpose(
                -2, -3
            ).contiguous()  # (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM)
            K = K.transpose(
                -2, -3
            ).contiguous()  # (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM)
            V = V.transpose(
                -2, -3
            ).contiguous()  # (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM)

            BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM = Q.shape
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

            block_size_q = 64

            grid = lambda args: (  # noqa: E731
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
                BATCH_SIZE * N_SEQ * HEAD,
                1,
            )

            # M is the logsumexp for the backward pass, one for each query
            M = torch.empty(
                (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN), device=Q.device, dtype=torch.float32
            )

            _attn_fwd[grid](
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
                BLOCK_SIZE_KV=16,
                num_warps=4,
                num_stages=1,
                **extra_kern_args,
            )

            ctx.save_for_backward(Q, K, V, res_mask, pair_bias, O, M)
            ctx.grid = grid
            ctx.softmax_scale = softmax_scale
            ctx.DIM = DIM

            O = O.transpose(-2, -3).contiguous()

            return O

        @staticmethod
        def backward(ctx, dO):
            """Run the backward pass of the attention mechanism."""

            Q, K, V, res_mask, pair_bias, O, M = ctx.saved_tensors
            dO = dO.transpose(
                -2, -3
            ).contiguous()  # (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM)

            assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
            dQ = torch.empty_like(Q)
            dK = torch.empty_like(K)
            dV = torch.empty_like(V)

            BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM = dQ.shape

            d_pair_bias = torch.empty(
                (BATCH_SIZE, 1, HEAD, SEQ_LEN, SEQ_LEN),
                device=pair_bias.device,
                dtype=torch.float32,
            ).zero_()

            BLOCK_DIM = max(triton.next_power_of_2(DIM), 32)

            D = torch.empty_like(M)  # Shape: (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN)

            preprocess_grid = lambda args: (  # noqa: E731
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
                BATCH_SIZE * N_SEQ * HEAD,
                1,
            )
            _attn_bwd_preprocess[preprocess_grid](
                O=O,
                dO=dO,
                D=D,
                SEQ_LEN=SEQ_LEN,
                DIM=DIM,
                BLOCK_DIM=BLOCK_DIM,
                BLOCK_SIZE_Q=16,
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
                BLOCK_SIZE_KV=64,
                num_warps=4,
                num_stages=1,
            )

            bwd_dq_grid = lambda args: (  # noqa: E731
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
                BATCH_SIZE * N_SEQ * HEAD,
                1,
            )
            _attn_bwd_dq[bwd_dq_grid](
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
                d_pair_bias=d_pair_bias,
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
                stride_d_pair_bias_batch=d_pair_bias.stride(0),
                stride_d_pair_bias_head=d_pair_bias.stride(2),
                stride_d_pair_bias_seq1=d_pair_bias.stride(3),
                stride_d_pair_bias_seq2=d_pair_bias.stride(4),
                HEAD=HEAD,
                N_SEQ=N_SEQ,
                SEQ_LEN=SEQ_LEN,
                BLOCK_DIM=BLOCK_DIM,
                DIM=ctx.DIM,
                BLOCK_SIZE_Q=16,
                BLOCK_SIZE_KV=16,
                num_warps=4,
                num_stages=1,
            )

            dQ = dQ.transpose(-2, -3).contiguous()
            dK = dK.transpose(-2, -3).contiguous()
            dV = dV.transpose(-2, -3).contiguous()

            return dQ, dK, dV, None, d_pair_bias.to(dO.dtype), None

    TritonEvoformer = EvoformerAttention.apply
