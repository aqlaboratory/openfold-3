# Copyright 2026 UIUC SSAIL
# Part of MegaFold project
#
# Licensed under the MIT License. See LICENSE file in the project root
# for full license text.
#
# If you use this code, please cite:
#
#   La, H., Gupta, A., Morehead, A., Cheng, J., & Zhang, M. (2025).
#   MegaFold: System-Level Optimizations for Accelerating Protein
#   Structure Prediction Models. arXiv:2506.20686
#   https://arxiv.org/abs/2506.20686
#
# BibTeX:
#   @misc{la2025megafoldsystemleveloptimizationsaccelerating,
#       title={MegaFold: System-Level Optimizations for Accelerating
#              Protein Structure Prediction Models},
#       author={Hoa La and Ahan Gupta and Alex Morehead
#               and Jianlin Cheng and Minjia Zhang},
#       year={2025},
#       eprint={2506.20686},
#       archivePrefix={arXiv},
#       primaryClass={q-bio.BM},
#       url={https://arxiv.org/abs/2506.20686},
#   }

# Instructions: autotune to get the best config for the right hardware and input sizes, then, manually write in the best block-sizes config, then comment out autotune (so that no autotuning happens in training)

import torch
import triton
import triton.language as tl


def is_hip():
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
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    offs_d: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    lo, hi = 0, SEQ_LEN
    Q_block = Q_block * tl.full((1,), softmax_scale, dtype=Q_block.dtype)

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # -- compute qk ----
        if EVEN_Q & EVEN_KV:
            pair_bias_block = tl.load(pair_bias_block_ptr)
            res_mask_block = tl.load(res_mask_block_ptr).broadcast_to(
                (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
            )  # (1, BLOCK_SIZE_KV) -> (BLOCK_SIZE_Q, BLOCK_SIZE_KV)
            if EVEN_DIM:
                K_block = tl.load(K_block_ptr)
                V_block = tl.load(V_block_ptr)
            else:
                K_block = tl.load(K_block_ptr, mask=offs_d[:, None] < DIM, other=0.0)
                V_block = tl.load(V_block_ptr, mask=offs_d[None, :] < DIM, other=0.0)
        else:
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
                    K_block_ptr, mask=(start_kv + offs_kv)[None, :] < SEQ_LEN, other=0.0
                )
                V_block = tl.load(
                    V_block_ptr, mask=(start_kv + offs_kv)[:, None] < SEQ_LEN, other=0.0
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

        QK_block = tl.dot(Q_block, K_block) + pair_bias_block + res_mask_block

        # Need to mask out otherwise the softmax is wrong
        if not EVEN_KV:
            QK_block += tl.where(
                (start_kv + offs_kv)[None, :] < SEQ_LEN, 0, float("-inf")
            )

        m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
        QK_block = QK_block - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        P_block = P_block.to(V_block.dtype)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr += BLOCK_SIZE_KV * stride_V_seq
        K_block_ptr += BLOCK_SIZE_KV * stride_K_seq
        pair_bias_block_ptr += BLOCK_SIZE_KV * stride_pair_bias_seq2
        res_mask_block_ptr += BLOCK_SIZE_KV * stride_mask_seq

    return O_block, l_i, m_i


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [16, 32, 64]
        for BLOCK_SIZE_KV in [16, 32, 64]
        for num_stages in [1, 3, 4, 7]
        for num_warps in [4, 8]
    ],
    key=["SEQ_LEN", "DIM"],
)
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
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    block_index_q = tl.program_id(0).to(tl.int64)

    index_batch_msa_head = tl.program_id(1).to(tl.int64)
    index_batch_msa = index_batch_msa_head // HEAD
    index_head = index_batch_msa_head % HEAD
    index_batch = index_batch_msa // N_SEQ
    index_msa = index_batch_msa % N_SEQ

    qvk_offset = (
        index_batch * stride_Q_batch
        + index_msa * stride_Q_msa
        + index_head * stride_Q_head
    )
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    offs_d = tl.arange(0, BLOCK_DIM)

    Q_block_ptr = Q + qvk_offset + (offs_q[:, None] * stride_Q_seq + offs_d[None, :])
    V_block_ptr = V + qvk_offset + (offs_kv[:, None] * stride_V_seq + offs_d[None, :])
    K_block_ptr = K + qvk_offset + (offs_kv[None, :] * stride_K_seq + offs_d[:, None])
    pair_bias_block_ptr = (
        pair_bias
        + index_batch * stride_pair_bias_batch
        + index_head * stride_pair_bias_head
        + (
            offs_q[:, None] * stride_pair_bias_seq1
            + offs_kv[None, :] * stride_pair_bias_seq2
        )
    )
    O_block_ptr = O + qvk_offset + (offs_q[:, None] * stride_O_seq + offs_d[None, :])

    res_mask_block_ptr = (
        res_mask
        + (index_batch * stride_mask_batch + index_msa * stride_mask_msa)
        + (offs_kv[None, :] * stride_mask_seq)
    )

    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i: the running sum. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, BLOCK_DIM], dtype=tl.float32)

    # load the blocks of Q: it will stay in SRAM throughout
    if EVEN_Q & EVEN_KV:
        if EVEN_DIM:
            Q_block = tl.load(Q_block_ptr)
        else:
            Q_block = tl.load(Q_block_ptr, mask=offs_d[None, :] < DIM, other=0.0)
    else:
        if EVEN_DIM:
            Q_block = tl.load(Q_block_ptr, mask=offs_q[:, None] < SEQ_LEN, other=0.0)
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


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_Q": BLOCK_SIZE_Q})
        for BLOCK_SIZE_Q in [16, 32, 64, 128]
    ],
    key=["SEQ_LEN"],
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
    block_index_q = tl.program_id(0).to(tl.int64)
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    index_batch_msa_head = tl.program_id(1).to(tl.int64)
    offs_dim = tl.arange(0, BLOCK_DIM)

    # Load a single block of BLOCK_SIZE_Q rows of O
    O_block = tl.load(
        O
        + index_batch_msa_head * SEQ_LEN * DIM
        + offs_q[:, None] * DIM
        + offs_dim[None, :],
        mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
        other=0.0,
    )
    # Load a single block of BLOCK_SIZE_Q rows of dO
    dO_block = tl.load(
        dO
        + index_batch_msa_head * SEQ_LEN * DIM
        + offs_q[:, None] * DIM
        + offs_dim[None, :],
        mask=(offs_q[:, None] < SEQ_LEN) & (offs_dim[None, :] < DIM),
        other=0.0,
    ).to(tl.float32)
    # Compute the D block
    D_block = tl.sum(dO_block * O_block, axis=1)  # Shape: (BLOCK_SIZE_Q,)
    # Store the D block
    D_block_ptrs = D + index_batch_msa_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block, mask=offs_q < SEQ_LEN)


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [16, 32, 64]
        for BLOCK_SIZE_KV in [16, 32, 64]
        for num_stages in [1, 3, 4, 7]
        for num_warps in [4, 8]
    ],
    key=["SEQ_LEN", "DIM"],
)
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
    index_batch_msa_head = tl.program_id(1).to(tl.int64)
    index_batch_msa = index_batch_msa_head // HEAD
    index_head = index_batch_msa_head % HEAD
    index_batch = index_batch_msa // N_SEQ
    index_msa = index_batch_msa % N_SEQ

    offset_batch_head_msa = (
        index_batch * stride_batch + index_head * stride_head + index_msa * stride_msa
    ).to(tl.int64)
    offset_batch_head_msa_seq = (index_batch_msa_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head_msa
    K += offset_batch_head_msa
    V += offset_batch_head_msa
    dO += offset_batch_head_msa
    dQ += offset_batch_head_msa
    dK += offset_batch_head_msa
    dV += offset_batch_head_msa

    # Make sure the pointers are in the right place w.r.t batch, head and sequence
    M += offset_batch_head_msa_seq
    D += offset_batch_head_msa_seq

    # load scales
    offs_dim = tl.arange(0, BLOCK_DIM)

    index_block_kv = tl.program_id(0).to(tl.int64)
    offs_q = index_block_kv * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)

    dQ_block = tl.zeros([BLOCK_SIZE_Q, BLOCK_DIM], dtype=tl.float32)

    if EVEN_Q & EVEN_KV:
        M_block = tl.load(M + offs_q)
        Di = tl.load(D + offs_q)
        if EVEN_DIM:
            Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :])
            dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :])
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
        + (index_batch * stride_pair_bias_batch + index_head * stride_pair_bias_head)
        + offs_q[:, None] * stride_pair_bias_seq1
        + offs_kv[None, :] * stride_pair_bias_seq2
    )

    d_pair_bias_block_ptr = (
        d_pair_bias
        + (
            index_batch * stride_d_pair_bias_batch
            + index_head * stride_d_pair_bias_head
        ).to(tl.int64)
        + (offs_q[:, None] * stride_d_pair_bias_seq1)
        + (offs_kv[None, :] * stride_d_pair_bias_seq2)
    )
    res_mask_block_ptr = (
        res_mask
        + (index_batch * stride_mask_batch + index_msa * stride_mask_msa)
        + (offs_kv[None, :] * stride_mask_seq)
    )

    # We access the K and V as transposed blocks
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
                K_T_block = tl.load(kT_ptrs, mask=offs_dim[:, None] < DIM, other=0.0)
                V_T_block = tl.load(vT_ptrs, mask=offs_dim[:, None] < DIM, other=0.0)
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
        # Need to mask out otherwise the softmax is wrong
        if not EVEN_KV:
            QK_block += tl.where(
                (blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN, 0, float("-inf")
            )

        P_block = tl.math.exp(QK_block - M_block)

        # Compute dP and dS.
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])

        # Update d_pair_bias atomic add with float32 precision
        tl.atomic_add(
            d_pair_bias_block_ptr,
            dS_block,
            mask=(offs_q[:, None] < SEQ_LEN)
            & ((blk_idx * BLOCK_SIZE_KV + offs_kv)[None, :] < SEQ_LEN),
            sem="relaxed",
        )
        dS_block = dS_block.to(K_T_block.dtype)

        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))

        # Increment pointers.
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


@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [16, 32, 64]
        for BLOCK_SIZE_KV in [16, 32, 64]
        for num_stages in [1, 3, 4, 7]
        for num_warps in [4, 8]
    ],
    key=["SEQ_LEN", "DIM"],
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
    index_batch_msa_head = tl.program_id(1).to(tl.int64)
    index_batch_msa = index_batch_msa_head // HEAD
    index_head = index_batch_msa_head % HEAD
    index_batch = index_batch_msa // N_SEQ
    index_msa = index_batch_msa % N_SEQ

    offset_batch_msa_head = (
        index_batch * stride_batch + index_msa * stride_msa + index_head * stride_head
    ).to(tl.int64)
    offset_batch_msa_head_seq = (index_batch_msa_head * SEQ_LEN).to(tl.int64)

    # Make sure the pointers are in the right place w.r.t batch and head
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

    index_block_kv = tl.program_id(0).to(tl.int64)
    offs_kv = index_block_kv * BLOCK_SIZE_KV + tl.arange(0, BLOCK_SIZE_KV)
    offs_q = tl.arange(0, BLOCK_SIZE_Q)

    dK_block = tl.zeros([BLOCK_SIZE_KV, BLOCK_DIM], dtype=tl.float32)
    dV_block = tl.zeros([BLOCK_SIZE_KV, BLOCK_DIM], dtype=tl.float32)

    res_mask_block_ptr = (
        res_mask
        + (index_batch * stride_mask_batch + index_msa * stride_mask_msa)
        + offs_kv[None, :] * stride_mask_seq
    )

    # load K and V: they stay in SRAM throughout the inner loop.
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
                res_mask_block_ptr, mask=offs_kv[None, :] < SEQ_LEN, other=float("-inf")
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

    # pointer to transposed pair bias
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

    for blk_idx in range(num_steps):
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
                qT_block = tl.load(qT_ptrs, mask=offs_q[None, :] < SEQ_LEN, other=0.0)
                dO_block = tl.load(dO_ptrs, mask=offs_q[:, None] < SEQ_LEN, other=0.0)
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

        # This gives us (QK^T)^T = (K^T)^T(Q^T) = K(Q^T) = P^T
        QK_T_block = (
            tl.dot(K_block, qT_block) + pair_bias_T_block + res_mask_T_block
        )  # [(BLOCK_SIZE_KV, BLOCK_SIZE_Q)]

        # Need to mask out otherwise the softmax is wrong
        if not (EVEN_Q & EVEN_KV):
            QK_T_block += tl.where(
                (offs_kv[:, None] < SEQ_LEN) & (offs_q[None, :] < SEQ_LEN),
                0,
                float("-inf"),
            )

        # We apply the softmax by using the logsumexp trick
        P_T_block = tl.math.exp(
            QK_T_block - m[None, :]
        )  # [(BLOCK_SIZE_KV, BLOCK_SIZE_Q)]

        # According to the formula: dV_new = dV_old + P^T x dO, where x is the matrix multiplication
        dV_block += tl.dot(
            P_T_block.to(K_block.dtype), dO_block
        )  # [(BLOCK_SIZE_KV, DIM)]

        # dP = dO x V^T, so dP^T = V x dO^T
        # Where x is the matrix multiplication
        dpT_block = tl.dot(
            V_block, tl.trans(dO_block)
        ).to(
            tl.float32
        )  # [(BLOCK_SIZE_KV, DIM) x (DIM, BLOCK_SIZE_Q)] -> [(BLOCK_SIZE_KV, BLOCK_SIZE_Q)]

        # We know that dS = P * (dP - Delta), so dS^T = P^T * (dP^T - Delta^T)
        # [(1, BLOCK_SIZE_Q)]
        dS_T_block = P_T_block * (
            dpT_block - Di[None, :]
        )  # [(BLOCK_SIZE_KV, BLOCK_SIZE_Q)]
        dS_T_block = dS_T_block.to(K_block.dtype)

        # According to the formula on the paper: dK_new = dK_old + dS^T x Q
        dK_block += (
            softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        )  # [(BLOCK_SIZE_KV, BLOCK_SIZE_Q) x (BLOCK_SIZE_Q, DIM) = (BLOCK_SIZE_KV, DIM)]

        # Increment pointers.
        curr_q += BLOCK_SIZE_Q
        qT_ptrs += BLOCK_SIZE_Q * stride_seq
        dO_ptrs += BLOCK_SIZE_Q * stride_seq
        pair_bias_T_block_ptr += BLOCK_SIZE_Q * stride_pair_bias_seq1

    # Write back dV & dK
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
    def forward(ctx, Q, K, V, res_mask, pair_bias):
        # Q, K, V: [Batch, N_seq, N_res, Head, Dim]
        # res_mask: [Batch, N_seq, 1, 1, N_res]
        # pair_bias: [Batch, 1, Head, N_res, N_res]

        DIM_Q, DIM_K, DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
        assert DIM_Q == DIM_K and DIM_K == DIM_V

        # Note: transpose is different from view
        # Note: use contiguous otherwise tiling and block ptrs are mixed up -- very important
        Q = Q.transpose(-2, -3).contiguous()  # (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM)
        K = K.transpose(-2, -3).contiguous()  # (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM)
        V = V.transpose(-2, -3).contiguous()  # (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM)

        BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM = Q.shape  # change here as well
        softmax_scale = DIM**-0.5
        BLOCK_DIM = max(triton.next_power_of_2(DIM), 16)

        O = torch.empty_like(Q)

        # Tuning for AMD target
        extra_kern_args = {}
        if is_hip():
            waves_per_eu = 3 if DIM <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        grid = lambda args: (
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
            **extra_kern_args,
        )

        ctx.save_for_backward(
            Q, K, V, res_mask, pair_bias, O, M
        )  # all tensors are in order: (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.DIM = DIM

        # change back to original shape
        O = O.transpose(
            -2, -3
        ).contiguous()  # (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM) -> (BATCH_SIZE, N_SEQ, SEQ_LEN, HEAD, DIM)

        return O

    @staticmethod
    def backward(ctx, dO):
        # dO: BATCH, N_SEQ, SEQ_LEN, HEAD, DIM

        Q, K, V, res_mask, pair_bias, O, M = ctx.saved_tensors
        dO = dO.transpose(
            -2, -3
        ).contiguous()  # (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN, DIM)

        # now, all are in "workspace shape"
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

        BLOCK_DIM = max(triton.next_power_of_2(DIM), 16)

        D = torch.empty_like(M)  # Shape: (BATCH_SIZE, N_SEQ, HEAD, SEQ_LEN)

        preprocess_grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * N_SEQ * HEAD,
            1,
        )
        # Compute all the elements Di
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            # BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            DIM=DIM,
            BLOCK_DIM=BLOCK_DIM,
        )

        bwd_dk_dv_grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_KV"]),
            BATCH_SIZE * N_SEQ * HEAD,
            1,
        )
        # Fix KV and iterate through all the Q blocks
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
            # BLOCK_SIZE_Q=BLOCK_SIZE_MICRO,
            # BLOCK_SIZE_KV=BLOCK_SIZE_MACRO,
            # num_warps=NUM_WARPS,
            # num_stages=NUM_STAGES,
        )

        bwd_dq_grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
            BATCH_SIZE * N_SEQ * HEAD,
            1,
        )
        # Fix Q and iterate through all the KV block
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
            # BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            # BLOCK_SIZE_KV=BLOCK_SIZE_MICRO,
            # num_warps=NUM_WARPS,
            # num_stages=NUM_STAGES,
        )

        # change shape to original
        dQ = dQ.transpose(-2, -3).contiguous()
        dK = dK.transpose(-2, -3).contiguous()
        dV = dV.transpose(-2, -3).contiguous()

        return dQ, dK, dV, None, d_pair_bias.to(dO.dtype)


TritonEvoformer = EvoformerAttention.apply


if __name__ == "__main__":
    # Run autotune for inputs
    N = 256
    BATCH, H, HEAD_DIM, N_SEQ, N_CTX = 4, 16, 64, 1, N  # attentionpairbias
    # BATCH, H, HEAD_DIM, N_SEQ, N_CTX = 1, 4, 32, N, N # triangle attention inputs
    dtype = torch.bfloat16
    device = "cuda"
    q = torch.randn(
        (BATCH, N_SEQ, N_CTX, H, HEAD_DIM),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    k = torch.randn(
        (BATCH, N_SEQ, N_CTX, H, HEAD_DIM),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    v = torch.randn(
        (BATCH, N_SEQ, N_CTX, H, HEAD_DIM),
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    res_mask = torch.randint(
        0, 2, (BATCH, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device=device
    )
    pair_bias = torch.randn(
        (BATCH, 1, H, N_CTX, N_CTX),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    o = TritonEvoformer(q, k, v, res_mask, pair_bias)
    do = torch.randn_like(o)
    o.backward(do, retain_graph=True)
