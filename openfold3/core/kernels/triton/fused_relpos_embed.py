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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: fused Triton gather-add for the
# relative-position embedding tables used by the input embedder.

"""Fused relpos gather-add kernel.

Fuses the 4 gather-add operations in ``_add_relpos_from_weights_`` into a
single Triton kernel that reads ``z``, looks up 3 index tables + 1 constant
row from ``w``, sums them, and writes back to ``z`` in place.

This avoids materializing any ``[N, N, c_z]`` intermediate from the gather.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_relpos_embed_kernel(
    Z_ptr,
    W_ptr,
    IDX1_ptr,
    IDX2_ptr,
    IDX3_ptr,
    SAME_ENTITY_ptr,
    C: tl.constexpr,
    SAME_ENTITY_OFFSET: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Keep flattened addressing valid when N²*C exceeds signed int32.
    ij_offset = tl.program_id(0).to(tl.int64)

    idx1 = tl.load(IDX1_ptr + ij_offset)
    idx2 = tl.load(IDX2_ptr + ij_offset)
    idx3 = tl.load(IDX3_ptr + ij_offset)
    same_ent = tl.load(SAME_ENTITY_ptr + ij_offset).to(tl.float32)

    c_offsets = tl.arange(0, BLOCK_C)
    mask = c_offsets < C

    z_offset = ij_offset * C + c_offsets
    z_vals = tl.load(Z_ptr + z_offset, mask=mask)

    w1 = tl.load(W_ptr + idx1 * C + c_offsets, mask=mask)
    w2 = tl.load(W_ptr + idx2 * C + c_offsets, mask=mask)
    w3 = tl.load(W_ptr + idx3 * C + c_offsets, mask=mask)
    w_ent = tl.load(W_ptr + SAME_ENTITY_OFFSET * C + c_offsets, mask=mask)

    z_vals += w1 + w2 + w3 + same_ent * w_ent

    tl.store(Z_ptr + z_offset, z_vals, mask=mask)


def fused_relpos_embed_add_(
    z: torch.Tensor,
    w: torch.Tensor,
    rel_pos_idx: torch.Tensor,
    rel_token_idx: torch.Tensor,
    rel_chain_idx: torch.Tensor,
    same_entity: torch.Tensor,
    same_entity_offset: int,
) -> None:
    """In-place add all relpos embeddings to z in a single kernel launch.

    Args:
        z: [*, N, N, C] pair representation (fp32, contiguous last dim).
        w: [vocab, C] transposed weight table (fp32).
        rel_pos_idx: [*, N, N] int64, pre-offset indices into w.
        rel_token_idx: [*, N, N] int64, pre-offset indices into w.
        rel_chain_idx: [*, N, N] int64, pre-offset indices into w.
        same_entity: [*, N, N] bool/int mask.
        same_entity_offset: row index in w for the same-entity embedding.
    """
    assert z.is_contiguous(), "z must be contiguous"
    C = z.shape[-1]
    M = z.numel() // C  # total number of (i, j) positions

    BLOCK_C = triton.next_power_of_2(C)

    z_flat = z.view(-1, C)
    idx1_flat = rel_pos_idx.reshape(-1)
    idx2_flat = rel_token_idx.reshape(-1)
    idx3_flat = rel_chain_idx.reshape(-1)
    se_flat = same_entity.reshape(-1)

    grid = (M,)
    _fused_relpos_embed_kernel[grid](
        z_flat,
        w,
        idx1_flat,
        idx2_flat,
        idx3_flat,
        se_flat,
        C=C,
        SAME_ENTITY_OFFSET=same_entity_offset,
        BLOCK_C=BLOCK_C,
    )
