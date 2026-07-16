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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: streamed projection, kernel
# dispatch, and residual-write handling for cap-blocked triangle attention.

"""Fused cap-blocked triangle attention dispatch for inference."""

from __future__ import annotations

import math
import os

import torch

from openfold3.core.kernels.triton.flash_tri_attn import (
    flash_tri_attn_v1_block,
)
from openfold3.core.kernels.triton.flash_tri_attn import (
    is_triton_available as _flash_triton_available,
)
from openfold3.core.kernels.triton.fused_ln_linear import (
    is_triton_available as _ln_triton_available,
)
from openfold3.core.kernels.triton.fused_ln_linear import (
    pair_ln_linear_inference,
)

_FLAG_TRUE = {"1", "true", "True"}


def is_fused_tri_attn_v1_enabled() -> bool:
    return (
        os.environ.get("OPENFOLD3_FUSED_TRI_ATTN_V1", "0") in _FLAG_TRUE
        and _ln_triton_available()
        and _flash_triton_available()
    )


def _eligible(z: torch.Tensor) -> bool:
    return (
        _ln_triton_available()
        and _flash_triton_available()
        and z.is_cuda
        and not torch.is_grad_enabled()
        and z.dim() == 4
    )


def _has_bias(*linears) -> bool:
    return any(m.bias is not None for m in linears)


def fused_tri_attn_v1_forward(
    module,
    z: torch.Tensor,
    mask: torch.Tensor | None,
    inplace_safe: bool,
    use_cueq_triangle_kernels: bool,
    chunk_size: int | None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Cap-blocked custom triangle attention with streamed LN prologue (Design B).

    The outer schedule streams row blocks of size ``chunk_size`` through a
    flash-attention-style Triton kernel (``flash_tri_attn_v1_block``).  Per row
    block: LN → packed cuBLAS QKVG matmul → flash kernel → ``linear_o`` write.
    The ``triangle_bias`` projection uses ``pair_ln_linear_inference`` (strided
    LN→linear Triton kernel in ``fused_ln_linear.py``) which decodes ``(i, j)``
    via independent strides so contig ``z`` (tri_att_start) and transpose-view
    ``z`` (tri_att_end in ``PairBlock``) share the same call path — no Python
    branch on the input stride pattern, no persistent 1U ``z_norm`` buffer.

    When ``residual`` is provided (must have the same shape / dtype / device as
    ``z``), the per-block ``linear_o`` write is fused with the residual add.
    The write dispatches on the residual's *write-view* layout, defined as
    ``residual_wv = residual if module.starting else residual.transpose(-2, -3)``
    (so ``residual_wv`` and ``z`` share (I, J) semantics after the internal
    ``if not starting`` transpose):

    * ``residual_wv`` contig  → ``torch.addmm(out=residual_wv[start:end])``
      fuses the add into the ``W_o`` matmul (0U extra allocation).
    * ``residual_wv`` non-contig → materialize per-block scratch
      (``chunk × J × c_z`` ≈ 0.1U at chunk=128, N=1264) and ``.add_()`` into
      the strided slice.  This handles ``PairBlock.tri_att_end`` (starting=True
      module on pre-transposed z with ``residual=z_transposed`` sharing z's
      storage) correctly, including the axis swap that the removed pre-Design-B
      case-(c) shortcut used to drop silently.

    Returns ``residual`` (mutated in place) when residual-fused, or the freshly
    allocated output tensor otherwise.  Returns ``None`` if the fused path is
    ineligible (grad enabled, unsupported dims, biases on gemms, or an
    unsupported residual layout that shares neither z's stride nor is contig).
    """
    del inplace_safe, use_cueq_triangle_kernels

    if (
        chunk_size is None
        or not _eligible(z)
        or torch.is_grad_enabled()
    ):
        return None

    mha = module.mha
    if _has_bias(mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_o):
        return None
    if mha.linear_g is None or mha.linear_g.bias is not None:
        return None

    c_z = module.c_in
    c_hidden = module.c_hidden
    no_heads = module.no_heads
    total_hidden = c_hidden * no_heads
    supported_shape = (c_z, c_hidden, no_heads) in {
        (64, 16, 4),
        (128, 32, 4),
    }
    if not supported_shape or total_hidden != c_z:
        return None

    # ``starting`` denotes the starting-node attention pattern.  For a
    # ``TriangleAttention(starting=False)`` module, we pre-transpose z so the
    # rest of the fused path can process both variants with the same flash-
    # kernel starting-node layout.  The tri_att_end pattern in ``PairBlock``
    # instead calls a ``starting=True`` module on a pre-transposed z (with
    # residual = z_transposed); that non-contig-input case is handled by the
    # ``residual_wv`` classifier below (correct axis-swapped write path) and
    # by the strided-input support in ``pair_ln_linear_inference`` for tri_bias.
    if not module.starting:
        z = z.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

    B, I_dim, J_dim, _ = z.shape
    if B != 1 or I_dim != J_dim:
        return None
    if I_dim <= chunk_size:
        return None

    # Residual write-view classification.  After the ``if not starting`` block
    # above, ``z`` is always in the flash-kernel's starting-node layout.  The
    # per-block flash output ``attn`` is indexed as ``(z-row-block, z-col)``,
    # so its write into ``residual`` must be into a view that matches ``z``'s
    # (I, J) axes.  Define ``residual_wv`` = ``residual`` if the caller passed
    # residual in z's layout (starting=True), else ``residual.transpose(-2,-3)``
    # (starting=False — the caller passed residual in the module's I/O layout,
    # which is z's transposed layout post the ``if not starting`` transpose).
    # Then ``residual_wv`` and ``z`` share (I, J) semantics and the per-block
    # slice ``residual_wv[:, start:end, :, :]`` is the correct write target.
    #
    # ``residual_wv`` can be contig or non-contig; dispatch below:
    # * contig write-view → per-block ``addmm(out=residual_wv[start:end])`` fuses
    #   the residual add into the ``W_o`` matmul (0U extra).
    # * non-contig write-view → materialize per-block scratch (~0.1U at
    #   chunk=128) and ``.add_()`` into the strided slice.
    #
    # The four practical cases:
    # (a) starting=True, residual contig  → residual_wv contig     (tri_att_start)
    # (b) starting=False, residual contig → residual_wv non-contig (native
    #     ending-node module)
    # (c) starting=True, residual is the same non-contig transpose view as ``z``
    #     (i.e. caller pre-transposed z, passed residual=z_transposed) →
    #     residual_wv non-contig (tri_att_end pattern in PairBlock)
    # (d) starting=False, residual non-contig transpose view of z's underlying
    #     storage → residual_wv contig (uncommon, but symmetric to (c))
    residual_wv: torch.Tensor | None = None
    residual_wv_contig: bool = False
    if residual is not None:
        if (
            residual.shape != z.shape
            or residual.dtype != z.dtype
            or residual.device != z.device
        ):
            return None
        # If ``starting=False`` fired the transpose on ``z`` above, we need to
        # apply the same transpose to ``residual`` so their (I, J) axes line up.
        residual_wv = residual if module.starting else residual.transpose(-2, -3)
        if residual_wv.is_contiguous():
            residual_wv_contig = True
        else:
            # Non-contig write-view is only supported when it lines up with
            # ``z``'s stride pattern — otherwise the ``.add_()`` broadcast /
            # slice semantics would be wrong.
            if not (
                residual_wv.data_ptr() == z.data_ptr()
                and residual_wv.stride() == z.stride()
            ):
                return None
            residual_wv_contig = False

    ln = module.layer_norm

    # tri_bias projection via strided LN→linear.  ``pair_ln_linear_inference``
    # decodes ``(i, j)`` via independent strides so the contig and transpose-
    # view ``z`` cases are handled by one call with no Python branch and no
    # full-N ``z_norm`` materialization.  Returns ``[I*J, no_heads]`` flat.
    tb_flat = pair_ln_linear_inference(
        z,
        ln.weight,
        ln.bias,
        module.linear_z.weight,
        None,
        ln.eps,
    )
    triangle_bias_full = (
        tb_flat.view(B, I_dim, J_dim, no_heads)
        .permute(0, 3, 1, 2)
        .contiguous()
        .unsqueeze(1)
    )
    del tb_flat

    W_qkvg = torch.cat(
        [
            mha.linear_q.weight,
            mha.linear_k.weight,
            mha.linear_v.weight,
            mha.linear_g.weight,
        ],
        dim=0,
    )

    if mask is not None and not mask.is_contiguous():
        mask = mask.contiguous()

    # Fresh ``out`` allocation policy:
    # * residual write-view provided (contig or non-contig) → aliases the write-
    #   view (0U extra).  Per-block dispatch below picks addmm vs scratch+add_.
    # * no residual → fresh ``[B, I_dim, J_dim, c_z]`` contig buffer (1U) in the
    #   flash-kernel's starting-node layout; post-loop we transpose it back to
    #   the caller's I/O layout if starting=False.
    if residual_wv is not None:
        out = residual_wv
    else:
        out = torch.empty((B, I_dim, J_dim, c_z), dtype=z.dtype, device=z.device)

    scale = 1.0 / math.sqrt(c_hidden)

    for start in range(0, I_dim, chunk_size):
        end = min(I_dim, start + chunk_size)
        rows = end - start

        # Per-block LN — replaces the persistent ``z_norm`` allocation.  A
        # ``.contiguous()`` on the block slice is a cheap ``chunk × N × c_z``
        # copy (~0.1U at chunk=128, N=1264) that turns any stride pattern
        # (contig from tri_att_start; transpose-view from tri_att_end) into a
        # contig block ``F.layer_norm`` can consume in one shot.
        z_block = z[:, start:end, :, :].contiguous()
        z_norm_block = torch.nn.functional.layer_norm(
            z_block, (c_z,), ln.weight, ln.bias, ln.eps,
        )
        del z_block

        qkvg_2d = torch.nn.functional.linear(
            z_norm_block.reshape(-1, c_z), W_qkvg, None,
        )
        del z_norm_block

        q_2d, k_2d, v_2d, g_2d = qkvg_2d.split(total_hidden, dim=-1)
        q = q_2d.view(B, rows, J_dim, no_heads, c_hidden)
        k = k_2d.view(B, rows, J_dim, no_heads, c_hidden)
        v = v_2d.view(B, rows, J_dim, no_heads, c_hidden)
        gate = g_2d.view(B, rows, J_dim, no_heads, c_hidden)
        attn = flash_tri_attn_v1_block(
            q, k, v, gate,
            triangle_bias_full,
            mask, start, scale,
            out=q,
            mask_inf=module.inf,
        )

        attn_2d = attn.reshape(-1, total_hidden)
        W_o_t = mha.linear_o.weight.t()

        if residual_wv is None:
            # Fresh contig ``out`` — plain ``torch.mm(out=slice)``.
            out_block_2d = out[:, start:end, :, :].reshape(-1, c_z)
            torch.mm(attn_2d, W_o_t, out=out_block_2d)
            del out_block_2d
        elif residual_wv_contig:
            # Contig write-view: fuse the residual add into ``W_o`` matmul.
            out_block_2d = out[:, start:end, :, :].reshape(-1, c_z)
            torch.addmm(out_block_2d, attn_2d, W_o_t, out=out_block_2d)
            del out_block_2d
        else:
            # Non-contig write-view: materialize per-block scratch, then
            # ``.add_()`` the strided slice.  Scratch is ``chunk × J × c_z``
            # fp32 ≈ 0.1U at chunk=128, N=1264 — dropped before the next iter.
            scratch = torch.mm(attn_2d, W_o_t).view(B, rows, J_dim, c_z)
            out[:, start:end, :, :].add_(scratch)
            del scratch

        del qkvg_2d, q_2d, k_2d, v_2d, g_2d
        del q, k, v, gate, attn, attn_2d

    del W_qkvg, triangle_bias_full

    if residual_wv is not None:
        # ``residual`` (the caller-visible tensor) has already been mutated in
        # place; return it unchanged so the caller can chain assignments.
        return residual

    if not module.starting:
        out = out.transpose(-2, -3)

    return out
