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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: inference dispatch for the fused
# low-memory triangle-attention implementation.

"""Fused triangle attention dispatch (inference, no chunking).

Replaces the separate LN → 3 linear projections → cuEq → gate → output
pipeline with:
  1. ``fused_ln_linear`` for LN→QKV (one kernel, no LN intermediate in HBM)
  2. ``fused_ln_linear`` for LN→triangle_bias (tiny output, no LN intermediate)
  3. cuEq (or eager) attention core (unchanged)
  4. ``fused_gated_attn_out`` for gate→output (gate in registers, no HBM alloc)

Memory: ~4U peak (3U for q/k/v + 1U resident z) vs ~8U eager.
No chunking — cuEq receives the full q/k/v tensors.

Activated by ``OPENFOLD3_FUSED_TRI_ATTN=1``.
"""

from __future__ import annotations

import math
import os

import torch

from openfold3.core.kernels.triton.fused_ln_linear import (
    fused_ln_linear,
)
from openfold3.core.kernels.triton.fused_ln_linear import (
    is_triton_available as _ln_triton_available,
)
from openfold3.core.kernels.triton.fused_tri_attn_out import (
    fused_gated_attn_out,
)
from openfold3.core.kernels.triton.fused_tri_attn_out import (
    is_triton_available as _out_triton_available,
)
from openfold3.core.model.primitives.attention import (
    _attention,
    _cueq_triangle_attn,
    cueq_is_installed,
)
from openfold3.core.utils.tensor_utils import permute_final_dims

if cueq_is_installed:
    from openfold3.core.model.primitives.attention import cueq_would_fall_back

_FLAG_TRUE = {"1", "true", "True"}


def is_fused_tri_attn_enabled() -> bool:
    return (
        os.environ.get("OPENFOLD3_FUSED_TRI_ATTN", "0") in _FLAG_TRUE
        and _ln_triton_available()
        and _out_triton_available()
    )


def _eligible(z: torch.Tensor) -> bool:
    return (
        _ln_triton_available()
        and _out_triton_available()
        and z.is_cuda
        and not torch.is_grad_enabled()
        and z.dim() == 4
    )


def _has_bias(*linears) -> bool:
    return any(m.bias is not None for m in linears)


def fused_tri_attn_forward(
    module,
    z: torch.Tensor,
    mask: torch.Tensor | None,
    use_cueq: bool,
    inplace_safe: bool,
) -> torch.Tensor | None:
    """Fused triangle attention forward (no chunking).

    Args:
        module: ``TriangleAttention`` instance.
        z: ``[B, I, J, c_z]`` raw pair tensor (pre-LN).
        mask: ``[B, I, J]`` or None.
        use_cueq: whether cuEq triangle attention is requested.
        inplace_safe: whether output can alias input (unused here;
            the fused path always allocates fresh output).

    Returns:
        ``[B, I, J, c_z]`` attention update, or ``None`` if ineligible.
    """
    if not _eligible(z):
        return None

    mha = module.mha
    if _has_bias(mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_o):
        return None
    if mha.linear_g is not None and mha.linear_g.bias is not None:
        return None
    if mha.linear_g is None:
        return None

    c_z = module.c_in
    c_hidden = module.c_hidden
    no_heads = module.no_heads
    starting = module.starting

    # ── Handle starting/ending transpose ────────────────────────────
    if not starting:
        z = z.transpose(-2, -3)
        if mask is not None:
            mask = mask.transpose(-1, -2)

    B, I, J, _ = z.shape

    if mask is None:
        mask = z.new_ones(z.shape[:-1])

    # ── Biases (computed from z, need LN'd z) ──────────────────────
    # mask_bias: [B, I, 1, 1, J]
    mask_bias = (module.inf * (mask - 1))[..., :, None, None, :]

    # triangle_bias: LN(z) → linear_z → permute → [B, 1, H, I, J]
    # Use fused_ln_linear to avoid materializing the full LN'd z.
    z_2d = z.reshape(-1, c_z)
    ln = module.layer_norm
    tb_2d = fused_ln_linear(
        z_2d, ln.weight, ln.bias, module.linear_z.weight,
        module.linear_z.bias, ln.eps,
    )  # [B*I*J, H]
    tb = tb_2d.view(B, I, J, no_heads)
    triangle_bias = permute_final_dims(tb, (2, 0, 1))  # [B, H, I, J]
    triangle_bias = triangle_bias.unsqueeze(-4)  # [B, 1, H, I, J]

    biases = [mask_bias, triangle_bias]

    # ── Fused LN → QKV projection ─────────────────────────────────
    W_qkv = torch.cat(
        [mha.linear_q.weight, mha.linear_k.weight, mha.linear_v.weight],
        dim=0,
    )  # [3*H*c_h, c_z]
    qkv_2d = fused_ln_linear(
        z_2d, ln.weight, ln.bias, W_qkv, None, ln.eps,
    )  # [B*I*J, 3*H*c_h]

    del W_qkv

    total_hidden = c_hidden * no_heads
    q, k, v = qkv_2d.split(total_hidden, dim=-1)
    # Reshape: [B*I*J, H*c_h] → [B, I, J, H, c_h] → [B, I, H, J, c_h]
    # .contiguous() so cuEq doesn't silently copy while qkv_2d is still live.
    q = q.view(B, I, J, no_heads, c_hidden).transpose(-2, -3).contiguous()
    k = k.view(B, I, J, no_heads, c_hidden).transpose(-2, -3).contiguous()
    v = v.view(B, I, J, no_heads, c_hidden).transpose(-2, -3).contiguous()
    del qkv_2d

    # ── Attention core ─────────────────────────────────────────────
    use_cueq_here = use_cueq
    if use_cueq_here and cueq_is_installed and cueq_would_fall_back(
        n_token=J, hidden_dim=c_hidden, dtype=z.dtype,
    ):
        use_cueq_here = False

    if use_cueq_here and cueq_is_installed:
        scale = 1.0 / math.sqrt(c_hidden)
        o = _cueq_triangle_attn(q, k, v, biases, scale=scale)
        # o: [B, I, J, H, c_h] (cuEq transposes internally)
    else:
        q = q / math.sqrt(c_hidden)
        o = _attention(q, k, v, biases)
        # o: [B, I, H, J, c_h] → transpose to [B, I, J, H, c_h]
        o = o.transpose(-2, -3)

    del q, k, v

    # ── Fused gated output ─────────────────────────────────────────
    # o: [B, I, J, H, c_h] → flatten to [M, H*c_h]
    o_2d = o.reshape(-1, total_hidden)
    del o

    out_2d = fused_gated_attn_out(
        z_2d,       # raw z for LN recompute inside kernel
        o_2d,
        ln.weight,
        ln.bias,
        mha.linear_g.weight,
        mha.linear_o.weight,
        ln.eps,
    )  # [M, c_z]

    out = out_2d.view(B, I, J, c_z)

    # ── Undo starting/ending transpose ─────────────────────────────
    if not starting:
        out = out.transpose(-2, -3)

    return out
