# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

"""
Attention layers. Includes standard multi-head attention and global attention.
Optimizations such as LMA and DeepSpeed EvoformerAttention are included.
"""

import importlib
import math
from typing import Optional

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.utils.checkpointing import get_checkpoint_fn
from openfold3.core.utils.tensor_utils import flatten_final_dims

from .linear import Linear

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
ds4s_is_installed = (
    deepspeed_is_installed
    and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
)
if deepspeed_is_installed:
    import deepspeed

if ds4s_is_installed:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

DEFAULT_LMA_Q_CHUNK_SIZE = 1024
DEFAULT_LMA_KV_CHUNK_SIZE = 4096


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of
    type bfloat16
    """
    d = t.dtype
    deepspeed_is_initialized = (
        deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
    )
    if d is torch.bfloat16 and not deepspeed_is_initialized:
        with torch.amp.autocast("cuda", enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


# @torch.jit.script
def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
    use_high_precision: bool = False,
) -> torch.Tensor:
    """Attention operation with bias terms.

    For clarity, the dimensions are as follows:
        *: Batch dimensions
        H: Number of heads
        K, Q, V: Key, query, value dimensions
        C_hidden: Hidden dimension

    Args:
        query (shape [*, H, Q, C_hidden]): query tensor
        key (shape [*, H, K, C_hidden]): key tensor
        value (shape [*, H, V, C_hidden]): value tensor
        biases : list of bias tensors
        use_high_precision: Whether to use high precision up until
            and including softmax

    Returns:
        shape [*, H, V, C_hidden]: attention output
    """
    attn_dtype = torch.float32 if use_high_precision else query.dtype
    with torch.amp.autocast("cuda", dtype=attn_dtype):
        # Generate attention scores
        scores = torch.einsum("...qc, ...kc->...qk", query, key)

        # Add the biases
        for b in biases:
            scores += b

        # Normalize the scores
        scores = softmax_no_cast(scores, dim=-1)

    # Multiply scores by values
    attention = torch.einsum("...qk, ...kc->...qc", scores.to(dtype=value.dtype), value)

    return attention


@torch.jit.ignore
def attention_chunked_trainable(
    query,
    key,
    value,
    biases,
    chunk_size,
    chunk_dim,
    checkpoint,
):
    if checkpoint and len(biases) > 2:
        raise ValueError("Checkpointed version permits only permits two bias terms")

    def _checkpointable_attention(q, k, v, b1, b2):
        bs = [b for b in [b1, b2] if b is not None]
        a = _attention(q, k, v, bs)
        return a

    o_chunks = []
    checkpoint_fn = get_checkpoint_fn()
    count = query.shape[chunk_dim]
    for start in range(0, count, chunk_size):
        end = start + chunk_size
        idx = [slice(None)] * len(query.shape)
        idx[chunk_dim] = slice(start, end)
        idx_tup = tuple(idx)
        q_chunk = query[idx_tup]
        k_chunk = key[idx_tup]
        v_chunk = value[idx_tup]

        def _slice_bias(b: torch.Tensor, i: list, s: int, e: int) -> torch.Tensor:
            """Slice bias tensor along chunk dimension."""
            i[chunk_dim] = slice(s, e) if b.shape[chunk_dim] != 1 else slice(None)
            return b[tuple(i)]

        if checkpoint:
            bias_1_chunk, bias_2_chunk = (
                _slice_bias(b, i=idx, s=start, e=end) if b is not None else None
                for b in (biases + [None, None])[:2]
            )

            o_chunk = checkpoint_fn(
                _checkpointable_attention,
                q_chunk,
                k_chunk,
                v_chunk,
                bias_1_chunk,
                bias_2_chunk,
            )
        else:
            bias_chunks = [_slice_bias(b, i=idx, s=start, e=end) for b in biases]

            o_chunk = _attention(q_chunk, k_chunk, v_chunk, bias_chunks)

        o_chunk = o_chunk.transpose(-2, -3)
        o_chunks.append(o_chunk)

    o = torch.cat(o_chunks, dim=chunk_dim)
    return o


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        linear_init_params: ConfigDict = lin_init.mha_init,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, **linear_init_params.linear_q
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, **linear_init_params.linear_k
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, **linear_init_params.linear_v
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, **linear_init_params.linear_o
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, **linear_init_params.linear_g
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[list[torch.Tensor]] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = DEFAULT_LMA_Q_CHUNK_SIZE,
        lma_kv_chunk_size: int = DEFAULT_LMA_KV_CHUNK_SIZE,
        use_high_precision: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory-efficient attention kernel.
                If none of the "use_<...>" flags are True, a stock PyTorch
                implementation is used instead
            use_lma:
                Whether to use low-memory attention (Staats & Rabe 2021). If
                none of the "use_<...>" flags are True, a stock PyTorch
                implementation is used instead
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
            use_high_precision:
                Whether to use high precision up until and including softmax.
                This requires using the default implementation and cannot be
                used with the above kernel options.
        Returns
            [*, Q, C_q] attention update
        """
        if use_lma and (lma_q_chunk_size is None or lma_kv_chunk_size is None):
            raise ValueError(
                "If use_lma is specified, lma_q_chunk_size and "
                "lma_kv_chunk_size must be provided"
            )

        # TODO: Make this more explicit
        # The EvoformerAttention kernel can only be used for sequence lengths > 16
        if use_deepspeed_evo_attention and q_x.shape[-2] <= 16:
            use_deepspeed_evo_attention = False

        attn_options = [
            use_deepspeed_evo_attention,
            use_lma,
            use_high_precision,
        ]
        if sum(attn_options) > 1:
            raise ValueError("Choose at most one alternative attention algorithm")

        if biases is None:
            biases = []

        # DeepSpeed attention kernel applies scaling internally
        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=not use_deepspeed_evo_attention)

        if use_deepspeed_evo_attention:
            if len(biases) > 2:
                raise ValueError(
                    "If use_deepspeed_evo_attention is True, you may only "
                    "provide up to two bias terms"
                )
            o = _deepspeed_evo_attn(q, k, v, biases)
        elif use_lma:
            biases = [
                b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],))
                for b in biases
            ]
            o = _lma(q, k, v, biases, lma_q_chunk_size, lma_kv_chunk_size)
            o = o.transpose(-2, -3)
        else:
            o = _attention(q, k, v, biases, use_high_precision=use_high_precision)
            o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


class GlobalAttention(nn.Module):
    def __init__(
        self, c_in, c_hidden, no_heads, inf, eps, linear_init_params=lin_init.mha_init
    ):
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.linear_q = Linear(c_in, c_hidden * no_heads, **linear_init_params.linear_q)

        self.linear_k = Linear(
            c_in,
            c_hidden,
            **linear_init_params.linear_k,
        )
        self.linear_v = Linear(c_in, c_hidden, **linear_init_params.linear_v)
        self.linear_g = Linear(c_in, c_hidden * no_heads, **linear_init_params.linear_g)
        self.linear_o = Linear(c_hidden * no_heads, c_in, **linear_init_params.linear_o)

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        use_lma: bool = False,
    ) -> torch.Tensor:
        # [*, N_res, C_in]
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
            torch.sum(mask, dim=-1)[..., None] + self.eps
        )

        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q *= self.c_hidden ** (-0.5)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)

        bias = (self.inf * (mask - 1))[..., :, None, :]
        if not use_lma:
            o = _attention(q, k, v, [bias])
        else:
            o = _lma(
                q, k, v, [bias], DEFAULT_LMA_Q_CHUNK_SIZE, DEFAULT_LMA_KV_CHUNK_SIZE
            )

        # [*, N_res, N_seq, C_hidden]
        g = self.sigmoid(self.linear_g(m))

        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g

        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)

        return m


@torch.compiler.disable
def _deepspeed_evo_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: list[torch.Tensor],
):
    """ ""
    Compute attention using the DeepSpeed DS4Sci_EvoformerAttention kernel.

    Args:
        q:
            [*, H, Q, C_hidden] query data
        k:
            [*, H, K, C_hidden] key data
        v:
            [*, H, V, C_hidden] value data
        biases:
            List of biases that broadcast to [*, H, Q, K]
    """

    if not ds4s_is_installed:
        raise ValueError(
            "_deepspeed_evo_attn requires that DeepSpeed be installed "
            "and that the deepspeed.ops.deepspeed4science package exists"
        )

    # [*, Q/K, H, C_hidden]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    def reshape_dims(x):
        no_batch_dims = len(x.shape[:-3])
        if no_batch_dims < 2:
            return x.reshape(*((1,) * (2 - no_batch_dims) + x.shape))
        if no_batch_dims > 2:
            return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
        return x

    # Reshape tensors to match expected input shape [B, N, Q/K, H, C_hidden]
    # for DS4Sci_EvoformerAttention() by adding or flattening batch dims as needed.
    orig_shape = q.shape
    if len(orig_shape[:-3]) != 2:
        q = reshape_dims(q)
        k = reshape_dims(k)
        v = reshape_dims(v)
        biases = [reshape_dims(b) for b in biases]

    def convert_dtype(x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in [torch.bfloat16, torch.float16]:
            return x.to(dtype=torch.bfloat16)
        return x

    # DeepSpeed attn. kernel requires inputs to be type bf16 or fp16
    # Cast to bf16 so kernel can be used during inference
    orig_dtype = q.dtype
    q = convert_dtype(q)
    k = convert_dtype(k)
    v = convert_dtype(v)
    biases = [convert_dtype(b) for b in biases]

    o = DS4Sci_EvoformerAttention(q, k, v, biases)

    # Convert back to original shape and dtype
    o = o.reshape(orig_shape).to(dtype=orig_dtype)

    return o


def _lma(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: list[torch.Tensor],
    q_chunk_size: int,
    kv_chunk_size: int,
):
    no_q, no_kv = q.shape[-2], k.shape[-2]

    # [*, H, Q, C_hidden]
    o = q.new_zeros(q.shape)
    for q_s in range(0, no_q, q_chunk_size):
        q_chunk = q[..., q_s : q_s + q_chunk_size, :]
        large_bias_chunks = [b[..., q_s : q_s + q_chunk_size, :] for b in biases]

        maxes = []
        weights = []
        values = []
        for kv_s in range(0, no_kv, kv_chunk_size):
            k_chunk = k[..., kv_s : kv_s + kv_chunk_size, :]
            v_chunk = v[..., kv_s : kv_s + kv_chunk_size, :]
            small_bias_chunks = [
                b[..., kv_s : kv_s + kv_chunk_size] for b in large_bias_chunks
            ]

            a = torch.einsum(
                "...hqd,...hkd->...hqk",
                q_chunk,
                k_chunk,
            )

            for b in small_bias_chunks:
                a += b

            max_a = torch.max(a, dim=-1, keepdim=True)[0]
            exp_a = torch.exp(a - max_a)
            exp_v = torch.einsum("...hvf,...hqv->...hqf", v_chunk, exp_a)

            maxes.append(max_a.detach().squeeze(-1))
            weights.append(torch.sum(exp_a, dim=-1))
            values.append(exp_v)

        chunk_max = torch.stack(maxes, dim=-3)
        chunk_weights = torch.stack(weights, dim=-3)
        chunk_values = torch.stack(values, dim=-4)

        global_max = torch.max(chunk_max, dim=-3, keepdim=True)[0]
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values = chunk_values * max_diffs.unsqueeze(-1)
        chunk_weights = chunk_weights * max_diffs

        all_values = torch.sum(chunk_values, dim=-4)
        all_weights = torch.sum(chunk_weights.unsqueeze(-1), dim=-4)

        q_chunk_out = all_values / all_weights

        o[..., q_s : q_s + q_chunk_size, :] = q_chunk_out

    return o
