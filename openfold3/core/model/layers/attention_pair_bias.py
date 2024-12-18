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

"""Attention layer with pair bias."""

import math
from typing import Optional

import torch
from ml_collections import ConfigDict
from torch import nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.primitives import (
    AdaLN,
    Attention,
    BlockSparseAttention,
    LayerNorm,
    Linear,
)
from openfold3.core.utils.tensor_utils import permute_final_dims, sparsify_tensor


class AttentionPairBias(nn.Module):
    """Attention layer with pair bias and neighborhood mask.

    Implements AF3 Algorithm 24.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        use_ada_layer_norm: bool = False,
        use_block_sparse_attn: bool = False,
        block_size: Optional[int] = 16,
        gating: bool = True,
        inf=1e9,
        linear_init_params: ConfigDict = lin_init.att_pair_bias_init,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_s:
                Single activation channel dimension
            c_z:
                Pair activation channel dimension
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            use_ada_layer_norm:
                Whether to apply AdaLN-Zero conditioning
            use_block_sparse_attn:
                Whether to use Triton block sparse attention kernels
            block_size:
                Block size to use in block sparse attention
            gating:
                Whether the output should be gated using query data
            inf:
                Large constant used to create mask for attention logits
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.use_ada_layer_norm = use_ada_layer_norm
        self.use_block_sparse_attn = use_block_sparse_attn
        self.block_size = block_size
        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf

        if self.use_ada_layer_norm:
            self.layer_norm_a = AdaLN(
                c_a=self.c_q, c_s=self.c_s, linear_init_params=linear_init_params.ada_ln
            )
            self.linear_ada_out = Linear(
                self.c_s, self.c_q, **linear_init_params.linear_ada_out
            )
        else:
            self.layer_norm_a = LayerNorm(c_in=self.c_q)

        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(self.c_z, no_heads, **linear_init_params.linear_z)

        if self.use_block_sparse_attn:
            self.mha = BlockSparseAttention(
                c_q=c_q,
                c_k=c_k,
                c_v=c_v,
                c_hidden=c_hidden,
                no_heads=no_heads,
                block_size=block_size,
                gating=gating,
                linear_init_params=linear_init_params.mha,
            )
        else:
            self.mha = Attention(
                c_q=c_q,
                c_k=c_k,
                c_v=c_v,
                c_hidden=c_hidden,
                no_heads=no_heads,
                gating=gating,
                linear_init_params=linear_init_params.mha,
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_sparse_bias(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        layout: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Args:
            a:
                [*, N, C_token] Token or atom-level embedding
            z:
                [*, N, N, C_z] Pair embedding
            layout:
                [N / block_size, N / block_size] Layout config for block sparse
                attention. Dictates which sections of the attention matrix
                to compute.
            mask:
                [*, N] Mask for token or atom-level embedding

        Returns:
            List of bias terms. Includes the pair bias and attention mask.
        """
        if mask is None:
            # [*, N]
            mask = a.new_ones(
                a.shape[:-1],
            )

        # [*, 1, 1, N]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :]

        batch_dims = z.shape[:-3]
        feat_dim = z.shape[-1]

        # [*, C_z, N, N]
        z = permute_final_dims(z, [2, 0, 1])

        z = z + mask_bias
        layout = layout[None].tile((feat_dim, 1, 1)).long()

        # [*, num_blocks, C_z, block_size, block_size]
        z = sparsify_tensor(z, layout, self.block_size, batch_dims=batch_dims).reshape(
            *batch_dims, -1, feat_dim, self.block_size, self.block_size
        )

        # [*, num_blocks, block_size, block_size, C_z]
        z = self.layer_norm_z(permute_final_dims(z, [1, 2, 0]))

        # [*, num_blocks, block_size, block_size, no_heads]
        z = self.linear_z(z)

        # [*, num_blocks *  no_heads, block_size, block_size]
        z = permute_final_dims(z, [2, 0, 1]).reshape(
            *batch_dims, -1, self.block_size, self.block_size
        )

        return [z]

    def _prep_all_inputs(
        self,
        a: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> tuple:
        """
        Args:
            a:
                [*, N, C_token] Token or atom-level embedding
            z:
                [*, N, N, C_z] Pair embedding
            beta:
                [*, N, N] Neighborhood mask
            mask:
                [*, N] Mask for token or atom-level embedding

        Returns:
            List of bias terms. Includes the pair bias and attention mask.
        """
        if mask is None:
            # [*, N]
            mask = a.new_ones(
                a.shape[:-1],
            )

        def convert_to_blocks_1d(x, dim, shift_interval, block_len, num_blocks):
            blocks = [
                x.narrow(dim, shift_interval * i, block_len) for i in range(num_blocks)
            ]
            return torch.stack(blocks, dim=dim - 1)

        def convert_to_blocks_2d(x, dims, shift_interval, block_lens, num_blocks):
            blocks = [
                x.narrow(dims[0], shift_interval * i, block_lens[0]).narrow(
                    dims[1], shift_interval * i, block_lens[1]
                )
                for i in range(num_blocks)
            ]
            return torch.stack(blocks, dim=min(dims) - 1)

        n_query = 32
        n_key = 128
        n_atom = a.shape[-2]

        offset = n_query // 2

        num_blocks = math.ceil(n_atom / n_query)

        # TODO: This is here for hanging on AWS AMD CPU + H200 nodes. Remove later.
        # subset_centers = [n_query * i + offset for i in range(num_blocks)]
        subset_centers = offset + torch.arange(num_blocks) * n_query

        pad_len_right_q = (n_query - n_atom % n_query) % n_query
        a_query = torch.nn.functional.pad(a, (0, 0, 0, pad_len_right_q), value=0.0)

        a_query = convert_to_blocks_1d(
            x=a_query,
            dim=-2,
            shift_interval=n_query,
            block_len=n_query,
            num_blocks=num_blocks,
        )

        pad_len_right_k = subset_centers[-1] + n_key // 2 - n_atom
        pad_len_left_k = n_key // 2 - subset_centers[0]
        a_key = torch.nn.functional.pad(
            a, (0, 0, pad_len_left_k, pad_len_right_k), value=0.0
        )
        a_key = convert_to_blocks_1d(
            x=a_key,
            dim=-2,
            shift_interval=n_query,
            block_len=n_key,
            num_blocks=num_blocks,
        )

        mask_q = torch.nn.functional.pad(mask, (0, pad_len_right_q), value=0.0)
        mask_q = convert_to_blocks_1d(
            x=mask_q,
            dim=-1,
            shift_interval=n_query,
            block_len=n_query,
            num_blocks=num_blocks,
        )

        mask_k = torch.nn.functional.pad(
            mask, (pad_len_left_k, pad_len_right_k), value=0.0
        )
        mask_k = convert_to_blocks_1d(
            x=mask_k,
            dim=-1,
            shift_interval=n_query,
            block_len=n_key,
            num_blocks=num_blocks,
        )

        mask_bias = mask_q[..., None] * mask_k[..., None, :]

        # [*, 1, 1, N]
        mask_bias = (self.inf * (mask_bias - 1))[..., None, :, :]
        biases = [mask_bias]

        z = torch.nn.functional.pad(
            z, (0, 0, pad_len_left_k, pad_len_right_k, 0, pad_len_right_q), value=0.0
        )
        z = convert_to_blocks_2d(
            z,
            dims=[-3, -2],
            shift_interval=n_query,
            block_lens=[n_query, n_key],
            num_blocks=num_blocks,
        )

        # [*, N, N, C_z]
        z = self.layer_norm_z(z)

        # [*, N, N, no_heads]
        z = self.linear_z(z)

        # [*, no_heads, N, N]
        z = permute_final_dims(z, [2, 0, 1])

        biases.append(z)

        return a_query, a_key, biases

    def _prep_bias(
        self,
        a: torch.Tensor,
        z: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Args:
            a:
                [*, N, C_token] Token or atom-level embedding
            z:
                [*, N, N, C_z] Pair embedding
            beta:
                [*, N, N] Neighborhood mask
            mask:
                [*, N] Mask for token or atom-level embedding

        Returns:
            List of bias terms. Includes the pair bias and attention mask.
        """
        if mask is None:
            # [*, N]
            mask = a.new_ones(
                a.shape[:-1],
            )

        # [*, 1, 1, N]
        mask_bias = (self.inf * (mask - 1))[..., None, None, :]
        biases = [mask_bias]

        # [*, N, N, C_z]
        z = self.layer_norm_z(z)

        # [*, N, N, no_heads]
        z = self.linear_z(z)

        # [*, no_heads, N, N]
        z = permute_final_dims(z, [2, 0, 1])

        if beta is not None:
            z = z + beta.unsqueeze(-3)

        biases.append(z)

        return biases

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        layout: Optional[torch.Tensor] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            a:
                [*, N, C_q] Token or atom-level embedding
            z:
                [*, N, N, C_z] Pair embedding
            s:
                [*, N, C_s] Single embedding. Used in AdaLN if use_ada_layer_norm is
                True
            beta:
                [*, N, N] Neighborhood mask. Used in Sequence-local atom
                attention for rectangular blocks along the diagonal.
            mask:
                [*, N] Mask for token or atom-level embedding
            layout:
                [N / block_size, N / block_size] Layout config for block sparse
                attention. Dictates which sections of the attention matrix
                to compute.
            use_memory_efficient_kernel:
                Whether to use memory efficient kernel
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_lma:
                Whether to use LMA
        Returns
            [*, N, C_q] attention updated token or atom-level embedding
        """
        a = self.layer_norm_a(a, s) if self.use_ada_layer_norm else self.layer_norm_a(a)

        # TODO: Make this less awkward, DS kernel has strict shape asserts
        #  and expects the mask to be tiled to the correct shape
        reshape_mask = all(
            [
                not self.use_block_sparse_attn,
                use_deepspeed_evo_attention,
                a.shape[1] != mask.shape[1],
            ]
        )
        if reshape_mask:
            mask = mask.tile((1, a.shape[1], 1))

        if self.use_block_sparse_attn:
            biases = self._prep_sparse_bias(a=a, z=z, layout=layout, mask=mask)
            a = self.mha(q_x=a, kv_x=a, biases=biases, layout=layout)
        elif beta is not None:
            orig_shape = a.shape
            a_q, a_k, biases = self._prep_all_inputs(a=a, z=z, mask=mask)

            a = self.mha(
                q_x=a_q,
                kv_x=a_k,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
            )
            a = torch.cat(torch.unbind(a, dim=-3), dim=-2)[..., : orig_shape[-2], :]
        else:
            biases = self._prep_bias(a=a, z=z, beta=beta, mask=mask)

            # TODO: Make this less awkward, DS kernel has strict shape asserts
            #  and expects batch and seq dims to exist
            #  Current reshape function only expects missing batch dim
            if use_deepspeed_evo_attention:
                a = a.unsqueeze(1)
                biases = [b.unsqueeze(1) for b in biases]

            # Do we support all the memory efficient kernel types?
            a = self.mha(
                q_x=a,
                kv_x=a,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
            )

            if use_deepspeed_evo_attention:
                a = a.squeeze(1)

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a
