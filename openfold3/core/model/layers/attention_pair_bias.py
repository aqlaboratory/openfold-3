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

from typing import Optional

import torch
from ml_collections import ConfigDict
from torch import nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.primitives import (
    AdaLN,
    Attention,
    LayerNorm,
    Linear,
)
from openfold3.core.utils.atom_attention_block_utils import convert_single_rep_to_blocks
from openfold3.core.utils.tensor_utils import permute_final_dims


class AttentionPairBias(nn.Module):
    """Attention layer with pair bias.

    Implements AF3 Algorithm 24 for the trunk, where no sequence local
    or adaptive layernorm are needed by default.
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
        gating: bool = True,
        inf=1e9,
        linear_init_params: Optional[ConfigDict] = None,
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
            gating:
                Whether the output should be gated using query data
            inf:
                Large constant used to create mask for attention logits
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf

        self.use_ada_layer_norm = use_ada_layer_norm

        if linear_init_params is None:
            linear_init_params = (
                lin_init.diffusion_att_pair_bias_init
                if self.use_ada_layer_norm
                else lin_init.att_pair_bias_init
            )

        if self.use_ada_layer_norm:
            self.layer_norm_a = AdaLN(
                c_a=self.c_q, c_s=self.c_s, linear_init_params=linear_init_params.ada_ln
            )

            self.linear_ada_out = Linear(
                self.c_s, self.c_q, **linear_init_params.linear_ada_out
            )
        else:
            self.layer_norm_a = LayerNorm(c_in=self.c_q)

        self.layer_norm_z = LayerNorm(self.c_z, **linear_init_params.layer_norm_z)
        self.linear_z = Linear(self.c_z, no_heads, **linear_init_params.linear_z)

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

    def _prep_bias(
        self,
        a: torch.Tensor,
        z: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Args:
            a:
                [*, N, C_token] Token or atom-level embedding
            z:
                [*, N, N, C_z] Pair embedding
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

        biases.append(z)

        return biases

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
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
            mask:
                [*, N] Mask for token or atom-level embedding
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_lma:
                Whether to use LMA
            use_high_precision_attention:
                Whether to run attention in high precision
        Returns
            [*, N, C_q] attention updated token or atom-level embedding
        """
        a = self.layer_norm_a(a, s) if self.use_ada_layer_norm else self.layer_norm_a(a)

        # TODO: Make this less awkward, DS kernel has strict shape asserts
        #  and expects the mask to be tiled to the correct shape
        if use_deepspeed_evo_attention and a.shape[1] != mask.shape[1]:
            mask = mask.expand((-1, a.shape[1], -1))

        biases = self._prep_bias(a=a, z=z, mask=mask)

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
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_lma=use_lma,
            use_high_precision=use_high_precision_attention,
        )

        if use_deepspeed_evo_attention:
            a = a.squeeze(1)

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a


class CrossAttentionPairBias(nn.Module):
    """Attention layer with pair bias and neighborhood mask.
    Unlike AttentionPairBias, inputs are blocked for sequence-local attention
    and AdaLN is applied by default.

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
        n_query: Optional[int] = None,
        n_key: Optional[int] = None,
        gating: bool = True,
        inf=1e9,
        linear_init_params: Optional[ConfigDict] = None,
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
            n_query:
                Number of queries (block height). If provided, inputs are split into
                q/k blocks of n_query and n_key prior to attention.
            n_key:
                Number of keys (block width). If provided, inputs are split into
                q/k blocks of n_query and n_key prior to attention.
            gating:
                Whether the output should be gated using query data
            inf:
                Large constant used to create mask for attention logits
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf

        self.use_ada_layer_norm = use_ada_layer_norm
        self.n_query = n_query
        self.n_key = n_key

        if linear_init_params is None:
            linear_init_params = (
                lin_init.diffusion_att_pair_bias_init
                if self.use_ada_layer_norm
                else lin_init.att_pair_bias_init
            )

        if self.use_ada_layer_norm:
            self.layer_norm_a_q = AdaLN(
                c_a=self.c_q, c_s=self.c_s, linear_init_params=linear_init_params.ada_ln
            )
            self.layer_norm_a_k = AdaLN(
                c_a=self.c_q, c_s=self.c_s, linear_init_params=linear_init_params.ada_ln
            )

            self.linear_ada_out = Linear(
                self.c_s, self.c_q, **linear_init_params.linear_ada_out
            )
        else:
            self.layer_norm_a_q = LayerNorm(c_in=self.c_q)
            self.layer_norm_a_k = LayerNorm(c_in=self.c_q)

        self.layer_norm_z = LayerNorm(self.c_z, **linear_init_params.layer_norm_z)
        self.linear_z = Linear(self.c_z, no_heads, **linear_init_params.linear_z)

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

    def _prep_block_inputs(
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

        a_query, a_key, mask = convert_single_rep_to_blocks(
            ql=a, n_query=self.n_query, n_key=self.n_key, atom_mask=mask
        )

        # [*, 1, 1, N]
        mask_bias = (self.inf * (mask - 1))[..., None, :, :]
        biases = [mask_bias]

        # [*, N, N, C_z]
        z = self.layer_norm_z(z)

        # [*, N, N, no_heads]
        z = self.linear_z(z)

        # [*, no_heads, N, N]
        z = permute_final_dims(z, [2, 0, 1])

        biases.append(z)

        return a_query, a_key, biases

    def forward(
        self,
        a: torch.Tensor,
        z: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_high_precision_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
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
            mask:
                [*, N] Mask for token or atom-level embedding
            use_high_precision_attention:
                Whether to run attention in high precision
        Returns
            [*, N, C_q] attention updated token or atom-level embedding
        """
        batch_dims = a.shape[:-2]
        n_atom, n_dim = a.shape[-2:]

        a_q, a_k, biases = self._prep_block_inputs(a=a, z=z, mask=mask)

        if self.use_ada_layer_norm:
            s_q, s_k, _ = convert_single_rep_to_blocks(
                ql=s, n_query=self.n_query, n_key=self.n_key
            )
            a_q = self.layer_norm_a_q(a_q, s_q)
            a_k = self.layer_norm_a_k(a_k, s_k)
        else:
            a_q = self.layer_norm_a_q(a_q)
            a_k = self.layer_norm_a_k(a_k)

        a = self.mha(
            q_x=a_q,
            kv_x=a_k,
            biases=biases,
            use_high_precision=use_high_precision_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
        )

        # Convert back to unpadded and flattened atom representation
        # [*, N_blocks, N_query, c_atom] -> [*, N_atom, c_atom]
        a = a.reshape((*batch_dims, -1, n_dim))[..., :n_atom, :]

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a
