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

from typing import Optional, List

import torch

from openfold3.core.model.primitives import (
    AdaLN,
    LayerNorm,
    Linear,
    Attention,
    DEFAULT_LMA_Q_CHUNK_SIZE,
    DEFAULT_LMA_KV_CHUNK_SIZE
)

from openfold3.core.utils.tensor_utils import permute_final_dims


class AttentionPairBias(Attention):
    """Attention layer with pair bias and neighborhood mask.

    Implements AF3 Algorithm 24.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf=1e9
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
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
        """
        super(AttentionPairBias, self).__init__(c_q=c_q, c_k=c_k, c_v=c_v, c_hidden=c_hidden,
                                                no_heads=no_heads, gating=gating)

        self.use_ada_layer_norm = use_ada_layer_norm
        self.c_z = c_z
        self.inf = inf

        if self.use_ada_layer_norm:
            self.layer_norm_a = AdaLN(c_in=c_q)
        else:
            self.layer_norm_a = LayerNorm(c_in=self.c_q)

        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(
            self.c_z, self.no_heads, bias=False, init="normal"
        )

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=True, init="glorot"
        )

        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, bias=False, init="final"
        )

        if self.use_ada_layer_norm:
            self.linear_ada_out = Linear(self.c_q, self.c_q, init="gating_ada_zero")

    def _prep_bias(
        self,
        a: torch.Tensor,
        z: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            a:
                [*, N_res, C_token] Token-level embedding
            z:
                [*, N_res, N_res, C_z] Pair embedding
            beta:
                [*, N_res, N_res] Neighborhood mask
            mask:
                [*, N_res] Mask for token-level embedding

        Returns:
            List of bias terms. Includes the pair bias and attention mask.
        """
        if mask is None:
            # [*, I, J]
            mask = a.new_ones(
                a.shape[:-1],
            )

        # [*, N_res, N_res]
        square_mask = mask[..., None] * mask[..., None, :]
        # [*, 1, N_res, N_res]
        mask_bias = (self.inf * (square_mask - 1))[..., None, :, :]
        biases = [mask_bias]

        # [*, N_res, N_res, C_z]
        z = self.layer_norm_z(z)

        # [*, N_res, N_res, no_heads]
        z = self.linear_z(z)

        # [*, no_heads, N_res, N_res]
        z = permute_final_dims(z, [2, 0, 1])

        # TODO: This is how it is written in the algorithm, but I need to actually select these indices
        # in the a, z, and mask terms to actually reduce the attention computation
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
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = DEFAULT_LMA_Q_CHUNK_SIZE,
        lma_kv_chunk_size: int = DEFAULT_LMA_KV_CHUNK_SIZE
    ) -> torch.Tensor:
        """
        Args:
            a:
                [*, N_res, C_token] Token-level embedding
            z:
                [*, N_res, N_res, C_z] Pair embedding
            s:
                [*, N_res, C_s] Single embedding. Used in AdaLN if use_ada_layer_norm is True
            beta:
                [*, N_res, N_res] Neighborhood mask. Used in Sequence-local atom attention
                for rectangular blocks along the diagonal.
            mask:
                [*, N_res] Mask for token-level embedding
            use_memory_efficient_kernel:
                Whether to use memory efficient kernel
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_lma:
                Whether to use LMA
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention updated token-level embedding
        """
        if self.use_ada_layer_norm:
            a = self.layer_norm_a(a, s)
        else:
            a = self.layer_norm_a(a)

        biases = self._prep_bias(a, z, beta, mask)

        # Do we support all the memory efficient kernel types?
        a = super(AttentionPairBias, self).forward(
            q_x=a,
            kv_x=a,
            biases=biases,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
        )

        if self.use_ada_layer_norm:
            a = self.sigmoid(self.linear_ada_out(s)) * a

        return a
