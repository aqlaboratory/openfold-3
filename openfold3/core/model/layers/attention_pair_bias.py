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

from typing import List, Optional

import torch
from ml_collections import ConfigDict
from torch import nn

from openfold3.core.model.primitives import (
    AdaLN,
    Attention,
    LayerNorm,
    Linear,
)
from openfold3.core.utils.tensor_utils import permute_final_dims


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
        linear_init_params: ConfigDict,
        use_ada_layer_norm: bool = False,
        gating: bool = True,
        inf=1e9,
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
            linear_init_params:
                Linear layer initialization parameters
            use_ada_layer_norm:
                Whether to apply AdaLN-Zero conditioning
            gating:
                Whether the output should be gated using query data
            inf:
                Large constant used to create mask for attention logits
        """
        super().__init__()

        self.use_ada_layer_norm = use_ada_layer_norm
        self.c_q = c_q
        self.c_s = c_s
        self.c_z = c_z
        self.inf = inf

        if self.use_ada_layer_norm:
            self.layer_norm_a = AdaLN(
                c_a=self.c_q, c_s=self.c_s, **linear_init_params.ada_ln
            )
            self.linear_ada_out = Linear(
                self.c_s, self.c_q, **linear_init_params.linear_ada_out
            )
        else:
            self.layer_norm_a = LayerNorm(c_in=self.c_q)

        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(self.c_z, no_heads, **linear_init_params.linear_z)

        self.mha = Attention(
            c_q=c_q,
            c_k=c_k,
            c_v=c_v,
            c_hidden=c_hidden,
            no_heads=no_heads,
            linear_init_params=linear_init_params.mha,
            gating=gating,
        )

        self.sigmoid = nn.Sigmoid()

    def _prep_bias(
        self,
        a: torch.Tensor,
        z: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
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

        biases = self._prep_bias(a, z, beta, mask)

        # Do we support all the memory efficient kernel types?
        a = self.mha(
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
