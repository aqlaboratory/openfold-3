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

"""Triangle attention layers."""

from functools import partial, partialmethod

import torch
import torch.nn as nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.primitives import Attention, LayerNorm, Linear
from openfold3.core.utils.chunk_utils import chunk_layer
from openfold3.core.utils.tensor_utils import (
    permute_final_dims,
)


class TriangleAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_hidden,
        no_heads,
        starting=True,
        inf=1e9,
        linear_init_params=lin_init.tri_att_init,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_z = Linear(c_in, self.no_heads, **linear_init_params.linear_z)

        self.mha = Attention(
            c_q=self.c_in,
            c_k=self.c_in,
            c_v=self.c_in,
            c_hidden=self.c_hidden,
            no_heads=self.no_heads,
            linear_init_params=linear_init_params.mha,
        )

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: list[torch.Tensor],
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(
                self.mha,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_lma=use_lma,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """

        if use_cueq_triangle_kernels and use_deepspeed_evo_attention:
            # VS: Upstream in the Pairformer, its valid for both to be
            # true. This would trigger an error in Attention, so we
            # assume here if both are true, we just use cueq
            raise ValueError("both deepspeed and cueq enabled!!")
        if mask is None:
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear_z(x), (2, 0, 1))

        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                inplace_safe=inplace_safe,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                biases=biases,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


# Implements AF2 Algorithm 13 / AF3 Algorithm 14
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements AF2 Algorithm 14 / AF3 Algorithm 15.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
