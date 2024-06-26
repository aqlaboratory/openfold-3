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

"""Template pointwise attention layer."""

from functools import partial
from typing import List, Optional

import torch
from torch import nn as nn

from openfold3.core.model.primitives import Attention
from openfold3.core.utils.chunk_utils import chunk_layer
from openfold3.core.utils.tensor_utils import permute_final_dims


class TemplatePointwiseAttention(nn.Module):
    """Cross attention between pair and template representations.

    Implements AF2 Algorithm 17.
    """

    def __init__(self, c_t, c_z, c_hidden, no_heads, inf, **kwargs):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            inf:
                Large constant for masking
        """
        super(TemplatePointwiseAttention, self).__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.mha = Attention(
            self.c_z,
            self.c_t,
            self.c_t,
            self.c_hidden,
            self.no_heads,
            gating=False,
        )

    def _chunk(self,
               z: torch.Tensor,
               t: torch.Tensor,
               biases: List[torch.Tensor],
               chunk_size: int,
               use_lma: bool = False,
               ) -> torch.Tensor:
        mha_inputs = {
            "q_x": z,
            "kv_x": t,
            "biases": biases,
        }
        return chunk_layer(
            partial(self.mha, use_lma=use_lma),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )

    def forward(self,
                t: torch.Tensor,
                z: torch.Tensor,
                template_mask: Optional[torch.Tensor] = None,
                # This module suffers greatly from a small chunk size
                chunk_size: Optional[int] = 256,
                use_lma: bool = False,
                ) -> torch.Tensor:
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] Template embedding
            z:
                [*, N_res, N_res, C_t] Pair embedding
            template_mask:
                [*, N_templ] Template mask
            chunk_size:
                Chunk size for large inputs
            use_lma:
                Whether to use low-memory attention during inference
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])

        bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)

        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)

        # [*, N_res, N_res, N_temp, C_t]
        t = permute_final_dims(t, (1, 2, 0, 3))

        # [*, N_res, N_res, 1, C_z]
        biases = [bias]
        if chunk_size is not None and not self.training:
            z = self._chunk(z, t, biases, chunk_size, use_lma=use_lma)
        else:
            z = self.mha(q_x=z, kv_x=t, biases=biases, use_lma=use_lma)

        # [*, N_res, N_res, C_z]
        z = z.squeeze(-2)

        return z
