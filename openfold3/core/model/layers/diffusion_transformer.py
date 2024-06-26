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

"""Diffusion transformer block and stack."""

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from .attention_pair_bias import AttentionPairBias
from .transition import ConditionedTransitionBlock


class DiffusionTransformerBlock(nn.Module):
    """Diffusion transformer block.

    Implements AF3 Algorithm 23.
    """
    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        n_transition: int,
        inf: float = 1e9
    ):
        """
        Args:
            c_s:
                Single activation channel dimension
            c_z:
                Pair activation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            n_transition:
                Dimension multiplication factor used in transition layer
            inf:
                Large constant used to create mask for attention logits
        """
        super(DiffusionTransformerBlock, self).__init__()

        self.attention_pair_bias = AttentionPairBias(c_q=c_a,
                                                     c_k=c_a,
                                                     c_v=c_a,
                                                     c_s=c_s,
                                                     c_z=c_z,
                                                     c_hidden=c_hidden,
                                                     no_heads=no_heads,
                                                     use_ada_layer_norm=True,
                                                     gating=True,
                                                     inf=inf)

        self.conditioned_transition = ConditionedTransitionBlock(c_a=c_a, c_s=c_s, n=n_transition)

    def forward(self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        beta: Optional[torch.Tensor],
        mask: torch.Tensor,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:
                [*, N_res, C_token] Token-level embedding
            s:
                [*, N_res, C_s] Single embedding
            z:
                [*, N_res, N_res, C_z] Pair embedding
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
            _mask_trans:
                Whether to mask the output of the transition layer
        """
        b = self.attention_pair_bias(a=a, z=z, s=s, beta=beta, mask=mask,
                                     use_memory_efficient_kernel=use_memory_efficient_kernel,
                                     use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                     use_lma=use_lma)

        trans_mask = mask if _mask_trans else None
        a = b + self.conditioned_transition(a=a, s=s, mask=trans_mask)

        return a


class DiffusionTransformer(nn.Module):
    """Diffusion transformer stack.

    Implements AF3 Algorithm 23.
    """
    def __init__(
        self,
        c_a: int,
        c_s: int,
        c_z: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        inf: float,
    ):
        """
        Args:
            c_a:
                Token activation channel dimension
            c_s:
                Single activation channel dimension
            c_z:
                Pair activation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            n_transition:
                Dimension multiplication factor used in transition layer
            inf:
                Large constant used to create mask for attention logits
        """
        super(DiffusionTransformer, self).__init__()

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                c_a=c_a,
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                n_transition=n_transition,
                inf=inf
            )
            for _ in range(no_blocks)])

    def forward(self,
                a: torch.Tensor,
                s: torch.Tensor,
                z: Optional[torch.Tensor],
                beta: Optional[torch.Tensor],
                mask: torch.Tensor,
                use_memory_efficient_kernel: bool = False,
                use_deepspeed_evo_attention: bool = False,
                use_lma: bool = False,
                _mask_trans: bool = True
                ) -> torch.Tensor:
        """
        Args:
            a:
                [*, N_res, C_token] Token-level embedding
            s:
                [*, N_res, C_s] Single embedding
            z:
                [*, N_res, N_res, C_z] Pair embedding
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
            _mask_trans:
                Whether to mask the output of the transition layer
        """
        # Do we need all the fancy checkpoint blocks from evoformer?
        blocks = [
            partial(
                b,
                s=s,
                z=z,
                beta=beta,
                mask=mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        for b in blocks:
            a = b(a)

        return a
