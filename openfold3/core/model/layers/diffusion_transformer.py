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
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.utils.checkpointing import checkpoint_blocks

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
        use_ada_layer_norm: bool,
        n_query: Optional[int],
        n_key: Optional[int],
        inf: float = 1e9,
        linear_init_params: ConfigDict = lin_init.diffusion_transformer_init,
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
            use_ada_layer_norm:
                Whether to apply AdaLN-Zero conditioning
            n_query:
                Number of queries (block height). If provided, inputs are split into
                q/k blocks of n_query and n_key prior to attention.
            n_key:
                Number of keys (block width). If provided, inputs are split into
                q/k blocks of n_query and n_key prior to attention.
            inf:
                Large constant used to create mask for attention logits
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.attention_pair_bias = AttentionPairBias(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden,
            no_heads=no_heads,
            use_ada_layer_norm=use_ada_layer_norm,
            n_query=n_query,
            n_key=n_key,
            gating=True,
            inf=inf,
            linear_init_params=linear_init_params.att_pair_bias,
        )

        self.conditioned_transition = ConditionedTransitionBlock(
            c_a=c_a,
            c_s=c_s,
            n=n_transition,
            linear_init_params=linear_init_params.cond_transition,
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:
                [*, N, C_token] Token-level embedding
            s:
                [*, N, C_s] Single embedding
            z:
                [*, N, N, C_z] Pair embedding
            mask:
                [*, N] Mask for token-level embedding
            chunk_size:
                Inference-time subbatch size
            use_memory_efficient_kernel:
                Whether to use memory efficient kernel
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_lma:
                Whether to use LMA
            use_high_precision_attention:
                Whether to run attention in high precision
            _mask_trans:
                Whether to mask the output of the transition layer
        """
        # Note: Differs from SI, residual connection added.
        a = a + self.attention_pair_bias(
            a=a,
            z=z,
            s=s,
            mask=mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            use_high_precision_attention=use_high_precision_attention,
        )

        trans_mask = mask if _mask_trans else None

        # Note: Differs from SI, updated a_i from AttentionPairBias is used
        # instead of previous a_i.
        a = a + self.conditioned_transition(
            a=a, s=s, mask=trans_mask, chunk_size=chunk_size
        )

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
        use_ada_layer_norm: bool,
        n_query: Optional[int],
        n_key: Optional[int],
        inf: float,
        blocks_per_ckpt: Optional[int] = None,
        linear_init_params: ConfigDict = lin_init.diffusion_transformer_init,
        use_reentrant: Optional[bool] = None,
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
            use_ada_layer_norm:
                Whether to apply AdaLN-Zero conditioning
            n_query:
                Number of queries (block height). If provided, inputs are split into
                q/k blocks of n_query and n_key prior to attention.
            n_key:
                Number of keys (block width). If provided, inputs are split into
                q/k blocks of n_query and n_key prior to attention.
            blocks_per_ckpt:
                Number of blocks per checkpoint. If set, checkpointing will
                be used to save memory.
            inf:
                Large constant used to create mask for attention logits
            linear_init_params:
                Linear layer initialization parameters
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
        """
        super().__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_reentrant = use_reentrant

        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    c_a=c_a,
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    no_heads=no_heads,
                    n_transition=n_transition,
                    use_ada_layer_norm=use_ada_layer_norm,
                    n_query=n_query,
                    n_key=n_key,
                    inf=inf,
                    linear_init_params=linear_init_params,
                )
                for _ in range(no_blocks)
            ]
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            a:
                [*, N, C_token] Token-level embedding
            s:
                [*, N, C_s] Single embedding
            z:
                [*, N, N, C_z] Pair embedding
            mask:
                [*, N] Mask for token-level embedding
            chunk_size:
                Inference-time subbatch size
            use_memory_efficient_kernel:
                Whether to use memory efficient kernel
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_lma:
                Whether to use LMA
            use_high_precision_attention:
                Whether to run attention in high precision
            _mask_trans:
                Whether to mask the output of the transition layer
        """
        # Do we need all the fancy checkpoint blocks from evoformer?
        blocks = [
            partial(
                b,
                s=s,
                z=z,
                mask=mask,
                chunk_size=chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                use_high_precision_attention=use_high_precision_attention,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        (a,) = checkpoint_blocks(
            blocks,
            args=(a,),
            blocks_per_ckpt=blocks_per_ckpt,
            use_reentrant=self.use_reentrant,
        )

        return a
