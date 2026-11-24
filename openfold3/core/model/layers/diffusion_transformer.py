# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
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
from typing import cast

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.layers.attention_pair_bias import (
    CrossAttentionPairBias,
    DiffusionAttentionPairBias,
    build_mask_bias,
    mask_bias_from_mask,
)
from openfold3.core.model.layers.transition import ConditionedTransitionBlock
from openfold3.core.model.primitives import LayerNorm
from openfold3.core.utils.checkpointing import checkpoint_blocks


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
        n_query: int | None,
        n_key: int | None,
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
        self.use_cross_attention = n_query is not None

        if not self.use_cross_attention:
            self.attention_pair_bias = DiffusionAttentionPairBias(
                c_q=c_a,
                c_k=c_a,
                c_v=c_a,
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                gating=True,
                inf=inf,
                linear_init_params=linear_init_params.att_pair_bias,
            )
        else:
            self.attention_pair_bias = CrossAttentionPairBias(
                c_q=c_a,
                c_k=c_a,
                c_v=c_a,
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
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
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
        mask_bias: torch.Tensor | None = None,
        pair_bias: torch.Tensor | None = None,
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
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_cueq_triangle_kernels:
                Whether to use cuEq triangle kernels
            use_triton_triangle_kernels:
                Whether to use Triton triangle attention kernel
            use_lma:
                Whether to use LMA
            use_high_precision_attention:
                Whether to run attention in high precision
            _mask_trans:
                Whether to mask the output of the transition layer
            mask_bias:
                [*, 1, 1, N] Mask bias shared by all blocks in the stack
            pair_bias:
                [*, no_heads, N, N] Pair bias precomputed for this block.
                When set, the token mask is already inside it, so attention
                is called with no mask.
        """
        # Note: Differs from SI, residual connection added.

        if not self.use_cross_attention:
            a = a + self.attention_pair_bias(
                a=a,
                z=z,
                s=s,
                mask=None if pair_bias is not None else mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                use_high_precision_attention=use_high_precision_attention,
                mask_bias=mask_bias,
                pair_bias=pair_bias,
            )
        else:
            a = a + self.attention_pair_bias(
                a=a,
                z=z,
                s=s,
                mask=mask,
                use_high_precision_attention=use_high_precision_attention,
            )

        trans_mask = mask if _mask_trans else None

        # Note: Differs from SI, updated a_i from AttentionPairBias is used
        # instead of previous a_i.
        a = a + self.conditioned_transition(
            a=a,
            s=s,
            mask=trans_mask,
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
        n_query: int | None,
        n_key: int | None,
        inf: float,
        blocks_per_ckpt: int | None = None,
        linear_init_params: ConfigDict = lin_init.diffusion_transformer_init,
        use_reentrant: bool | None = None,
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
        self.inf = inf

        self.layer_norm_z = LayerNorm(c_z, create_offset=False)

        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    c_a=c_a,
                    c_s=c_s,
                    c_z=c_z,
                    c_hidden=c_hidden,
                    no_heads=no_heads,
                    n_transition=n_transition,
                    n_query=n_query,
                    n_key=n_key,
                    inf=inf,
                    linear_init_params=linear_init_params,
                )
                for _ in range(no_blocks)
            ]
        )

    def prepare_pair_bias_cache(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """Project the pair bias of every block once per rollout.

        Args:
            z:
                [*, N, N, C_z] Normalized pair embedding
            mask:
                [*, N] Token mask. When given, its additive bias is folded in
                here so ``forward`` passes no separate mask into attention.

        Returns:
            One [*, no_heads, N, N] pair bias per block, in block order.

        Raises:
            ValueError: A block uses cross attention. That bias is built from
                sequence-local blocks inside the block, so it cannot be cached
                from this token pair.
        """
        if any(b.use_cross_attention for b in self.blocks):
            raise ValueError(
                "pair bias cache requires self-attention blocks; "
                "cross-attention projects its bias inside the block"
            )

        mask_bias = None
        if mask is not None:
            # Built directly from the mask so it keeps the mask's own batch
            # dims and broadcasts over the rollout sample dim later. The token
            # mask carries its batch dims already, so there is no ``a`` to
            # consult for the leading singleton count.
            mask_bias = mask_bias_from_mask(mask=mask, inf=self.inf)

        cache: list[torch.Tensor] = []
        for module in self.blocks:
            # ModuleList yields nn.Module, which hides this block's methods.
            block = cast(DiffusionTransformerBlock, module)
            bias = block.attention_pair_bias.project_cached_pair_bias(z)
            cache.append(bias if mask_bias is None else bias + mask_bias)
        return cache

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
        pair_bias_cache: list[torch.Tensor] | None = None,
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
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_cueq_triangle_kernels:
                Whether to use cuEq triangle kernels
            use_triton_triangle_kernels:
                Whether to use Triton triangle attention kernel
            use_lma:
                Whether to use LMA
            use_high_precision_attention:
                Whether to run attention in high precision
            _mask_trans:
                Whether to mask the output of the transition layer
            pair_bias_cache:
                Output of ``prepare_pair_bias_cache``. Only valid for inference
                at a fixed ``z``; training omits it and every block projects
                its own bias. When set, each entry already includes the token
                mask, so no separate mask bias is built.
        """
        # Single layer norm for atom attention enc/dec diffusion transformer.
        # A pair bias cache was projected from the normalized pair, so
        # normalizing again would only repeat work every rollout step.
        if pair_bias_cache is None:
            z = self.layer_norm_z(z)
            # The mask bias is block-invariant, so build it once for the whole
            # stack. It is None when there is no mask, and then no bias is
            # added.
            mask_bias = build_mask_bias(
                a=a,
                mask=mask,
                inf=self.inf,
                tile_batch=(
                    use_deepspeed_evo_attention
                    or use_cueq_triangle_kernels
                    or use_triton_triangle_kernels
                ),
            )
        else:
            mask_bias = None

        blocks = [
            partial(
                b,
                s=s,
                z=z,
                mask=mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                use_high_precision_attention=use_high_precision_attention,
                _mask_trans=_mask_trans,
                mask_bias=mask_bias,
                pair_bias=(pair_bias_cache[i] if pair_bias_cache is not None else None),
            )
            for i, b in enumerate(self.blocks)
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
