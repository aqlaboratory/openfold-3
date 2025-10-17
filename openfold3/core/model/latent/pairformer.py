# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""PairFormer block and stack."""

from functools import partial

import torch
from ml_collections import ConfigDict
from torch import nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.latent.base_blocks import PairBlock
from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias
from openfold3.core.model.layers.transition import SwiGLUTransition
from openfold3.core.utils.checkpointing import checkpoint_blocks
from openfold3.core.utils.chunk_utils import (
    CUEQ_MAX_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
    ChunkSizeTuner,
)
from openfold3.core.utils.tensor_utils import add


class PairFormerBlock(nn.Module):
    """Implements block of AF3 Algorithm 17."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        transition_type: str,
        transition_n: int,
        pair_dropout: float,
        fuse_projection_weights: bool,
        inf: float,
        linear_init_params: ConfigDict = lin_init.pairformer_init,
    ):
        """
        Args:
            c_s:
                Single embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden_pair_bias:
                Hidden channel dimension for AttentionPairBias module
            no_heads_pair_bias:
                Number of heads for AttentionPairBias module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            no_heads_pair:
                Number of heads in triangular attention
            transition_type:
                String 'relu' or 'swiglu' to determine activation for the transition
                function
            transition_n:
                Factor by which to multiply c_z to obtain the transition layer
                hidden dimension
            pair_dropout:
                Dropout rate used throughout the stack
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            inf:
                Large constant used for masking
            linear_init_params:
                Parameters for initializing linear layers
        """
        super().__init__()

        self.pair_stack = PairBlock(
            c_z=c_z,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_pair=no_heads_pair,
            transition_type=transition_type,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
            linear_init_params=linear_init_params.pair_block,
        )

        self.attn_pair_bias = AttentionPairBias(
            c_q=c_s,
            c_k=c_s,
            c_v=c_s,
            c_s=c_s,
            c_z=c_z,
            c_hidden=c_hidden_pair_bias,
            no_heads=no_heads_pair_bias,
            use_ada_layer_norm=False,
            gating=True,
            inf=inf,
            linear_init_params=linear_init_params.att_pair_bias,
        )

        self.single_transition = SwiGLUTransition(
            c_in=c_s,
            n=transition_n,
            linear_init_params=linear_init_params.transition,
        )

    def forward(
        self,
        s: torch.Tensor | None,
        z: torch.Tensor | None,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, N_token, C_s] Single embedding
            z:
                [*, N_token, N_token, C_z] Pair embedding
            single_mask:
                [*, N_token] Single mask
            pair_mask:
                [*, N_token, N_token] Pair mask
            chunk_size:
                Inference-time subbatch size. Acts as a minimum if
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_cueq_triangle_kernels:
                Whether to use cuEquivariance triangle multiplicative
                update kernel and attention kernel. When both this and
                use_deepspeed_evo_attention are True, the cuEquivariance
                kernel is only used for triangle attention
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
            _mask_trans:
                Whether to mask the output of the transition layers
            _attn_chunk_size:
                Inference-time subbatch size for attention. If None, uses chunk
        Returns:
            s:
                [*, N_token, C_s] Single embedding
            z:
                [*, N_token, N_token, C_z] Pair embedding
        """

        single_trans_mask = single_mask if _mask_trans else None

        z = self.pair_stack(
            z=z,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
        )

        s = add(
            s,
            self.attn_pair_bias(
                a=s,
                z=z,
                s=None,
                mask=single_mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_lma=use_lma,
            ),
            inplace=inplace_safe,
        )

        s = add(
            s,
            self.single_transition(
                s,
                mask=single_trans_mask,
                chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )

        return s, z


# TODO: Make this inherit from MSAStack/CheckpointStack
class PairFormerStack(nn.Module):
    """Implements AF3 Algorithm 17."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden_pair_bias: int,
        no_heads_pair_bias: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_type: str,
        transition_n: int,
        pair_dropout: float,
        fuse_projection_weights: bool,
        blocks_per_ckpt: int | None,
        inf: float,
        linear_init_params: ConfigDict = lin_init.pairformer_init,
        use_reentrant: bool | None = None,
        clear_cache_between_blocks: bool = False,
        tune_chunk_size: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden_pair_bias:
                Hidden channel dimension for AttentionPairBias module
            no_heads_pair_bias:
                Number of heads for AttentionPairBias module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            no_heads_pair:
                Number of heads in triangular attention
            no_blocks:
                Number of PairFormer blocks
            transition_type:
                String 'relu' or 'swiglu' to determine activation for the transition
                function
            transition_n:
                Factor by which to multiply c_z to obtain the transition layer
                hidden dimension
            pair_dropout:
                Dropout rate used throughout the stack
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            blocks_per_ckpt:
                Number of blocks per activation checkpoint. None disables
                activation checkpointing
            inf:
                Large constant used for masking
            linear_init_params:
                Parameters for initializing linear layers
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super().__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_reentrant = use_reentrant
        self.clear_cache_between_blocks = clear_cache_between_blocks

        self.blocks = nn.ModuleList()

        for _ in range(no_blocks):
            block = PairFormerBlock(
                c_s=c_s,
                c_z=c_z,
                c_hidden_pair_bias=c_hidden_pair_bias,
                no_heads_pair_bias=no_heads_pair_bias,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair,
                transition_type=transition_type,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
                linear_init_params=linear_init_params,
            )
            self.blocks.append(block)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def _prep_blocks(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor | None,
        pair_mask: torch.Tensor | None,
        chunk_size: int | None,
        use_deepspeed_evo_attention: bool,
        use_cueq_triangle_kernels: bool,
        use_lma: bool,
        inplace_safe: bool,
        _mask_trans: bool,
    ):
        """
        Partially initialize the PairFormer blocks. Optionally add cache clearing
        between blocks and chunk size tuning. Arguments are the same as forward
        function.

        Returns:
            Partially initialized PairFormer blocks.
        """
        blocks = [
            partial(
                b,
                single_mask=single_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if self.clear_cache_between_blocks:

            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            max_chunk_size = (
                CUEQ_MAX_CHUNK_SIZE
                if use_cueq_triangle_kernels
                else DEFAULT_MAX_CHUNK_SIZE
            )
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                # We don't want to write in-place during chunk tuning runs
                args=(
                    s.clone(),
                    z.clone(),
                ),
                min_chunk_size=chunk_size,
                max_chunk_size=max_chunk_size,
            )
            attn_chunk = (
                tuned_chunk_size
                if use_cueq_triangle_kernels
                else (tuned_chunk_size // 4)
            )
            blocks = [
                partial(
                    b,
                    chunk_size=tuned_chunk_size,
                    # A temporary measure to address torch's occasional
                    # inability to allocate large tensors
                    _attn_chunk_size=max(chunk_size, attn_chunk),
                )
                for b in blocks
            ]

        return blocks

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, N_token, C_s] Single embedding
            z:
                [*, N_token, N_token, C_z] Pair embedding
            single_mask:
                [*, N_token] Single mask
            pair_mask:
                [*, N_token, N_token] Pair mask
            chunk_size:
                Inference-time subbatch size. Acts as a minimum if
                self.tune_chunk_size is True
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
            _mask_trans:
                Whether to mask the output of the transition layers
        Returns:
            s:
                [*, N_token, C_s] Single embedding
            z:
                [*, N_token, N_token, C_z] Pair embedding
        """
        blocks = self._prep_blocks(
            s=s,
            z=z,
            single_mask=single_mask,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
            use_reentrant=self.use_reentrant,
        )

        return s, z
