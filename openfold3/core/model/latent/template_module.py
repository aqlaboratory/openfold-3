# Copyright 2026 AlQuraishi Laboratory
# Copyright 2025 NVIDIA Corporation
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

"""Template embedding layers.

These modules embed templates into pair embeddings. Note that this includes the template
feature embedding functions in openfold3.core.model.feature_embedders.
"""

from functools import partial

import torch
from ml_collections import ConfigDict
from torch import nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.feature_embedders.template_embedders import (
    TemplatePairEmbedderAllAtom,
)
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.checkpointing import checkpoint_blocks, checkpoint_section
from openfold3.core.utils.chunk_utils import (
    CUEQ_MAX_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_SIZE,
    ChunkSizeTuner,
)
from openfold3.core.utils.tensor_utils import add

from .base_blocks import PairBlock


# TODO: Make arguments match PairBlock
class TemplatePairBlock(PairBlock):
    """Implements one block of AF2 Algorithm 16."""

    def __init__(
        self,
        c_t: int,
        c_hidden_tri_mul: int,
        c_hidden_tri_att: int,
        no_heads: int,
        transition_type: str,
        pair_transition_n: int,
        dropout_rate: float,
        tri_mul_first: bool,
        fuse_projection_weights: bool,
        ckpt_per_template: bool,
        inf: float,
        linear_init_params: ConfigDict = lin_init.pair_block_init,
        use_reentrant: bool | None = None,
        **kwargs,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_mul:
                Hidden dimension for triangular multiplication
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            no_heads:
                Number of heads in the attention mechanism
            transition_type:
                String 'relu' or 'swiglu' to determine activation for the transition
                function
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
            tri_mul_first:
                Whether to perform triangular multiplication before attention
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            ckpt_per_template:
                Whether to use activation checkpointing per template
            inf:
                Large constant used for masking
            linear_init_params:
                Configuration for linear initialization
        """
        super().__init__(
            c_z=c_t,
            c_hidden_mul=c_hidden_tri_mul,
            c_hidden_pair_att=c_hidden_tri_att,
            no_heads_pair=no_heads,
            transition_type=transition_type,
            transition_n=pair_transition_n,
            pair_dropout=dropout_rate,
            fuse_projection_weights=fuse_projection_weights,
            inf=inf,
            linear_init_params=linear_init_params,
        )

        self.tri_mul_first = tri_mul_first
        self.ckpt_per_template = ckpt_per_template
        self.use_reentrant = use_reentrant

    def _forward_single_template(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int | None,
        use_deepspeed_evo_attention: bool,
        use_cueq_triangle_kernels: bool,
        use_lma: bool,
        inplace_safe: bool,
        _mask_trans: bool,
        _attn_chunk_size: int | None,
    ):
        """
        Helper function to process exactly one template slice.
        """

        # t: [1, N, N, C]
        if self.tri_mul_first:
            t = self.tri_att_start_end(
                z=self.tri_mul_out_in(z=t, pair_mask=mask, inplace_safe=inplace_safe),
                _attn_chunk_size=_attn_chunk_size,
                pair_mask=mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
        else:
            t = self.tri_mul_out_in(
                z=self.tri_att_start_end(
                    z=t,
                    _attn_chunk_size=_attn_chunk_size,
                    pair_mask=mask,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                ),
                pair_mask=mask,
                inplace_safe=inplace_safe,
            )

        t = add(
            t,
            self.pair_transition(
                t,
                mask=mask if _mask_trans else None,
                chunk_size=chunk_size,
            ),
            inplace_safe,
        )

        return t

    def forward(
        self,
        t: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: int | None = None,
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] Template embedding
            mask:
                [*, N_templ, N_res, N_res] Template mask
            chunk_size:
                Inference-time subbatch size
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
            _attn_chunk_size:
                Inference-time subbatch size for attention. If None, uses chunk.

        Returns:
            [*, N_templ, N_res, N_res, C_t] Template embedding update
        """

        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        single_templates = [t_i.unsqueeze(-4) for t_i in torch.unbind(t, dim=-4)]
        single_templates_masks = [m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)]

        apply_ckpt = self.training and self.ckpt_per_template
        single_templ_fn = partial(
            self._forward_single_template,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
        )

        for i in range(len(single_templates)):
            t_in = single_templates[i]
            mask_in = single_templates_masks[i]

            # Make contiguous to avoid activation checkpointing on a views of tensors
            # If inplace_safe is true (inference), avoid making a copy of the tensor
            if not inplace_safe and apply_ckpt:
                t_in = t_in.contiguous()
                mask_in = mask_in.contiguous()

            t_out = checkpoint_section(
                single_templ_fn,
                args=(t_in, mask_in),
                apply_ckpt=apply_ckpt,
                use_reentrant=self.use_reentrant,
            )

            if inplace_safe:
                # t_in is the view into t at index i
                # Copy here to safely update t if t_out has any non-inplace updates
                t_in.copy_(t_out)
            else:
                single_templates[i] = t_out

        if not inplace_safe:
            t = torch.cat(single_templates, dim=-4)

        return t


class TemplatePairStack(nn.Module):
    """Implements AF2 Algorithm 16."""

    def __init__(
        self,
        c_t,
        c_hidden_tri_att,
        c_hidden_tri_mul,
        no_blocks,
        no_heads,
        transition_type,
        pair_transition_n,
        dropout_rate,
        tri_mul_first,
        fuse_projection_weights,
        blocks_per_ckpt,
        ckpt_per_template,
        inf=1e9,
        linear_init_params=lin_init.pair_block_init,
        use_reentrant: bool | None = None,
        tune_chunk_size: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_mul:
                Hidden dimension for triangular multiplication
            no_blocks:
                Number of blocks in the stack
            no_heads:
                Number of heads in the attention mechanism
            transition_type:
                String 'relu' or 'swiglu' to determine activation for the transition
                function
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
            tri_mul_first:
                Whether to perform triangular multiplication before attention
            fuse_projection_weights:
                When True, uses FusedTriangleMultiplicativeUpdate variant in
                the Pair Stack. Used in Multimer pipeline.
            blocks_per_ckpt:
                Number of blocks per activation checkpoint. None disables
                activation checkpointing
            ckpt_per_template:
                Whether to do activation checkpointing per template.
                This will disable the per-block checkpointing.
            inf:
                Large constant used for masking
            linear_init_params:
                Configuration for linear initialization
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
            tune_chunk_size:
                 Whether to dynamically tune the module's chunk size
        """
        super().__init__()

        self.blocks_per_ckpt = None if ckpt_per_template else blocks_per_ckpt
        self.use_reentrant = use_reentrant

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = TemplatePairBlock(
                c_t=c_t,
                c_hidden_tri_mul=c_hidden_tri_mul,
                c_hidden_tri_att=c_hidden_tri_att,
                no_heads=no_heads,
                transition_type=transition_type,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                tri_mul_first=tri_mul_first,
                fuse_projection_weights=fuse_projection_weights,
                ckpt_per_template=ckpt_per_template,
                inf=inf,
                linear_init_params=linear_init_params,
                use_reentrant=use_reentrant,
            )
            self.blocks.append(block)

        self.layer_norm = LayerNorm(c_t)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def _prep_blocks(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ):
        blocks = [
            partial(
                b,
                mask=mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            max_chunk_size = (
                CUEQ_MAX_CHUNK_SIZE
                if use_cueq_triangle_kernels
                else DEFAULT_MAX_CHUNK_SIZE
            )
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                args=(t.clone(),),
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
                    _attn_chunk_size=max(chunk_size, attn_chunk),
                )
                for b in blocks
            ]

        return blocks

    def forward(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
            chunk_size:
                Inference-time subbatch size
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
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        if mask.shape[-3] == 1:
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)

        blocks = self._prep_blocks(
            t=t,
            mask=mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        (t,) = checkpoint_blocks(
            blocks=blocks,
            args=(t,),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
            use_reentrant=self.use_reentrant,
        )

        t = self.layer_norm(t)

        return t


class TemplateEmbedderAllAtom(nn.Module):
    """Implements AF3 Algorithm 16."""

    def __init__(self, config: ConfigDict):
        """
        Args:
            config:
                ConfigDict with template config.
        """
        super().__init__()

        self.config = config
        self.template_pair_embedder = TemplatePairEmbedderAllAtom(
            **config.template_pair_embedder,
        )
        self.template_pair_stack = TemplatePairStack(
            **config.template_pair_stack,
        )

        templ_init = config.get(
            "linear_init_params", lin_init.all_atom_templ_module_init
        )
        self.linear_t = Linear(config.c_t, config.c_z, **templ_init.linear_t)

    def forward(
        self,
        batch: dict,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None = None,
        _mask_trans: bool = True,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Input feature dictionary
            z:
                [*, N_token, N_token, C_z] Pair embedding
            pair_mask:
                [*, N_token, N_token] Pair mask
            chunk_size:
                Inference-time subbatch size.
            _mask_trans:
                Whether to mask the output of the transition layers
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with and use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed

        Returns:
            t:
                [*, N_token, N_token, C_z] Template embedding
        """

        # [*, N_templ, N_token, N_token, C_t]
        template_embeds = self.template_pair_embedder(batch, z)
        n_templ = template_embeds.shape[-4]

        # [*, 1, N_token, N_token]
        pair_mask = pair_mask[..., None, :, :].to(dtype=z.dtype)

        # [*, N_templ, N_token, N_token, C_z]
        t = self.template_pair_stack(
            template_embeds,
            pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        # [*, N_token, N_token, C_z]
        t = torch.sum(t, dim=-4) / n_templ
        t = torch.nn.functional.relu(t)
        t = self.linear_t(t)

        return t
