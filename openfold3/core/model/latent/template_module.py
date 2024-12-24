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

"""Template embedding layers.

These modules embed templates into pair embeddings. Note that this includes the template
feature embedding functions in openfold3.core.model.feature_embedders.
"""

import math
import sys
from functools import partial
from typing import Optional

import torch
from ml_collections import ConfigDict
from torch import nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.feature_embedders.template_embedders import (
    TemplatePairEmbedderAllAtom,
    TemplatePairEmbedderMonomer,
    TemplatePairEmbedderMultimer,
    TemplateSingleEmbedderMonomer,
    TemplateSingleEmbedderMultimer,
)
from openfold3.core.model.layers.template_pointwise_attention import (
    TemplatePointwiseAttention,
)
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.checkpointing import checkpoint_blocks
from openfold3.core.utils.chunk_utils import ChunkSizeTuner
from openfold3.core.utils.tensor_utils import add, tensor_tree_map

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
        inf: float,
        linear_init_params: ConfigDict = lin_init.pair_block_init,
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

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
    ):
        """
        Args:
            z:
                [*, N_templ, N_res, N_res, C_t] Template embedding
            mask:
                [*, N_templ, N_res, N_res] Template mask
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
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

        single_templates = [t.unsqueeze(-4) for t in torch.unbind(z, dim=-4)]
        single_templates_masks = [m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)]

        for i in range(len(single_templates)):
            t = single_templates[i]
            t_pair_mask = single_templates_masks[i]

            if self.tri_mul_first:
                t = self.tri_att_start_end(
                    z=self.tri_mul_out_in(
                        z=t, pair_mask=t_pair_mask, inplace_safe=inplace_safe
                    ),
                    _attn_chunk_size=_attn_chunk_size,
                    pair_mask=t_pair_mask,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            else:
                t = self.tri_mul_out_in(
                    z=self.tri_att_start_end(
                        z=t,
                        _attn_chunk_size=_attn_chunk_size,
                        pair_mask=t_pair_mask,
                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                        use_lma=use_lma,
                        inplace_safe=inplace_safe,
                    ),
                    pair_mask=t_pair_mask,
                    inplace_safe=inplace_safe,
                )

            t = add(
                t,
                self.pair_transition(
                    t,
                    mask=t_pair_mask if _mask_trans else None,
                    chunk_size=chunk_size,
                ),
                inplace_safe,
            )

            if not inplace_safe:
                single_templates[i] = t

        if not inplace_safe:
            z = torch.cat(single_templates, dim=-4)

        return z


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
        inf=1e9,
        linear_init_params=lin_init.pair_block_init,
        use_reentrant: Optional[bool] = None,
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

        self.blocks_per_ckpt = blocks_per_ckpt
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
                inf=inf,
                linear_init_params=linear_init_params,
            )
            self.blocks.append(block)

        self.layer_norm = LayerNorm(c_t)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def forward(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
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
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
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

        blocks = [
            partial(
                b,
                mask=mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                args=(t.clone(),),
                min_chunk_size=chunk_size,
            )
            blocks = [
                partial(
                    b,
                    chunk_size=tuned_chunk_size,
                    _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4),
                )
                for b in blocks
            ]

        (t,) = checkpoint_blocks(
            blocks=blocks,
            args=(t,),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
            use_reentrant=self.use_reentrant,
        )

        t = self.layer_norm(t)

        return t


class TemplateEmbedderMonomer(nn.Module):
    """Implements AF2 Algorithm 2 lines 7-13."""

    def __init__(self, config: ConfigDict):
        """
        Args:
            config:
                ConfigDict with template config.
        """
        super().__init__()

        self.config = config
        self.template_single_embedder = TemplateSingleEmbedderMonomer(
            **config.template_single_embedder,
        )
        self.template_pair_embedder = TemplatePairEmbedderMonomer(
            **config.template_pair_embedder,
        )
        self.template_pair_stack = TemplatePairStack(
            **config.template_pair_stack,
        )
        self.template_pointwise_att = TemplatePointwiseAttention(
            **config.template_pointwise_attention,
        )

    def forward(
        self,
        batch,
        z,
        pair_mask,
        templ_dim,
        chunk_size,
        _mask_trans=True,
        use_deepspeed_evo_attention=False,
        use_lma=False,
        inplace_safe=False,
    ):
        """
        Args:
            batch:
                Input feature dictionary
            z:
                [*, N_res, N_res, C_z] Pair embedding
            pair_mask:
                [*, N_res, N_res] Pair mask
            templ_dim:
                The template dimension of the template tensors in batch
            chunk_size:
                Inference-time subbatch size.
            _mask_trans:
                Whether to mask the output of the transition layers
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed

        Returns:
            t:
                [*, N_res, N_res, C_z] Template embedding
        """
        # Embed the templates one at a time (with a poor man's vmap)
        pair_embeds = []
        n = z.shape[-2]
        n_templ = batch["template_aatype"].shape[templ_dim]

        if inplace_safe:
            # We'll preallocate the full pair tensor now to avoid manifesting
            # a second copy during the stack later on
            t_pair = z.new_zeros(
                z.shape[:-3] + (n_templ, n, n, self.config.template_pair_embedder.c_out)
            )

        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),  # noqa: B023
                batch,
            )

            # [*, N, N, C_t]
            t = self.template_pair_embedder(
                batch=single_template_feats,
                distogram_config=self.config.distogram,
                use_unit_vector=self.config.use_unit_vector,
                inf=self.config.inf,
                eps=self.config.eps,
            )

            if inplace_safe:
                t_pair[..., i, :, :, :] = t
            else:
                pair_embeds.append(t)

            del t

        if not inplace_safe:
            t_pair = torch.stack(pair_embeds, dim=templ_dim)

        del pair_embeds

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            t_pair,
            pair_mask.unsqueeze(-3).to(dtype=z.dtype),
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )
        del t_pair

        # [*, N, N, C_z]
        t = self.template_pointwise_att(
            t,
            z,
            template_mask=batch["template_mask"].to(dtype=z.dtype),
            use_lma=use_lma,
        )

        t_mask = torch.sum(batch["template_mask"], dim=-1) > 0
        # Append singletons
        t_mask = t_mask.reshape(
            *t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape)))
        )

        if inplace_safe:
            t *= t_mask
        else:
            t = t * t_mask

        ret = {}

        ret.update({"template_pair_embedding": t})

        del t

        if self.config.embed_angles:
            # [*, S_t, N, C_m]
            a = self.template_single_embedder(batch)

            ret["template_single_embedding"] = a

        return ret


class TemplateEmbedderMultimer(nn.Module):
    """Implements AF2-Multimer version of Algorithm 2 lines 7-13."""

    def __init__(self, config: ConfigDict):
        """
        Args:
            config:
                ConfigDict with template config.
        """
        super().__init__()

        self.config = config
        self.template_single_embedder = TemplateSingleEmbedderMultimer(
            **config.template_single_embedder,
        )
        self.template_pair_embedder = TemplatePairEmbedderMultimer(
            **config.template_pair_embedder,
        )
        self.template_pair_stack = TemplatePairStack(
            **config.template_pair_stack,
        )

        templ_init = config.get(
            "linear_init_params", lin_init.multimer_templ_module_init
        )
        self.linear_t = Linear(config.c_t, config.c_z, **templ_init.linear_t)

    def forward(
        self,
        batch,
        z,
        pair_mask,
        templ_dim,
        chunk_size,
        multichain_mask_2d,
        _mask_trans=True,
        use_deepspeed_evo_attention=False,
        use_lma=False,
        inplace_safe=False,
    ):
        """
        Args:
            batch:
                Input feature dictionary
            z:
                [*, N_res, N_res, C_z] Pair embedding
            pair_mask:
                [*, N_res, N_res] Pair mask
            templ_dim:
                The template dimension of the template tensors in batch
            chunk_size:
                Inference-time subbatch size.
            multichain_mask_2d:
                [*, N_res, N_res] Multichain mask built from asym IDs
            _mask_trans:
                Whether to mask the output of the transition layers
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed

        Returns:
            t:
                [*, N_res, N_res, C_z] Template embedding
        """
        template_embeds = {}
        n_templ = batch["template_aatype"].shape[templ_dim]

        pair_act = self.template_pair_embedder(
            batch=batch,
            distogram_config=self.config.distogram,
            query_embedding=z,
            multichain_mask_2d=multichain_mask_2d,
            inf=self.config.inf,
        )

        template_embeds["template_pair_embedding"] = pair_act
        template_embeds.update(
            self.template_single_embedder(
                batch,
            )
        )

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            template_embeds["template_pair_embedding"],
            pair_mask.unsqueeze(-3).to(dtype=z.dtype),
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )
        # [*, N, N, C_z]
        t = torch.sum(t, dim=-4) / n_templ
        t = torch.nn.functional.relu(t)
        t = self.linear_t(t)
        template_embeds["template_pair_embedding"] = t

        return template_embeds


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
        chunk_size: Optional[int] = None,
        _mask_trans: bool = True,
        use_deepspeed_evo_attention: bool = False,
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
                Mutually exclusive with use_lma and use_flash.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_flash and use_deepspeed_evo_attention.
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
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
        )

        # [*, N_token, N_token, C_z]
        t = torch.sum(t, dim=-4) / n_templ
        t = torch.nn.functional.relu(t)
        t = self.linear_t(t)

        return t


def embed_templates_offload(
    model,
    batch,
    z,
    pair_mask,
    templ_dim,
    template_chunk_size=256,
    inplace_safe=False,
):
    """
    Args:
        model:
            An AlphaFold model object
        batch:
            An AlphaFold input batch. See documentation of AlphaFold.
        z:
            A [*, N, N, C_z] pair embedding
        pair_mask:
            A [*, N, N] pair mask
        templ_dim:
            The template dimension of the template tensors in batch
        template_chunk_size:
            Integer value controlling how quickly the offloaded pair embedding
            tensor is brought back into GPU memory. In dire straits, can be
            lowered to reduce memory consumption of this function even more.
        inplace_safe:
            Whether inplace operations can be performed
    Returns:
        A dictionary of template pair and angle embeddings.

    A version of the "embed_templates" method of the AlphaFold class that
    offloads the large template pair tensor to CPU. Slower but more frugal
    with GPU memory than the original. Useful for long-sequence inference.
    """
    # Embed the templates one at a time (with a poor man's vmap)
    pair_embeds_cpu = []
    n = z.shape[-2]
    n_templ = batch["template_aatype"].shape[templ_dim]
    for i in range(n_templ):
        idx = batch["template_aatype"].new_tensor(i)
        single_template_feats = tensor_tree_map(
            lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),  # noqa: B023
            batch,
        )

        # [*, N, N, C_t]
        t = model.template_embedder.template_pair_embedder(
            batch=single_template_feats,
            distogram_config=model.config.template.distogram,
            use_unit_vector=model.config.template.use_unit_vector,
            inf=model.config.template.inf,
            eps=model.config.template.eps,
        )

        # [*, 1, N, N, C_z]
        t = model.template_embedder.template_pair_stack(
            t.unsqueeze(templ_dim),
            pair_mask.unsqueeze(-3).to(dtype=z.dtype),
            chunk_size=model.globals.chunk_size,
            use_deepspeed_evo_attention=model.globals.use_deepspeed_evo_attention,
            use_lma=model.globals.use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=model.config._mask_trans,
        )

        assert sys.getrefcount(t) == 2

        pair_embeds_cpu.append(t.cpu())

        del t

    # Preallocate the output tensor
    t = z.new_zeros(z.shape)

    for i in range(0, n, template_chunk_size):
        pair_chunks = [
            p[..., i : i + template_chunk_size, :, :] for p in pair_embeds_cpu
        ]
        pair_chunk = torch.cat(pair_chunks, dim=templ_dim).to(device=z.device)
        z_chunk = z[..., i : i + template_chunk_size, :, :]
        att_chunk = model.template_embedder.template_pointwise_att(
            pair_chunk,
            z_chunk,
            template_mask=batch["template_mask"].to(dtype=z.dtype),
            use_lma=model.globals.use_lma,
        )

        t[..., i : i + template_chunk_size, :, :] = att_chunk

    del pair_chunks

    if inplace_safe:
        t = t * (torch.sum(batch["template_mask"], dim=-1) > 0)
    else:
        t *= torch.sum(batch["template_mask"], dim=-1) > 0

    ret = {}
    if model.config.template.embed_angles:
        # [*, N, C_m]
        a = model.template_embedder.template_single_embedder(batch)

        ret["template_single_embedding"] = a

    ret.update({"template_pair_embedding": t})

    return ret


def embed_templates_average(
    model,
    batch,
    z,
    pair_mask,
    templ_dim,
    templ_group_size=2,
    inplace_safe=False,
):
    """
    Args:
        model:
            An AlphaFold model object
        batch:
            An AlphaFold input batch. See documentation of AlphaFold.
        z:
            A [*, N, N, C_z] pair embedding
        pair_mask:
            A [*, N, N] pair mask
        templ_dim:
            The template dimension of the template tensors in batch
        templ_group_size:
            Granularity of the approximation. Larger values trade memory for
            greater proximity to the original function
        inplace_safe:
            Whether inplace operations can be performed
    Returns:
        A dictionary of template pair and angle embeddings.

    A memory-efficient approximation of the "embed_templates" method of the
    AlphaFold class. Instead of running pointwise attention over pair
    embeddings for all of the templates at the same time, it splits templates
    into groups of size templ_group_size, computes embeddings for each group
    normally, and then averages the group embeddings. In our experiments, this
    approximation has a minimal effect on the quality of the resulting
    embedding, while its low memory footprint allows the number of templates
    to scale almost indefinitely.
    """
    # Embed the templates one at a time (with a poor man's vmap)
    n_templ = batch["template_aatype"].shape[templ_dim]
    out_tensor = z.new_zeros(z.shape)

    def slice_template_tensor(t, i):
        s = [slice(None) for _ in t.shape]
        s[templ_dim] = slice(i, i + templ_group_size)
        return t[s]

    for i in range(0, n_templ, templ_group_size):
        template_feats = tensor_tree_map(
            partial(slice_template_tensor, i=i),
            batch,
        )

        # [*, N, N, C_t]
        t = model.template_embedder.template_pair_embedder(
            batch=template_feats,
            distogram_config=model.config.template.distogram,
            use_unit_vector=model.config.template.use_unit_vector,
            inf=model.config.template.inf,
            eps=model.config.template.eps,
        )

        t = model.template_embedder.template_pair_stack(
            t,
            pair_mask.unsqueeze(-3).to(dtype=z.dtype),
            chunk_size=model.globals.chunk_size,
            use_deepspeed_evo_attention=model.globals.use_deepspeed_evo_attention,
            use_lma=model.globals.use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=model.config._mask_trans,
        )

        t = model.template_embedder.template_pointwise_att(
            t,
            z,
            template_mask=template_feats["template_mask"].to(dtype=z.dtype),
            use_lma=model.globals.use_lma,
        )

        denom = math.ceil(n_templ / templ_group_size)
        if inplace_safe:
            t /= denom
        else:
            t = t / denom

        if inplace_safe:
            out_tensor += t
        else:
            out_tensor = out_tensor + t

        del t

    if inplace_safe:
        out_tensor *= torch.sum(batch["template_mask"], dim=-1) > 0
    else:
        out_tensor = out_tensor * (torch.sum(batch["template_mask"], dim=-1) > 0)

    ret = {}
    if model.config.template.embed_angles:
        # [*, N, C_m]
        a = model.template_embedder.template_single_embedder(batch)

        ret["template_single_embedding"] = a

    ret.update({"template_pair_embedding": out_tensor})

    return ret
