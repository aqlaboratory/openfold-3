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
import math
import sys
from functools import partial

import torch
from torch import nn as nn

from openfold3.core.model.feature_embedders import (
    TemplateSingleEmbedderMonomer,
    TemplatePairEmbedderMonomer,
    TemplatePairEmbedderMultimer,
    TemplateSingleEmbedderMultimer,
    TemplatePairEmbedderAllAtom
)
from openfold3.core.model.latent.pair_stacks import TemplatePairStack
from openfold3.core.model.layers.template_pointwise_attention import TemplatePointwiseAttention
from openfold3.core.model.primitives import Linear
from openfold3.core.utils.tensor_utils import tensor_tree_map, dict_multimap


class TemplateEmbedderMonomer(nn.Module):
    def __init__(self, config):
        super(TemplateEmbedderMonomer, self).__init__()

        self.config = config
        self.template_single_embedder = TemplateSingleEmbedderMonomer(
            **config["template_single_embedder"],
        )
        self.template_pair_embedder = TemplatePairEmbedderMonomer(
            **config["template_pair_embedder"],
        )
        self.template_pair_stack = TemplatePairStack(
            **config["template_pair_stack"],
        )
        self.template_pointwise_att = TemplatePointwiseAttention(
            **config["template_pointwise_attention"],
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
        inplace_safe=False
    ):
        # Embed the templates one at a time (with a poor man's vmap)
        pair_embeds = []
        n = z.shape[-2]
        n_templ = batch["template_aatype"].shape[templ_dim]

        if (inplace_safe):
            # We'll preallocate the full pair tensor now to avoid manifesting
            # a second copy during the stack later on
            t_pair = z.new_zeros(
                z.shape[:-3] +
                (n_templ, n, n, self.config.template_pair_embedder.c_out)
            )

        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),
                batch,
            )

            # [*, N, N, C_t]
            t = self.template_pair_embedder(batch=single_template_feats,
                                            distogram_config=self.config.distogram,
                                            use_unit_vector=self.config.use_unit_vector,
                                            inf=self.config.inf,
                                            eps=self.config.eps)

            if (inplace_safe):
                t_pair[..., i, :, :, :] = t
            else:
                pair_embeds.append(t)

            del t

        if (not inplace_safe):
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

        if (inplace_safe):
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
    def __init__(self, config):
        super(TemplateEmbedderMultimer, self).__init__()

        self.config = config
        self.template_single_embedder = TemplateSingleEmbedderMultimer(
            **config["template_single_embedder"],
        )
        self.template_pair_embedder = TemplatePairEmbedderMultimer(
            **config["template_pair_embedder"],
        )
        self.template_pair_stack = TemplatePairStack(
            **config["template_pair_stack"],
        )

        self.linear_t = Linear(config.c_t, config.c_z)

    def forward(self,
        batch,
        z,
        padding_mask_2d,
        templ_dim,
        chunk_size,
        multichain_mask_2d,
        _mask_trans=True,
        use_deepspeed_evo_attention=False,
        use_lma=False,
        inplace_safe=False
    ):
        template_embeds = []
        n_templ = batch["template_aatype"].shape[templ_dim]
        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx),
                batch,
            )

            single_template_embeds = {}

            pair_act = self.template_pair_embedder(
                batch=single_template_feats,
                distogram_config=self.config.distogram,
                query_embedding=z,
                multichain_mask_2d=multichain_mask_2d,
                inf=self.config.inf
            )

            single_template_embeds["template_pair_embedding"] = pair_act
            single_template_embeds.update(
                self.template_single_embedder(
                    single_template_feats,
                )
            )
            template_embeds.append(single_template_embeds)

        template_embeds = dict_multimap(
            partial(torch.cat, dim=templ_dim),
            template_embeds,
        )

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            template_embeds["template_pair_embedding"],
            padding_mask_2d.unsqueeze(-3).to(dtype=z.dtype),
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
    def __init__(self, config):
        super(TemplateEmbedderAllAtom, self).__init__()

        self.config = config
        self.template_pair_embedder = TemplatePairEmbedderAllAtom(
            **config["template_pair_embedder"],
        )
        self.template_pair_stack = TemplatePairStack(
            **config["template_pair_stack"],
        )

        self.linear_t = Linear(config.c_t, config.c_z, bias=False)

    def forward(self,
        batch,
        z,
        padding_mask_2d,
        templ_dim,
        chunk_size,
        multichain_mask_2d,
        _mask_trans=True,
        use_deepspeed_evo_attention=False,
        use_lma=False,
        inplace_safe=False
    ):
        template_embeds = []
        n_templ = batch["template_aatype"].shape[templ_dim]
        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx),
                batch,
            )

            pair_act = self.template_pair_embedder(
                batch=single_template_feats,
                distogram_config=self.config.distogram,
                query_embedding=z,
                multichain_mask_2d=multichain_mask_2d,
                inf=self.config.inf
            )

            template_embeds.append(pair_act)

        template_embeds = torch.cat(template_embeds, dim=templ_dim)

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            template_embeds["template_pair_embedding"],
            padding_mask_2d.unsqueeze(-3).to(dtype=z.dtype),
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
            lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),
            batch,
        )

        # [*, N, N, C_t]
        t = model.template_embedder.template_pair_embedder(batch=single_template_feats,
                                                           distogram_config=model.config.template.distogram,
                                                           use_unit_vector=model.config.template.use_unit_vector,
                                                           inf=model.config.template.inf,
                                                           eps=model.config.template.eps)

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

        assert (sys.getrefcount(t) == 2)

        pair_embeds_cpu.append(t.cpu())

        del t

    # Preallocate the output tensor
    t = z.new_zeros(z.shape)

    for i in range(0, n, template_chunk_size):
        pair_chunks = [
            p[..., i: i + template_chunk_size, :, :] for p in pair_embeds_cpu
        ]
        pair_chunk = torch.cat(pair_chunks, dim=templ_dim).to(device=z.device)
        z_chunk = z[..., i: i + template_chunk_size, :, :]
        att_chunk = model.template_embedder.template_pointwise_att(
            pair_chunk,
            z_chunk,
            template_mask=batch["template_mask"].to(dtype=z.dtype),
            use_lma=model.globals.use_lma,
        )

        t[..., i: i + template_chunk_size, :, :] = att_chunk

    del pair_chunks

    if inplace_safe:
        t = t * (torch.sum(batch["template_mask"], dim=-1) > 0)
    else:
        t *= (torch.sum(batch["template_mask"], dim=-1) > 0)

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
    for i in range(0, n_templ, templ_group_size):
        def slice_template_tensor(t):
            s = [slice(None) for _ in t.shape]
            s[templ_dim] = slice(i, i + templ_group_size)
            return t[s]

        template_feats = tensor_tree_map(
            slice_template_tensor,
            batch,
        )

        # [*, N, N, C_t]
        t = model.template_embedder.template_pair_embedder(batch=template_feats,
                                                           distogram_config=model.config.template.distogram,
                                                           use_unit_vector=model.config.template.use_unit_vector,
                                                           inf=model.config.template.inf,
                                                           eps=model.config.template.eps)

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
        out_tensor *= (torch.sum(batch["template_mask"], dim=-1) > 0)
    else:
        out_tensor = out_tensor * (torch.sum(batch["template_mask"], dim=-1) > 0)

    ret = {}
    if model.config.template.embed_angles:
        # [*, N, C_m]
        a = model.template_embedder.template_single_embedder(batch)

        ret["template_single_embedding"] = a

    ret.update({"template_pair_embedding": out_tensor})

    return ret
