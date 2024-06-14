from functools import partial

import torch
import torch.nn as nn

from openfold3.core.model.latent.template import TemplatePairStack, TemplatePointwiseAttention
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils import all_atom_multimer, geometry
from openfold3.core.utils.feats import (build_template_pair_feat, build_template_angle_feat,
                                        pseudo_beta_fn, dgram_from_positions)
from openfold3.core.utils.tensor_utils import tensor_tree_map, dict_multimap


class TemplateSingleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements AF2 Algorithm 2, line 7.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(TemplateSingleEmbedder, self).__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = Linear(self.c_in, self.c_out, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, init="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x


class TemplatePairEmbedder(nn.Module):
    """
    Embeds "template_pair_feat" features.

    Implements AF2 Algorithm 2, line 9.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:

            c_out:
                Output channel dimension
        """
        super(TemplatePairEmbedder, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        # Despite there being no relu nearby, the source uses that initializer
        self.linear = Linear(self.c_in, self.c_out, init="relu")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        x = self.linear(x)

        return x


class TemplateEmbedder(nn.Module):
    def __init__(self, config):
        super(TemplateEmbedder, self).__init__()

        self.config = config
        self.template_single_embedder = TemplateSingleEmbedder(
            **config["template_single_embedder"],
        )
        self.template_pair_embedder = TemplatePairEmbedder(
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
            t = build_template_pair_feat(
                single_template_feats,
                use_unit_vector=self.config.use_unit_vector,
                inf=self.config.inf,
                eps=self.config.eps,
                **self.config.distogram,
            ).to(z.dtype)
            t = self.template_pair_embedder(t)

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
            template_angle_feat = build_template_angle_feat(
                batch
            )

            # [*, S_t, N, C_m]
            a = self.template_single_embedder(template_angle_feat)

            ret["template_single_embedding"] = a

        return ret


class TemplatePairEmbedderMultimer(nn.Module):
    def __init__(self,
        c_in: int,
        c_out: int,
        c_dgram: int,
        c_aatype: int,
    ):
        super(TemplatePairEmbedderMultimer, self).__init__()

        self.dgram_linear = Linear(c_dgram, c_out, init='relu')
        self.aatype_linear_1 = Linear(c_aatype, c_out, init='relu')
        self.aatype_linear_2 = Linear(c_aatype, c_out, init='relu')
        self.query_embedding_layer_norm = LayerNorm(c_in)
        self.query_embedding_linear = Linear(c_in, c_out, init='relu')

        self.pseudo_beta_mask_linear = Linear(1, c_out, init='relu')
        self.x_linear = Linear(1, c_out, init='relu')
        self.y_linear = Linear(1, c_out, init='relu')
        self.z_linear = Linear(1, c_out, init='relu')
        self.backbone_mask_linear = Linear(1, c_out, init='relu')

    def forward(self,
                template_dgram: torch.Tensor,
                aatype_one_hot: torch.Tensor,
                query_embedding: torch.Tensor,
                pseudo_beta_mask: torch.Tensor,
                backbone_mask: torch.Tensor,
                multichain_mask_2d: torch.Tensor,
                unit_vector: geometry.Vec3Array,
                ) -> torch.Tensor:
        act = 0.

        pseudo_beta_mask_2d = (
            pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]
        )
        pseudo_beta_mask_2d *= multichain_mask_2d
        template_dgram *= pseudo_beta_mask_2d[..., None]
        act += self.dgram_linear(template_dgram)
        act += self.pseudo_beta_mask_linear(pseudo_beta_mask_2d[..., None])

        aatype_one_hot = aatype_one_hot.to(template_dgram.dtype)
        act += self.aatype_linear_1(aatype_one_hot[..., None, :, :])
        act += self.aatype_linear_2(aatype_one_hot[..., None, :])

        backbone_mask_2d = (
            backbone_mask[..., None] * backbone_mask[..., None, :]
        )
        backbone_mask_2d *= multichain_mask_2d
        x, y, z = [(coord * backbone_mask_2d).to(dtype=query_embedding.dtype) for coord in unit_vector]
        act += self.x_linear(x[..., None])
        act += self.y_linear(y[..., None])
        act += self.z_linear(z[..., None])

        act += self.backbone_mask_linear(backbone_mask_2d[..., None].to(dtype=query_embedding.dtype))

        query_embedding = self.query_embedding_layer_norm(query_embedding)
        act += self.query_embedding_linear(query_embedding)

        return act


class TemplateSingleEmbedderMultimer(nn.Module):
    def __init__(self,
        c_in: int,
        c_out: int,
    ):
        super(TemplateSingleEmbedderMultimer, self).__init__()
        self.template_single_embedder = Linear(c_in, c_out)
        self.template_projector = Linear(c_out, c_out)

    def forward(self,
        batch,
        atom_pos,
        aatype_one_hot,
    ):
        out = {}

        dtype = batch["template_all_atom_positions"].dtype

        template_chi_angles, template_chi_mask = (
            all_atom_multimer.compute_chi_angles(
                atom_pos,
                batch["template_all_atom_mask"],
                batch["template_aatype"],
            )
        )

        template_features = torch.cat(
            [
                aatype_one_hot,
                torch.sin(template_chi_angles) * template_chi_mask,
                torch.cos(template_chi_angles) * template_chi_mask,
                template_chi_mask,
            ],
            dim=-1,
        ).to(dtype=dtype)

        template_mask = template_chi_mask[..., 0].to(dtype=dtype)

        template_activations = self.template_single_embedder(
            template_features
        )
        template_activations = torch.nn.functional.relu(
            template_activations
        )
        template_activations = self.template_projector(
            template_activations,
        )

        out["template_single_embedding"] = (
            template_activations
        )
        out["template_mask"] = template_mask

        return out


class TemplateEmbedderMultimer(nn.Module):
    def __init__(self, config):
        super(TemplateEmbedderMultimer, self).__init__()

        self.config = config
        self.template_pair_embedder = TemplatePairEmbedderMultimer(
            **config["template_pair_embedder"],
        )
        self.template_single_embedder = TemplateSingleEmbedderMultimer(
            **config["template_single_embedder"],
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
            act = 0.

            template_positions, pseudo_beta_mask = pseudo_beta_fn(
                single_template_feats["template_aatype"],
                single_template_feats["template_all_atom_positions"],
                single_template_feats["template_all_atom_mask"])

            template_dgram = dgram_from_positions(
                template_positions,
                inf=self.config.inf,
                **self.config.distogram,
            )

            aatype_one_hot = torch.nn.functional.one_hot(
                single_template_feats["template_aatype"], 22,
            )

            raw_atom_pos = single_template_feats["template_all_atom_positions"]

            # Vec3Arrays are required to be float32
            atom_pos = geometry.Vec3Array.from_array(raw_atom_pos.to(dtype=torch.float32))

            rigid, backbone_mask = all_atom_multimer.make_backbone_affine(
                atom_pos,
                single_template_feats["template_all_atom_mask"],
                single_template_feats["template_aatype"],
            )
            points = rigid.translation
            rigid_vec = rigid[..., None].inverse().apply_to_point(points)
            unit_vector = rigid_vec.normalized()

            pair_act = self.template_pair_embedder(
                template_dgram,
                aatype_one_hot,
                z,
                pseudo_beta_mask,
                backbone_mask,
                multichain_mask_2d,
                unit_vector,
            )

            single_template_embeds["template_pair_embedding"] = pair_act
            single_template_embeds.update(
                self.template_single_embedder(
                    single_template_feats,
                    atom_pos,
                    aatype_one_hot,
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
