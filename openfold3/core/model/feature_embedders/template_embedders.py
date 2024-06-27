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

"""
Template feature embedders. Used in the template stack to build the final template
embeddings.
"""

from typing import Dict

import torch
import torch.nn as nn

from openfold3.core.data.data_transforms import pseudo_beta_fn
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.np import residue_constants as rc
from openfold3.core.utils import all_atom_multimer, geometry
from openfold3.core.utils.feats import dgram_from_positions
from openfold3.core.utils.rigid_utils import Rigid


class TemplateSingleEmbedderMonomer(nn.Module):
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
        super().__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = Linear(self.c_in, self.c_out, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, init="relu")

    def forward(self, template_feats: Dict) -> torch.Tensor:
        """
        Args:
            template_feats: Dict with template features
        Returns:
            x: [*, N_templ, N_res, C_out] Template single feature embedding
        """
        template_aatype = template_feats["template_aatype"]
        torsion_angles_sin_cos = template_feats["template_torsion_angles_sin_cos"]
        alt_torsion_angles_sin_cos = template_feats[
            "template_alt_torsion_angles_sin_cos"
        ]
        torsion_angles_mask = template_feats["template_torsion_angles_mask"]
        x = torch.cat(
            [
                nn.functional.one_hot(template_aatype, 22),
                torsion_angles_sin_cos.reshape(*torsion_angles_sin_cos.shape[:-2], 14),
                alt_torsion_angles_sin_cos.reshape(
                    *alt_torsion_angles_sin_cos.shape[:-2], 14
                ),
                torsion_angles_mask,
            ],
            dim=-1,
        )

        # [*, N_templ, N_res, c_in]
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x


class TemplatePairEmbedderMonomer(nn.Module):
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
                Final dimension of template pair features
            c_out:
                Output channel dimension
        """
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        # Despite there being no relu nearby, the source uses that initializer
        self.linear = Linear(self.c_in, self.c_out, init="relu")

    def forward(
        self,
        batch: Dict,
        distogram_config: Dict,
        use_unit_vector: bool,
        inf: float,
        eps: float,
    ) -> torch.Tensor:
        """

        Args:
            batch:
                Input template feature dictionary
            distogram_config:
                Configuration for distogram computation
            use_unit_vector:
                Whether to use unit vector in template features
            inf:
                Large value for distogram computation
            eps:
                Small constant for numerical stability
        Returns:
            # [*, N_res, N_res, C_out] Template pair feature embedding
        """
        # [*, C_in]
        dtype = batch["template_all_atom_positions"].dtype

        template_mask = batch["template_pseudo_beta_mask"]
        template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

        # Compute distogram (this seems to differ slightly from Alg. 5)
        tpb = batch["template_pseudo_beta"]
        dgram = dgram_from_positions(pos=tpb, inf=inf, **distogram_config)

        to_concat = [dgram, template_mask_2d[..., None]]

        aatype_one_hot = nn.functional.one_hot(
            batch["template_aatype"],
            rc.restype_num + 2,
        )

        n_res = batch["template_aatype"].shape[-1]
        to_concat.append(
            aatype_one_hot[..., None, :, :].expand(
                *aatype_one_hot.shape[:-2], n_res, -1, -1
            )
        )
        to_concat.append(
            aatype_one_hot[..., None, :].expand(
                *aatype_one_hot.shape[:-2], -1, n_res, -1
            )
        )

        n, ca, c = (rc.atom_order[a] for a in ["N", "CA", "C"])
        rigids = Rigid.make_transform_from_reference(
            n_xyz=batch["template_all_atom_positions"][..., n, :],
            ca_xyz=batch["template_all_atom_positions"][..., ca, :],
            c_xyz=batch["template_all_atom_positions"][..., c, :],
            eps=eps,
        )
        points = rigids.get_trans()[..., None, :, :]
        rigid_vec = rigids[..., None].invert_apply(points)

        inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec**2, dim=-1))

        t_aa_masks = batch["template_all_atom_mask"]
        template_mask = t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
        template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

        inv_distance_scalar = inv_distance_scalar * template_mask_2d
        unit_vector = rigid_vec * inv_distance_scalar[..., None]

        if not use_unit_vector:
            unit_vector = unit_vector * 0.0

        to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
        to_concat.append(template_mask_2d[..., None])

        act = torch.cat(to_concat, dim=-1)
        act = act * template_mask_2d[..., None]

        # [*, C_out]
        act = self.linear(act.to(dtype=dtype))

        return act


class TemplateSingleEmbedderMultimer(nn.Module):
    """
    Embeds the "template_single_feat" feature.

    Implements Multimer version of AF2 Algorithm 2, line 7.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
    ):
        """
        Args:
            c_in:
                Final dimension of single template features
            c_out:
                Output channel dimension
        """
        super().__init__()
        self.template_single_embedder = Linear(c_in, c_out)
        self.template_projector = Linear(c_out, c_out)

    def forward(self, batch: Dict):
        """
        Args:
            batch:
                Input template feature dictionary

        Returns:
            [*, N_templ, N_res, C_out] Template single feature embedding
        """
        out = {}

        dtype = batch["template_all_atom_positions"].dtype

        aatype_one_hot = torch.nn.functional.one_hot(
            batch["template_aatype"],
            22,
        )

        raw_atom_pos = batch["template_all_atom_positions"]

        # Vec3Arrays are required to be float32
        atom_pos = geometry.Vec3Array.from_array(raw_atom_pos.to(dtype=torch.float32))

        template_chi_angles, template_chi_mask = all_atom_multimer.compute_chi_angles(
            atom_pos,
            batch["template_all_atom_mask"],
            batch["template_aatype"],
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

        template_activations = self.template_single_embedder(template_features)
        template_activations = torch.nn.functional.relu(template_activations)
        template_activations = self.template_projector(
            template_activations,
        )

        out["template_single_embedding"] = template_activations
        out["template_mask"] = template_mask

        return out


class TemplatePairEmbedderMultimer(nn.Module):
    """
    Embeds template pair features.

    Implements Multimer version of AF2 Algorithm 2, line 9.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        c_dgram: int,
        c_aatype: int,
    ):
        """
        Args:
            c_in:
                Pair embedding dimension
            c_out:
                Template pair embedding dimension
            c_dgram:
                Distogram feature embedding dimension
            c_aatype:
                Template aatype feature embedding dimension
        """
        super().__init__()

        self.dgram_linear = Linear(c_dgram, c_out, init="relu")
        self.aatype_linear_1 = Linear(c_aatype, c_out, init="relu")
        self.aatype_linear_2 = Linear(c_aatype, c_out, init="relu")
        self.query_embedding_layer_norm = LayerNorm(c_in)
        self.query_embedding_linear = Linear(c_in, c_out, init="relu")

        self.pseudo_beta_mask_linear = Linear(1, c_out, init="relu")
        self.x_linear = Linear(1, c_out, init="relu")
        self.y_linear = Linear(1, c_out, init="relu")
        self.z_linear = Linear(1, c_out, init="relu")
        self.backbone_mask_linear = Linear(1, c_out, init="relu")

    def forward(
        self,
        batch: Dict,
        distogram_config: Dict,
        query_embedding: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
        inf: float,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Input template feature dictionary
            distogram_config:
                Configuration for distogram computation
            query_embedding:
                [*, N_res, N_res, C_z] Pair embedding (z)
            multichain_mask_2d:
                [*, N_res, N_res] Multichain mask built from asym IDs
            inf:
                Large value for distogram computation
        Returns:
            # [*, N_templ, N_res, N_res, C_out] Template pair feature embedding
        """
        query_embedding = query_embedding.unsqueeze(-4)
        multichain_mask_2d = multichain_mask_2d.unsqueeze(-3)

        template_positions, pseudo_beta_mask = pseudo_beta_fn(
            batch["template_aatype"],
            batch["template_all_atom_positions"],
            batch["template_all_atom_mask"],
        )

        template_dgram = dgram_from_positions(
            template_positions,
            inf=inf,
            **distogram_config,
        )

        pseudo_beta_mask_2d = (
            pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]
        )
        pseudo_beta_mask_2d *= multichain_mask_2d
        template_dgram *= pseudo_beta_mask_2d[..., None]
        act = self.dgram_linear(template_dgram)
        act += self.pseudo_beta_mask_linear(pseudo_beta_mask_2d[..., None])

        aatype_one_hot = torch.nn.functional.one_hot(
            batch["template_aatype"],
            22,
        )
        aatype_one_hot = aatype_one_hot.to(template_dgram.dtype)
        act += self.aatype_linear_1(aatype_one_hot[..., None, :, :])
        act += self.aatype_linear_2(aatype_one_hot[..., None, :])

        raw_atom_pos = batch["template_all_atom_positions"]

        # Vec3Arrays are required to be float32
        atom_pos = geometry.Vec3Array.from_array(raw_atom_pos.to(dtype=torch.float32))

        rigid, backbone_mask = all_atom_multimer.make_backbone_affine(
            atom_pos,
            batch["template_all_atom_mask"],
            batch["template_aatype"],
        )
        points = rigid.translation
        rigid_vec = rigid[..., None].inverse().apply_to_point(points[..., None, :])
        unit_vector = rigid_vec.normalized()
        backbone_mask_2d = backbone_mask[..., None] * backbone_mask[..., None, :]
        backbone_mask_2d *= multichain_mask_2d
        x, y, z = (
            (coord * backbone_mask_2d).to(dtype=query_embedding.dtype)
            for coord in unit_vector
        )
        act += self.x_linear(x[..., None])
        act += self.y_linear(y[..., None])
        act += self.z_linear(z[..., None])

        act += self.backbone_mask_linear(
            backbone_mask_2d[..., None].to(dtype=query_embedding.dtype)
        )

        query_embedding = self.query_embedding_layer_norm(query_embedding)
        act += self.query_embedding_linear(query_embedding)

        return act


class TemplatePairEmbedderAllAtom(nn.Module):
    """
    Implements AF3 Algorithm 16 lines 1-5. Also includes line 8. The resulting embedded
    template will go into the TemplatePairStack.
    """

    def __init__(self, c_in: int, c_z: int, c_out: int):
        """
        Args:
            c_in:
                Final dimension of template pair features
            c_z:
                Pair embedding dimension
            c_out:
                Output channel dimension
        """
        super().__init__()

        self.linear_feats = Linear(c_in, c_out, bias=False)
        self.query_embedding_layer_norm = LayerNorm(c_z)
        self.query_embedding_linear = Linear(c_z, c_out, bias=False, init="relu")

    def forward(
        self,
        batch: Dict,
        distogram_config: Dict,
        query_embedding: torch.Tensor,
        multichain_mask_2d: torch.Tensor,
        inf: float,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Input template feature dictionary
            distogram_config:
                Configuration for distogram computation
            query_embedding:
                [*, N_token, N_token, C_z] Pair embedding (z)
            multichain_mask_2d:
                [*, N_token, N_token] Multichain mask built from asym IDs
            inf:
                Large value for distogram computation
        Returns:
            # [*, N_templ, N_token, N_token, C_out] Template pair feature embedding
        """
        query_embedding = query_embedding.unsqueeze(-4)
        multichain_mask_2d = multichain_mask_2d.unsqueeze(-3)

        template_positions, pseudo_beta_mask = pseudo_beta_fn(
            batch["template_aatype"],
            batch["template_all_atom_positions"],
            batch["template_all_atom_mask"],
        )

        template_dgram = dgram_from_positions(
            template_positions,
            inf=inf,
            **distogram_config,
        )

        pseudo_beta_mask_2d = (
            pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]
        )
        template_dgram *= pseudo_beta_mask_2d[..., None]

        raw_atom_pos = batch["template_all_atom_positions"]

        # Vec3Arrays are required to be float32
        atom_pos = geometry.Vec3Array.from_array(raw_atom_pos.to(dtype=torch.float32))

        rigid, backbone_mask = all_atom_multimer.make_backbone_affine(
            atom_pos,
            batch["template_all_atom_mask"],
            batch["template_aatype"],
        )
        points = rigid.translation
        rigid_vec = rigid[..., None].inverse().apply_to_point(points[..., None, :])
        unit_vector = rigid_vec.normalized()
        backbone_mask_2d = backbone_mask[..., None] * backbone_mask[..., None, :]

        x, y, z = ((coord * backbone_mask_2d)[..., None] for coord in unit_vector)

        act = torch.cat(
            [
                template_dgram,
                backbone_mask_2d[..., None],
                x,
                y,
                z,
                pseudo_beta_mask_2d[..., None],
            ],
            dim=-1,
        )
        act = act * multichain_mask_2d[..., None]

        aatype_one_hot = torch.nn.functional.one_hot(
            batch["template_aatype"],
            22,
        )

        n_res = batch["template_aatype"].shape[-1]
        aatype_ti = aatype_one_hot[..., None, :, :].expand(
            *aatype_one_hot.shape[:-2], n_res, -1, -1
        )
        aatype_tj = aatype_one_hot[..., None, :].expand(
            *aatype_one_hot.shape[:-2], -1, n_res, -1
        )

        act = torch.cat([act, aatype_ti, aatype_tj], dim=-1)

        act = self.linear_feats(act.to(dtype=query_embedding.dtype))

        query_embedding = self.query_embedding_layer_norm(query_embedding)
        act += self.query_embedding_linear(query_embedding)

        return act
