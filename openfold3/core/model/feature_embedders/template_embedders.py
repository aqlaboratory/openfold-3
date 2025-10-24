# Copyright 2025 AlQuraishi Laboratory
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

import torch.nn as nn
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.primitives import LayerNorm, Linear


class TemplatePairEmbedderAllAtom(nn.Module):
    """
    Implements AF3 Algorithm 16 lines 1-5. Also includes line 8.
    The resulting embedded template will go into the TemplatePairStack.
    """

    def __init__(
        self,
        c_in: int,
        c_dgram: int,
        c_aatype: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.all_atom_templ_pair_feat_emb_init,
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
            c_out:
                Output channel dimension
            linear_init_params:
                Linear layer initialization
        """
        super().__init__()
        self.dgram_linear = Linear(c_dgram, c_out, **linear_init_params.linear_a)
        self.aatype_linear_1 = Linear(c_aatype, c_out, **linear_init_params.linear_a)
        self.aatype_linear_2 = Linear(c_aatype, c_out, **linear_init_params.linear_a)
        self.pseudo_beta_mask_linear = Linear(1, c_out, **linear_init_params.linear_a)
        self.x_linear = Linear(1, c_out, **linear_init_params.linear_a)
        self.y_linear = Linear(1, c_out, **linear_init_params.linear_a)
        self.z_linear = Linear(1, c_out, **linear_init_params.linear_a)
        self.backbone_mask_linear = Linear(1, c_out, **linear_init_params.linear_a)

        self.layer_norm_z = LayerNorm(c_in)
        self.linear_z = Linear(c_in, c_out, **linear_init_params.linear_z)

    def _embed_feats(self, batch: dict):
        dtype = batch["template_unit_vector"].dtype

        # [*, N_token, N_token]
        multichain_pair_mask = (
            batch["asym_id"][..., None] * batch["asym_id"][..., None, :]
        )
        multichain_pair_mask = multichain_pair_mask[..., None, :, :, None]

        # [*, N_templ, N_token, N_token]
        pseudo_beta_pair_mask = (
            batch["template_pseudo_beta_mask"][..., None]
            * batch["template_pseudo_beta_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        template_distogram = batch["template_distogram"] * multichain_pair_mask

        backbone_frame_pair_mask = (
            batch["template_backbone_frame_mask"][..., None]
            * batch["template_backbone_frame_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        template_unit_vector = batch["template_unit_vector"] * multichain_pair_mask
        x, y, z = template_unit_vector.unbind(dim=-1)

        # [*, N_templ, N_token, N_token, 32]
        template_restype = batch["template_restype"]
        n_token = batch["template_restype"].shape[-2]
        template_restype_ti = template_restype[..., None, :].expand(
            *template_restype.shape[:-2], -1, n_token, -1
        )
        template_restype_tj = template_restype[..., None, :, :].expand(
            *template_restype.shape[:-2], n_token, -1, -1
        )

        a = self.dgram_linear(template_distogram)
        a = a + self.pseudo_beta_mask_linear(pseudo_beta_pair_mask)
        a = a + self.aatype_linear_1(template_restype_ti.to(dtype=dtype))
        a = a + self.aatype_linear_2(template_restype_tj.to(dtype=dtype))
        a = a + self.x_linear(x[..., None])
        a = a + self.y_linear(y[..., None])
        a = a + self.z_linear(z[..., None])
        a = a + self.backbone_mask_linear(backbone_frame_pair_mask)

        return a

    def forward(self, batch, z):
        """
        Args:
            batch:
                Input template feature dictionary
            z:
                Pair embedding
        Returns:
            # [*, N_templ, N_token, N_token, C_out] Template pair feature embedding
        """
        a = self._embed_feats(batch=batch)

        # [*, N_templ, N_token, N_token, C_out]
        z = self.linear_z(self.layer_norm_z(z))
        z = z[..., None, :, :, :] + a

        return z
