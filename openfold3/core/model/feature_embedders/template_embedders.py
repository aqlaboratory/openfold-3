# Copyright 2026 AlQuraishi Laboratory
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.data.resources.residues import STANDARD_RESIDUES_WITH_GAP_3
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.all_atom_multimer import make_transform_from_reference
from openfold3.core.utils.geometry.vector import Vec3Array


def dgram_from_positions(
    positions: torch.Tensor,
    min_bin: float,
    max_bin: float,
    num_bins: int,
    inf: float = 1e8,
) -> torch.Tensor:
    """Compute distogram from amino acid pseudo-beta coordinates.

    Args:
        positions: (..., num_res, 3) Pseudo-beta coordinates.
        min_bin: Lower distogram bin edge (Angstrom).
        max_bin: Upper distogram bin edge (Angstrom).
        num_bins: Number of distogram bins.
        inf: Value to set last bin entries to. Defaults to 1e8.

    Returns:
        Distogram (..., num_res, num_res, num_bins) of dtype float32.
    """
    lower = torch.linspace(min_bin, max_bin, num_bins, device=positions.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dist2 = ((positions[..., None, :] - positions[..., None, :, :]) ** 2).sum(
        dim=-1, keepdim=True
    )
    return ((dist2 > lower) & (dist2 < upper)).float()


def unit_vector_from_frame_atom_coords(
    frame_atom_coords: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build per-residue rigid frames from N, CA, C atom coords and return the
    pairwise unit vector between residues in each residue's local frame.

    Args:
        frame_atom_coords: (..., n_template, n_token, 3, 3) coordinates of
            the N, CA, C atoms (along dim -2) for each token. NaNs should
            already be replaced with zeros by the data pipeline.

    Returns:
        Tuple of (x, y, z) tensors, each of shape
        (..., n_template, n_token, n_token), giving the components of the
        unit vector from residue j to residue i in residue i's local frame.
    """
    n_vec = Vec3Array.from_array(frame_atom_coords[..., 0, :])
    ca_vec = Vec3Array.from_array(frame_atom_coords[..., 1, :])
    c_vec = Vec3Array.from_array(frame_atom_coords[..., 2, :])

    rigid = make_transform_from_reference(a_xyz=n_vec, b_xyz=ca_vec, c_xyz=c_vec)
    unit_vec = (
        rigid[..., None]
        .inverse()
        .apply_to_point(rigid.translation[..., None, :])
        .normalized()
    )
    return unit_vec.x, unit_vec.y, unit_vec.z


class TemplatePairEmbedderAllAtom(nn.Module):
    """
    Implements AF3 Algorithm 16 lines 1-5. Also includes line 8.
    The resulting embedded template will go into the TemplatePairStack.
    """

    def __init__(
        self,
        c_in: int,
        c_aatype: int,
        c_out: int,
        distogram: ConfigDict,
        linear_init_params: ConfigDict = lin_init.all_atom_templ_pair_feat_emb_init,
    ):
        """
        Args:
            c_in:
                Pair embedding dimension
            c_aatype:
                Template aatype feature embedding dimension
            c_out:
                Output channel dimension
            distogram:
                Distogram bin configuration with fields:
                    min_bin (float): Lower distogram bin edge (Angstrom).
                    max_bin (float): Upper distogram bin edge (Angstrom).
                    n_bins  (int):   Number of distogram bins.
            linear_init_params:
                Linear layer initialization
        """
        super().__init__()

        self.distogram = distogram

        self.dgram_linear = Linear(
            self.distogram.n_bins, c_out, **linear_init_params.linear_a
        )
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
        # Needed when training with DeepSpeed so that the features are cast
        # to the correct model precision prior to the linear layers
        dtype = self.dgram_linear.weight.dtype

        # [*, N_token, N_token]
        multichain_pair_mask = (
            batch["asym_id"][..., None] == batch["asym_id"][..., None, :]
        )
        multichain_pair_mask = multichain_pair_mask[..., None, :, :, None]

        # [*, N_templ, N_token, N_token]
        pseudo_beta_pair_mask = (
            batch["template_pseudo_beta_mask"][..., None]
            * batch["template_pseudo_beta_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        # Compute distogram on-the-fly from pseudo-beta coords. Apply the
        # per-template pair mask the data pipeline used to bake in.
        template_distogram = dgram_from_positions(
            batch["template_pseudo_beta_coords"].to(torch.float32),
            min_bin=self.distogram.min_bin,
            max_bin=self.distogram.max_bin,
            num_bins=self.distogram.n_bins,
        )
        template_distogram = template_distogram * pseudo_beta_pair_mask

        backbone_frame_pair_mask = (
            batch["template_backbone_frame_mask"][..., None]
            * batch["template_backbone_frame_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        # Compute unit vector between residue pairs in each residue's local
        # frame from N/CA/C atom coordinates, mask, and unbind into x/y/z.
        x, y, z = unit_vector_from_frame_atom_coords(
            batch["template_frame_atom_coords"].to(torch.float32)
        )
        backbone_frame_2d = (
            batch["template_backbone_frame_mask"][..., None]
            * batch["template_backbone_frame_mask"][..., None, :]
        )
        x = (x * backbone_frame_2d)[..., None] * multichain_pair_mask
        y = (y * backbone_frame_2d)[..., None] * multichain_pair_mask
        z = (z * backbone_frame_2d)[..., None] * multichain_pair_mask

        # One-hot encode restype indices, then expand to [*, N_templ, N_token,
        # N_token, num_classes] along the pair axes for the two restype linears.
        template_restype_onehot = F.one_hot(
            batch["template_restype"].long(),
            num_classes=len(STANDARD_RESIDUES_WITH_GAP_3),
        ).to(dtype)
        n_token = template_restype_onehot.shape[-2]
        template_restype_ti = template_restype_onehot[..., None, :].expand(
            *template_restype_onehot.shape[:-2], -1, n_token, -1
        )
        template_restype_tj = template_restype_onehot[..., None, :, :].expand(
            *template_restype_onehot.shape[:-2], n_token, -1, -1
        )

        a = self.dgram_linear(template_distogram.to(dtype))
        a = a + self.pseudo_beta_mask_linear(pseudo_beta_pair_mask.to(dtype))
        a = a + self.aatype_linear_1(template_restype_ti)
        a = a + self.aatype_linear_2(template_restype_tj)
        a = a + self.x_linear(x.to(dtype))
        a = a + self.y_linear(y.to(dtype))
        a = a + self.z_linear(z.to(dtype))
        a = a + self.backbone_mask_linear(backbone_frame_pair_mask.to(dtype))

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
