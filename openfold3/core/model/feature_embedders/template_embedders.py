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
from openfold3.core.kernels.triton.fused_ln_linear import fused_ln_linear_inference
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.model.primitives.fused_ln_linear import (
    is_fused_ln_linear_enabled,
)
from openfold3.core.model.primitives.fused_template_pair_embedder import (
    fused_template_pair_embedder_inference,
    is_fused_template_pair_embedder_enabled,
)


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

    def _linear_scalar_stack(
        self, scalar_feats: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """One matmul for pseudo-beta mask, unit vector, and backbone-mask feats."""
        weight = torch.cat(
            (
                self.pseudo_beta_mask_linear.weight,
                self.x_linear.weight,
                self.y_linear.weight,
                self.z_linear.weight,
                self.backbone_mask_linear.weight,
            ),
            dim=-1,
        )
        biases = (
            self.pseudo_beta_mask_linear.bias,
            self.x_linear.bias,
            self.y_linear.bias,
            self.z_linear.bias,
            self.backbone_mask_linear.bias,
        )
        bias = None
        if any(b is not None for b in biases):
            bias = sum(b for b in biases if b is not None)
        if weight.dtype != dtype:
            weight = weight.to(dtype=dtype)
        if bias is not None and bias.dtype != dtype:
            bias = bias.to(dtype=dtype)
        return F.linear(scalar_feats, weight, bias)

    def _embed_feats(self, batch: dict):
        dtype = batch["template_unit_vector"].dtype

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

        template_distogram = batch["template_distogram"]

        backbone_frame_pair_mask = (
            batch["template_backbone_frame_mask"][..., None]
            * batch["template_backbone_frame_mask"][..., None, :]
        )[..., None] * multichain_pair_mask

        template_unit_vector = batch["template_unit_vector"]

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
        # In inference, mutate `a` in place so each successive contribution
        # is `a + temp` rather than `(old a) + temp = new a` (which keeps all
        # three tensors alive simultaneously). Each `a` tensor at multimer
        # N=590, n_templ=4, c_t=64 is ~2U; the eager pattern's triple-alive
        # peak is the second-largest transient inside template_embedder
        # after the pair_stack itself.
        inplace_add = not torch.is_grad_enabled()

        def _add(tmp):
            nonlocal a
            if inplace_add:
                a.add_(tmp)
            else:
                a = a + tmp

        _add(self.aatype_linear_1(template_restype_ti.to(dtype=dtype)))
        _add(self.aatype_linear_2(template_restype_tj.to(dtype=dtype)))
        scalar_feats = torch.cat(
            (
                pseudo_beta_pair_mask.to(dtype=dtype),
                template_unit_vector,
                backbone_frame_pair_mask.to(dtype=dtype),
            ),
            dim=-1,
        )
        _add(self._linear_scalar_stack(scalar_feats, dtype=dtype))

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
        # Fused inference path: LN(z)@Wz allocated straight into ``a``
        # (chunked, no 0.5U ``z_projected`` transient) + in-place addmm
        # for dgram / scalar + broadcast-add for the tiny per-i and per-j
        # aatype projections (no 0.25U .to(dtype) expand copies).  Handles
        # exactly one template at a time — this dispatch fires only when
        # configured low-memory offload has reduced ``batch`` to n_templ=1.
        # For multi-template batches (grad path or non-streaming inference),
        # fall through to the eager loop below.
        if (
            not torch.is_grad_enabled()
            and is_fused_template_pair_embedder_enabled()
            and z.is_cuda
            and z.dim() == 4
            and z.shape[0] == 1
            and batch["template_restype"].shape[-3] == 1
        ):
            return fused_template_pair_embedder_inference(
                module=self, batch=batch, z=z, template_index=0
            )

        a = self._embed_feats(batch=batch)

        # [*, N_templ, N_token, N_token, C_out]
        if is_fused_ln_linear_enabled() and not torch.is_grad_enabled():
            z = fused_ln_linear_inference(
                z,
                self.layer_norm_z.weight,
                self.layer_norm_z.bias,
                self.linear_z.weight,
                self.linear_z.bias,
                self.layer_norm_z.eps,
            )
        else:
            z = self.linear_z(self.layer_norm_z(z))
        if not torch.is_grad_enabled():
            a.add_(z[..., None, :, :, :])
            return a

        return z[..., None, :, :, :] + a
