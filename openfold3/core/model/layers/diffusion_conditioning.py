# Copyright 2025 AlQuraishi Laboratory
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

import torch
from ml_collections import ConfigDict
from torch import nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.feature_embedders.input_embedders import FourierEmbedding
from openfold3.core.model.layers.transition import SwiGLUTransition
from openfold3.core.model.primitives.linear import Linear
from openfold3.core.model.primitives.normalization import LayerNorm
from openfold3.core.utils.chunk_utils import ChunkSizeTuner
from openfold3.core.utils.relpos import relpos_complex


class DiffusionConditioning(nn.Module):
    """
    Implements AF3 Algorithm 21.
    """

    def __init__(
        self,
        c_s_input: int,
        c_s: int,
        c_z: int,
        c_fourier_emb: int,
        max_relative_idx: int,
        max_relative_chain: int,
        sigma_data: float,
        linear_init_params: ConfigDict = lin_init.diffusion_cond_init,
        tune_chunk_size: bool = False,
    ):
        """
        Args:
            c_s_input:
                Per token input representation channel dimension
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_fourier_emb:
                Fourier embedding channel dimension
            max_relative_idx:
                Maximum relative position and token indices clipped
            max_relative_chain:
                Maximum relative chain indices clipped
            sigma_data:
                Constant determined by data variance
            linear_init_params:
                Linear layer initialization parameters
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super().__init__()

        self.c_s_input = c_s_input
        self.c_s = c_s
        self.c_z = c_z
        self.c_fourier_emb = c_fourier_emb
        self.max_relative_idx = max_relative_idx
        self.max_relative_chain = max_relative_chain
        self.sigma_data = sigma_data

        num_rel_pos_bins = 2 * max_relative_idx + 2
        num_rel_token_bins = 2 * max_relative_idx + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (
            num_rel_pos_bins
            + num_rel_token_bins
            + num_rel_chain_bins
            + num_same_entity_features
        )

        self.layer_norm_z = LayerNorm(num_relpos_dims + self.c_z, create_offset=False)
        self.linear_z = Linear(
            num_relpos_dims + self.c_z, self.c_z, **linear_init_params.linear_z
        )

        self.transition_z = nn.ModuleList(
            [
                SwiGLUTransition(
                    c_in=self.c_z,
                    n=2,
                    linear_init_params=linear_init_params.transition_z,
                )
                for _ in range(2)
            ]
        )

        self.layer_norm_s = LayerNorm(self.c_s + self.c_s_input, create_offset=False)
        self.linear_s = Linear(
            self.c_s + self.c_s_input, self.c_s, **linear_init_params.linear_z
        )

        self.fourier_emb = FourierEmbedding(c=c_fourier_emb)
        self.layer_norm_n = LayerNorm(self.c_fourier_emb, create_offset=False)
        self.linear_n = Linear(
            self.c_fourier_emb, self.c_s, **linear_init_params.linear_n
        )

        self.transition_s = nn.ModuleList(
            [
                SwiGLUTransition(
                    c_in=self.c_s,
                    n=2,
                    linear_init_params=linear_init_params.transition_s,
                )
                for _ in range(2)
            ]
        )

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def _embed_trunk_inputs(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pair conditioning
        relpos_zij = relpos_complex(
            batch=batch,
            max_relative_idx=self.max_relative_idx,
            max_relative_chain=self.max_relative_chain,
        ).to(dtype=zij_trunk.dtype)

        zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
        zij = self.linear_z(self.layer_norm_z(zij))

        # Single conditioning
        si = torch.cat([si_trunk, si_input], dim=-1)
        si = self.linear_s(self.layer_norm_s(si))

        n = 0.25 * torch.log(t / self.sigma_data)
        n = self.fourier_emb(n.unsqueeze(-1))

        si = si + self.linear_n(self.layer_norm_n(n)).unsqueeze(-2)

        return si, zij

    def _forward(
        self,
        si: torch.Tensor,
        zij: torch.Tensor,
        token_mask: torch.Tensor,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_token_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)

        # Pair conditioning
        for l in self.transition_z:
            zij = zij + l(zij, mask=pair_token_mask, chunk_size=chunk_size)

        # Single conditioning
        for l in self.transition_s:
            si = si + l(si, mask=token_mask, chunk_size=chunk_size)

        return si, zij

    def _chunk_forward(
        self,
        si: torch.Tensor,
        zij: torch.Tensor,
        token_mask: torch.Tensor,
        chunk_size: int,
    ):
        assert not self.training

        if self.chunk_size_tuner is not None:
            chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=self._forward,
                # We don't want to write in-place during chunk tuning runs
                args=(
                    si.clone(),
                    zij.clone(),
                    token_mask,
                ),
                min_chunk_size=chunk_size,
                max_chunk_size=2048,
            )

        si, zij = self._forward(
            si=si, zij=zij, token_mask=token_mask, chunk_size=chunk_size
        )

        return si, zij

    def forward(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Feature dictionary
            t:
                [*] Noise level at a diffusion timestep
            si_input:
                [*, N_token, c_s_input] Input embedding
            si_trunk:
                [*, N_token, c_s] Single representation
            zij_trunk:
                [*, N_token, N_token, c_z] Pair representation
            chunk_size:
                Inference-time subbatch size. Acts as a minimum if
                self.tune_chunk_size is True
        Returns:
            si:
                [*, N_token, c_s] Conditioned single representation
            zij:
                [*, N_token, N_token, c_z] Conditioned pair representation
        """
        token_mask = batch["token_mask"]
        si, zij = self._embed_trunk_inputs(
            batch=batch, t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk
        )

        if chunk_size is not None:
            si, zij = self._chunk_forward(
                si=si, zij=zij, token_mask=token_mask, chunk_size=chunk_size
            )
        else:
            si, zij = self._forward(si=si, zij=zij, token_mask=token_mask)

        return si, zij
