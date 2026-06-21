# Copyright 2026 AlQuraishi Laboratory
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
from openfold3.core.model.primitives.fused_ln_linear import (
    FusedLNLinear,
    is_fused_ln_linear_enabled,
    register_legacy_remap_hook,
)
from openfold3.core.model.primitives.fused_swiglu_transition import (
    is_fused_swiglu_transition_enabled,
)
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
        seed_fourier_emb: int = 42,
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
            seed_fourier_emb:
                Random seed for initializing fourier embedding parameters
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

        self._use_fused_ln_linear = is_fused_ln_linear_enabled()
        if self._use_fused_ln_linear:
            self.fused_ln_linear_z = FusedLNLinear(
                num_relpos_dims + self.c_z, self.c_z,
                ln_create_offset=False,
                linear_bias=linear_init_params.linear_z.get("bias", True),
                linear_init=linear_init_params.linear_z.get("init", "default"),
                legacy_ln_name="layer_norm_z", legacy_linear_name="linear_z",
            )
        else:
            self.layer_norm_z = LayerNorm(
                num_relpos_dims + self.c_z, create_offset=False
            )
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

        if self._use_fused_ln_linear:
            self.fused_ln_linear_s = FusedLNLinear(
                self.c_s + self.c_s_input, self.c_s,
                ln_create_offset=False,
                linear_bias=linear_init_params.linear_z.get("bias", True),
                linear_init=linear_init_params.linear_z.get("init", "default"),
                legacy_ln_name="layer_norm_s", legacy_linear_name="linear_s",
            )
        else:
            self.layer_norm_s = LayerNorm(
                self.c_s + self.c_s_input, create_offset=False
            )
            self.linear_s = Linear(
                self.c_s + self.c_s_input, self.c_s, **linear_init_params.linear_z
            )

        self.fourier_emb = FourierEmbedding(c=c_fourier_emb, seed=seed_fourier_emb)
        if self._use_fused_ln_linear:
            self.fused_ln_linear_n = FusedLNLinear(
                self.c_fourier_emb, self.c_s,
                ln_create_offset=False,
                linear_bias=linear_init_params.linear_n.get("bias", True),
                linear_init=linear_init_params.linear_n.get("init", "default"),
                legacy_ln_name="layer_norm_n", legacy_linear_name="linear_n",
            )
        else:
            self.layer_norm_n = LayerNorm(self.c_fourier_emb, create_offset=False)
            self.linear_n = Linear(
                self.c_fourier_emb, self.c_s, **linear_init_params.linear_n
            )

        register_legacy_remap_hook(
            self,
            [
                ("layer_norm_z", "linear_z", "fused_ln_linear_z"),
                ("layer_norm_s", "linear_s", "fused_ln_linear_s"),
                ("layer_norm_n", "linear_n", "fused_ln_linear_n"),
            ],
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

    _EMBED_ZIJ_CHUNK_ROWS: int = 128

    def _embed_zij_invariant(
        self,
        batch: dict,
        zij_trunk: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self._embed_zij_invariant_chunked(batch, zij_trunk)

        relpos_zij = relpos_complex(
            batch=batch,
            max_relative_idx=self.max_relative_idx,
            max_relative_chain=self.max_relative_chain,
        ).to(dtype=zij_trunk.dtype)

        zij = torch.cat([zij_trunk, relpos_zij], dim=-1)
        del relpos_zij
        if self._use_fused_ln_linear:
            zij = self.fused_ln_linear_z(zij)
        else:
            zij = self.linear_z(self.layer_norm_z(zij))
        return zij

    # by Liang Hong <lhong22@cse.cuhk.edu.hk>: row-chunked invariant pair
    # embedding that avoids materializing the full relpos concat transient.
    def _embed_zij_invariant_chunked(
        self,
        batch: dict,
        zij_trunk: torch.Tensor,
    ) -> torch.Tensor:
        """Row-chunked variant that avoids the full [N, N, 267] concat transient.

        Processes ``_EMBED_ZIJ_CHUNK_ROWS`` rows at a time. LN normalizes over
        the last dim (267 channels) so each row is independent — chunking is
        mathematically exact.
        """
        N = zij_trunk.shape[-3]
        chunk = self._EMBED_ZIJ_CHUNK_ROWS
        out = torch.empty(
            *zij_trunk.shape[:-1], self.c_z,
            dtype=zij_trunk.dtype, device=zij_trunk.device,
        )
        for i in range(0, N, chunk):
            row_slice = slice(i, min(i + chunk, N))
            relpos_chunk = relpos_complex(
                batch=batch,
                max_relative_idx=self.max_relative_idx,
                max_relative_chain=self.max_relative_chain,
                row_slice=row_slice,
            ).to(dtype=zij_trunk.dtype)
            cat_chunk = torch.cat(
                [zij_trunk[..., row_slice, :, :], relpos_chunk], dim=-1
            )
            del relpos_chunk
            if self._use_fused_ln_linear:
                out[..., row_slice, :, :] = self.fused_ln_linear_z(cat_chunk)
            else:
                out[..., row_slice, :, :] = self.linear_z(
                    self.layer_norm_z(cat_chunk)
                )
            del cat_chunk
        return out

    def _embed_si_base(
        self,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
    ) -> torch.Tensor:
        si = torch.cat([si_trunk, si_input], dim=-1)
        if self._use_fused_ln_linear:
            si = self.fused_ln_linear_s(si)
        else:
            si = self.linear_s(self.layer_norm_s(si))
        return si

    def _embed_n(self, t: torch.Tensor) -> torch.Tensor:
        n = 0.25 * torch.log(t / self.sigma_data)
        n = self.fourier_emb(n.unsqueeze(-1))
        if self._use_fused_ln_linear:
            return self.fused_ln_linear_n(n)
        return self.linear_n(self.layer_norm_n(n))

    def _transition_z(
        self,
        zij: torch.Tensor,
        token_mask: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        pair_token_mask = token_mask.unsqueeze(-1) * token_mask.unsqueeze(-2)
        for l in self.transition_z:
            if chunk_size is None and not torch.is_grad_enabled():
                zij = l._transition_inplace(
                    x=zij,
                    mask=pair_token_mask.unsqueeze(-1),
                    residual=zij,
                )
            else:
                zij = zij + l(zij, mask=pair_token_mask, chunk_size=chunk_size)
        return zij

    def _transition_s(
        self,
        si: torch.Tensor,
        token_mask: torch.Tensor,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        for l in self.transition_s:
            si = si + l(si, mask=token_mask, chunk_size=chunk_size)
        return si

    def prepare_invariants(
        self,
        batch: dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        token_mask: torch.Tensor,
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the rollout-invariant outputs of DiffusionConditioning.

        Returns ``{"zij_conditioned", "si_base"}``. Performs the
        ``use_conditioning`` zero-out exactly once. Tunes the
        ``transition_z`` chunk size if ``chunk_size`` is set and the tuner
        is active.
        """
        if not use_conditioning:
            si_trunk = si_trunk * 0
            zij_trunk = zij_trunk * 0

        zij = self._embed_zij_invariant(batch=batch, zij_trunk=zij_trunk)
        si_base = self._embed_si_base(si_input=si_input, si_trunk=si_trunk)

        z_chunk_size = chunk_size
        if is_fused_swiglu_transition_enabled() and not torch.is_grad_enabled():
            z_chunk_size = None
        if z_chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            z_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=self._transition_z,
                args=(zij.clone(), token_mask),
                min_chunk_size=z_chunk_size,
                max_chunk_size=2048,
            )

        zij = self._transition_z(
            zij=zij, token_mask=token_mask, chunk_size=z_chunk_size
        )

        return {"zij_conditioned": zij, "si_base": si_base}

    def apply_t(
        self,
        cache: dict[str, torch.Tensor],
        t: torch.Tensor,
        token_mask: torch.Tensor,
        chunk_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the t-dependent path on top of a prepared cache.

        Reads ``cache["si_base"]`` and ``cache["zij_conditioned"]``. The
        cached pair tensor is returned by reference; ``apply_t`` never
        mutates the cache.
        """
        si_base = cache["si_base"]
        zij = cache["zij_conditioned"]

        n = self._embed_n(t)
        si = si_base + n.unsqueeze(-2)

        s_chunk_size = chunk_size
        if s_chunk_size is not None and self.chunk_size_tuner is not None:
            assert not self.training
            s_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=self._transition_s,
                args=(si.clone(), token_mask),
                min_chunk_size=s_chunk_size,
                max_chunk_size=2048,
            )

        si = self._transition_s(si=si, token_mask=token_mask, chunk_size=s_chunk_size)
        return si, zij

    def forward(
        self,
        batch: dict,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
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
            use_conditioning:
                Whether to condition with the trunk representations
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
        cache = self.prepare_invariants(
            batch=batch,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=use_conditioning,
            token_mask=token_mask,
            chunk_size=chunk_size,
        )
        return self.apply_t(
            cache=cache, t=t, token_mask=token_mask, chunk_size=chunk_size
        )
