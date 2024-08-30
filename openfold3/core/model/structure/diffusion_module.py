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
Diffusion module. Implements the algorithms in section 3.7 of the
Supplementary Information.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.layers import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)
from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.core.utils.rigid_utils import quat_to_rot


def sample_rotations(shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Sample random quaternions"""
    q = torch.randn(*shape, 4, dtype=dtype, device=device)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)

    rots = quat_to_rot(q)

    return rots


def centre_random_augmentation(
    xl: torch.Tensor, atom_mask: torch.Tensor, scale_trans: float = 1.0
) -> torch.Tensor:
    """
    Implements AF3 Algorithm 19.

    Args:
        xl:
            [*, N_atom, 3] Atom positions
        atom_mask:
            [*, N_atom] Atom mask
        scale_trans:
            Translation scaling factor
    Returns:
        Updated atom position with random global rotation and translation
    """
    rots = sample_rotations(shape=xl.shape[:-2], dtype=xl.dtype, device=xl.device)

    trans = scale_trans * torch.randn(
        (*xl.shape[:-2], 3), dtype=xl.dtype, device=xl.device
    )

    mean_xl = torch.sum(
        xl * atom_mask[..., None],
        dim=-2,
        keepdim=True,
    ) / torch.sum(atom_mask[..., None], dim=-2, keepdim=True)

    # center coordinates
    pos_centered = xl - mean_xl
    return pos_centered @ rots.transpose(-1, -2) + trans[..., None, :]


# Move this somewhere else?
def create_noise_schedule(
    no_rollout_steps: float, sigma_data: float, s_max: float, s_min: float, p: int
):
    """
    Implements AF3 noise schedule (Page 24).

     Args:
        no_rollout_steps:
            Number of diffusion rollout steps
        sigma_data:
            Constant determined by data variance
        s_max:
            Maximum standard deviation of noise
        s_min:
            Minimum standard deviation of noise
        p:
            Constant controlling the extent steps near s_min are shortened
            at the cost of longer steps near s_max
    Returns:
        Noise schedule
    """
    t = torch.arange(0, 1 + no_rollout_steps) / no_rollout_steps
    return (
        sigma_data * (s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p
    )


class DiffusionModule(nn.Module):
    """
    Implements AF3 Algorithm 20.
    """

    def __init__(self, config):
        """
        Args:
            config:
                Configuration dictionary for diffusion module
        """
        super().__init__()
        self.c_s = config.diffusion_module.c_s
        self.c_token = config.diffusion_module.c_token
        self.sigma_data = config.diffusion_module.sigma_data

        self.diffusion_conditioning = DiffusionConditioning(
            **config.diffusion_conditioning
        )

        self.atom_attn_enc = AtomAttentionEncoder(
            **config.atom_attn_enc, add_noisy_pos=True
        )

        diff_mod_init = config.diffusion_module.get(
            "linear_init_params", lin_init.diffusion_module_init
        )

        self.layer_norm_s = LayerNorm(self.c_s)
        self.linear_s = Linear(
            self.c_s,
            self.c_token,
            **diff_mod_init.linear_s,
        )

        self.diffusion_transformer = DiffusionTransformer(
            **config.diffusion_transformer
        )

        self.layer_norm_a = LayerNorm(self.c_token)

        self.atom_attn_dec = AtomAttentionDecoder(**config.atom_attn_dec)

    def forward(
        self,
        batch: Dict,
        xl_noisy: torch.Tensor,
        atom_mask: torch.Tensor,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Feature dictionary
            xl_noisy:
                [*, N_atom, 3] Noisy atom positions
            atom_mask:
                [*, N_atom] Atom mask
            t:
                [*] Noise level at a diffusion step
            si_input:
                [*, N_token, c_s_input] Input embedding
            si_trunk:
                [*, N_token, c_s] Single representation
            zij_trunk:
                [*, N_token, c_s] Pair representation
            chunk_size:
                Inference-time subbatch size
        Returns:
            [*, N_atom, 3] Denoised atom positions
        """
        si, zij = self.diffusion_conditioning(
            batch=batch, t=t, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk
        )

        rl_noisy = xl_noisy / torch.sqrt(t[..., None, None] ** 2 + self.sigma_data**2)

        ai, ql, cl, plm = self.atom_attn_enc(
            batch=batch,
            atom_mask=atom_mask,
            rl=rl_noisy,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            chunk_size=chunk_size,
        )  # differ from AF3

        ai = ai + self.linear_s(self.layer_norm_s(si))

        ai = self.diffusion_transformer(
            a=ai, s=si, z=zij, mask=batch["token_mask"], chunk_size=chunk_size
        )

        ai = self.layer_norm_a(ai)

        rl_update = self.atom_attn_dec(
            batch=batch,
            atom_mask=atom_mask,
            ai=ai,
            ql=ql,
            cl=cl,
            plm=plm,
            chunk_size=chunk_size,
        )

        xl_out = (
            self.sigma_data**2
            / (self.sigma_data**2 + t[..., None, None] ** 2)
            * xl_noisy
            + self.sigma_data
            * t[..., None, None]
            / torch.sqrt(self.sigma_data**2 + t[..., None, None] ** 2)
            * rl_update
        )

        return xl_out


class SampleDiffusion(nn.Module):
    """
    Implements AF3 Algorithm 18.
    """

    def __init__(
        self,
        gamma_0: float,
        gamma_min: float,
        noise_scale: float,
        step_scale: float,
        diffusion_module: DiffusionModule,
    ):
        """
        Args:
            gamma_0:
                Schedule controlling factor
            gamma_min:
                Minimum schedule threshold to apply schedule control
            noise_scale:
                Noise scaling factor
            step_scale:
                Step scaling factor
            diffusion_module:
                An instantiated DiffusionModule
        """
        super().__init__()
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.diffusion_module = diffusion_module

    def forward(
        self,
        batch: Dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        noise_schedule: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Feature dictionary
            si_input:
                [*, N_token, c_s_input] Input embedding
            si_trunk:
                [*, N_token, c_s] Single representation
            zij_trunk:
                [*, N_token, N_token, c_z] Pair representation
            noise_schedule:
                [no_rollout_steps] Noise schedule
            chunk_size:
                Inference-time subbatch size
        Returns:
            [*, N_atom, 3] Sampled atom positions
        """
        atom_mask = broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=batch["token_mask"],
        )

        xl = noise_schedule[0] * torch.randn(
            (*atom_mask.shape, 3), device=atom_mask.device, dtype=atom_mask.dtype
        )

        for tau, c_tau in enumerate(noise_schedule[1:]):
            xl = centre_random_augmentation(xl=xl, atom_mask=atom_mask)

            gamma = self.gamma_0 if c_tau > self.gamma_min else 0

            t = noise_schedule[tau] * (gamma + 1)

            noise = (
                self.noise_scale
                * torch.sqrt(t**2 - noise_schedule[tau] ** 2)
                * torch.randn_like(xl)
            )

            xl_noisy = xl + noise

            xl_denoised = self.diffusion_module(
                batch=batch,
                xl_noisy=xl_noisy,
                atom_mask=atom_mask,
                t=t.to(xl_noisy.device),
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                chunk_size=chunk_size,
            )

            delta = (xl - xl_denoised) / t
            dt = c_tau - t
            xl = xl_noisy + self.step_scale * dt * delta

        return xl
