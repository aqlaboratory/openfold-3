from typing import Dict, Tuple

import torch
import torch.nn as nn

from openfold3.core.model.feature_embedders.input_embedders import RelposAllAtom
from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
from openfold3.core.model.layers.sequence_local_atom_attention import AtomAttentionEncoder, AtomAttentionDecoder
from openfold3.core.model.layers.transition import SwiGLUTransition
from openfold3.core.model.primitives import LayerNorm, Linear


def centre_random_augmentation(pos: torch.Tensor, pos_mask: torch.Tensor, scale_trans: float = 1.) -> torch.Tensor:
    """
    Implements AF3 Algorithm 19.

    Args:
        pos:
            [*, N_atom, 3] Atom positions
        pos_mask:
            [*, N_atom] Atom mask
        scale_trans:
            Translation scaling factor
    Returns:
        Updated atom position with random global rotation and translation
    """
    m = torch.rand((*pos.shape[:-2], 3, 3), dtype=pos.dtype)
    rots, __ = torch.linalg.qr(m)

    trans = scale_trans * torch.randn((*pos.shape[:-2], 3), dtype=pos.dtype)

    mean_pos = (
        torch.mean(
            pos * pos_mask[..., None],
            dim=-2,
            keepdim=True,
        )
    )

    # center coordinates
    pos_centered = pos - mean_pos
    return pos_centered @ rots.transpose(-1, -2) + trans[..., None, :]


# Move this somewhere else?
def create_noise_schedule(
    step_size: float,
    sigma_data: float,
    s_max: float = 160.,
    s_min: float = 4e-4,
    p: int = 7
):
    """
    Implements AF3 noise schedule (Page 24).

     Args:
        step_size:
            Diffusion step size
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
    t = torch.arange(0, 1+step_size, step=step_size)
    return sigma_data * (s_max ** (1/p) + t * (s_min ** (1/p) - s_max ** (1/p))) ** p


# Should this be moved to embedders?
class FourierEmbedding(nn.Module):
    """
    Implements AF3 Algorithm 22.
    """
    def __init__(self, c: int):
        """
        Args:
            c:
                Embedding dimension
        """
        super(FourierEmbedding, self).__init__()

        self.linear = Linear(in_dim=1, out_dim=c, bias=True, init='fourier')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, 1] Input tensor
        Returns:
            [*, c] Embedding
        """
        return torch.cos(2 * torch.pi * self.linear(x))


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
                Fourier embedding channel diemnsion
            max_relative_idx:
                Maximum relative position and token indices clipped
            max_relative_chain:
                Maximum relative chain indices clipped
            sigma_data:
                Constant determined by data variance
        """
        super(DiffusionConditioning, self).__init__()

        self.c_s_input = c_s_input
        self.c_s = c_s
        self.c_z = c_z
        self.c_fourier_emb = c_fourier_emb
        self.sigma_data = sigma_data

        self.relpos = RelposAllAtom(c_z=self.c_z,
                                    max_relative_idx=max_relative_idx,
                                    max_relative_chain=max_relative_chain)

        self.layer_norm_z = LayerNorm(2 * self.c_z)
        self.linear_z = Linear(2 * self.c_z, self.c_z, bias=False)

        self.transition_z = nn.ModuleList([SwiGLUTransition(c_in=self.c_z, n=2)
                                           for _ in range(2)])

        self.layer_norm_s = LayerNorm(self.c_s + self.c_s_input)
        self.linear_s = Linear(self.c_s + self.c_s_input, self.c_s, bias=False)

        self.fourier_emb = FourierEmbedding(c=c_fourier_emb)
        self.layer_norm_n = LayerNorm(self.c_fourier_emb)
        self.linear_n = Linear(self.c_fourier_emb, self.c_s, bias=False)

        self.transition_s = nn.ModuleList([SwiGLUTransition(c_in=self.c_s, n=2)
                                           for _ in range(2)])

    def forward(
        self,
        batch: Dict,
        t: torch.Tensor,
        s_input: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        token_mask: torch.Tensor,
        pair_token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Feature dictionary
            t:
                [*] Noise level at a diffusion timestep
            s_input:
                [*, N_token, c_s_input] Input embedding
            s_trunk:
                [*, N_token, c_s] Single representation
            z_trunk:
                [*, N_token, N_token, c_z] Pair representation
            token_mask:
                [*, N_token] Token mask
            pair_token_mask:
                [*, N_token, N_token] Token pair mask
        Returns:
            s:
                [*, N_token, c_s] Conditioned single representation
            z:
                [*, N_token, N_token, c_z] Conditioned pair representation
        """

        z = torch.cat([z_trunk, self.relpos(batch)], dim=-1)
        z = self.linear_z(self.layer_norm_z(z))

        for l in self.transition_z:
            z = z + l(z, pair_token_mask)

        s = torch.cat([s_trunk, s_input], dim=-1)
        s = self.linear_s(self.layer_norm_s(s))

        n = 0.25 * torch.log(t / self.sigma_data)
        n = self.fourier_emb(n.unsqueeze(-1))

        s = s + self.linear_n(self.layer_norm_n(n))[..., None, :]

        for l in self.transition_s:
            s = s + l(s, token_mask)

        return s, z


class DiffusionModule(nn.Module):
    """
    Implements AF3 Algorithm 20.
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
        c_atom_ref: int,
        c_atom: int,
        c_atom_pair: int,
        c_token: int,
        c_hidden_att: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        inf: float,
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
            c_atom_ref:
                Reference per-atom feature dimension
            c_atom:
                Atom emebedding channel dimension
            c_atom_pair:
                Atom pair embedding channel dimension
            c_token:
                Token representation channel dimension
            c_hidden_att:
                Hidden channel dimension
            no_heads:
                Number of attention heads (for diffusion transformer)
            no_blocks:
                Number of attention blocks (for diffusion transformer)
            n_transition:
                Dimension multiplication factor used in transition layer (for diffusion transformer)
            inf:
                Large number used for attention masking
        """
        super(DiffusionModule, self).__init__()
        self.sigma_data = sigma_data

        self.diffusion_conditioning = DiffusionConditioning(c_s_input=c_s_input,
                                                            c_s=c_s,
                                                            c_z=c_z,
                                                            c_fourier_emb=c_fourier_emb,
                                                            max_relative_idx=max_relative_idx,
                                                            max_relative_chain=max_relative_chain,
                                                            sigma_data=sigma_data)

        self.atom_attn_enc = AtomAttentionEncoder(c_s=c_s,
                                                  c_z=c_z,
                                                  c_atom_ref=c_atom_ref,
                                                  c_atom=c_atom,
                                                  c_atom_pair=c_atom_pair,
                                                  c_token=c_token,
                                                  add_noisy_pos=True,
                                                  c_hidden=c_hidden_att,
                                                  # no_heads=encoder_no_heads,
                                                  # no_blocks=no_blocks,
                                                  # n_transition=n_transition,
                                                  inf=inf)

        self.layer_norm_s = LayerNorm(c_s)
        self.linear_s = Linear(c_s, c_token, bias=False)

        self.diffusion_transformer = DiffusionTransformer(c_a=c_token,
                                                          c_s=c_s,
                                                          c_z=c_z,
                                                          c_hidden=c_hidden_att,
                                                          no_heads=no_heads,
                                                          no_blocks=no_blocks,
                                                          n_transition=n_transition,
                                                          inf=inf)

        self.layer_norm_a = LayerNorm(c_token)

        self.atom_attn_dec = AtomAttentionDecoder(c_atom=c_atom,
                                                  c_atom_pair=c_atom_pair,
                                                  c_token=c_token,
                                                  c_hidden=c_hidden_att,  # shared across encoder, transformer and decoder? Intended?
                                                  # no_heads=no_heads,
                                                  # no_blocks=no_blocks,
                                                  # n_transition=n_transition,
                                                  inf=inf)

    def forward(
        self,
        batch: Dict,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        s_input: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        token_mask: torch.Tensor,
        pair_token_mask: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Feature dictionary
            x_noisy:
                [*, N_atom, 3] Noisy atom positions
            t:
                [*] Noise level at a diffusion step
            s_input:
                [*, N_token, c_s_input] Input embedding
            s_trunk:
                [*, N_token, c_s] Single representation
            z_trunk:
                [*, N_token, c_s] Pair representation
            token_mask:
                [*, N_token] Token mask
            pair_token_mask:
                [*, N_token, N_token] Token pair mask
        Returns:
            [*, N_atom, 3] Denoised atom positions
        """
        s, z = self.diffusion_conditioning(batch=batch,
                                           t=t,
                                           s_input=s_input,
                                           s_trunk=s_trunk,
                                           z_trunk=z_trunk,
                                           token_mask=token_mask,
                                           pair_token_mask=pair_token_mask)

        r_noisy = x_noisy / torch.sqrt(t ** 2 + self.sigma_data ** 2)

        a, q, c, p = self.atom_attn_enc(atom_feats=batch,
                                        atom_mask=atom_mask,
                                        rl=r_noisy,
                                        si_trunk=s_trunk, 
                                        zij=z_trunk) # differ from AF3

        a = a + self.linear_s(self.layer_norm_s(s))

        a = self.diffusion_transformer(a=a,
                                       s=s,
                                       z=z,
                                       beta=None,
                                       mask=token_mask)

        a = self.layer_norm_a(a)

        r_update = self.atom_attn_dec(atom_feats=batch,
                                      ai=a, 
                                      ql_skip=q,
                                      cl_skip=c, 
                                      plm=p,
                                      atom_mask=atom_mask)

        x_out = self.sigma_data ** 2 / (self.sigma_data ** 2 + t ** 2) * x_noisy + \
            self.sigma_data * t / torch.sqrt(self.sigma_data ** 2 + t ** 2) * r_update

        return x_out

class SampleDiffusion(nn.Module):
    """
    Implements AF3 Algorithm 18.
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
        c_atom_ref: int,
        c_atom: int,
        c_atom_pair: int,
        c_token: int,
        c_hidden_att: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        inf: float,
        gamma_0: float,
        gamma_min: float,
        noise_scale: float,
        step_scale: float,
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
            c_atom_ref:
                Reference per-atom feature dimension
            c_atom:
                Atom emebedding channel dimension
            c_atom_pair:
                Atom pair embedding channel dimension
            c_token:
                Token representation channel dimension
            c_hidden_att:
                Hidden channel dimension
            no_heads:
                Number of attention heads (for diffusion transformer)
            no_blocks:
                Number of attention blocks (for diffusion transformer)
            n_transition:
                Number of transition blocks (for diffusion transformer)
            inf:
                Large number used for attention masking
            gamma_0:
                Schedule controlling factor
            gamma_min:
                Minimum schedule threshold to apply schedule control
            noise_scale:
                Noise scaling factor
            step_scale:
                Step scaling factor
        """
        super(SampleDiffusion, self).__init__()
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale

        self.diffusion_module = DiffusionModule(c_s_input=c_s_input,
                                                c_s=c_s,
                                                c_z=c_z,
                                                c_fourier_emb=c_fourier_emb,
                                                max_relative_idx=max_relative_idx,
                                                max_relative_chain=max_relative_chain,
                                                sigma_data=sigma_data,
                                                c_atom_ref=c_atom_ref,
                                                c_atom=c_atom,
                                                c_atom_pair=c_atom_pair,
                                                c_token=c_token,
                                                c_hidden_att=c_hidden_att,
                                                no_heads=no_heads,
                                                no_blocks=no_blocks,
                                                n_transition=n_transition,
                                                inf=inf)

    def forward(
        self,
        batch: Dict,
        s_input: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        token_mask: torch.Tensor,
        pair_token_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        noise_schedule: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Feature dictionary
            s_input:
                [*, N_token, c_s_input] Input embedding
            s_trunk:
                [*, N_token, c_s] Single representation
            z_trunk:
                [*, N_token, N_token, c_z] Pair represnetation
            token_mask:
                [*, N_token] Token mask
            pair_token_mask:
                [*, N_token, N_token] Token pair mask
            atom_mask:
                [*, N_atom, N_atom] Atom mask
            noise_schedule:
                [T+1] Noise schedule with a step size of 1/T
        Returns:
            [*, N_atom, 3] Sampled atom positions
        """
        x = noise_schedule[0] * torch.randn((*atom_mask.shape, 3), device=atom_mask.device)

        for tau, c_tau in enumerate(noise_schedule[1:]):

            x = centre_random_augmentation(pos=x, pos_mask=atom_mask)

            gamma = self.gamma_0 if c_tau > self.gamma_min else 0

            t = noise_schedule[tau-1] * (gamma + 1)

            xi = self.noise_scale * torch.sqrt(t ** 2 - noise_schedule[tau-1] ** 2) * torch.randn_like(x)

            x_noisy = x + xi

            x_denoised = self.diffusion_module(batch=batch,
                                               x_noisy=x_noisy,
                                               t=t,
                                               s_input=s_input,
                                               s_trunk=s_trunk,
                                               z_trunk=z_trunk,
                                               token_mask=token_mask,
                                               pair_token_mask=pair_token_mask,
                                               atom_mask=atom_mask)

            delta = (x - x_denoised) / t
            dt = c_tau - t
            x = x_noisy + self.step_scale * dt * delta

        return x