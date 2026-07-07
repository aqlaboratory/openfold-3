# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
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

import logging

import torch
import torch.nn as nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.layers.diffusion_conditioning import DiffusionConditioning
from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
from openfold3.core.model.layers.sequence_local_atom_attention import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
)
from openfold3.core.model.primitives import LayerNorm, Linear

# Re-exported from a lightweight module so that data-pipeline imports do not pull
# in the heavy model stack (e.g. cuequivariance). See augmentation.py / issue #268.
from openfold3.core.model.structure.augmentation import (  # noqa: F401
    centre_random_augmentation,
    sample_rotations,
)

logger = logging.getLogger(__name__)


# Move this somewhere else?
def create_noise_schedule(
    no_rollout_steps: float,
    sigma_data: float,
    s_max: float,
    s_min: float,
    p: int,
    dtype: torch.dtype,
    device: torch.device,
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
        dtype:
            Dtype of noise schedule
        device:
            Device of noise schedule
    Returns:
        Noise schedule
    """
    t = (
        torch.arange(0, 1 + no_rollout_steps, dtype=dtype, device=device)
        / no_rollout_steps
    )
    return (
        sigma_data * (s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p
    )


def _batch_scalar(batch: dict, name: str, default, cast=float):
    value = batch.get(name)
    if value is None:
        return default
    return cast(value.flatten()[0].item())


def _feature_mask(batch: dict, atom_mask: torch.Tensor, name: str) -> torch.Tensor:
    mask = batch[name].to(device=atom_mask.device).bool()
    batch_dim = atom_mask.shape[0]
    while mask.dim() > 2 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and batch_dim > 1:
        mask = mask.expand(batch_dim, -1)
    return mask


def _score_ligand_pose(
    lig_pose: torch.Tensor,
    protein: torch.Tensor,
    pocket: torch.Tensor,
    lig_vdw: torch.Tensor,
    protein_vdw: torch.Tensor,
    contact_distance: float,
    vdw_buffer: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank a ligand proposal by site entry."""
    d_prot = torch.cdist(lig_pose.unsqueeze(0), protein.unsqueeze(0)).squeeze(0)
    vdw_lower = (lig_vdw[:, None] + protein_vdw[None, :]) * (1.0 - vdw_buffer)
    overlap = torch.relu(vdw_lower - d_prot)
    vdw_overlap = torch.square(overlap).sum()
    min_prot = d_prot.min()

    d_pocket = torch.cdist(lig_pose.unsqueeze(0), pocket.unsqueeze(0)).squeeze(0)
    lig_to_pocket = d_pocket.min(dim=1).values
    contact = torch.square(torch.relu(lig_to_pocket - contact_distance)).mean()
    lig_atoms_in_pocket = (lig_to_pocket < contact_distance).sum().to(lig_pose.dtype)
    pocket_com_dist = torch.linalg.vector_norm(
        lig_pose.mean(dim=0) - pocket.mean(dim=0)
    )

    return pocket_com_dist, vdw_overlap, min_prot, lig_atoms_in_pocket, contact


def _candidate_sort_key(item) -> tuple[float, float, float, float, float]:
    return (
        float(item[0]),
        float(item[3]),
        float(item[4]),
        float(item[1]),
        -float(item[2]),
    )


def _build_pocket_sampling_seeds(
    batch: dict,
    xl_base: torch.Tensor,
    atom_mask: torch.Tensor,
    no_rollout_samples: int,
) -> torch.Tensor:
    """Generate ligand seeds in a user-specified pocket for partial diffusion."""
    batch_dim, num_atoms = atom_mask.shape[0], atom_mask.shape[-1]
    lig_mask = _feature_mask(batch, atom_mask, "pocket_sampling_ligand_atom_mask")
    pocket_mask = _feature_mask(batch, atom_mask, "pocket_sampling_pocket_atom_mask")
    lig_idx = torch.nonzero(lig_mask[0], as_tuple=False).flatten()
    pocket_idx = torch.nonzero(pocket_mask[0], as_tuple=False).flatten()
    protein_idx = torch.nonzero(atom_mask[0].bool() & ~lig_mask[0], as_tuple=False)
    protein_idx = protein_idx.flatten()

    radii = batch.get("pocket_sampling_vdw_radii")
    if radii is None:
        radii = torch.full(
            (num_atoms,), 1.70, dtype=atom_mask.dtype, device=atom_mask.device
        )
    else:
        radii = radii.to(device=atom_mask.device, dtype=atom_mask.dtype)
        while radii.dim() > 1:
            radii = radii[0]
    lig_vdw = radii[lig_idx]
    protein_vdw = radii[protein_idx]

    conformers = batch.get("pocket_sampling_conformer_rels")
    if conformers is not None:
        conformers = conformers.to(device=atom_mask.device, dtype=atom_mask.dtype)
        while conformers.dim() > 3 and conformers.shape[0] == 1:
            conformers = conformers[0]

    n_parents = max(
        1,
        min(
            no_rollout_samples,
            _batch_scalar(batch, "pocket_sampling_num_parents", 16, int),
        ),
    )
    n_candidates = max(
        no_rollout_samples,
        _batch_scalar(batch, "pocket_sampling_candidates", 1024, int),
    )
    contact_distance = _batch_scalar(
        batch, "pocket_sampling_contact_distance", 4.0, float
    )
    translate = _batch_scalar(batch, "pocket_sampling_translate", 1.0, float)
    center_jitter = _batch_scalar(batch, "pocket_sampling_center_jitter", 4.0, float)
    surface_jitter = _batch_scalar(batch, "pocket_sampling_surface_jitter", 1.5, float)
    vdw_buffer = _batch_scalar(batch, "pocket_sampling_vdw_buffer", 0.225, float)
    diversity_rmsd = _batch_scalar(batch, "pocket_sampling_diversity_rmsd", 0.5, float)

    all_seed_batches = []
    for b in range(batch_dim):
        parent_scores = []
        for s in range(no_rollout_samples):
            score, _, _, _, _ = _score_ligand_pose(
                xl_base[b, s, lig_idx],
                xl_base[b, s, protein_idx],
                xl_base[b, s, pocket_idx],
                lig_vdw,
                protein_vdw,
                contact_distance,
                vdw_buffer,
            )
            parent_scores.append(score)
        parent_order = torch.argsort(torch.stack(parent_scores))[:n_parents]

        candidates = []
        for i in range(n_candidates):
            parent_slot = int(parent_order[i % n_parents])
            parent = xl_base[b, parent_slot]
            pocket_parent = parent[pocket_idx]
            lig_parent = parent[lig_idx]
            lig_com = lig_parent.mean(dim=0, keepdim=True)
            parent_rel = lig_parent - lig_com

            if i < n_parents:
                pose = lig_parent
            else:
                if conformers is None or conformers.shape[0] == 0:
                    lig_rel = parent_rel
                else:
                    lig_rel = conformers[
                        torch.randint(conformers.shape[0], (), device=atom_mask.device)
                    ]
                rot = sample_rotations(
                    shape=(),
                    dtype=atom_mask.dtype,
                    device=atom_mask.device,
                )
                if torch.rand((), device=atom_mask.device) < 0.5:
                    target = pocket_parent.mean(dim=0, keepdim=True)
                    target = target + center_jitter * torch.randn(
                        (1, 3), dtype=atom_mask.dtype, device=atom_mask.device
                    )
                else:
                    target = pocket_parent[
                        torch.randint(
                            pocket_parent.shape[0], (), device=atom_mask.device
                        )
                    ].unsqueeze(0)
                    target = target + surface_jitter * torch.randn(
                        (1, 3), dtype=atom_mask.dtype, device=atom_mask.device
                    )
                pose = lig_rel @ rot.transpose(-1, -2) + target
                if translate > 0:
                    pose = pose + translate * torch.randn(
                        (1, 3), dtype=atom_mask.dtype, device=atom_mask.device
                    )

            score, vdw_overlap, min_prot, lig_atoms, contact = _score_ligand_pose(
                pose,
                parent[protein_idx],
                parent[pocket_idx],
                lig_vdw,
                protein_vdw,
                contact_distance,
                vdw_buffer,
            )
            candidates.append(
                (
                    score,
                    vdw_overlap,
                    min_prot,
                    -lig_atoms,
                    contact,
                    pose,
                    parent_slot,
                )
            )

        candidates.sort(key=_candidate_sort_key)
        selected = []
        for cand in candidates:
            if all(
                torch.sqrt(torch.square(cand[5] - prev[5]).mean()).item()
                >= diversity_rmsd
                for prev in selected
            ):
                selected.append(cand)
                if len(selected) == no_rollout_samples:
                    break
        if len(selected) < no_rollout_samples:
            selected.extend(candidates[: no_rollout_samples - len(selected)])

        seed_list = []
        for cand in selected[:no_rollout_samples]:
            seed = xl_base[b, cand[6]].clone()
            seed[lig_idx] = cand[5]
            seed_list.append(seed)
        all_seed_batches.append(torch.stack(seed_list, dim=0))

        best = selected[0]
        logger.info(
            "[pocket_sampling] parents=%s/%s candidates=%s "
            "best_pocket_com=%.3g best_vdw=%.3g "
            "best_min_protein=%.3g best_lig_atoms=%s",
            n_parents,
            no_rollout_samples,
            n_candidates,
            float(best[0]),
            float(best[1]),
            float(best[2]),
            int(-best[3].item()),
        )

    return torch.stack(all_seed_batches, dim=0)


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

        self.layer_norm_s = LayerNorm(self.c_s, create_offset=False)
        self.linear_s = Linear(
            self.c_s,
            self.c_token,
            **diff_mod_init.linear_s,
        )

        self.diffusion_transformer = DiffusionTransformer(
            **config.diffusion_transformer
        )

        self.layer_norm_a = LayerNorm(self.c_token, create_offset=False)

        self.atom_attn_dec = AtomAttentionDecoder(**config.atom_attn_dec)

    def forward(
        self,
        batch: dict,
        xl_noisy: torch.Tensor,
        token_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        t: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        use_conditioning: bool,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Feature dictionary
            xl_noisy:
                [*, N_atom, 3] Noisy atom positions
            token_mask:
                [*, N_token] Token mask
            atom_mask:
                [*, N_atom] Atom mask. In the training step this is the
                ground truth mask, but in the mini/full rollout this is
                the padding mask.
            t:
                [*] Noise level at a diffusion step
            si_input:
                [*, N_token, c_s_input] Input embedding
            si_trunk:
                [*, N_token, c_s] Single representation
            zij_trunk:
                [*, N_token, c_s] Pair representation
            use_conditioning:
                Whether to condition with the trunk representations
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_triton_triangle_kernels:
                Whether to use Triton triangle attention kernel
            use_lma:
                Whether to use LMA
            use_high_precision_attention:
                Whether to run attention in high precision
            _mask_trans:
                Whether to mask the output of the transition layer
        Returns:
            [*, N_atom, 3] Denoised atom positions
        """
        si, zij = self.diffusion_conditioning(
            batch=batch,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_conditioning=use_conditioning,
            chunk_size=chunk_size,
        )

        xl_noisy = xl_noisy * atom_mask[..., None]

        rl_noisy = xl_noisy / torch.sqrt(t[..., None, None] ** 2 + self.sigma_data**2)

        # Note: These modules are not memory-intensive compared to other parts of the
        # model (i.e. TemplateStack) so chunking is unnecessary for now.
        ai, ql, cl, plm = self.atom_attn_enc(
            batch=batch,
            rl=rl_noisy,
            si_trunk=si_trunk,
            zij_trunk=zij,  # Use conditioned trunk representation
            use_high_precision_attention=use_high_precision_attention,
        )

        ai = ai + self.linear_s(self.layer_norm_s(si))

        ai = self.diffusion_transformer(
            a=ai,
            s=si,
            z=zij,
            mask=token_mask,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_triton_triangle_kernels=use_triton_triangle_kernels,
            use_lma=use_lma,
            use_high_precision_attention=use_high_precision_attention,
            _mask_trans=_mask_trans,
        )

        ai = self.layer_norm_a(ai)

        rl_update = self.atom_attn_dec(
            batch=batch,
            ai=ai,
            ql=ql,
            cl=cl,
            plm=plm,
            use_high_precision_attention=use_high_precision_attention,
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

        xl_out = xl_out * atom_mask[..., None]

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

    def _sample_rollout(
        self,
        batch: dict,
        xl: torch.Tensor,
        atom_mask: torch.Tensor,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        noise_schedule: torch.Tensor,
        start_step: int,
        use_conditioning: bool,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
    ) -> torch.Tensor:
        """Run the standard OF3 denoising loop from a chosen schedule index."""
        for tau, c_tau in enumerate(noise_schedule[1:]):
            if tau < start_step:
                continue
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
                token_mask=batch["token_mask"],
                atom_mask=atom_mask,
                t=t.to(xl_noisy.device),
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                use_conditioning=use_conditioning,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                use_high_precision_attention=use_high_precision_attention,
                _mask_trans=_mask_trans,
            )

            delta = (xl_noisy - xl_denoised) / t
            dt = c_tau - t
            xl = xl_noisy + self.step_scale * dt * delta

        return xl

    def forward(
        self,
        batch: dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        noise_schedule: torch.Tensor,
        no_rollout_samples: int,
        use_conditioning: bool = True,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        use_high_precision_attention: bool = False,
        _mask_trans: bool = True,
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
            no_rollout_samples:
                [no_rollout_samples] Number of samples to generate for rollout
            use_conditioning:
                Whether to condition with the trunk representations
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_triton_triangle_kernels:
                Whether to use Triton triangle attention kernel
            use_lma:
                Whether to use LMA
            use_high_precision_attention:
                Whether to run attention in high precision
            _mask_trans:
                Whether to mask the output of the transition layer
        Returns:
            [*, N_atom, 3] Sampled atom positions
        """
        atom_mask = batch["atom_mask"]
        batch_dim, num_atoms = atom_mask.shape[0], atom_mask.shape[-1]

        total_steps = len(noise_schedule) - 1

        xl = noise_schedule[0] * torch.randn(
            (batch_dim, no_rollout_samples, num_atoms, 3),
            device=atom_mask.device,
            dtype=atom_mask.dtype,
        )

        rollout_kwargs = {
            "batch": batch,
            "atom_mask": atom_mask,
            "si_input": si_input,
            "si_trunk": si_trunk,
            "zij_trunk": zij_trunk,
            "noise_schedule": noise_schedule,
            "use_conditioning": use_conditioning,
            "chunk_size": chunk_size,
            "use_deepspeed_evo_attention": use_deepspeed_evo_attention,
            "use_cueq_triangle_kernels": use_cueq_triangle_kernels,
            "use_triton_triangle_kernels": use_triton_triangle_kernels,
            "use_lma": use_lma,
            "use_high_precision_attention": use_high_precision_attention,
            "_mask_trans": _mask_trans,
        }

        xl = self._sample_rollout(xl=xl, start_step=0, **rollout_kwargs)

        pocket_sampling_enabled = bool(
            _batch_scalar(batch, "pocket_sampling_enabled", False, bool)
        )
        if pocket_sampling_enabled:
            if batch_dim != 1:
                raise ValueError(
                    "Pocket proposal/refinement currently supports one query per "
                    "model batch"
                )
            seed = _build_pocket_sampling_seeds(
                batch=batch,
                xl_base=xl.detach(),
                atom_mask=atom_mask,
                no_rollout_samples=no_rollout_samples,
            )
            pocket_sampling_start_frac = _batch_scalar(
                batch, "pocket_sampling_start_frac", 0.75, float
            )
            pocket_sampling_jitter = _batch_scalar(
                batch, "pocket_sampling_ligand_jitter", 0.25, float
            )
            pocket_sampling_start_step = max(
                0,
                min(
                    total_steps - 1,
                    int(round(pocket_sampling_start_frac * total_steps)),
                ),
            )
            lig_mask = _feature_mask(
                batch, atom_mask, "pocket_sampling_ligand_atom_mask"
            )
            lig = lig_mask[:, None, :, None]
            lig_jitter = pocket_sampling_jitter * torch.randn(
                (batch_dim, no_rollout_samples, 1, 3),
                device=atom_mask.device,
                dtype=atom_mask.dtype,
            )
            seed = torch.where(lig, seed + lig_jitter, seed)
            xl = seed + noise_schedule[pocket_sampling_start_step] * torch.randn_like(
                seed
            )
            logger.info(
                "[pocket_sampling] start_frac=%.3f start_step=%s/%s "
                "sigma=%.3g jitter=%.3g seed_proposals=%s",
                pocket_sampling_start_frac,
                pocket_sampling_start_step,
                total_steps,
                float(noise_schedule[pocket_sampling_start_step]),
                pocket_sampling_jitter,
                seed.shape[1],
            )
            xl = self._sample_rollout(
                xl=xl,
                start_step=pocket_sampling_start_step,
                **rollout_kwargs,
            )

        return xl
