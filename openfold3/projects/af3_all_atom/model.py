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
The main inference and training loops for AlphaFold3.
"""

import random

import torch
from ml_collections import ConfigDict
from torch import nn

from openfold3.core.model.feature_embedders.input_embedders import (
    InputEmbedderAllAtom,
    MSAModuleEmbedder,
)
from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
from openfold3.core.model.latent.msa_module import MSAModuleStack
from openfold3.core.model.latent.pairformer import PairFormerStack
from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.model.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
    centre_random_augmentation,
    create_noise_schedule,
)
from openfold3.core.utils.permutation_alignment import (
    safe_multi_chain_permutation_alignment,
)
from openfold3.core.utils.tensor_utils import add, tensor_tree_map


class AlphaFold3(nn.Module):
    """
    Alphafold 3.

    Implements AF3 Algorithm 1 main loop (but with training).
    """

    def __init__(self, config: ConfigDict):
        """
        Args:
            config:
                The model configuration as a ConfigDict object.
        """
        super().__init__()
        self.config = config
        self.settings = self.config.settings
        self.shared = self.config.architecture.shared

        self.input_embedder = InputEmbedderAllAtom(
            **self.config.architecture.input_embedder
        )

        self.layer_norm_z = LayerNorm(self.shared.c_z)
        self.linear_z = Linear(self.shared.c_z, self.shared.c_z, bias=False)

        self.template_embedder = TemplateEmbedderAllAtom(
            config=self.config.architecture.template
        )

        self.msa_module_embedder = MSAModuleEmbedder(
            **self.config.architecture.msa.msa_module_embedder
        )
        self.msa_module = MSAModuleStack(**self.config.architecture.msa.msa_module)

        self.layer_norm_s = LayerNorm(self.shared.c_s)
        self.linear_s = Linear(self.shared.c_s, self.shared.c_s, bias=False)

        self.pairformer_stack = PairFormerStack(**self.config.architecture.pairformer)

        self.diffusion_module = DiffusionModule(
            config=self.config.architecture.diffusion_module
        )

        self.sample_diffusion = SampleDiffusion(
            **self.config.architecture.sample_diffusion,
            diffusion_module=self.diffusion_module,
        )

        # Confidence and Distogram Heads
        self.aux_heads = AuxiliaryHeadsAllAtom(config=self.config.architecture.heads)

    def _disable_activation_checkpointing(self):
        """
        Disable activation checkpointing for the TemplateEmbedder, MSAModule,
        and Pairformer.
        """
        self.template_embedder.template_pair_stack.blocks_per_ckpt = None
        self.msa_module.blocks_per_ckpt = None
        self.pairformer_stack.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        """
        Enable activation checkpointing for the TemplateEmbedder, MSAModule,
        and Pairformer.
        """
        self.template_embedder.template_pair_stack.blocks_per_ckpt = (
            self.config.architecture.template.template_pair_stack.blocks_per_ckpt
        )
        self.msa_module.blocks_per_ckpt = (
            self.config.architecture.msa.msa_module.blocks_per_ckpt
        )
        self.pairformer_stack.blocks_per_ckpt = (
            self.config.architecture.pairformer.blocks_per_ckpt
        )

    def run_trunk(
        self, batch: dict, num_cycles: int, inplace_safe: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implements Algorithm 1 lines 1-14.

        Args:
            batch:
                Input feature dictionary
            num_cycles:
                Number of cycles to run
            inplace_safe:
                Whether inplace operations can be performed

        Returns:
            s_input:
                [*, N_token, C_s_input] Single (input) representation
            s:
                [*, N_token, C_s] Single representation
            z:
                [*, N_token, N_token, C_z] Pair representation
        """
        s_input, s_init, z_init = self.input_embedder(
            batch=batch,
            inplace_safe=inplace_safe,
            use_high_precision_attention=True,
        )

        # s: [*, N_token, C_s]
        # z: [*, N_token, N_token, C_z]
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        # token_mask: [*, N_token]
        # pair_mask: [*, N_token, N_token]
        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        is_grad_enabled = torch.is_grad_enabled()

        for cycle_no in range(num_cycles):
            is_final_iter = cycle_no == (num_cycles - 1)

            # Enable grad when we're training, only enable grad on the last cycle
            enable_grad = is_grad_enabled and is_final_iter
            with torch.set_grad_enabled(enable_grad):
                if is_final_iter and torch.is_autocast_enabled():
                    # Sidestep AMP bug (PyTorch issue #65766)
                    torch.clear_autocast_cache()

                # [*, N_token, N_token, C_z]
                z = z_init + self.linear_z(self.layer_norm_z(z))

                z = add(
                    z,
                    self.template_embedder(
                        batch=batch,
                        z=z,
                        pair_mask=pair_mask,
                        chunk_size=self.settings.chunk_size,
                        _mask_trans=True,
                        use_deepspeed_evo_attention=self.settings.use_deepspeed_evo_attention,
                        use_lma=self.settings.use_lma,
                        inplace_safe=inplace_safe,
                    ),
                    inplace=inplace_safe,
                )

                m, msa_mask = self.msa_module_embedder(batch=batch, s_input=s_input)

                # Run MSA + pair embeddings through the MsaModule
                # m: [*, N_seq, N_token, C_m]
                # z: [*, N_token, N_token, C_z]
                if self.settings.offload_inference:
                    input_tensors = [m, z]
                    del m, z
                    z = self.msa_module.forward_offload(
                        input_tensors,
                        msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                        pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                        chunk_size=self.settings.chunk_size,
                        use_deepspeed_evo_attention=self.settings.use_deepspeed_evo_attention,
                        use_lma=self.settings.use_lma,
                        _mask_trans=True,
                    )

                    del input_tensors, msa_mask
                else:
                    z = self.msa_module(
                        m,
                        z,
                        msa_mask=msa_mask.to(dtype=m.dtype),
                        pair_mask=pair_mask.to(dtype=z.dtype),
                        chunk_size=self.settings.chunk_size,
                        use_deepspeed_evo_attention=self.settings.use_deepspeed_evo_attention,
                        use_lma=self.settings.use_lma,
                        inplace_safe=inplace_safe,
                        _mask_trans=True,
                    )

                    del m, msa_mask

                s = s_init + self.linear_s(self.layer_norm_s(s))
                s, z = self.pairformer_stack(
                    s=s,
                    z=z,
                    single_mask=token_mask.to(dtype=z.dtype),
                    pair_mask=pair_mask.to(dtype=s.dtype),
                    chunk_size=self.settings.chunk_size,
                    use_deepspeed_evo_attention=self.settings.use_deepspeed_evo_attention,
                    use_lma=self.settings.use_lma,
                    inplace_safe=inplace_safe,
                    _mask_trans=True,
                )

        del s_init, z_init

        return s_input, s, z

    def _rollout(
        self,
        batch: dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        inplace_safe: bool = False,
    ) -> dict:
        """
        Mini diffusion rollout described in section 4.1.
        Implements Algorithm 1 lines 15-18.

        Args:
            batch:
                Input feature dictionary
            si_input:
                [*, N_token, C_s_input] Single (input) representation
            si_trunk:
                [*, N_token, C_s] Single representation output from model trunk
            zij_trunk:
                [*, N_token, N_token, C_z] Pair representation output from model trunk
            inplace_safe:
                Whether inplace operations can be performed

        Returns:
            Output dictionary containing the predicted trunk embeddings,
            all-atom positions, and confidence/distogram head logits
        """
        # Determine number of rollout steps and samples depending on training/eval mode
        no_rollout_steps = (
            self.shared.diffusion.no_mini_rollout_steps
            if self.training
            else self.shared.diffusion.no_full_rollout_steps
        )

        no_rollout_samples = (
            self.shared.diffusion.no_mini_rollout_samples
            if self.training
            else self.shared.diffusion.no_full_rollout_samples
        )

        # Compute atom positions
        with (
            torch.no_grad(),
            torch.amp.autocast(device_type="cuda", dtype=torch.float32),
        ):
            noise_schedule = create_noise_schedule(
                no_rollout_steps=no_rollout_steps,
                **self.config.architecture.noise_schedule,
                dtype=si_input.dtype,
                device=si_input.device,
            )

            atom_positions_predicted = self.sample_diffusion(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                noise_schedule=noise_schedule,
                no_rollout_samples=no_rollout_samples,
                use_deepspeed_evo_attention=self.settings.use_deepspeed_evo_attention,
                use_lma=self.settings.use_lma,
                _mask_trans=True,
            )

        output = {
            "si_trunk": si_trunk,
            "zij_trunk": zij_trunk,
            "atom_positions_predicted": atom_positions_predicted,
        }

        # Compute confidence logits
        aux_heads_out = self.aux_heads(
            batch=batch,
            si_input=si_input,
            output=output,
            use_deepspeed_evo_attention=self.settings.use_deepspeed_evo_attention,
            use_lma=self.settings.use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=True,
        )

        output.update(
            {
                k: v.to(dtype=atom_positions_predicted.dtype)
                for k, v in aux_heads_out.items()
            }
        )

        return output

    def _train_diffusion(
        self,
        batch: dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> dict:
        """
        Run diffusion training over no_samples noised versions of the input structure.

        Args:
            batch:
                Input feature dictionary
            si_input:
                [*, N_token, C_s_input] Single (input) representation
            si_trunk:
                [*, N_token, C_s] Single representation output from model trunk
            zij_trunk:
                [*, N_token, N_token, C_z] Pair representation output from model trunk

        Returns:
            Output dictionary containing the following keys:
                "noise_level" ([*])
                    Noise level at a diffusion step
                "atom_positions_diffusion" ([*, N_samples, N_atom, 3]):
                    Predicted atom positions
        """
        xl_gt = batch["ground_truth"]["atom_positions"]
        atom_mask_gt = batch["ground_truth"]["atom_resolved_mask"]

        # Sample noise schedule for training
        no_samples = self.shared.diffusion.no_samples
        batch_size, n_atom = xl_gt.shape[0], xl_gt.shape[-2]
        device, dtype = xl_gt.device, xl_gt.dtype

        xl_gt = xl_gt.tile((1, no_samples, 1, 1))
        xl_gt = centre_random_augmentation(xl=xl_gt, atom_mask=atom_mask_gt)
        n = torch.randn((batch_size, no_samples), device=device, dtype=dtype)
        t = self.shared.diffusion.sigma_data * torch.exp(-1.2 + 1.5 * n)

        # Sample noise
        noise = t[..., None, None] * torch.randn(
            (batch_size, no_samples, n_atom, 3), device=device, dtype=dtype
        )

        # Sample atom positions
        xl_noisy = xl_gt + noise

        token_mask = batch["token_mask"]

        # Run diffusion module
        xl = self.diffusion_module(
            batch=batch,
            xl_noisy=xl_noisy,
            token_mask=token_mask,
            atom_mask=atom_mask_gt,
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            use_high_precision_attention=True,
            _mask_trans=True,
        )

        output = {
            "noise_level": t,
            "atom_positions_diffusion": xl,
        }

        return output

    def forward(self, batch: dict) -> tuple[dict, dict]:
        """
        Args:
            batch:
                Dictionary of arguments outlined in supplement section 2.8. Keys must
                include the official names of the features in Table 5, as well as
                additional features noted with a *.

                Features:
                    "residue_index" ([*, N_token])
                        Residue number in the token’s original input chain
                    "token_index" ([*, N_token])
                        Token number
                    "asym_id" ([*, N_token])
                        Unique integer for each distinct chain
                    "entity_id" ([*, N_token])
                        Unique integer for each distinct sequence
                    "sym_id" ([*, N_token])
                        Unique integer within chains of this sequence
                    "restype" ([*, N_token, 32])
                        One-hot encoding of the sequence
                    "is_protein" ([*, N_token])
                        Molecule type mask
                    "is_rna" ([*, N_token])
                        Molecule type mask
                    "is_dna" ([*, N_token])
                        Molecule type mask
                    "is_ligand" ([*, N_token])
                        Molecule type mask
                    "ref_pos" ([*, N_atom, 3])
                        Atom positions (reference conformer)
                    "ref_mask" ([*, N_atom])
                        Atom mask (reference conformer)
                    "ref_element" ([*, N_atom, 128])
                        One-hot encoding of the element atomic
                        number (reference conformer)
                    "ref_charge" ([*, N_atom])
                        Atom charge (reference conformer)
                    "ref_atom_name_chars" ([*, N_atom, 4, 64])
                        One-hot encoding of the unique atom names (reference conformer)
                    "ref_space_uid" ([*, N_atom])
                        Numerical encoding of the chain id and residue
                        index (reference conformer)
                    "msa": ([*, N_msa, N_token, 32])
                        One-hot encoding of the processed MSA
                    "has_deletion" ([*, N_msa, N_token])
                        Binary feature indicating if there is a deletion to
                        the left of each MSA position
                    "deletion_value" ([*, N_msa, N_token])
                        Raw deletion counts
                    "profile" ([*, N_token, 32])
                        Distribution across restypes in the main MSA
                    "deletion_mean" ([*, N_token])
                        Mean number of deletions at each position in the main MSA
                    "template_restype": ([*, N_templ, N_token, 32])
                        One-hot encoding of the template sequence
                    "template_pseudo_beta_mask" ([*, N_templ, N_token])
                        Mask for template C_beta atoms (C_alpha for glycine)
                    "template_backbone_frame_mask" ([*, N_templ, N_token])
                        Mask indicating if required template atoms exist to
                        compute frames
                    "template_distogram" ([*, N_templ, N_token, N_token, 39])
                        A one-hot pairwise feature indicating the distance between
                        C_beta atoms (C_alpha for glycine) in the template
                    "template_unit_vector"([*, N_templ, N_token, N_token, 3])
                        The unit vector between pairs of C_alpha atoms within
                        the local frame of each template residue
                    "token_bonds" ([*, N_token, N_token])
                        A 2D matrix indicating if there is a bond between
                        any atom in token i and token j
                    *"num_atoms_per_token" ([*, N_token])
                        Number of atoms per token
                    *"start_atom_index" ([*, N_token])
                        Starting atom index in each token
                    *"token_mask" ([*, N_token])
                        Token-level mask
                    *"msa_mask" ([*, N_msa, N_token])
                        MSA mask
                    *"num_paired_seqs" ([*])
                        Number of main MSA seqs used in MSA sampling (non-uniprot)
                    "ground_truth" (Dict):
                        "residue_index" ([*, N_token])
                            Residue number in the token’s original input chain
                        "token_index" ([*, N_token])
                            Token number
                        "asym_id" ([*, N_token])
                            Unique integer for each distinct chain
                        "entity_id" ([*, N_token])
                            Unique integer for each distinct sequence
                        "is_protein" ([*, N_token])
                            Molecule type mask
                        "is_ligand" ([*, N_token])
                            Molecule type mask
                        "token_bonds" ([*, N_token, N_token])
                            A 2D matrix indicating if there is a bond between
                            any atom in token i and token j
                        *"num_atoms_per_token" ([*, N_token])
                            Number of atoms per token
                        *"token_mask" ([*, N_token])
                            Token-level mask
                        *"atom_positions" ([*, N_atom, 3])
                            Ground truth atom positions for training
                        *"atom_resolved_mask" ([*, N_atom])
                            Mask for ground truth atom positions
        Returns:
            batch: Updated batch dictionary post permutation alignment
            output: Output dictionary containing the following keys:
                "si_trunk" ([*, N_token, C_s]):
                    Single representation output from model trunk
                "zij_trunk" ([*, N_token, N_token, C_z]):
                    Pair representation output from model trunk
                "atom_positions_predicted" ([*, N_atom, 3]):
                    Predicted atom positions
                "plddt_logits" ([*, N_atom, 50]):
                    Predicted binned PLDDT logits
                "pae_logits" ([*, N_token, N_token, 64]):
                    Predicted binned PAE logits
                "pde_logits" ([*, N_token, N_token, 64]):
                    Predicted binned PDE logits
                "experimentally_resolved_logits" ([*, N_atom, 2]):
                    Predicted binned experimentally resolved logits
                "distogram_logits" ([*, N_token, N_token, 64]):
                    Predicted binned distogram logits
                "noise_level" ([*])
                    Training only, noise level at a diffusion step
                "atom_positions_diffusion" ([*, N_samples, N_atom, 3]):
                    Training only, predicted atom positions

        """
        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        num_cycles = (
            random.randint(1, self.shared.max_cycles)
            if self.training
            else self.shared.max_cycles
        )

        output = {"recycles": num_cycles}

        # Compute representations
        si_input, si_trunk, zij_trunk = self.run_trunk(
            batch=batch, num_cycles=num_cycles, inplace_safe=inplace_safe
        )

        # Expand sampling dimension for rollout and diffusion
        si_input = si_input.unsqueeze(1)
        si_trunk = si_trunk.unsqueeze(1)
        zij_trunk = zij_trunk.unsqueeze(1)

        # Expand sampling dimension for batch features
        # Exclude ref_space_uid_to_perm feature since this
        # does not have a proper batch dimension
        ref_space_uid_to_perm = batch.pop("ref_space_uid_to_perm", None)
        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
        batch["ref_space_uid_to_perm"] = ref_space_uid_to_perm

        # Mini rollout
        rollout_output = self._rollout(
            batch=batch,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            inplace_safe=inplace_safe,
        )

        output.update(rollout_output)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            # Apply permutation alignment for training and validation
            if "ground_truth" in batch:
                # Update the ground-truth coordinates/mask in-place with the correct
                # permutation (and optionally disable losses in case of a
                # critical error)
                with torch.no_grad():
                    safe_multi_chain_permutation_alignment(
                        batch=batch,
                        atom_positions_predicted=output["atom_positions_predicted"],
                    )

            if self.training:  # noqa: SIM102
                # Run training step (if necessary)
                if self.settings.diffusion_training_enabled:
                    if torch.is_autocast_enabled():
                        # Sidestep AMP bug (PyTorch issue #65766)
                        # Needed due to no_grad in _rollout
                        torch.clear_autocast_cache()

                    diffusion_output = self._train_diffusion(
                        batch=batch,
                        si_input=si_input,
                        si_trunk=si_trunk,
                        zij_trunk=zij_trunk,
                    )

                    output.update(diffusion_output)

        return batch, output
