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

from typing import Dict, Tuple

import torch
from ml_collections import ConfigDict
from torch import nn

from openfold3.core.model.feature_embedders import InputEmbedderAllAtom
from openfold3.core.model.feature_embedders.input_embedders import MSAModuleEmbedder
from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
from openfold3.core.model.latent.msa_module import MSAModuleStack
from openfold3.core.model.latent.pairformer import PairFormerStack
from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.model.structure.diffusion_module import (
    DiffusionModule,
    SampleDiffusion,
)

# from openfold3.core.utils.multi_chain_permutation import multi_chain_permutation_align
from openfold3.core.utils.tensor_utils import add


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
        self.globals = self.config.globals

        self.input_embedder = InputEmbedderAllAtom(**self.config.model.input_embedder)

        self.layer_norm_z = LayerNorm(self.globals.c_z)
        self.linear_z = Linear(self.globals.c_z, self.globals.c_z, bias=False)

        self.template_embedder = TemplateEmbedderAllAtom(
            config=self.config.model.template
        )

        self.msa_module_embedder = MSAModuleEmbedder(
            **self.config.model.msa.msa_module_embedder
        )
        self.msa_module = MSAModuleStack(**self.config.model.msa.msa_module)

        self.layer_norm_s = LayerNorm(self.globals.c_s)
        self.linear_s = Linear(self.globals.c_s, self.globals.c_s, bias=False)

        self.pairformer_stack = PairFormerStack(**self.config.model.pairformer)

        self.diffusion_module = DiffusionModule(
            config=self.config.model.diffusion_module
        )

        self.sample_diffusion = SampleDiffusion(
            **self.config.model.sample_diffusion, diffusion_module=self.diffusion_module
        )

        # Confidence and Distogram Heads
        self.aux_heads = AuxiliaryHeadsAllAtom(config=self.config.model.heads)

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
            self.config.template.template_pair_stack.blocks_per_ckpt
        )
        self.msa_module.blocks_per_ckpt = self.config.msa.msa_module.blocks_per_ckpt
        self.pairformer_stack.blocks_per_ckpt = self.config.pairformer.blocks_per_ckpt

    def run_trunk(
        self, batch: Dict, inplace_safe: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implements Algorithm 1 lines 1-14.

        Args:
            batch:
                Input feature dictionary
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
        s_input, s_init, z_init = self.input_embedder(batch=batch)

        # s: [*, N_token, C_s]
        # z: [*, N_token, N_token, C_z]
        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        # token_mask: [*, N_token]
        # pair_mask: [*, N_token, N_token]
        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        for _ in range(self.globals.no_cycles):
            # [*, N_token, N_token, C_z]
            z = z_init + self.linear_z(self.layer_norm_z(z))

            z = add(
                z,
                self.template_embedder(
                    batch=batch,
                    z=z,
                    pair_mask=pair_mask,
                    chunk_size=self.globals.chunk_size,
                    _mask_trans=True,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_lma=self.globals.use_lma,
                    inplace_safe=inplace_safe,
                ),
                inplace=inplace_safe,
            )

            m, msa_mask = self.msa_module_embedder(batch=batch, s_input=s_input)

            # Run MSA + pair embeddings through the MsaModule
            # m: [*, N_seq, N_token, C_m]
            # z: [*, N_token, N_token, C_z]
            if self.globals.offload_inference:
                input_tensors = [m, z]
                del m, z
                z = self.msa_module.forward_offload(
                    input_tensors,
                    msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                    pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_lma=self.globals.use_lma,
                    _mask_trans=True,
                )

                del input_tensors, msa_mask
            else:
                z = self.msa_module(
                    m,
                    z,
                    msa_mask=msa_mask.to(dtype=m.dtype),
                    pair_mask=pair_mask.to(dtype=z.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                    use_lma=self.globals.use_lma,
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
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=True,
            )

        del s_init, z_init

        return s_input, s, z

    def _rollout(
        self,
        batch: Dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> Dict:
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

        Returns:
            Output dictionary containing the predicted trunk embeddings,
            all-atom positions, and confidence/distogram head logits
        """
        # Compute atom positions
        with torch.no_grad():
            x_pred = self.sample_diffusion(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                chunk_size=self.globals.chunk_size,
            )

        output = {
            "si_trunk": si_trunk,
            "zij_trunk": zij_trunk,
            "x_pred": x_pred,
        }

        # Compute confidences
        output.update(
            self.aux_heads(
                batch=batch,
                si_input=si_input,
                output=output,
                chunk_size=self.globals.chunk_size,
            )
        )

        return output

    def _train_diffusion(
        self,
        batch: Dict,
        si_input: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> Dict:
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
                "x_sample" ([*, N_samples, N_atom, 3]):
                    Predicted atom positions
        """
        # Expand sampling dimension
        # Is this ideal?
        si_input = si_input.unsqueeze(1)
        si_trunk = si_trunk.unsqueeze(1)
        zij_trunk = zij_trunk.unsqueeze(1)
        for key in batch:
            batch[key] = batch[key].unsqueeze(1)

        # Ground truth positions
        xl_gt = batch["gt_atom_positions"]

        # Sample noise schedule for training
        no_samples = self.globals.no_samples
        batch_size, n_atom, device = xl_gt.shape[0], xl_gt.shape[1], xl_gt.device
        n = torch.randn((batch_size, no_samples), device=device)
        t = self.globals.sigma_data * torch.exp(-1.2 + 1.5 * n)

        # Sample noise
        noise = (t[..., None, None] ** 2) * torch.randn(
            (batch_size, no_samples, n_atom, 3), device=device
        )

        # Sample atom positions
        xl_noisy = xl_gt + noise

        # Run diffusion module
        xl = self.diffusion_module(
            batch=batch,
            xl_noisy=xl_noisy,
            atom_mask=batch["gt_atom_mask"],
            t=t,
            si_input=si_input,
            si_trunk=si_trunk,
            zij_trunk=zij_trunk,
            chunk_size=self.globals.chunk_size,
        )

        output = {
            "noise_level": t,
            "x_sample": xl,
        }

        return output

    def forward(self, batch: Dict) -> Dict:
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
                    *"num_main_msa_seqs" ([*])
                        Number of main MSA seqs used in MSA sampling (non-uniprot)
                    *"gt_atom_positions" ([*, N_atom, 3])
                        Ground truth atom positions for training
                    *"gt_atom_mask" ([*, N_atom])
                        Mask for ground truth atom positions
        Returns:
            Output dictionary containing the following keys:
                "si_trunk" ([*, N_token, C_s]):
                    Single representation output from model trunk
                "zij_trunk" ([*, N_token, N_token, C_z]):
                    Pair representation output from model trunk
                "x_pred" ([*, N_atom, 3]):
                    Predicted atom positions
                "p_plddt" ([*, N_atom, 50]):
                    Predicted binned PLDDT logits
                "p_pae" ([*, N_token, N_token, 64]):
                    Predicted binned PAE logits
                "p_pde" ([*, N_token, N_token, 64]):
                    Predicted binned PDE logits
                "p_resolved" ([*, N_atom, 2]):
                    Predicted binned experimentally resolved logits
                "p_distogram" ([*, N_token, N_token, 64]):
                    Predicted binned distogram logits
                "noise_level" ([*])
                    Training only, noise level at a diffusion step
                "x_sample" ([*, N_samples, N_atom, 3]):
                    Training only, predicted atom positions

        """
        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in batch:
            if batch[k].dtype == torch.float32:
                batch[k] = batch[k].to(dtype=dtype)

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Compute representations
        si_input, si_trunk, zij_trunk = self.run_trunk(
            batch=batch, inplace_safe=inplace_safe
        )

        # Mini rollout
        output = self._rollout(
            batch=batch, si_input=si_input, si_trunk=si_trunk, zij_trunk=zij_trunk
        )

        # Run training step (if necessary)
        if self.training:
            # TODO: Add multi-chain permutation alignment here
            #  Permutation code needs to be updated first
            #  Needs to happen before losses and training diffusion step
            # ground_truth = {k: v for k, v in batch.items() if k.startswith("gt_")}
            # batch = multi_chain_permutation_align(
            #     out=output, features=batch, ground_truth=ground_truth
            # )

            diffusion_output = self._train_diffusion(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
            )

            output.update(diffusion_output)

        return output
