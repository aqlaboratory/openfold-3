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

import importlib

import torch
import torch.nn as nn

from openfold3.core.model.heads.prediction_heads import (
    DistogramHead,
    ExperimentallyResolvedHeadAllAtom,
    PairformerEmbedding,
    PerResidueLDDTAllAtom,
    PredictedAlignedErrorHead,
    PredictedDistanceErrorHead,
)
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_representative_atoms,
)

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed


class AuxiliaryHeadsAllAtom(nn.Module):
    """
    Auxiliary head for OF3
    Implements AF3 Algorithm 31 with main inference loop (Algorithm 1) line 16 - 17.
    """

    def __init__(self, config):
        """
        Args:
            config: ConfigDict with following keys
                "pairformer_embedding": Pairformer embedding config
                "pae": PAE config
                "pde": PDE config
                "lddt": LDDT config
                "distogram": Distogram config
                "experimentally_resolved": Experimentally_resolved config
        """
        super().__init__()
        self.config = config
        self.max_atoms_per_token = config.max_atoms_per_token
        self.per_sample_token_cutoff = config.per_sample_token_cutoff

        self.pairformer_embedding = PairformerEmbedding(
            **self.config["pairformer_embedding"],
        )

        self.pde = PredictedDistanceErrorHead(
            **self.config["pde"],
        )

        self.plddt = PerResidueLDDTAllAtom(
            **self.config["lddt"],
        )

        self.distogram = DistogramHead(
            **self.config["distogram"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHeadAllAtom(
            **self.config["experimentally_resolved"],
        )

        # DDP does not allow unused parameters without performance hit
        # Only initialize PAE head if enabled or using DeepSpeed
        deepspeed_is_initialized = (
            deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
        )
        if self.config.pae.enabled or deepspeed_is_initialized:
            self.pae = PredictedAlignedErrorHead(
                **self.config["pae"],
            )

    def _pair_logits_per_sample(
        self,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        repr_x_pred: torch.Tensor,
        repr_x_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int | None,
        use_deepspeed_evo_attention: bool,
        use_cueq_triangle_kernels: bool,
        use_triton_triangle_kernels: bool,
        use_lma: bool,
        inplace_safe: bool,
        _mask_trans: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        no_samples = repr_x_pred.shape[-3]
        si_out = None
        pde_logits = None
        pae_logits = None

        for i in range(no_samples):
            si_chunk, zij_chunk = self.pairformer_embedding(
                si_input=si_input,
                si=si,
                zij=zij.clone(),
                x_pred=repr_x_pred[..., i : i + 1, :, :],
                single_mask=repr_x_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
                apply_per_sample=False,
            )

            if si_out is None:
                si_out = torch.empty(
                    (*si_chunk.shape[:-3], no_samples, *si_chunk.shape[-2:]),
                    device=si_chunk.device,
                    dtype=si_chunk.dtype,
                )
            si_out[..., i : i + 1, :, :] = si_chunk
            del si_chunk

            pde_chunk = self.pde(zij_chunk, apply_per_sample=False)
            if pde_logits is None:
                pde_logits = torch.empty(
                    (*pde_chunk.shape[:-4], no_samples, *pde_chunk.shape[-3:]),
                    device="cpu",
                    dtype=pde_chunk.dtype,
                )
            pde_logits[..., i : i + 1, :, :, :] = pde_chunk.to(device="cpu")
            del pde_chunk

            if self.config.pae.enabled:
                pae_chunk = self.pae(zij_chunk, apply_per_sample=False)
                if pae_logits is None:
                    pae_logits = torch.empty(
                        (*pae_chunk.shape[:-4], no_samples, *pae_chunk.shape[-3:]),
                        device="cpu",
                        dtype=pae_chunk.dtype,
                    )
                pae_logits[..., i : i + 1, :, :, :] = pae_chunk.to(device="cpu")
                del pae_chunk

            del zij_chunk

        if si_out is None or pde_logits is None:
            raise RuntimeError("per-sample confidence embedding produced no samples")

        return si_out, pde_logits, pae_logits

    def forward(
        self,
        batch: dict,
        si_input: torch.Tensor,
        output: dict,
        use_zij_trunk_embedding: bool,
        chunk_size: int | None = None,
        use_deepspeed_evo_attention: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        offload_inference: bool = False,
        _mask_trans: bool = True,
    ):
        """
        Args:
            batch:
                Input feature dictionary
            si_input:
                [*, N_token, C_s_input] Single (input) representation
            output:
                Dict containing outputs
                    "si_trunk" ([*, N_token, C_s]):
                        Single representation output from model trunk
                    "zij_trunk" ([*, N_token, N_token, C_z]):
                        Pair representation output from model trunk
                    "atom_positions_predicted" ([*, N_atom, 3]):
                        Predicted atom positions
            use_zij_trunk_embedding:
                Whether to use the zij trunk embedding in the confidence Pairformer
                embedding.
            chunk_size:
                Inference-time subbatch size. Associated with PairFormer embedding.
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_cueq_triangle_kernels:
                Whether to use cuEq triangle attention kernel.
                Mutually exclusive with use_lma
            use_triton_triangle_kernels:
                Whether to use Triton triangle attention kernel.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
            offload_inference:
                Whether to offload some computation to CPU
            _mask_trans:
                Whether to mask the output of the transition layers

        Returns:
            aux_out:
                Dict containing following keys:
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
        Note:
            Previous implementations of losses include softmax so all
            heads return logits.
        """
        aux_out = {}

        out_dtype = output["atom_positions_predicted"].dtype
        si = output["si_trunk"]
        zij = output["zij_trunk"]
        atom_positions_predicted = output["atom_positions_predicted"].to(dtype=si.dtype)

        # Distogram head: Main loop (Algorithm 1), line 17
        distogram_logits = self.distogram(z=zij)
        if inplace_safe and not torch.is_grad_enabled() and distogram_logits.is_cuda:
            distogram_logits = distogram_logits.to(device="cpu")
        aux_out["distogram_logits"] = distogram_logits

        # Stop grad
        si_input = si_input.detach().clone()
        si = si.detach().clone()
        if inplace_safe and not torch.is_grad_enabled():
            zij = zij.detach()
        else:
            zij = zij.detach().clone()
        del output["zij_trunk"]
        atom_positions_predicted = atom_positions_predicted.detach().clone()

        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        # Get representative atoms
        repr_x_pred, repr_x_mask = get_token_representative_atoms(
            batch=batch, x=atom_positions_predicted, atom_mask=batch["atom_mask"]
        )

        num_samples = repr_x_pred.shape[-3]
        apply_per_sample = (
            not torch.is_grad_enabled()
            and num_samples > 1
            and self.per_sample_token_cutoff is not None
            and repr_x_pred.shape[-2] > self.per_sample_token_cutoff
        )
        out_device = atom_positions_predicted.device
        final_out_device = (
            torch.device("cpu")
            if offload_inference and not torch.is_grad_enabled()
            else out_device
        )

        if not use_zij_trunk_embedding:
            zij = zij * 0

        if apply_per_sample:
            si, pde_logits, pae_logits = self._pair_logits_per_sample(
                si_input=si_input,
                si=si,
                zij=zij,
                repr_x_pred=repr_x_pred,
                repr_x_mask=repr_x_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            del zij
        else:
            # Embed trunk outputs
            # If offload_inference is enabled, si and zij will be returned on the CPU
            si, zij = self.pairformer_embedding(
                si_input=si_input,
                si=si,
                zij=zij,
                x_pred=repr_x_pred,
                single_mask=repr_x_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                offload_inference=offload_inference,
                _mask_trans=_mask_trans,
                apply_per_sample=False,
            )

        # Get atom mask padded to MAX_ATOMS_PER_TOKEN
        # Required to extract pLDDT and experimentally resolved logits for
        # the flat atom representation
        max_atom_per_token_mask = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=token_mask,
            max_num_atoms_per_token=self.max_atoms_per_token,
        )
        # Expand to match sample dimension
        max_atom_per_token_mask = max_atom_per_token_mask.expand(
            (*atom_positions_predicted.shape[:-2], -1)
        )

        si = si.to(device=out_device)
        aux_out["plddt_logits"] = self.plddt(
            s=si, max_atom_per_token_mask=max_atom_per_token_mask
        )

        experimentally_resolved_logits = self.experimentally_resolved(
            si, max_atom_per_token_mask
        )
        aux_out["experimentally_resolved_logits"] = experimentally_resolved_logits

        if not apply_per_sample:
            # zij is moved back to GPU after the single rep confidence heads
            # because building the max_atom_per_token_mask uses a lot of memory
            zij = zij.to(device=out_device)
            pde_logits = self.pde(zij, apply_per_sample=False)

            if self.config.pae.enabled:
                # Offload pde logits to not keep all three pairwise tensors
                # in GPU memory at once
                offload_device = "cpu" if offload_inference else out_device
                pde_logits = pde_logits.to(device=offload_device)
                aux_out["pae_logits"] = self.pae(zij, apply_per_sample=False)

            del zij
        elif pae_logits is not None:
            aux_out["pae_logits"] = pae_logits

        aux_out["pde_logits"] = pde_logits

        if apply_per_sample:
            aux_out = {k: v.to(dtype=out_dtype) for k, v in aux_out.items()}
        else:
            aux_out = {
                k: v.to(device=final_out_device, dtype=out_dtype)
                for k, v in aux_out.items()
            }

        return aux_out
