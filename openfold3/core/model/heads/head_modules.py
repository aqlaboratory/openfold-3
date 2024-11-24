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

from typing import Optional

import torch
import torch.nn as nn

from openfold3.core.model.heads.prediction_heads import (
    DistogramHead,
    ExperimentallyResolvedHead,
    ExperimentallyResolvedHeadAllAtom,
    MaskedMSAHead,
    PairformerEmbedding,
    PerResidueLDDAllAtom,
    PerResidueLDDTCaPredictor,
    PredictedAlignedErrorHead,
    PredictedDistanceErrorHead,
    TMScoreHead,
)
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_representative_atoms,
)


class AuxiliaryHeadsAF2(nn.Module):
    """
    Auxiliary head for OF2
    Implements section 1.9 (AF2)

    Source: OpenFold
    """

    def __init__(self, config):
        super().__init__()

        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.masked_msa = MaskedMSAHead(
            **config["masked_msa"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config["experimentally_resolved"],
        )

        if config.tm.enabled:
            self.tm = TMScoreHead(
                **config.tm,
            )

        self.config = config

    def forward(self, outputs):
        """
        Args:
            outputs: Dict containing following keys and tensors:
                "sm":
                    "single": Single embedding
                "pair": Pair embedding
                "msa": MSA embedding
        Returns:
            aux_out: Dict containing:
                "lddt_logits" ([*, N_res, bins_plddt]):
                    pLDDT head out
                "distogram_logits" ([*, N_res, N_res, bins_distogram]):
                    Distogram head out
                "masked_msa_logits" ([*, N_seq, N_res, bins_masked_msa]):
                    Masked msa head out
                "experimentally_resolved_logits" ([*, N_res, bins_resolved]):
                    Resolved head out
                "tm_logits" ([*, N_res, N_res, bins_pae]):
                    Values identical to pae_logits
        """
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        masked_msa_logits = self.masked_msa(outputs["msa"])
        aux_out["masked_msa_logits"] = masked_msa_logits

        experimentally_resolved_logits = self.experimentally_resolved(outputs["single"])
        aux_out["experimentally_resolved_logits"] = experimentally_resolved_logits

        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits

        return aux_out


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

        self.pairformer_embedding = PairformerEmbedding(
            **self.config["pairformer_embedding"],
        )

        self.pde = PredictedDistanceErrorHead(
            **self.config["pde"],
        )

        self.plddt = PerResidueLDDAllAtom(
            **self.config["lddt"],
        )

        self.distogram = DistogramHead(
            **self.config["distogram"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHeadAllAtom(
            **self.config["experimentally_resolved"],
        )

        if self.config.pae.enabled:
            self.pae = PredictedAlignedErrorHead(
                **self.config["pae"],
            )

    def forward(
        self,
        batch: dict,
        si_input: torch.Tensor,
        output: dict,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
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
            chunk_size:
                Inference-time subbatch size. Associated with PairFormer embedding.
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory efficient kernel.
                Mutually exclusive with use_lma.
            use_lma:
                Whether to use low-memory attention during inference.
                Mutually exclusive with use_deepspeed_evo_attention.
            inplace_safe:
                Whether inplace operations can be performed
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

        si_trunk = output["si_trunk"]
        zij_trunk = output["zij_trunk"]
        atom_positions_predicted = output["atom_positions_predicted"]

        # Distogram head: Main loop (Algorithm 1), line 17
        # Not enabled in finetuning 3 stage
        if self.config["distogram"]["enabled"]:
            distogram_logits = self.distogram(z=zij_trunk)
            aux_out["distogram_logits"] = distogram_logits

        # Stop grad
        si_input = si_input.detach().clone()
        si_trunk = si_trunk.detach().clone()
        zij_trunk = zij_trunk.detach().clone()
        atom_positions_predicted = atom_positions_predicted.detach().clone()

        token_mask = batch["token_mask"]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]

        # Expand token mask to atom mask
        atom_mask = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=token_mask,
        )

        # TODO: Check why this gets a CUDA index error sometimes
        # Get representative atoms
        repr_x_pred, repr_x_mask = get_token_representative_atoms(
            batch=batch, x=atom_positions_predicted, atom_mask=atom_mask
        )

        # Embed trunk outputs
        si, zij = self.pairformer_embedding(
            si_input=si_input,
            si=si_trunk,
            zij=zij_trunk,
            x_pred=repr_x_pred,
            single_mask=repr_x_mask,
            pair_mask=pair_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
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

        aux_out["plddt_logits"] = self.plddt(
            s=si, max_atom_per_token_mask=max_atom_per_token_mask
        )

        experimentally_resolved_logits = self.experimentally_resolved(
            si, max_atom_per_token_mask
        )
        aux_out["experimentally_resolved_logits"] = experimentally_resolved_logits

        aux_out["pde_logits"] = self.pde(zij)

        if self.config.pae.enabled:
            aux_out["pae_logits"] = self.pae(zij)

        return aux_out
