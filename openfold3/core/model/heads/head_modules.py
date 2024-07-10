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

import torch
import torch.nn as nn

from openfold3.core.loss.loss import (
    compute_plddt,
    compute_tm,
    compute_predicted_aligned_error,
)

from openfold3.core.model.heads.prediction_heads import (
    Pairformer_Embedding, 
    PredictedAlignedErrorHead, 
    PredictedDistanceErrorHead, 
    PerResidueLDDAllAtom, 
    ExperimentallyResolvedHeadAllAtom,
    DistogramHead,
    PerResidueLDDTCaPredictor, 
    ExperimentallyResolvedHead,
    TMScoreHead, 
    MaskedMSAHead
)

from typing import Dict

class AuxiliaryHeadsAF2(nn.Module):
    """
    Auxiliary head for OF2
    Implements section 1.9 (AF2)

    Source: OpenFold    
    """
    def __init__(self, config):
        super(AuxiliaryHeadsAF2, self).__init__()

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
            outputs: a dict containing following keys and tensors: 
                'sm':
                    'single': single embedding 
                'pair': pair embedding 
                'msa': msa embedding
        Returns: 
            aux_out: a dict containing: 
                'lddt_logits': plddt head out [*, n_atoms, bins_plddt]
                'plddt': computed plddt [*, n_atoms, bins_plddt]
                'distogram_logits': distogram head out [*, n_token, n_token, bins_distogram]
                'masked_msa_logits': masked msa head out[]
                'experimentally_resolved_logits': resolved head out [*, n_atoms, bins_resolved]
                'tm_logits': values identical to pae_logits [*, n_token, n_token, bins_pae]
                'ptm_scores': 
                'iptm_score': 
                'weighted_ptm_score': 
        """
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        masked_msa_logits = self.masked_msa(outputs["msa"])
        aux_out["masked_msa_logits"] = masked_msa_logits

        experimentally_resolved_logits = self.experimentally_resolved(
            outputs["single"]
        )
        aux_out[
            "experimentally_resolved_logits"
        ] = experimentally_resolved_logits

        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits
            aux_out["ptm_score"] = compute_tm(
                tm_logits, **self.config.tm
            )
            asym_id = outputs.get("asym_id")
            if asym_id is not None:
                aux_out["iptm_score"] = compute_tm(
                    tm_logits, asym_id=asym_id, interface=True, **self.config.tm
                )
                aux_out["weighted_ptm_score"] = (self.config.tm["iptm_weight"] * aux_out["iptm_score"]
                                                 + self.config.tm["ptm_weight"] * aux_out["ptm_score"])

            aux_out.update(
                compute_predicted_aligned_error(
                    tm_logits,
                    **self.config.tm,
                )
            )

        return aux_out

class AuxiliaryHeadsAllAtom(nn.Module):
    """ 
    Auxiliary head for OF3
    Implements Algorithm 31 with main inference loop (Algorithm 1) line 16 - 17. 
    """
    def __init__(self, config):
        """ 
        Args:
            config: ConfigDict with following keys
                'pairformer_embedding': pairformer embedding config
                'pae': pae config 
                'pde': pde config
                'lddt': lddt config 
                'distogram': distogram config 
                'experimentally_resolved': experimentally_resolved config 
        """
        super(AuxiliaryHeadsAllAtom, self).__init__()
        self.pairformer_embedding = Pairformer_Embedding(
            **config['pairformer_embedding'], 
        )

        self.pae = PredictedAlignedErrorHead(
            **config['pae'],
        )

        self.pde = PredictedDistanceErrorHead(
            **config['pde'],
        )

        self.plddt = PerResidueLDDAllAtom(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHeadAllAtom(
            **config["experimentally_resolved"],
        )

        self.config = config
    
    def forward(self, 
                si_input: torch.Tensor, 
                outputs: Dict, 
                x_pred: torch.Tensor,
                token_representative_atom_idx: torch.Tensor, 
                token_to_atom_idx: torch.Tensor,
                single_mask: torch.Tensor,
                pair_mask: torch.Tensor,
                chuck_size: int,
                ):
        """ 
        Args: 
            si_input: single, token embedding [*, n_token, c_s]
            outputs: TensorDict containing outputs 
                'single': single out [*, n_token, c_s]
                'pair': pair out [*, n_token, n_token, c_z]
            x_pred: coordinate of representative atom per each token [*, n_atom, 3] 
            token_representative_atom_idx: index of representative atom index for each token: [*, n_token, n_atom]
            token_to_atom_idx: feature of token to atom idx [*, n_token, n_atom,]
            single_mask: single mask feat associated with pairformer stack [*, n_token]
            pair_mask: pair mask feat associated with pairformer stack [*, n_token, n_token]
            chuck_size: feat associated with pairformer stack (int)

        Returns: 
            aux_out: dict containing following keys: 
                'distogram_logits': distogram head out [*, n_token, n_token, bins_distogram]
                'pae_logits': pae head out[*, n_token, n_token, bins_pae]
                'pde_logits': pde head out[*, n_token, n_token, bins_pde]
                'plddt_logits': plddt head out [*, n_atoms, bins_plddt]
                'experimentally_resolved_logits': resolved head out [*, n_atoms, bins_resolved]
                'tm_logits': values identical to pae_logits [*, n_token, n_token, bins_pae]
                'ptm_scores': 
                'iptm_score': 
                'weighted_ptm_score': 
        """
        aux_out = {}        
        # 1. distogram head: distogram head needs outputs['pair'] before passing pairformer (main loop: line 17)
        distogram_logits = self.distogram(outputs['pair'])
        aux_out["distogram_logits"] = distogram_logits

        # 2. stop grad
        pair = outputs['pair'].detach()
        single = outputs['single'].detach()
        coords = x_pred.detach()
        coords = torch.sum(coords.unsqueeze(-3) * token_representative_atom_idx.unsqueeze(-1), dim = -2) #using the representative atom for each token (Ca, C1', and heavy atom)

        #3. si, zij from pairformer stack outs
        si, zij = self.pairformer_embedding(si_input,
                                            single,
                                            pair,
                                            coords,
                                            single_mask,
                                            pair_mask,
                                            chuck_size,
                                            )

        lddt_logits = self.plddt(si, token_to_atom_idx)
        aux_out['plddt_logits'] = lddt_logits

        experimentally_resolved_logits = self.experimentally_resolved(si, token_to_atom_idx)
        aux_out["experimentally_resolved_logits"] = experimentally_resolved_logits

        pae_logits = self.pae(zij)
        aux_out['pae_logits'] = pae_logits
        
        pde_logits = self.pde(zij)
        aux_out['pde_logits'] = pde_logits
        
        if self.config['tm']['enabled']:
            tm_logits = pae_logits #uses pae_logits (SI pg. 27)
            aux_out["tm_logits"] = tm_logits
            aux_out["ptm_score"] = compute_tm(
                tm_logits, **self.config.tm
            )
            asym_id = outputs.get("asym_id")
            if asym_id is not None:
                aux_out["iptm_score"] = compute_tm(
                    tm_logits, asym_id=asym_id, interface=True, **self.config.tm
                )
                aux_out["weighted_ptm_score"] = (self.config.tm["iptm_weight"] * aux_out["iptm_score"]
                                                 + self.config.tm["ptm_weight"] * aux_out["ptm_score"])

            aux_out.update(
                compute_predicted_aligned_error(
                    tm_logits,
                    **self.config.tm,
                )
            )
        
        return aux_out