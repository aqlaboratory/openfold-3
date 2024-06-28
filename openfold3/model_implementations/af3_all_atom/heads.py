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

from openfold3.core.model.primitives import Linear
from openfold3.core.utils.loss import (
    compute_plddt,
    compute_tm,
    compute_predicted_aligned_error,
)
from openfold3.core.utils.precision_utils import is_fp16_enabled

from openfold3.core.model.latent.pairformer import PairFormerStack 


class AuxiliaryHeads(nn.Module):
    """ 
    Auxiliary head for OF3. Implements Algorithm 31 with main inference loop (Algorithm 1) line 16 - 17. 
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
        super(AuxiliaryHeads, self).__init__()
        self.pairformer_embedding = Pairformer_Embedding(
            **config['pairformer_embedding'], 
        )

        self.pae = PAEHead(
            **config['pae'],
        )

        self.pde = PDEHead(
            **config['pde'],
        )

        self.plddt = PerResidueLDDTHead(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config["experimentally_resolved"],
        )

        self.config = config
    
    def forward(self, 
                si_input, 
                outputs, 
                token_to_atom_idx,
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
                'coordinates': coordinate of representative atom per each token [*, n_token, 3] 
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

        #2. si, zij from pairformer stack outs
        si, zij = self.pairformer_embedding(si_input,
                                            outputs['single'],
                                            outputs['pair'],
                                            outputs['coordinates'],
                                            single_mask,
                                            pair_mask,
                                            chuck_size,
                                            )

        lddt_logits = self.plddt(si, token_to_atom_idx)
        aux_out['plddt_logits'] = lddt_logits

        experimentally_resolved_logits = self.experimentally_resolved(outputs["single"], token_to_atom_idx)
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

class Pairformer_Embedding(nn.Module):
    """ 
    Implements Algorithm 31, line 1 - 6
    """
    def __init__(self, min_bin, max_bin, no_bin, inf, c_s, c_z, config):
        """ 
        Args: 
            min_bin: minimum value for bin (3.25)
            max_bin: maximum value for bin (20.75)
            no_bin: number of bins (15)
            inf: inf (1e8)
            c_s: single embedding dimension 
            c_z: pair embedding dimension 
            config: config for PairFormerStack used 
        """
        super(Pairformer_Embedding, self).__init__() 
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bin = no_bin
        self.inf = inf 

        self.linear_i = Linear(c_s, c_z, bias=False, init = 'relu')
        self.linear_j = Linear(c_s, c_z, bias=False, init = 'relu')

        self.linear_distance = Linear(self.no_bin, c_z, bias=False, init = 'relu')
        self.pairformer_stack = PairFormerStack(**config)
    
    def forward(self, 
                si_input: torch.Tensor, 
                si: torch.Tensor, 
                zij: torch.Tensor, 
                x_pred: torch.Tensor,
                single_mask: torch.Tensor, 
                pair_mask: torch.Tensor, 
                chuck_size: int,
                ): 
        """ 
        Args: 
            si_input: output of InputFeatureEmbedder [*, n_token, c_s]
            si: single embedding, [*, n_token, c_s]
            zij: pairwise embedding, [*, n_token, n_token, c_z]
            x_pred: representative atom predicted coordinates per token, [*, n_token, 3]
            single_mask: single mask feat associated with pairformer stack [*, n_token]
            pair_mask: pair mask feat associated with pairformer stack [*, n_token, n_token]
            chuck_size: feat associated with pairformer stack

        Returns: 
            si: pairformer stack out single [*, n_token, c_s]
            zij: pairformer stack out pair [*, n_token, n_token, c_z]
        """
        #1. si projection to zij 
        zij = zij + self.linear_i(si_input.unsqueeze(-2)) + self.linear_j(si_input.unsqueeze(-3))
        
        #2. embed pair distances of representative atoms. 
        # Implementation from RecyclingEmbedder: https://github.com/aqlaboratory/openfold3/blob/f6c875b3c8e3e873a932cbe3b31f94ae011f6fd4/openfold/model/embedders.py#L406   
        bins = torch.linspace(self.min_bin, self.max_bin, self.no_bin)
        squared_bins = bins ** 2
        upper = torch.cat([squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1)
        dij = torch.sum((x_pred[..., None, :] - x_pred[..., None, :, :]) ** 2, dim=-1, keepdims=True)
        dij = ((dij > squared_bins) * (dij < upper)).type(x_pred.dtype) 
        zij = zij + self.linear_distance(dij)

        #3. call pairformer
        si, zij = self.pairformer_stack(si, zij, single_mask, pair_mask, chuck_size)
        
        return si, zij

class PAEHead(nn.Module):
    """
    Implements PAE Head (Algorithm 31, Line 5)
    """
    def __init__(self, c_z, c_out, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of PAE bins
        """
        super(PAEHead, self).__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(self.c_z, self.c_out, init="final")

    def forward(self, zij):
        """
        Args:
            zij:
                [*, n_res, n_res, c_z] pair embedding
        Returns:
            logits:
                [*, n_res, n_res, c_out] logits
        """
        logits = self.linear(zij)
        return logits

class PDEHead(nn.Module):
    """
    Implements PDE Head (Algorithm 31, Line 6)
    """
    def __init__(self, c_z, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of PDE bins
        """
        super(PDEHead, self).__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(self.c_z, self.c_out, init="final")

    def forward(self, zij):
        """
        Args:
            zij:
                [*, n_res, n_res, c_z] pair embedding
        Returns:
            logits: to be fair, it isn't logit (but before applying softmax). 
                [*, n_res, n_res, c_out] 

        Note: 
            Previous implementations of losses happened to include softmax. change if necessary. 
        """
        logits = self.linear(zij + zij.transpose(-2, -3))
        return logits

class PerResidueLDDTHead(nn.Module):
    """
    Implements Plddt Head (Algorithm 31, Line 7)
    """

    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(PerResidueLDDTHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s, token_to_atom_idx):
        """
        Args:
            s:
                [*, n_res, c_s] single embedding
            token_to_atom_idx: one hot encoding of token to atom
                [*, n_res, n_atom,] 
        Returns:
            logits: 
                [*, n_atom, c_out] 

        Note: 
            Previous implementations of losses happened to include softmax. change if necessary. 
        """
        logits = self.linear(torch.sum(s[..., None, :, :] * token_to_atom_idx.unsqueeze(-1), dim = -2)) 
        return logits

class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss
    """
    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s, token_to_atom_idx):
        """
        Args:
            s:
                [*, n_res, c_s] single embedding
            token_to_atom_idx: one hot encoding of token to atom
                [*, n_res, n_atom,] 
        Returns:
            logits: 
                [*, n_atom, c_out] 

        Note: 
            Previous implementations of losses happened to include softmax. change if necessary. 
        """
        logits = self.linear(torch.sum(s[..., None, :, :] * token_to_atom_idx.unsqueeze(-1), dim = -2)) 
        return logits

class DistogramHead(nn.Module):
    """
    Just directly copied from OF implementation. As stated in SI, no changes made in DistogramHead 

    Computes a distogram probability distribution.
    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, c_out, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = c_out

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits