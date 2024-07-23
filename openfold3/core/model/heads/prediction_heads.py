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
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.latent.pairformer import PairFormerStack
from openfold3.core.model.primitives import LayerNorm, Linear


class PairformerEmbedding(nn.Module):
    """
    Implements AF3 Algorithm 31, line 1 - 6
    """

    def __init__(
        self,
        pairformer: ConfigDict,
        c_s: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bin: int,
        inf: float,
        linear_init_params: ConfigDict = lin_init.pairformer_head_init,
    ):
        """
        Args:
            pairformer:
                Config for PairFormerStack used
            c_s:
                Single embedding dimension
            c_z:
                Pair embedding dimension
            min_bin:
                Minimum value for bin (3.25). The value is slightly
                different from SI. Previous AF2 implementation utilized these values
                for bins.
            max_bin:
                Maximum value for bin (20.75). ibid
            no_bin:
                Number of bins (15). ibid
            inf:
                Inf (1e8). ibid
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bin = no_bin
        self.inf = inf

        self.linear_i = Linear(c_s, c_z, **linear_init_params.linear_i)
        self.linear_j = Linear(c_s, c_z, **linear_init_params.linear_j)

        self.linear_distance = Linear(
            self.no_bin, c_z, **linear_init_params.linear_distance
        )
        self.pairformer_stack = PairFormerStack(**pairformer)

    def forward(
        self,
        si_input: torch.Tensor,
        si: torch.Tensor,
        zij: torch.Tensor,
        x_pred: torch.Tensor,
        single_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
    ):
        """
        Args:
            si_input: output of InputFeatureEmbedder [*, n_token, c_s]
            si: single embedding, [*, n_token, c_s]
            zij: pairwise embedding, [*, n_token, n_token, c_z]
            x_pred: representative atom predicted coordinates per token, [*, n_token, 3]
            single_mask: single mask feat associated with pairformer stack [*, n_token]
            pair_mask: pair mask feat associated with pairformer stack
                [*, n_token, n_token]
            chunk_size: feat associated with pairformer stack

        Returns:
            si: pairformer stack out single [*, n_token, c_s]
            zij: pairformer stack out pair [*, n_token, n_token, c_z]
        """
        # 1. si projection to zij
        zij = (
            zij
            + self.linear_i(si_input.unsqueeze(-2))
            + self.linear_j(si_input.unsqueeze(-3))
        )

        # 2. embed pair distances of representative atoms
        bins = torch.linspace(self.min_bin, self.max_bin, self.no_bin)
        squared_bins = bins**2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        dij = torch.sum(
            (x_pred[..., None, :] - x_pred[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )
        dij = ((dij > squared_bins) * (dij < upper)).type(x_pred.dtype)
        zij = zij + self.linear_distance(dij)

        # 3. call pairformer
        si, zij = self.pairformer_stack(si, zij, single_mask, pair_mask, chunk_size)

        return si, zij


class PredictedAlignedErrorHead(nn.Module):
    """
    Implements PredictedAlignedError Head (Algorithm 31, Line 5) for
    AF3 (subsection 4.3.2)
    """

    def __init__(
        self, c_z: int, c_out: int, linear_init_params: ConfigDict = lin_init.pae_init
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of PredictedAlignedError (PAE) bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(self.c_z, self.c_out, **linear_init_params.linear)

    def forward(self, zij):
        """
        Args:
            zij:
                [*, n_res, n_res, c_z] pair embedding
        Returns:
            logits:
                [*, n_res, n_res, c_out] logits

        Note:
            Previous implementations of losses happened to include softmax so
            head returns logits. change if necessary.
        """
        logits = self.linear(zij)
        return logits


class PredictedDistanceErrorHead(nn.Module):
    """
    Implements PredictedDistanceError Head (Algorithm 31, Line 6) for
    AF3 (subsection 4.3.3)
    """

    def __init__(
        self, c_z: int, c_out: int, linear_init_params: ConfigDict = lin_init.pde_init
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of PredictedDistanceError (PDE) bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(self.c_z, self.c_out, **linear_init_params.linear)

    def forward(self, zij):
        """
        Args:
            zij:
                [*, n_res, n_res, c_z] pair embedding
        Returns:
            logits: to be fair, it isn't logit (but before applying softmax).
                [*, n_res, n_res, c_out]

        Note:
            Previous implementations of losses happened to include softmax so
            head returns logits. change if necessary.
        """
        logits = self.linear(zij + zij.transpose(-2, -3))
        return logits


class PerResidueLDDAllAtom(nn.Module):
    """
    Implements Plddt Head (Algorithm 31, Line 7) for AF3 (subsection 4.3.1)
    """

    def __init__(
        self, c_s: int, c_out: int, linear_init_params: ConfigDict = lin_init.lddt_init
    ):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of PLDDT bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, **linear_init_params.linear)

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
            Previous implementations of losses happened to include softmax so
            head returns logits. change if necessary.
        """
        logits = self.linear(
            torch.sum(s[..., None, :, :] * token_to_atom_idx.unsqueeze(-1), dim=-2)
        )
        return logits


class PerResidueLDDTCaPredictor(nn.Module):
    """
    Implements plddtHead for AF2, subsection 1.9.10

    Source: OpenFold
    """

    def __init__(
        self,
        no_bins: int,
        c_in: int,
        c_hidden: int,
        linear_init_params: ConfigDict = lin_init.lddt_ca_init,
    ):
        super().__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, **linear_init_params.linear_1)
        self.linear_2 = Linear(
            self.c_hidden, self.c_hidden, **linear_init_params.linear_2
        )
        self.linear_3 = Linear(
            self.c_hidden, self.no_bins, **linear_init_params.linear_3
        )

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class ExperimentallyResolvedHeadAllAtom(nn.Module):
    """
    Implements resolvedHeads for AF3, subsection 4.3.3
    """

    def __init__(
        self,
        c_s: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.exp_res_all_atom_init,
    ):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of ExperimentallyResolved Head AllAtom bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, **linear_init_params.linear)

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
            Previous implementations of losses happened to include softmax so
            head returns logits. change if necessary.
        """
        logits = self.linear(
            torch.sum(s[..., None, :, :] * token_to_atom_idx.unsqueeze(-1), dim=-2)
        )
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """
    Implements resolvedHeads for AF2.
    For use in computation of experimentally resolved loss, subsection 1.9.10 (AF2)

    Source: OpenFold
    """

    def __init__(
        self,
        c_s: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.exp_res_init,
    ):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of experimentally resolved atom bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, **linear_init_params.linear)

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            logits:
                [*, N, C_out] logits
        """

        logits = self.linear(s)
        return logits


class DistogramHead(nn.Module):
    """
    Implementation of distogram head for both AF2 and AF3.

    Computes a distogram probability distribution.
    For use in computation of distogram loss, subsection 1.9.8 (AF2), section 4.4 (AF3)
    """

    def __init__(
        self,
        c_z: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.distogram_init,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of distogram bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(self.c_z, self.c_out, **linear_init_params.linear)

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            logit:
                [*, N, N, c_out] distogram probability distribution

        Note:
            Previous implementations of losses happened to include softmax so
            head returns logits. change if necessary. For symmetric pairwise
            PairDistanceError loss (PDE), logits are calculated by linear(zij +
            zij.transpose(-2, -3))
        """

        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7 (AF2)
    """

    def __init__(
        self,
        c_z: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.tm_score_init,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_out:
                Number of bins
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_z = c_z
        self.c_out = c_out

        self.linear = Linear(self.c_z, self.c_out, **linear_init_params.linear)

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            logits:
                [*, N_res, N_res, c_out] prediction
        """

        logits = self.linear(z)
        return logits


class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9 (AF2)

    Source: OpenFold
    """

    def __init__(
        self,
        c_m: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.masked_msa_init,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_out:
                Output channel dimension
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_m = c_m
        self.c_out = c_out

        self.linear = Linear(self.c_m, self.c_out, **linear_init_params.linear)

    def forward(self, m):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
        Returns:
            logits:
                [*, N_seq, N_res, c_out] reconstruction
        """

        logits = self.linear(m)
        return logits
