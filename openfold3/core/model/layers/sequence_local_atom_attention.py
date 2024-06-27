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

"""Sequence-local atom attention modules. Includes AtomAttentionEncoder, AtomAttentionDecoder, and AtomTransformer."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from openfold3.core.model.primitives import LayerNorm, Linear

from .diffusion_transformer import DiffusionTransformer

TensorDict = Dict[str, torch.Tensor]


class AtomTransformer(nn.Module):
    """Atom Transformer: neighborhood-blocked (32 * 128) diffusion transformer.

    Implements AF3 Algorithm 7
    """

    def __init__(
        self,
        c_q: int,
        c_p: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        inf: float,
    ):
        """
        Args:
            c_q:
                Atom embedding channel dimension
            c_p:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_blocks:
                Number of blocks
            n_transition:
                Dimension multiplication factor used in transition layer
            inf:
                Large constant used to create mask for attention logits
        """
        super(AtomTransformer, self).__init__()
        self.diffusion_transformer = DiffusionTransformer(
            c_a=c_q,
            c_s=c_q,
            c_z=c_p,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            inf=inf,
        )

    def forward(
        self,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
        atom_mask: torch.Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
    ):
        """
        Args:
            ql:
                [*, N_atom, C_atom] atom embedding
            cl:
                [*, N_atom, C_atom] atom embedding
            plm:
                [*, N_atom, N_atom, C_atompair] pairwise embedding
            atom_mask:
                [*, N_atom] atom mask
            n_queries:
                Neighborhood block height
            n_keys:
                Neighborhood block width

        Returns:
            ql:
                Diffusion transformer output
        """
        # 1. Get subset centers
        n_atom = ql.shape[-2]
        subset_centers = torch.arange(int(n_atom // n_queries) + 1) * n_queries + (
            n_queries // 2 - 0.5
        )

        # 2. Make Blm: (32 * 128) blocks
        # [N_atom, Subset]
        row_condition = (
            torch.abs(torch.arange(n_atom).unsqueeze(1) - subset_centers.unsqueeze(0))
            < n_queries / 2
        )
        col_condition = (
            torch.abs(torch.arange(n_atom).unsqueeze(1) - subset_centers.unsqueeze(0))
            < n_keys / 2
        )
        # [N_atom, N_atom]
        blm = (
            torch.sum(
                torch.logical_and(
                    row_condition.unsqueeze(1), col_condition.unsqueeze(0)
                ).to(ql.dtype),
                dim=-1,
            )
            * 1e10
            - 1e10
        )

        # 3. Call diffusion_transformer
        ql = self.diffusion_transformer(a=ql, s=cl, z=plm, beta=blm, mask=atom_mask)

        return ql


class AtomFeatureEmbedder(nn.Module):
    """Implements atom feature embedding (Algorithm 5, line 1 - 6)"""

    def __init__(self, c_in: int, c_atom: int, c_atom_pair: int):
        """
        Args:
            c_in:
                Parsed feature dims (390)
            c_atom:
                Atom embedding dimension
            c_atom_pair:
                Atom pair embedding dimension
        """
        super(AtomFeatureEmbedder, self).__init__()
        self.linear_feats = Linear(c_in, c_atom, bias=False)
        self.linear_ref_offset = Linear(3, c_atom_pair, bias=False)
        self.linear_inv_dists = Linear(1, c_atom_pair, bias=False)
        self.linear_valid_mask = Linear(1, c_atom_pair, bias=False)

    def forward(
        self,
        atom_feats: TensorDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_feats: Dictionary with following keys/features:
                - "ref_pos":  [*, N_atom, 3] atom positions, given in Angstrom
                - "ref_mask": [*, N_atom] atom mask
                - "ref_element": [*, N_atom, 128] one-hot encoding of atomic number
                - "ref_charge": [*, N_atom] atom charge
                - "ref_atom_name_chars": [*, N_atom, 4, 64] one-hot encoding of unicode integers
                    representing unique atom names
                - "ref_space_uid": numerical encoding of the chain id and residue index [*, n_atom,]
        Returns:
            cl:
                [*, N_atom, C_atom] atom embedding
            plm:
                [*, N_atom, N_atom, C_atom_pair] pair embedding
        """
        # 1. Embed atom features
        # [*, N_atom, C_in] -> [*, N_atom, C_atom]
        # TODO: CONFIRM THIS FORMAT ONCE DATALOADER/FEATURIZER DON
        cl = self.linear_feats(
            torch.cat(
                (
                    atom_feats["ref_pos"],
                    atom_feats["ref_mask"].unsqueeze(-1),
                    atom_feats["ref_element"],
                    atom_feats["ref_charge"].unsqueeze(-1),
                    torch.flatten(atom_feats["ref_atom_name_chars"], start_dim=-2),
                    atom_feats["ref_space_uid"].unsqueeze(-1),
                ),
                dim=-1,
            )
        )

        # 2. Embed offsets
        # [*, N_atom, 3] -> [*, N_atom, N_atom, 3]
        dlm = atom_feats["ref_pos"].unsqueeze(-3) - atom_feats["ref_pos"].unsqueeze(-2)

        # [*, N_atom] --> [*, N_atom, N_atom]
        vlm = (
            atom_feats["ref_space_uid"].unsqueeze(-2)
            == atom_feats["ref_space_uid"].unsqueeze(-1)
        ).to(dlm.dtype)

        # [*, N_atom, N_atom, 3] -> [*, N_atom, N_atom, C_atom_pair]
        plm = self.linear_ref_offset(dlm) * vlm.unsqueeze(-1)

        # 3. Embed pairwise inverse squared distance
        # [*, N_atom, N_atom, 3] -> [*, N_atom, N_atom, C_atom_pair]
        inv_dists = self.linear_inv_dists(
            torch.pow(1.0 + torch.norm(dlm, dim=-1).unsqueeze(-1), -1.0)
        )
        plm = plm + inv_dists * vlm.unsqueeze(-1)

        # [*, N_atom, N_atom, C_atom_pair]
        plm = plm + self.linear_valid_mask(vlm.unsqueeze(-1)) * vlm.unsqueeze(-1)
        return cl, plm


class NoisyPositionEmbedder(nn.Module):
    def __init__(self, c_s: int, c_z: int, c_atom: int, c_atom_pair: int):
        """
        Implements AF3 Algorithm 5 (line 8 - 12)

        Args:
            c_s:
                Single activation channel dimension
            c_z:
                Pair activation channel dimension
            c_atom:
                Atom embedding dimension
            c_atom_pair:
                Atom pair embedding dimension
        """
        super(NoisyPositionEmbedder, self).__init__()
        self.c_s = c_s
        self.c_z = c_z

        self.layer_norm_s = LayerNorm(c_s)
        self.linear_s = Linear(c_s, c_atom, bias=False)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, c_atom_pair, bias=False)
        self.linear_r = Linear(3, c_atom, bias=False)

    def forward(
        self,
        atom_feats: TensorDict,
        cl: torch.Tensor,
        plm: torch.Tensor,
        ql: torch.Tensor,
        s_trunk: torch.Tensor,
        zij: torch.Tensor,
        rl: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_feats: TensorDict with features:
                - "atom_to_token_index": atom to token index feature [*, n_atom, n_token,]
            cl: atom embedding [*, n_atom, c_atom]
            plm: atom pair embedding [*, n_atom, n_atom, c_atom]
            ql: token embedding [*, n_atom, c_atom]
            s_trunk: trunk embedding (per tokens) [*, n_token, c_s]
            zij: pairwise token embedding [*, n_token, n_token, c_z]
            rl: noisy coordinates [*, n_atom, 3]

        Returns:
            cl: atom embedding with token embd projection [*, n_atom, c_atom]
            plm: atom pair embedding [*, n_atom, n_atom, c_atom_pair]
            ql: atom embedding with noisy coordinate projection [*, n_atom, c_atom]
        """

        # 1. project trunk embedding (s_trunk) onto atom embedding
        cl = cl + self.linear_s(
            self.layer_norm_s(
                torch.sum(
                    s_trunk.unsqueeze(-3)
                    * atom_feats["atom_to_token_index"].unsqueeze(-1),
                    dim=-2,
                )
            )
        )

        # 2. project pair embedding
        # [b, n_token, n_token, c_z]
        # [b, n_atom, 1, n_token, 1] * [b, 1, n_atom, 1, n_token]
        atom_pair_to_token_index = (
            atom_feats["atom_to_token_index"][..., None, :, None]
            * atom_feats["atom_to_token_index"][..., None, :, None, :]
        )  # [*, n_atom, n_atom, n_token, n_token]
        plm = plm + self.linear_z(
            self.layer_norm_z(
                torch.sum(
                    zij[..., None, None, :, :, :] * atom_pair_to_token_index[..., None],
                    dim=(-2, -3),
                )
            )
        )  # [*, n_atom, n_atom, c_z]

        # 3. noisy coordinate projection
        ql = ql + self.linear_r(rl)
        return cl, plm, ql


class AtomAttentionEncoder(nn.Module):
    """
    Implements AF3 Algorithm 5.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_atom_ref: int,
        c_atom: int,
        c_atom_pair: int,
        c_token: int,
        c_hidden: int,
        add_noisy_pos: bool,
        no_heads: int = 4,
        no_blocks: int = 3,
        n_transition: int = 2,
        inf: float = 1e9,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_atom_ref:
                Reference per-atom feature dimension
            c_atom:
                Atom emebedding channel dimension
            c_atom_pair:
                Atom pair embedding channel dimension
            c_token:
                Token representation channel dimension
            c_hidden:
                Hidden channel dimension
            add_noisy_pos:
                Whether to add noised positions and trunk embeddings
            no_heads:
                Number of attention heads
            no_blocks:
                Number of attention blocks
            n_transition:
                Dimension multiplication factor used in transition layer
            inf:
                Large number used for attention masking
        """
        super(AtomAttentionEncoder, self).__init__()

        self.add_noisy_pos = add_noisy_pos

        self.atom_feature_emb = AtomFeatureEmbedder(
            c_in=c_atom_ref, c_atom=c_atom, c_atom_pair=c_atom_pair
        )

        if add_noisy_pos:
            self.noisy_position_emb = NoisyPositionEmbedder(
                c_s=c_s, c_z=c_z, c_atom=c_atom, c_atom_pair=c_atom_pair
            )

        self.relu = nn.ReLU()
        self.linear_l = Linear(
            c_atom, c_atom_pair, bias=False, init="relu"
        )  # ** changed to (c_atom, c_atompair) from (c_atom, c_atom)
        self.linear_m = Linear(c_atom, c_atom_pair, bias=False, init="relu")  # ibid

        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False, init="relu"),
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False, init="relu"),
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, bias=False, init="relu"),
        )

        self.atom_transformer = AtomTransformer(
            c_q=c_atom,
            c_p=c_atom_pair,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            inf=inf,
        )

        self.c_token = c_token
        self.linear_q = nn.Sequential(
            Linear(c_atom, c_token, bias=False, init="relu"), nn.ReLU()
        )

    def forward(
        self,
        atom_feats: TensorDict,
        atom_mask: torch.Tensor,
        rl: Optional[torch.Tensor] = None,
        si_trunk: Optional[torch.Tensor] = None,
        zij: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_feats: TensorDict with following keys/features:
                - "ref_pos": atom position, given in Angstrom [*, n_atom, 3]
                - "ref_mask": atom mask [*, n_atom, 1]
                - "ref_element": one hot encoding of atomic number (up to 128) [*, n_atom, 128]
                - "ref_charge": atom charge [*, n_atom, 1]
                - "ref_atom_name_chars" WHAT IS THIS? [*, n_atom, 4, 64]
                - "ref_space_uid": numerical encoding of the chain id and residue index [*, n_atom, 1]
                ...
                - "atom_to_token_index": atom to token index [*, n_atom, n_token,]
            atom_mask:
                [*, N_atom] atom mask
            rl:
                [*, N_atom, 3] noised coordinates
            si_trunk:
                [*, N_at, 3] trunk embedding
            zij:
                pair embedding

        Returns:
            ai:
                token level embedding [*, n_token, c_token]
            ql: atom level embedding with noisy coordinate info projection [*, n_atom, c_atom]
            cl: atom level embedding with atom features [*, n_atom, c_atom]
            plm: atom pairwise embedding [*, n_atom, n_atom, c_atom_pair]
        """
        # 1. atom feature projection (line 1- 6)
        cl, plm = self.atom_feature_emb(
            atom_feats
        )  # [*, n_atom, c_atom], [*, n_atom, n_atom, c_atom_pair]

        ql = cl.detach().clone()
        # 2. noisy pos projection (line 8 - 12)
        if rl is not None:
            cl, plm, ql = self.noisy_position_emb(
                atom_feats, cl, plm, ql, si_trunk, zij, rl
            )

        # 3. add the combined single conditioning to the pair representation (line 13 - 14)
        plm = (
            plm
            + self.linear_l(self.relu(cl.unsqueeze(-3)))
            + self.linear_m(self.relu(cl.unsqueeze(-2)))
        )  # [*, n_atom, c_atom] --> [*, n_atom, n_atom, atom_pair]
        plm = plm + self.pair_mlp(plm)

        # 4. cross attention transformer (line 15)
        ql = self.atom_transformer(ql=ql, cl=cl, plm=plm, atom_mask=atom_mask)

        # 5. aggregate
        token_to_atom_index = atom_feats["atom_to_token_index"].transpose(-1, -2)
        ai = torch.sum(
            self.linear_q(ql).unsqueeze(-3) * token_to_atom_index.unsqueeze(-1), dim=-2
        ) / torch.sum(
            token_to_atom_index, dim=-1, keepdim=True
        )  # [b, n_token, c_token]

        return ai, ql, cl, plm


class AtomAttentionDecoder(nn.Module):
    """
    Implements AF3 Algorithm 6.
    """

    def __init__(
        self,
        c_atom: int,
        c_atom_pair: int,
        c_token: int,
        c_hidden: int,
        no_heads: int = 4,
        no_blocks: int = 3,
        n_transition: int = 2,
        inf: float = 1e9,
    ):
        """
        args:
            c_token: token embedding dimension
            c_atom: atom embedding dimension
            c_atom_pair: atom pair embedding dimension
            c_hidden: hidden dimension (for atom transformer)
            no_heads: number of heads (for atom transformer)
            no_blocks: number of blocks (for atom transformer)
            n_transition: transition (for aotm transformer)
            inf:
        """
        super(AtomAttentionDecoder, self).__init__()

        self.linear_q_in = Linear(c_token, c_atom, bias=False)

        self.atom_transformer = AtomTransformer(
            c_q=c_atom,
            c_p=c_atom_pair,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            inf=inf,
        )

        self.layer_norm = LayerNorm(c_in=c_atom)
        self.linear_q_out = Linear(c_atom, 3, bias=False, init="final")

    def forward(
        self,
        atom_feats: TensorDict,
        ai: torch.Tensor,
        ql_skip: torch.Tensor,
        cl_skip: torch.Tensor,
        plm: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            atom_feats: TensorDict
                - "atom_to_token_index: atom to token index [*, n_atom, n_token]
            ai: token embedding [*, n_token, c_token]
            ql_skip: atom embedding [*, n_atom, c_atom]
            cl_skip: atom embedding [*, n_atom, c_atom]
            plm: pairwise embedding [*, n_atom, n_atom, c_atom_pair]

        returns:
            r_update: predicted coordinate noise [*, n_atom, 3]
        """
        # 'atom_to_token_index': [b, n_atom, n_token]

        # 1. broadcast projected token embd
        ql_skip = ql_skip + torch.sum(
            self.linear_q_in(ai).unsqueeze(-3)
            * atom_feats["atom_to_token_index"].unsqueeze(-1),
            dim=-2,
        )  # [*, n_atom, c_atom]

        # 2. atom transformer
        q_out = self.atom_transformer(
            ql_skip, cl_skip, plm, atom_mask
        )  # q_out: shape [*, n_atom, c_atom]

        # 3. predict the noise
        r_update = self.linear_q_out(
            self.layer_norm(q_out)
        )  # [*, n_atom, c_atom] -> [*, n_atom, 3]

        return r_update
