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
Sequence-local atom attention modules. Includes AtomAttentionEncoder,
AtomAttentionDecoder, and AtomTransformer.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from openfold3.core.model.layers import DiffusionTransformer
from openfold3.core.model.primitives import LayerNorm, Linear

TensorDict = Dict[str, torch.Tensor]


class AtomTransformer(nn.Module):
    """
    Atom Transformer: neighborhood-blocked (32 * 128) diffusion transformer.

    Implements AF3 Algorithm 7.
    """

    def __init__(
        self,
        c_q: int,
        c_p: int,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        n_query: int,
        n_key: int,
        inf: float,
    ):
        """
        Args:
            c_q:
                Atom single representation channel dimension
            c_p:
                Atom pair representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_blocks:
                Number of attention blocks
            n_transition:
                Number of transition blocks
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
            inf:
                Large number used for attention masking
        """
        super().__init__()
        self.n_query = n_query
        self.n_key = n_key
        self.inf = inf
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
        batch: TensorDict,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
    ):
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "token_mask": [*, N_token] token mask
                    - "atom_to_token_index": [*, N_atom, N_token] one-hot encoding
                        of token index per atom
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair representation
        Returns:
            ql:
                [*, N_atom, c_atom] Updated atom single representation
        """
        # Define subset centers
        # [N_center]
        n_atom = ql.shape[-2]
        offset = self.n_query // 2 - 0.5  # TODO: check this
        n_center = int(n_atom // self.n_query) + 1
        subset_centers = offset + torch.arange(n_center) * self.n_query

        # Compute beta
        # [*, N_atom, N_atom]
        row_mask = torch.abs(
            torch.arange(n_atom).unsqueeze(1) - subset_centers.unsqueeze(0)
        ) < (self.n_query / 2)
        col_mask = torch.abs(
            torch.arange(n_atom).unsqueeze(1) - subset_centers.unsqueeze(0)
        ) < (self.n_key / 2)
        blm = (
            torch.einsum("li,mi->lm", row_mask.to(ql.dtype), col_mask.to(ql.dtype))
            - 1.0
        ) * self.inf  # TODO: check this
        blm = blm.reshape(len(plm[:-3]) * (1,) + (n_atom, n_atom))

        # Create atom mask
        # [*, N_atom]
        atom_mask = torch.einsum(
            "...li,...i->...l", batch["atom_to_token_index"], batch["token_mask"]
        )

        # Run diffusion transformer
        # [*, N_atom, c_atom]
        ql = self.diffusion_transformer(a=ql, s=cl, z=plm, beta=blm, mask=atom_mask)

        return ql


class RefAtomFeatureEmbedder(nn.Module):
    """
    Implements AF3 Algorithm 5 (line 1 - 6).
    """

    def __init__(self, c_atom_ref: int, c_atom: int, c_atom_pair: int):
        """
        Args:
            c_atom_ref:
                Reference atom feature channel dimension (390)
            c_atom:
                Atom single conditioning channel dimension
            c_atom_pair:
                Atom pair conditioning channel dimension
        """
        super().__init__()
        self.linear_feats = Linear(c_atom_ref, c_atom, bias=False)
        self.linear_ref_offset = Linear(3, c_atom_pair, bias=False)
        self.linear_inv_sq_dists = Linear(1, c_atom_pair, bias=False)
        self.linear_valid_mask = Linear(1, c_atom_pair, bias=False)

    def forward(
        self,
        batch: TensorDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "ref_pos": [*, N_atom, 3] atom positions in the
                        reference conformer
                    - "ref_mask": [*, N_atom] atom mask for the reference conformer
                    - "ref_element": [*, N_atom, 128] one-hot encoding of atomic number
                        in the reference conformer
                    - "ref_charge": [*, N_atom] atom charge in the reference conformer
                    - "ref_atom_name_chars": [*, N_atom, 4, 64] one-hot encoding of
                        unicode integers representing unique atom names in the
                        reference conformer
                    - "ref_space_uid": [*, n_atom,] numerical encoding of the chain id
                        and residue index in the reference conformer
        Returns:
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair conditioning
        """
        # Embed atom features
        # [*, N_atom, c_atom]
        cl = self.linear_feats(
            torch.cat(
                [
                    batch["ref_pos"],
                    batch["ref_mask"].unsqueeze(-1),
                    batch["ref_element"],
                    batch["ref_charge"].unsqueeze(-1),
                    batch["ref_atom_name_chars"].flatten(start_dim=-2),
                    batch["ref_space_uid"].unsqueeze(-1),
                ],
                dim=-1,
            )
        )  # CONFIRM THIS FORMAT ONCE DATALOADER/FEATURIZER DONE

        # Embed offsets
        # dlm: [*, N_atom, N_atom, 3]
        # vlm: [*, N_atom, N_atom]
        # plm: [*, N_atom, N_atom, c_atom_pair]
        dlm = batch["ref_pos"].unsqueeze(-3) - batch["ref_pos"].unsqueeze(-2)
        vlm = (
            batch["ref_space_uid"].unsqueeze(-2) == batch["ref_space_uid"].unsqueeze(-1)
        ).to(dlm.dtype)
        plm = self.linear_ref_offset(dlm) * vlm.unsqueeze(-1)

        # Embed pairwise inverse squared distances
        # [*, N_atom, N_atom, c_atom_pair]
        inv_sq_dists = 1.0 / (1 + torch.sum(dlm**2, dim=-1, keepdim=True))
        plm = plm + self.linear_inv_sq_dists(inv_sq_dists) * vlm.unsqueeze(-1)
        plm = plm + self.linear_valid_mask(vlm.unsqueeze(-1)) * vlm.unsqueeze(-1)

        return cl, plm


class NoisyPositionEmbedder(nn.Module):
    """
    Implements AF3 Algorithm 5 (line 8 - 12).
    """

    def __init__(self, c_s: int, c_z: int, c_atom: int, c_atom_pair: int):
        """
        Args:
            c_atom:
                Atom single conditioning channel dimension
            c_atom_pair:
                Atom pair conditioning channel dimension
        """
        super().__init__()
        self.layer_norm_s = LayerNorm(c_s)
        self.linear_s = Linear(c_s, c_atom, bias=False)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, c_atom_pair, bias=False)
        self.linear_r = Linear(3, c_atom, bias=False)

    def forward(
        self,
        batch: TensorDict,
        cl: torch.Tensor,
        plm: torch.Tensor,
        ql: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        rl: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "atom_to_token_index": [*, N_atom, N_token] one-hot encoding
                        of token index per atom
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair conditioning
            ql:
                [*, N_atom, c_atom] Atom single representation
            si_trunk:
                [*, N_token, c_s] Trunk single representation
            zij_trunk:
                [*, N_token, N_token, c_z] Trunk pair representation
            rl:
                [*, N_atom, 3] Noisy atom positions
        Returns:
            cl:
                [*, N_atom, c_atom] Atom single conditioning with trunk single
                    representation embedded
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair conditioning with trunk pair
                    representation embedded
            ql:
                [*, N_atom, c_atom] Atom single representation with noisy coordinate
                    projection
        """
        # Broadcast trunk single representation into atom single conditioning
        # [*, N_atom, c_atom]
        sl_trunk = torch.einsum(
            "...ic,...li->...lc", si_trunk, batch["atom_to_token_index"]
        )
        cl = cl + self.linear_s(self.layer_norm_s(sl_trunk))

        # Broadcast trunk pair representation into atom pair conditioning
        # [*, N_atom, N_atom, c_atom_pair]
        zlj_trunk = torch.einsum(
            "...ijc,...li->...ljc", zij_trunk, batch["atom_to_token_index"]
        )
        zlm_trunk = torch.einsum(
            "...ljc,...mj->...lmc", zlj_trunk, batch["atom_to_token_index"]
        )
        plm = plm + self.linear_z(self.layer_norm_z(zlm_trunk))

        # Add noisy coordinate projection
        # [*, N_atom, c_atom]
        ql = ql + self.linear_r(rl)

        return cl, plm, ql


class AtomAttentionEncoder(nn.Module):
    """
    Implements AF3 Algorithm 5.
    """

    def __init__(
        self,
        c_atom_ref: int,
        c_atom: int,
        c_atom_pair: int,
        c_token: int,
        add_noisy_pos: bool,
        c_hidden: int,
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        n_query: int,
        n_key: int,
        inf: float,
        c_s: Optional[int] = None,
        c_z: Optional[int] = None,
    ):
        """
        Args:
            c_atom_ref:
                Reference atom feature channel dimension (390)
            c_atom:
                Atom single representation channel dimension
            c_atom_pair:
                Atom pair representation channel dimension
            c_token:
                Token single representation channel dimension
            add_noisy_pos:
                Whether to add noisy positions and trunk embeddings
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_blocks:
                Number of attention blocks
            n_transition:
                Number of transition blocks
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
            inf:
                Large number used for attention masking
            c_s:
                Single representation channel dimension (optional)
            c_z:
                Pair representation channel dimension (optional)
        """
        super().__init__()
        self.ref_atom_feature_embedder = RefAtomFeatureEmbedder(
            c_atom_ref=c_atom_ref, c_atom=c_atom, c_atom_pair=c_atom_pair
        )

        if add_noisy_pos:
            self.noisy_position_embedder = NoisyPositionEmbedder(
                c_s=c_s, c_z=c_z, c_atom=c_atom, c_atom_pair=c_atom_pair
            )

        self.relu = nn.ReLU()
        self.linear_l = Linear(c_atom, c_atom_pair, bias=False, init="relu")
        self.linear_m = Linear(
            c_atom, c_atom_pair, bias=False, init="relu"
        )  # TODO: check initialization

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
            n_query=n_query,
            n_key=n_key,
            inf=inf,
        )

        self.linear_q = nn.Sequential(
            Linear(c_atom, c_token, bias=False, init="relu"), nn.ReLU()
        )

    def forward(
        self,
        batch: TensorDict,
        rl: Optional[torch.Tensor] = None,
        si_trunk: Optional[torch.Tensor] = None,
        zij_trunk: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "ref_pos": [*, N_atom, 3] atom positions in the
                        reference conformer
                    - "ref_mask": [*, N_atom] atom mask for the reference conformer
                    - "ref_element": [*, N_atom, 128] one-hot encoding of atomic number
                        in the reference conformer
                    - "ref_charge": [*, N_atom] atom charge in the reference conformer
                    - "ref_atom_name_chars": [*, N_atom, 4, 64] one-hot encoding of
                        unicode integers representing unique atom names in the
                        reference conformer
                    - "ref_space_uid": [*, n_atom,] numerical encoding of the chain id
                        and residue index in the reference conformer
                    - "token_mask":
                        [*, N_token] token mask
                    - "atom_to_token_index":
                        [*, N_atom, N_token] one-hot encoding of token index per atom
            rl:
                [*, N_atom, 3] Noisy atom positions (optional)
            si_trunk:
                [*, N_atom, c_s] Trunk single representation (optional)
            zij_trunk:
                [*, N_atom, N_atom, c_z] Trunk pair representation (optional)
        Returns:
            ai:
                [*, N_token, c_token] Token representation
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair representation
        """
        # Embed reference atom features
        # cl: [*, N_atom, c_atom]
        # plm: [*, N_atom, N_atom, c_atom_pair]
        cl, plm = self.ref_atom_feature_embedder(batch)

        # Initialize atom single representation
        # [*, N_atom, c_atom]
        ql = cl
        # why detach? ql = cl.detach().clone()

        # Embed noisy atom positions and trunk embeddings
        # cl: [*, N_atom, c_atom]
        # plm: [*, N_atom, N_atom, c_atom_pair]
        # ql: [*, N_atom, c_atom]
        if rl is not None:
            cl, plm, ql = self.noisy_position_embedder(
                batch=batch,
                cl=cl,
                plm=plm,
                ql=ql,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                rl=rl,
            )

        # Add the combined single conditioning to the pair rep (line 13 - 14)
        # [*, N_atom, N_atom, c_atom_pair]
        plm = (
            plm
            + self.linear_l(self.relu(cl.unsqueeze(-3)))
            + self.linear_m(self.relu(cl.unsqueeze(-2)))
        )
        plm = plm + self.pair_mlp(plm)

        # Cross attention transformer (line 15)
        # [*, N_atom, c_atom]
        ql = self.atom_transformer(batch=batch, ql=ql, cl=cl, plm=plm)

        # Create atom to token index conversion matrix
        # [*, N_token, N_atom]
        token_to_atom_index = batch["atom_to_token_index"].transpose(-1, -2)

        # Aggregate per-atom representation to per-token representation
        # [*, N_token, c_token]
        ai = torch.einsum(
            "...lc,...il->...ic", self.linear_q(ql), token_to_atom_index
        ) / torch.sum(token_to_atom_index, dim=-1, keepdim=True)  # TODO: check this

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
        no_heads: int,
        no_blocks: int,
        n_transition: int,
        n_query: int,
        n_key: int,
        inf: float,
    ):
        """
        Args:
            c_atom:
                Atom single representation channel dimension
            c_atom_pair:
                Atom pair representation channel dimension
            c_token:
                Token single representation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_blocks:
                Number of attention blocks
            n_transition:
                Number of transition blocks
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
            inf:
                Large number used for attention masking
        """
        super().__init__()

        self.linear_q_in = Linear(c_token, c_atom, bias=False)

        self.atom_transformer = AtomTransformer(
            c_q=c_atom,
            c_p=c_atom_pair,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            n_query=n_query,
            n_key=n_key,
            inf=inf,
        )

        self.layer_norm = LayerNorm(c_in=c_atom)
        self.linear_q_out = Linear(c_atom, 3, bias=False, init="final")

    def forward(
        self,
        batch: TensorDict,
        ai: torch.Tensor,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "token_mask": [*, N_token] token mask
                    - "atom_to_token_index": [*, N_atom, N_token] one-hot encoding
                        of token index per atom
            ai:
                [*, N_token, c_token] Token representation
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair representation
        Returns:
            rl_update:
                [*, N_atom, 3] Atom position updates
        """
        # Broadcast per-token activations to atoms
        # [*, N_atom, c_atom]
        ql = ql + torch.einsum(
            "...ic,...li->...lc", self.linear_q_in(ai), batch["atom_to_token_index"]
        )

        # Atom transformer
        # [*, N_atom, c_atom]
        ql = self.atom_transformer(batch=batch, ql=ql, cl=cl, plm=plm)

        # Compute updates for atom positions
        # [*, N_atom, 3]
        rl_update = self.linear_q_out(self.layer_norm(ql))

        return rl_update
