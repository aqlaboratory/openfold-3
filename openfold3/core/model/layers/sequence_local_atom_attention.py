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

Note: The DeepSpeed EvoAttention kernel is not enabled for the atom attention
encoder/decoder currently as this does not pass the shape asserts. Ignoring asserts
results in NaNs. The option is still available here in case of future improvements,
but it is not recommended to use it at the moment.
"""

from typing import Optional

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.atom_attention_block_utils import (
    convert_single_rep_to_blocks,
    convert_trunk_pair_rep_to_blocks,
)
from openfold3.core.utils.atomize_utils import (
    aggregate_atom_feat_to_tokens,
    broadcast_token_feat_to_atoms,
)
from openfold3.core.utils.checkpointing import checkpoint_section

TensorDict = dict[str, torch.Tensor]


class RefAtomFeatureEmbedder(nn.Module):
    """
    Implements AF3 Algorithm 5 (line 1 - 6).
    """

    def __init__(
        self,
        c_atom_ref: int,
        c_atom: int,
        c_atom_pair: int,
        linear_init_params: ConfigDict = lin_init.ref_atom_emb_init,
    ):
        """
        Args:
            c_atom_ref:
                Reference atom feature channel dimension (390)
            c_atom:
                Atom single conditioning channel dimension
            c_atom_pair:
                Atom pair conditioning channel dimension
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()
        # Ref conformer feats
        self.linear_ref_pos = Linear(3, c_atom, **linear_init_params.linear_feats)
        self.linear_ref_charge = Linear(1, c_atom, **linear_init_params.linear_feats)
        self.linear_ref_mask = Linear(1, c_atom, **linear_init_params.linear_feats)
        self.linear_ref_element = Linear(119, c_atom, **linear_init_params.linear_feats)
        self.linear_ref_atom_chars = Linear(
            256, c_atom, **linear_init_params.linear_feats
        )

        self.linear_ref_offset = Linear(
            3, c_atom_pair, **linear_init_params.linear_ref_offset
        )
        self.linear_inv_sq_dists = Linear(
            1, c_atom_pair, **linear_init_params.linear_inv_sq_dists
        )
        self.linear_valid_mask = Linear(
            1, c_atom_pair, **linear_init_params.linear_valid_mask
        )

    def forward(
        self,
        batch: TensorDict,
        n_query: int,
        n_key: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
        Returns:
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning
        """
        dtype = batch["ref_pos"].dtype

        # Embed atom features
        # [*, N_atom, c_atom]
        cl = self.linear_ref_pos(batch["ref_pos"])
        cl = cl + self.linear_ref_charge(
            torch.arcsinh(batch["ref_charge"].unsqueeze(-1))
        )
        cl = cl + self.linear_ref_mask(batch["ref_mask"].unsqueeze(-1).to(dtype=dtype))
        cl = cl + self.linear_ref_element(batch["ref_element"].to(dtype=dtype))
        cl = cl + self.linear_ref_atom_chars(
            batch["ref_atom_name_chars"].flatten(start_dim=-2).to(dtype=dtype)
        )

        # Embed offsets
        # Convert all atom rep to block format ahead of time due to
        # reduce memory cost
        # dl, dm: [*, N_blocks, N_query, 3], [*, N_blocks, N_key, 3]
        # vl, vm: [*, N_blocks, N_query, 1], [*, N_blocks, N_key, 1]
        # atom_mask: [*, N_blocks, N_query, N_key]
        d_l, d_m, atom_mask = convert_single_rep_to_blocks(
            ql=batch["ref_pos"],
            n_query=n_query,
            n_key=n_key,
            atom_mask=batch["atom_mask"],
        )
        v_l, v_m, _ = convert_single_rep_to_blocks(
            ql=batch["ref_space_uid"].unsqueeze(-1), n_query=n_query, n_key=n_key
        )

        # dlm: [*, N_blocks, N_query, N_key, 3]
        # vlm: [*, N_blocks, N_query, N_key, 1]
        dlm = (d_l.unsqueeze(-2) - d_m.unsqueeze(-3)) * atom_mask.unsqueeze(-1)
        vlm = (v_l.unsqueeze(-2) == v_m.unsqueeze(-3)).to(
            dtype=dlm.dtype
        ) * atom_mask.unsqueeze(-1)

        plm = self.linear_ref_offset(dlm) * vlm

        # Embed pairwise inverse squared distances
        # [*, N_blocks, N_query, N_key, c_atom_pair]
        inv_sq_dists = 1.0 / (1 + torch.sum(dlm**2, dim=-1, keepdim=True))
        plm = plm + self.linear_inv_sq_dists(inv_sq_dists) * vlm
        plm = plm + self.linear_valid_mask(vlm) * vlm

        return cl, plm


class NoisyPositionEmbedder(nn.Module):
    """
    Implements AF3 Algorithm 5 (line 8 - 12).
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_atom: int,
        c_atom_pair: int,
        linear_init_params: ConfigDict = lin_init.noisy_pos_emb_init,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_atom:
                Atom single conditioning channel dimension
            c_atom_pair:
                Atom pair conditioning channel dimension
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()
        self.layer_norm_s = LayerNorm(c_s, create_offset=False)
        self.linear_s = Linear(c_s, c_atom, **linear_init_params.linear_s)
        self.layer_norm_z = LayerNorm(c_z, create_offset=False)
        self.linear_z = Linear(c_z, c_atom_pair, **linear_init_params.linear_z)
        self.linear_r = Linear(3, c_atom, **linear_init_params.linear_r)

    def forward(
        self,
        batch: TensorDict,
        cl: torch.Tensor,
        plm: torch.Tensor,
        ql: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
        rl: torch.Tensor,
        n_query: int,
        n_key: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "token_mask": [*, N_token] Token mask
                    - "num_atoms_per_token": [*, N_token] Number of atoms per token
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning
            ql:
                [*, N_atom, c_atom] Atom single representation
            si_trunk:
                [*, N_token, c_s] Trunk single representation
            zij_trunk:
                [*, N_token, N_token, c_z] Trunk pair representation
            rl:
                [*, N_atom, 3] Noisy atom positions
            n_query:
                Number of queries (block height)
            n_key:
                Number of keys (block width)
        Returns:
            cl:
                [*, N_atom, c_atom] Atom single conditioning with trunk single
                    representation embedded
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning with
                    trunk pair representation embedded
            ql:
                [*, N_atom, c_atom] Atom single representation with noisy coordinate
                    projection
        """

        # Broadcast trunk single representation into atom single conditioning
        # [*, N_atom, c_atom]
        si_trunk = broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=si_trunk,
            token_dim=-2,
        )
        cl = cl + self.linear_s(self.layer_norm_s(si_trunk))

        # Broadcast trunk pair representation into atom pair conditioning
        zij_trunk = self.linear_z(self.layer_norm_z(zij_trunk))

        zij_trunk = convert_trunk_pair_rep_to_blocks(
            batch=batch, zij_trunk=zij_trunk, n_query=n_query, n_key=n_key
        )

        plm = plm + zij_trunk

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
        use_ada_layer_norm: bool,
        c_s: Optional[int] = None,
        c_z: Optional[int] = None,
        blocks_per_ckpt: Optional[int] = None,
        ckpt_intermediate_steps: bool = False,
        inf: float = 1e9,
        linear_init_params: ConfigDict = lin_init.atom_att_enc_init,
        use_reentrant: Optional[bool] = None,
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
            use_ada_layer_norm:
                Whether to apply AdaLN-Zero conditioning
            c_s:
                Single representation channel dimension (optional)
            c_z:
                Pair representation channel dimension (optional)
            blocks_per_ckpt:
                Number of blocks per checkpoint. If set, checkpointing will
                be used to save memory.
            ckpt_intermediate_steps:
                Whether to checkpoint intermediate steps in the module, including
                RefAtomFeatureEmbedder, NoisyPositionEmbedder, and feature aggregation
            inf:
                Large number used for attention masking
            linear_init_params:
                Linear layer initialization parameters
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
        """
        super().__init__()
        self.n_query = n_query
        self.n_key = n_key
        self.ckpt_intermediate_steps = ckpt_intermediate_steps
        self.inf = inf
        self.use_reentrant = use_reentrant

        self.ref_atom_feature_embedder = RefAtomFeatureEmbedder(
            c_atom_ref=c_atom_ref,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            linear_init_params=linear_init_params.ref_atom_emb,
        )

        if add_noisy_pos:
            self.noisy_position_embedder = NoisyPositionEmbedder(
                c_s=c_s,
                c_z=c_z,
                c_atom=c_atom,
                c_atom_pair=c_atom_pair,
                linear_init_params=linear_init_params.noisy_pos_emb,
            )

        self.relu = nn.ReLU()
        self.linear_l = Linear(c_atom, c_atom_pair, **linear_init_params.linear_l)
        self.linear_m = Linear(
            c_atom, c_atom_pair, **linear_init_params.linear_m
        )  # TODO: check initialization

        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, **linear_init_params.pair_mlp_1),
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, **linear_init_params.pair_mlp_2),
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, **linear_init_params.pair_mlp_3),
        )

        self.atom_transformer = DiffusionTransformer(
            c_a=c_atom,
            c_s=c_atom,
            c_z=c_atom_pair,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            use_ada_layer_norm=use_ada_layer_norm,
            n_query=self.n_query,
            n_key=self.n_key,
            blocks_per_ckpt=blocks_per_ckpt,
            inf=self.inf,
            linear_init_params=linear_init_params.diffusion_transformer,
            use_reentrant=use_reentrant,
        )

        self.c_token = c_token
        self.linear_q = nn.Sequential(
            Linear(c_atom, c_token, **linear_init_params.linear_q), nn.ReLU()
        )

    def get_atom_reps(
        self,
        batch: TensorDict,
        rl: Optional[torch.Tensor] = None,
        si_trunk: Optional[torch.Tensor] = None,
        zij_trunk: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary
            rl:
                [*, N_atom, 3] Noisy atom positions (optional)
            si_trunk:
                [*, N_atom, c_s] Trunk single representation (optional)
            zij_trunk:
                [*, N_atom, N_atom, c_z] Trunk pair representation (optional)
        Returns:
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair representation
                Note: Converted to block format ahead of time due to reduce memory cost
        """
        # Embed reference atom features
        # cl: [*, N_atom, c_atom]
        # plm: [*, N_blocks, N_query, N_key, c_atom_pair]
        cl, plm = self.ref_atom_feature_embedder(
            batch=batch, n_query=self.n_query, n_key=self.n_key
        )

        # Initialize atom single representation
        # [*, N_atom, c_atom]
        ql = cl.clone()

        # Embed noisy atom positions and trunk embeddings
        # cl: [*, N_atom, c_atom]
        # plm: [*, N_blocks, N_query, N_key, c_atom_pair]
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
                n_query=self.n_query,
                n_key=self.n_key,
            )

        # Add the combined single conditioning to the pair rep (line 13 - 14)
        cl_l, cl_m, atom_mask = convert_single_rep_to_blocks(
            ql=cl, n_query=self.n_query, n_key=self.n_key, atom_mask=batch["atom_mask"]
        )

        # Note to devs: in previous checkpoints before v13, linear_l and linear_m
        #  were reversed. Changed it for consistent naming.
        cl_lm = (
            self.linear_l(self.relu(cl_l.unsqueeze(-2)))
            + self.linear_m(self.relu(cl_m.unsqueeze(-3)))
        ) * atom_mask.unsqueeze(-1)

        # [*, N_blocks, N_query, N_key, c_atom_pair]
        plm = plm + cl_lm

        plm = plm + self.pair_mlp(plm)

        plm = plm * atom_mask.unsqueeze(-1)

        return ql, cl, plm

    def forward(
        self,
        batch: TensorDict,
        atom_mask: torch.Tensor,
        rl: Optional[torch.Tensor] = None,
        si_trunk: Optional[torch.Tensor] = None,
        zij_trunk: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_high_precision_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
                    - "token_mask": [*, N_token] token mask
                    - "num_atoms_per_token": [*, N_token] Number of atoms per token
            atom_mask:
                [*, N_atom] Atom mask
            rl:
                [*, N_atom, 3] Noisy atom positions (optional)
            si_trunk:
                [*, N_atom, c_s] Trunk single representation (optional)
            zij_trunk:
                [*, N_atom, N_atom, c_z] Trunk pair representation (optional)
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_high_precision_attention:
                Whether to run attention in high precision
        Returns:
            ai:
                [*, N_token, c_token] Token representation
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair representation
                Note: Converted to block format ahead of time due to reduce memory cost
        """
        atom_feat_args = (
            batch,
            rl,
            si_trunk,
            zij_trunk,
        )
        ql, cl, plm = checkpoint_section(
            fn=self.get_atom_reps,
            args=atom_feat_args,
            apply_ckpt=self.ckpt_intermediate_steps,
            use_reentrant=self.use_reentrant,
        )

        # Cross attention transformer (line 15)
        # [*, N_blocks, N_query, c_atom]
        ql = self.atom_transformer(
            a=ql,
            s=cl,
            z=plm,
            mask=atom_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_high_precision_attention=use_high_precision_attention,
        )

        agg_args = (
            batch["token_mask"],
            batch["atom_to_token_index"],
            atom_mask,
            self.linear_q(ql),
            -2,
        )
        ai = checkpoint_section(
            fn=aggregate_atom_feat_to_tokens,
            args=agg_args,
            apply_ckpt=self.ckpt_intermediate_steps,
            use_reentrant=self.use_reentrant,
        )

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
        use_ada_layer_norm: bool,
        blocks_per_ckpt: Optional[int] = None,
        inf: float = 1e9,
        linear_init_params: ConfigDict = lin_init.atom_att_dec_init,
        use_reentrant: Optional[bool] = None,
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
            use_ada_layer_norm:
                Whether to apply AdaLN-Zero conditioning
            blocks_per_ckpt:
                Number of blocks per checkpoint. If set, checkpointing will
                be used to save memory.
            inf:
                Large number used for attention masking
            linear_init_params:
                Linear layer initialization parameters
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
        """
        super().__init__()

        self.inf = inf

        self.linear_q_in = Linear(c_token, c_atom, **linear_init_params.linear_q_in)

        self.atom_transformer = DiffusionTransformer(
            c_a=c_atom,
            c_s=c_atom,
            c_z=c_atom_pair,
            c_hidden=c_hidden,
            no_heads=no_heads,
            no_blocks=no_blocks,
            n_transition=n_transition,
            use_ada_layer_norm=use_ada_layer_norm,
            n_query=n_query,
            n_key=n_key,
            blocks_per_ckpt=blocks_per_ckpt,
            inf=self.inf,
            linear_init_params=linear_init_params.diffusion_transformer,
            use_reentrant=use_reentrant,
        )

        self.layer_norm = LayerNorm(c_in=c_atom, create_offset=False)
        self.linear_q_out = Linear(c_atom, 3, **linear_init_params.linear_q_out)

    def forward(
        self,
        batch: TensorDict,
        atom_mask: torch.Tensor,
        ai: torch.Tensor,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_deepspeed_evo_attention: bool = False,
        use_high_precision_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "token_mask": [*, N_token] Token mask
                    - "num_atoms_per_token": [*, N_token] Number of atoms per token
            atom_mask:
                [*, N_atom] Atom mask
            ai:
                [*, N_token, c_token] Token representation
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair representation
                Note: Converted to block format in AtomAttentionEncoder
            chunk_size:
                Inference-time subbatch size
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
            use_high_precision_attention:
                Whether to run attention in high precision
        Returns:
            rl_update:
                [*, N_atom, 3] Atom position updates
        """
        # Broadcast per-token activations to atoms
        # [*, N_atom, c_atom]
        ql = ql + broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=self.linear_q_in(ai),
            token_dim=-2,
        )

        # Atom transformer
        # [*, N_atom, c_atom]
        ql = self.atom_transformer(
            a=ql,
            s=cl,
            z=plm,
            mask=atom_mask,
            chunk_size=chunk_size,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_high_precision_attention=use_high_precision_attention,
        )

        # Compute updates for atom positions
        # [*, N_atom, 3]
        rl_update = self.linear_q_out(self.layer_norm(ql))

        return rl_update
