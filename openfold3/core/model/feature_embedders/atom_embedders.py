from typing import Optional

import torch
from ml_collections import ConfigDict
from torch import nn as nn

from openfold3.core.config import default_linear_init_config as lin_init
from openfold3.core.model.layers.sequence_local_atom_attention import TensorDict
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.core.utils.checkpointing import checkpoint_section


def convert_to_blocks_1d(x, dim, shift_interval, block_len, num_blocks):
    if shift_interval == block_len:
        blocks = torch.chunk(x, num_blocks, dim=dim)
    else:
        blocks = [
            x.narrow(dim, shift_interval * i, block_len) for i in range(num_blocks)
        ]
    return torch.stack(blocks, dim=dim - 1)


def convert_to_blocks_2d(x, dims, shift_interval, block_lens, num_blocks):
    blocks = [
        x.narrow(dims[0], shift_interval * i, block_lens[0]).narrow(
            dims[1], shift_interval * i, block_lens[1]
        )
        for i in range(num_blocks)
    ]
    return torch.stack(blocks, dim=min(dims) - 1)


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
        self.linear_feats = Linear(
            c_atom_ref, c_atom, **linear_init_params.linear_feats
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
                    batch["ref_charge"].unsqueeze(-1),
                    batch["ref_mask"].unsqueeze(-1),
                    batch["ref_element"],
                    batch["ref_atom_name_chars"].flatten(start_dim=-2),
                ],
                dim=-1,
            )
        )  # CONFIRM THIS FORMAT ONCE DATALOADER/FEATURIZER DONE

        # Embed offsets
        # dlm: [*, N_atom, N_atom, 3]
        # vlm: [*, N_atom, N_atom]
        # plm: [*, N_atom, N_atom, c_atom_pair]
        dlm = batch["ref_pos"].unsqueeze(-2) - batch["ref_pos"].unsqueeze(-3)
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


class AtomTrunkEmbedder(nn.Module):
    """
    Implements AF3 Algorithm 5 (line 8 - 12).
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_atom: int,
        c_atom_pair: int,
        linear_init_params: ConfigDict = lin_init.atom_trunk_emb_init,
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
        self.layer_norm_s = LayerNorm(c_s)
        self.linear_s = Linear(c_s, c_atom, **linear_init_params.linear_s)
        self.layer_norm_z = LayerNorm(c_z)
        self.linear_z = Linear(c_z, c_atom_pair, **linear_init_params.linear_z)

    def forward(
        self,
        batch: TensorDict,
        cl: torch.Tensor,
        plm: torch.Tensor,
        si_trunk: torch.Tensor,
        zij_trunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "token_mask": [*, N_token] Token mask
                    - "num_atoms_per_token": [*, N_token] Number of atoms per token
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair conditioning
            si_trunk:
                [*, N_token, c_s] Trunk single representation
            zij_trunk:
                [*, N_token, N_token, c_z] Trunk pair representation
        Returns:
            cl:
                [*, N_atom, c_atom] Atom single conditioning with trunk single
                    representation embedded
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair conditioning with trunk pair
                    representation embedded
        """

        # Broadcast trunk single representation into atom single conditioning
        # [*, N_atom, c_atom]
        sl_trunk = broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=si_trunk,
            token_dim=-2,
        )
        cl = cl + self.linear_s(self.layer_norm_s(sl_trunk))

        # Broadcast trunk pair representation into atom pair conditioning
        # [*, N_atom, N_atom, c_atom_pair]
        zij_trunk = self.linear_z(self.layer_norm_z(zij_trunk))
        zlj_trunk = broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=zij_trunk,
            token_dim=-3,
        )
        zlm_trunk = broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=zlj_trunk.transpose(-2, -3),
            token_dim=-3,
        )
        zlm_trunk = zlm_trunk.transpose(-2, -3)

        plm = plm + zlm_trunk

        return cl, plm


class SeqLocalAtomEmbedder(nn.Module):
    """
    Implements AF3 Algorithm 5 (line 1 - 14).
    """

    def __init__(
        self,
        c_atom_ref: int,
        c_atom: int,
        c_atom_pair: int,
        add_trunk_emb: bool,
        c_s: Optional[int] = None,
        c_z: Optional[int] = None,
        ckpt_intermediate_steps: bool = False,
        linear_init_params: ConfigDict = lin_init.seq_local_atom_emb_init,
        use_reentrant: Optional[bool] = None,
    ):
        """
        Args:
            c_atom_ref:
                Reference atom feature channel dimension (390)
            c_atom:
                Atom single conditioning channel dimension
            c_atom_pair:
                Atom pair conditioning channel dimension
            add_trunk_emb:
                Whether to add trunk embeddings
            c_s:
                Single representation channel dimension (optional)
            c_z:
                Pair representation channel dimension (optional)
            ckpt_intermediate_steps:
                Whether to checkpoint intermediate steps in the module, including
                RefAtomFeatureEmbedder, NoisyPositionEmbedder, and feature aggregation
            linear_init_params:
                Linear layer initialization parameters
            use_reentrant:
                Whether to use reentrant variant of checkpointing. If set,
                torch checkpointing will be used (DeepSpeed does not support
                this feature)
        """
        super().__init__()
        self.add_trunk_emb = add_trunk_emb
        self.ckpt_intermediate_steps = ckpt_intermediate_steps
        self.use_reentrant = use_reentrant

        self.ref_atom_feature_embedder = RefAtomFeatureEmbedder(
            c_atom_ref=c_atom_ref,
            c_atom=c_atom,
            c_atom_pair=c_atom_pair,
            linear_init_params=linear_init_params.ref_atom_emb,
        )

        if self.add_trunk_emb:
            self.atom_trunk_embedder = AtomTrunkEmbedder(
                c_s=c_s,
                c_z=c_z,
                c_atom=c_atom,
                c_atom_pair=c_atom_pair,
                linear_init_params=linear_init_params.atom_trunk_emb,
            )

        self.relu = nn.ReLU()
        self.linear_l = Linear(c_atom, c_atom_pair, **linear_init_params.linear_l)
        self.linear_m = Linear(
            c_atom, c_atom_pair, **linear_init_params.linear_m
        )  # TODO: check initialization

        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, **linear_init_params.pair_mlp),
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, **linear_init_params.pair_mlp),
            nn.ReLU(),
            Linear(c_atom_pair, c_atom_pair, **linear_init_params.pair_mlp),
        )

    def get_atom_reps(
        self,
        batch: TensorDict,
        si_trunk: Optional[torch.Tensor] = None,
        zij_trunk: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            batch:
                Input feature dictionary
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
                [*, N_atom, N_atom, c_atom_pair] Atom pair representation
        """
        # Embed reference atom features
        # cl: [*, N_atom, c_atom]
        # plm: [*, N_atom, N_atom, c_atom_pair]
        cl, plm = self.ref_atom_feature_embedder(batch)

        # Initialize atom single representation
        # [*, N_atom, c_atom]
        ql = cl.clone()

        # Embed noisy atom positions and trunk embeddings
        # cl: [*, N_atom, c_atom]
        # plm: [*, N_atom, N_atom, c_atom_pair]
        # ql: [*, N_atom, c_atom]
        if self.add_trunk_emb:
            cl, plm = self.atom_trunk_embedder(
                batch=batch,
                cl=cl,
                plm=plm,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
            )

        # Add the combined single conditioning to the pair rep (line 13 - 14)
        # [*, N_atom, N_atom, c_atom_pair]
        plm = (
            plm
            + self.linear_l(self.relu(cl.unsqueeze(-3)))
            + self.linear_m(self.relu(cl.unsqueeze(-2)))
        )
        plm = plm + self.pair_mlp(plm)

        return ql, cl, plm

    def forward(
        self,
        batch: TensorDict,
        si_trunk: Optional[torch.Tensor] = None,
        zij_trunk: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            ql:
                [*, N_atom, c_atom] Atom single representation
            cl:
                [*, N_atom, c_atom] Atom single conditioning
            plm:
                [*, N_atom, N_atom, c_atom_pair] Atom pair conditioning
        """
        atom_feat_args = (batch, si_trunk, zij_trunk)
        ql, cl, plm = checkpoint_section(
            fn=self.get_atom_reps,
            args=atom_feat_args,
            apply_ckpt=self.ckpt_intermediate_steps,
            use_reentrant=self.use_reentrant,
        )

        return ql, cl, plm
