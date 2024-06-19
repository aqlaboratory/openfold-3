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
from typing import Tuple

from openfold3.core.model.layers.sequence_local_atom_attention import AtomAttentionEncoder
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.tensor_utils import add, one_hot


class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements AF2 Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedder, self).__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, ri: torch.Tensor):
        """
        Computes relative positional encodings

        Implements AF2 Algorithm 4.

        Args:
            ri:
                "residue_index" features of shape [*, N]
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        ) 
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)
        return self.linear_relpos(d)

    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        msa: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: Dict containing
                "target_feat":
                    Features of shape [*, N_res, tf_dim]
                "residue_index":
                    Features of shape [*, N_res]
                "msa_feat":
                    Features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = self.relpos(ri.type(tf_emb_i.dtype))
        pair_emb = add(pair_emb, 
            tf_emb_i[..., None, :], 
            inplace=inplace_safe
        )
        pair_emb = add(pair_emb, 
            tf_emb_j[..., None, :, :], 
            inplace=inplace_safe
        )

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb


class InputEmbedderMultimer(nn.Module):
    """
    Embeds a subset of the input features.

    Implements AF2-Multimer Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        max_relative_idx: int,
        use_chain_relative: bool,
        max_relative_chain: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
            relpos_k:
                Window size used in relative positional encoding
        """
        super(InputEmbedderMultimer, self).__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        # RPE stuff
        self.max_relative_idx = max_relative_idx
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if(self.use_chain_relative):
            self.no_bins = (
                2 * max_relative_idx + 2 +
                1 +
                2 * max_relative_chain + 2
            )
        else:
            self.no_bins = 2 * max_relative_idx + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, batch):
        pos = batch["residue_index"]
        asym_id = batch["asym_id"]
        asym_id_same = (asym_id[..., None] == asym_id[..., None, :])
        offset = pos[..., None] - pos[..., None, :]

        clipped_offset = torch.clamp(
            offset + self.max_relative_idx, 0, 2 * self.max_relative_idx
        )

        rel_feats = []
        if(self.use_chain_relative):
            final_offset = torch.where(
                asym_id_same, 
                clipped_offset,
                (2 * self.max_relative_idx + 1) * 
                torch.ones_like(clipped_offset)
            )
            boundaries = torch.arange(
                start=0, end=2 * self.max_relative_idx + 2, device=final_offset.device
            )
            rel_pos = one_hot(
                final_offset,
                boundaries,
            )

            rel_feats.append(rel_pos)

            entity_id = batch["entity_id"]
            entity_id_same = (entity_id[..., None] == entity_id[..., None, :])
            rel_feats.append(entity_id_same[..., None].to(dtype=rel_pos.dtype))

            sym_id = batch["sym_id"]
            rel_sym_id = sym_id[..., None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain
            clipped_rel_chain = torch.clamp(
                rel_sym_id + max_rel_chain,
                0,
                2 * max_rel_chain,
            )

            final_rel_chain = torch.where(
                entity_id_same,
                clipped_rel_chain,
                (2 * max_rel_chain + 1) *
                torch.ones_like(clipped_rel_chain)
            )

            boundaries = torch.arange(
                start=0, end=2 * max_rel_chain + 2, device=final_rel_chain.device
            )
            rel_chain = one_hot(
                final_rel_chain,
                boundaries,
            )

            rel_feats.append(rel_chain)
        else:
            boundaries = torch.arange(
                start=0, end=2 * self.max_relative_idx + 1, device=clipped_offset.device
            )
            rel_pos = one_hot(
                clipped_offset, boundaries,
            )
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, dim=-1).to(
            self.linear_relpos.weight.dtype
        )

        return self.linear_relpos(rel_feat)

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        tf = batch["target_feat"]
        msa = batch["msa_feat"]

        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]
        pair_emb = pair_emb + self.relpos(batch)

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb


class RelposAllAtom(nn.Module):
    def __init__(
        self,
        c_z: int,
        max_relative_idx: int = 2,
        max_relative_chain: int = 32,
        **kwargs,
    ):
        """

        Args:
            c_z:
            max_relative_idx:
            max_relative_chain:
            **kwargs:
        """
        super(RelposAllAtom, self).__init__()

        # RPE stuff
        self.max_relative_idx = max_relative_idx
        self.max_relative_chain = max_relative_chain

        self.no_bins = (
            2 * max_relative_idx + 2 +
            1 +
            2 * max_relative_chain + 2
        )

        self.linear_relpos = Linear(self.no_bins, c_z, bias=False)

    @staticmethod
    def relpos(pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int):
        """

        Args:
            pos:
            condition:
            rel_clip_idx:

        Returns:

        """
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(
            offset + rel_clip_idx, 0, 2 * rel_clip_idx
        )
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) *
            torch.ones_like(clipped_offset)
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device
        )
        rel_pos = one_hot(
            final_offset,
            boundaries,
        )

        return rel_pos

    def forward(self, batch):
        """

        Args:
            batch:

        Returns:

        """
        res_idx = batch["residue_index"]
        asym_id = batch["asym_id"]
        entity_id = batch["entity_id"]
        same_chain = (asym_id[..., None] == asym_id[..., None, :])
        same_res = (res_idx[..., None] == res_idx[..., None, :])

        rel_pos = self.relpos(pos=res_idx,
                              condition=same_chain,
                              rel_clip_idx=self.max_relative_idx)
        rel_token = self.relpos(pos=batch["token_index"],
                                condition=same_chain & same_res,
                                rel_clip_idx=self.max_relative_idx)

        same_entity = (entity_id[..., None] == entity_id[..., None, :])
        same_entity = same_entity[..., None].to(dtype=rel_pos.dtype)

        rel_chain = self.relpos(pos=batch["sym_id"],
                                condition=~same_chain,
                                rel_clip_idx=self.max_relative_chain)

        rel_feat = torch.cat([rel_pos,
                              rel_token,
                              same_entity,
                              rel_chain], dim=-1).to(self.linear_relpos.weight.dtype)

        return self.linear_relpos(rel_feat)


class InputEmbedderAllAtom(nn.Module):
    """
    Embeds a subset of the input features.

    AF3 Algorithm 1 lines 1-5. Includes Algorithms 2 (InputFeatureEmbedder)
    and 3 (RelativePositionEncoding).
    """

    def __init__(
        self,
        atom_ref_dim: int,
        tok_bonds_dim: int,
        c_s: int,
        c_z: int,
        c_atom: int,
        c_atom_pair: int,
        c_token: int,
        c_hidden_att: int,
        max_relative_idx: int,
        max_relative_chain: int,
        **kwargs,
    ):
        """

        Args:
            atom_ref_dim:
            tok_bonds_dim:
            c_s:
            c_z:
            c_atom:
            c_atom_pair:
            c_token:
            c_hidden_att:
            max_relative_idx:
            max_relative_chain:
            **kwargs:
        """
        super(InputEmbedderAllAtom, self).__init__()

        self.c_s = c_s
        self.c_z = c_z

        self.atom_attn_enc = AtomAttentionEncoder(c_in=atom_ref_dim,
                                                  c_atom=c_atom,
                                                  c_atom_pair=c_atom_pair,
                                                  c_token=c_token,
                                                  c_hidden=c_hidden_att,
                                                  add_noisy_pos=False)

        self.linear_s = Linear(self.c_s, self.c_s, bias=False)
        self.linear_z_i = Linear(self.c_s, self.c_z, bias=False)
        self.linear_z_j = Linear(self.c_s, self.c_z, bias=False)

        self.relpos = RelposAllAtom(c_z=self.c_z,
                                    max_relative_idx=max_relative_idx,
                                    max_relative_chain=max_relative_chain)

        self.linear_tok_bonds = Linear(tok_bonds_dim, self.c_z, bias=False)

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            batch:

        Returns:

        """
        a, _, _, _ = self.atom_attn_enc(batch=batch)

        s = torch.cat([a,
                       batch["target_feat"],
                       batch['msa_profile'],
                       batch['deletion_mean'].unsqueeze(-1)], dim=-2)

        s = self.linear_s(s)
        z = self.linear_z_i(s) + self.linear_z_j(s)
        z = z + self.relpos(batch)
        z = z + self.linear_tok_bonds(batch['token_bonds'].unsqueeze(-1))

        return s, z


class PreembeddingEmbedder(nn.Module):
    """
    Embeds the sequence pre-embedding passed to the model and the target_feat features.
    """

    def __init__(
        self,
        tf_dim: int,
        preembedding_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                End channel dimension of the incoming target features
            preembedding_dim:
                End channel dimension of the incoming embeddings
            c_z:
                Pair embedding dimension
            c_m:
                Single-Seq embedding dimension
            relpos_k:
                Window size used in relative position encoding
        """
        super(PreembeddingEmbedder, self).__init__()

        self.tf_dim = tf_dim
        self.preembedding_dim = preembedding_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_preemb_m = Linear(self.preembedding_dim, c_m)
        self.linear_preemb_z_i = Linear(self.preembedding_dim, c_z)
        self.linear_preemb_z_j = Linear(self.preembedding_dim, c_z)

        # Relative Positional Encoding
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.no_bins, c_z)

    def relpos(self, ri: torch.Tensor):
        """
        Computes relative positional encodings
        Args:
            ri:
                "residue_index" feature of shape [*, N]
        Returns:
                Relative positional encoding of protein using the
                residue_index feature
        """
        d = ri[..., None] - ri[..., None, :]
        boundaries = torch.arange(
            start=-self.relpos_k, end=self.relpos_k + 1, device=d.device
        )
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        d = d.to(ri.dtype)
        return self.linear_relpos(d)

    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        preemb: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
        )
        preemb_emb = self.linear_preemb_m(preemb[..., None, :, :]) + tf_m
        preemb_emb_i = self.linear_preemb_z_i(preemb)
        preemb_emb_j = self.linear_preemb_z_j(preemb)

        pair_emb = self.relpos(ri.type(preemb_emb_i.dtype))
        pair_emb = add(pair_emb,
                       preemb_emb_i[..., None, :],
                       inplace=inplace_safe)
        pair_emb = add(pair_emb,
                       preemb_emb_j[..., None, :, :],
                       inplace=inplace_safe)

        return preemb_emb, pair_emb


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements AF2 Algorithm 32.
    """
    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N, C_m]
        m_update = self.layer_norm_m(m)
        if(inplace_safe):
            m.copy_(m_update)
            m_update = m

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)
        if(inplace_safe):
            z.copy_(z_update)
            z_update = z

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = add(z_update, d, inplace_safe)

        return m_update, z_update


class ExtraMSAEmbedder(nn.Module):
    """
    Embeds unclustered MSA sequences.

    Implements AF2 Algorithm 2, line 15
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
        """
        super(ExtraMSAEmbedder, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear = Linear(self.c_in, self.c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_extra_seq, N_res, C_in] "extra_msa_feat" features
        Returns:
            [*, N_extra_seq, N_res, C_out] embedding
        """
        x = self.linear(x)

        return x

