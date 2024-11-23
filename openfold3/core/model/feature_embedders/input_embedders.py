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
Embedders for input features. Includes InputEmbedders for monomer, multimer, soloseq,
and all-atom models. Also includes the RecyclingEmbedder and ExtraMSAEmbedder.
"""

from typing import Optional

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.layers.sequence_local_atom_attention import (
    AtomAttentionEncoder,
)
from openfold3.core.model.primitives import LayerNorm, Linear, normal_init_
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.core.utils.tensor_utils import add, binned_one_hot


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
        linear_init_params: ConfigDict = lin_init.monomer_input_emb_init,
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
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z, **linear_init_params.linear_tf_z_i)
        self.linear_tf_z_j = Linear(tf_dim, c_z, **linear_init_params.linear_tf_z_j)
        self.linear_tf_m = Linear(tf_dim, c_m, **linear_init_params.linear_tf_m)
        self.linear_msa_m = Linear(msa_dim, c_m, **linear_init_params.linear_msa_m)

        # RPE stuff
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(
            self.no_bins, c_z, **linear_init_params.linear_relpos
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                [*, N_res, tf_dim] Target features
            ri:
                [*, N_res] Residue index
            msa:
                [*, N_clust, N_res, msa_dim] MSA features
            inplace_safe:
                Bool determining if operations can be done in place (inference only)
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
        pair_emb = add(pair_emb, tf_emb_i[..., None, :], inplace=inplace_safe)
        pair_emb = add(pair_emb, tf_emb_j[..., None, :, :], inplace=inplace_safe)

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1))
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
        linear_init_params: ConfigDict = lin_init.multimer_input_emb_init,
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
            max_relative_idx:
                Maximum relative position and token indices clipped
            use_chain_relative:
                Whether to add relative chain encoding
            max_relative_chain:
                Maximum relative chain indices clipped
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z, **linear_init_params.linear_tf_z_i)
        self.linear_tf_z_j = Linear(tf_dim, c_z, **linear_init_params.linear_tf_z_j)
        self.linear_tf_m = Linear(tf_dim, c_m, **linear_init_params.linear_tf_m)
        self.linear_msa_m = Linear(msa_dim, c_m, **linear_init_params.linear_msa_m)

        # RPE stuff
        self.max_relative_idx = max_relative_idx
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if self.use_chain_relative:
            self.no_bins = 2 * max_relative_idx + 2 + 1 + 2 * max_relative_chain + 2
        else:
            self.no_bins = 2 * max_relative_idx + 1
        self.linear_relpos = Linear(
            self.no_bins, c_z, **linear_init_params.linear_relpos
        )

    def relpos(self, batch):
        pos = batch["residue_index"]
        asym_id = batch["asym_id"]
        asym_id_same = asym_id[..., None] == asym_id[..., None, :]
        offset = pos[..., None] - pos[..., None, :]

        clipped_offset = torch.clamp(
            offset + self.max_relative_idx, 0, 2 * self.max_relative_idx
        )

        rel_feats = []
        if self.use_chain_relative:
            final_offset = torch.where(
                asym_id_same,
                clipped_offset,
                (2 * self.max_relative_idx + 1) * torch.ones_like(clipped_offset),
            )
            boundaries = torch.arange(
                start=0, end=2 * self.max_relative_idx + 2, device=final_offset.device
            )
            rel_pos = binned_one_hot(
                final_offset,
                boundaries,
            )

            rel_feats.append(rel_pos)

            entity_id = batch["entity_id"]
            entity_id_same = entity_id[..., None] == entity_id[..., None, :]
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
                (2 * max_rel_chain + 1) * torch.ones_like(clipped_rel_chain),
            )

            boundaries = torch.arange(
                start=0, end=2 * max_rel_chain + 2, device=final_rel_chain.device
            )
            rel_chain = binned_one_hot(
                final_rel_chain,
                boundaries,
            )

            rel_feats.append(rel_chain)
        else:
            boundaries = torch.arange(
                start=0, end=2 * self.max_relative_idx + 1, device=clipped_offset.device
            )
            rel_pos = binned_one_hot(
                clipped_offset,
                boundaries,
            )
            rel_feats.append(rel_pos)

        rel_feat = torch.cat(rel_feats, dim=-1).to(self.linear_relpos.weight.dtype)

        return self.linear_relpos(rel_feat)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary

        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding
        """
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
            .expand((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb


class RelposAllAtom(nn.Module):
    """Relative Positional Encoding. Implements AF3 Algorithm 3."""

    def __init__(
        self,
        c_z: int,
        max_relative_idx: int,
        max_relative_chain: int,
        linear_init_params: ConfigDict = lin_init.relpos_emb_init,
    ):
        """
        Args:
            c_z:
                Pair embedding dimension
            max_relative_idx:
                Maximum relative position and token indices clipped
            max_relative_chain:
                Maximum relative chain indices clipped
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.max_relative_idx = max_relative_idx
        self.max_relative_chain = max_relative_chain

        num_rel_pos_bins = 2 * max_relative_idx + 2
        num_rel_token_bins = 2 * max_relative_idx + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        self.num_dims = (
            num_rel_pos_bins
            + num_rel_token_bins
            + num_rel_chain_bins
            + num_same_entity_features
        )

        self.linear_relpos = Linear(
            self.num_dims, c_z, **linear_init_params.linear_relpos
        )

    @staticmethod
    def relpos(
        pos: torch.Tensor, condition: torch.BoolTensor, rel_clip_idx: int
    ) -> torch.Tensor:
        """
        Args:
            pos:
                [*, N_token] Token index
            condition:
                [*, N_token, N_token] Condition for clipping
            rel_clip_idx:
                Max idx for clipping (max_relative_idx or max_relative_chain)
        Returns:
            rel_pos:
                [*, N_token, N_token, 2 * rel_clip_idx + 2] Relative position embedding
        """
        offset = pos[..., None] - pos[..., None, :]
        clipped_offset = torch.clamp(offset + rel_clip_idx, min=0, max=2 * rel_clip_idx)
        final_offset = torch.where(
            condition,
            clipped_offset,
            (2 * rel_clip_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * rel_clip_idx + 2, device=final_offset.device
        )
        rel_pos = binned_one_hot(
            final_offset,
            boundaries,
        )

        return rel_pos

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch:
                Input feature dictionary

        Returns:
            [*, N_token, N_token, C_z] Relative position embedding
        """
        res_idx = batch["residue_index"]
        asym_id = batch["asym_id"]
        entity_id = batch["entity_id"]
        same_chain = asym_id[..., None] == asym_id[..., None, :]
        same_res = res_idx[..., None] == res_idx[..., None, :]

        rel_pos = self.relpos(
            pos=res_idx, condition=same_chain, rel_clip_idx=self.max_relative_idx
        )
        rel_token = self.relpos(
            pos=batch["token_index"],
            condition=same_chain & same_res,
            rel_clip_idx=self.max_relative_idx,
        )

        same_entity = entity_id[..., None] == entity_id[..., None, :]
        same_entity = same_entity[..., None].to(dtype=rel_pos.dtype)

        rel_chain = self.relpos(
            pos=batch["sym_id"],
            condition=~same_chain,
            rel_clip_idx=self.max_relative_chain,
        )

        rel_feat = torch.cat([rel_pos, rel_token, same_entity, rel_chain], dim=-1).to(
            self.linear_relpos.weight.dtype
        )

        return self.linear_relpos(rel_feat)


class InputEmbedderAllAtom(nn.Module):
    """
    Embeds a subset of the input features.

    AF3 Algorithm 1 lines 1-5. Includes Algorithms 2 (InputFeatureEmbedder)
    and 3 (RelativePositionEncoding).
    """

    def __init__(
        self,
        c_s_input: int,
        c_s: int,
        c_z: int,
        max_relative_idx: int,
        max_relative_chain: int,
        atom_attn_enc: dict,
        linear_init_params: ConfigDict = lin_init.all_atom_input_emb_init,
    ):
        """
        Args:
            c_s_input:
                Per token input representation channel dimension
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            max_relative_idx:
                Maximum relative position and token indices clipped
            max_relative_chain:
                Maximum relative chain indices clipped
            atom_attn_enc:
                Config for the AtomAttentionEncoder
            linear_init_params:
                Linear layer initialization parameters
            **kwargs:
        """
        super().__init__()

        self.atom_attn_enc = AtomAttentionEncoder(
            **atom_attn_enc,
            add_noisy_pos=False,
        )

        self.linear_s = Linear(c_s_input, c_s, **linear_init_params.linear_s)
        self.linear_z_i = Linear(c_s_input, c_z, **linear_init_params.linear_z_i)
        self.linear_z_j = Linear(c_s_input, c_z, **linear_init_params.linear_z_j)

        self.relpos = RelposAllAtom(
            c_z=c_z,
            max_relative_idx=max_relative_idx,
            max_relative_chain=max_relative_chain,
            linear_init_params=linear_init_params.relpos_emb,
        )

        # Expecting binary feature "token_bonds" of shape [*, N_token, N_token, 1]
        self.linear_token_bonds = Linear(
            1, c_z, **linear_init_params.linear_token_bonds
        )

    def forward(
        self,
        batch: dict,
        inplace_safe: bool = False,
        use_deepspeed_evo_attention: Optional[bool] = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary
            inplace_safe:
                Whether inplace operations can be performed
        Returns:
            s_input:
                [*, N_token, C_s_input] Single (input) representation
            s:
                [*, N_token, C_s] Single representation
            z:
                [*, N_token, N_token, C_z] Pair representation
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed Evo Attention kernel
        """
        atom_mask = broadcast_token_feat_to_atoms(
            token_mask=batch["token_mask"],
            num_atoms_per_token=batch["num_atoms_per_token"],
            token_feat=batch["token_mask"],
        )

        a, _, _, _ = self.atom_attn_enc(
            batch=batch,
            atom_mask=atom_mask,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
        )

        # [*, N_token, C_s_input]
        s_input = torch.cat(
            [
                a,
                batch["restype"],
                batch["profile"],
                batch["deletion_mean"].unsqueeze(-1),
            ],
            dim=-1,
        )

        # [*, N_token, C_s]
        s = self.linear_s(s_input)

        s_input_emb_i = self.linear_z_i(s_input)
        s_input_emb_j = self.linear_z_j(s_input)
        token_bonds_emb = self.linear_token_bonds(
            batch["token_bonds"].unsqueeze(-1).to(dtype=s.dtype)
        )

        # [*, N_token, N_token, C_z]
        z = self.relpos(batch)
        z = add(z, s_input_emb_i[..., None, :], inplace=inplace_safe)
        z = add(z, s_input_emb_j[..., None, :, :], inplace=inplace_safe)
        z = add(z, token_bonds_emb, inplace=inplace_safe)

        return s_input, s, z


class MSAModuleEmbedder(nn.Module):
    """Sample MSA features and embed them. Implements AF3 Algorithm 8 lines 1-4.
    This section of the MSAModule is separated from the main stack to allow for
    tensor offloading during inference.
    """

    def __init__(
        self,
        c_m_feats: int,
        c_m: int,
        c_s_input: int,
        linear_init_params: ConfigDict = lin_init.msa_module_emb_init,
    ):
        """
        Args:
            c_m_feats:
                MSA input features channel dimension
            c_m:
                MSA channel dimension
            c_s_input:
                Single (s_input) channel dimension
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.linear_m = Linear(c_m_feats, c_m, **linear_init_params.linear_m)
        self.linear_s_input = Linear(
            c_s_input, c_m, **linear_init_params.linear_s_input
        )

    @staticmethod
    def subsample_msa(
        msa_feat: torch.Tensor,
        msa_mask: torch.Tensor,
        num_paired_seqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subsample main MSA features for a single element in the batch.

        Args:
            msa_feat:
                [N_seq, N_token, c_m_feats] MSA features
            msa_mask:
                [N_seq, N_token] MSA mask
            num_paired_seqs:
                Number of paired MSA sequences
        Returns:
            sampled_msa:
                [N_seq_sampled, N_token, c_m_feats] Sampled MSA features
            msa_mask:
                [N_seq_sampled, N_token] Sampled MSA mask
        """
        total_msa_seq = msa_feat.shape[-3]

        # Split uniprot and main MSA sequences. Only main MSA seqs will be sampled.
        # All uniprot seqs are in the final MSA representation.
        num_main_msa_seqs = total_msa_seq - num_paired_seqs

        if num_main_msa_seqs.any():
            split_sections = [num_paired_seqs, num_main_msa_seqs]
            uniprot_msa, main_msa = torch.split(msa_feat, split_sections, dim=-3)
            uniprot_msa_mask, main_msa_mask = torch.split(
                msa_mask, split_sections, dim=-2
            )

            # Sample Uniform[1, num_main_msa_seqs] sequences from the main MSA
            n_seq_sample = torch.randint(low=1, high=num_main_msa_seqs + 1, size=(1,))
            index_order = torch.randperm(num_main_msa_seqs, device=msa_feat.device)
            index_order = index_order[:n_seq_sample]

            main_msa = torch.index_select(main_msa, dim=-3, index=index_order)
            main_msa_mask = torch.index_select(main_msa_mask, dim=-2, index=index_order)

            # Combine uniprot and sampled main MSA sequences
            sampled_msa = torch.cat([uniprot_msa, main_msa], dim=-3)
            msa_mask = torch.cat([uniprot_msa_mask, main_msa_mask], dim=-2)

        else:
            sampled_msa = msa_feat

        return sampled_msa, msa_mask

    def forward(
        self, batch: dict, s_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary. Features used in this function:
                    - "msa": [*, N_msa, N_token, 32]
                    - "has_deletion": [*, N_msa, N_token]
                    - "deletion_value": [*, N_msa, N_token]
                    - "msa_mask": [*, N_msa, N_token]
                    - "num_paired_seqs": []
            s_input:
                [*, N_token, C_s_input] single embedding

        Returns:
            m:
                [*, N_seq, N_token, C_m] MSA embedding
            msa_mask:
                [*, N_seq, N_token] MSA mask
        """
        batch_dims = batch["msa"].shape[:-3]
        msa_feat = torch.cat(
            [
                batch["msa"],
                batch["has_deletion"].unsqueeze(-1),
                batch["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        msa_mask = batch["msa_mask"]

        # Unbind batch dim if it exists, and subsample msa seqs per batch
        if len(batch_dims) > 0:
            per_batch_msa = torch.unbind(msa_feat, dim=0)
            per_batch_mask = torch.unbind(msa_mask, dim=0)
            num_paired_seqs = torch.unbind(batch["num_paired_seqs"], dim=0)

            per_batch_sampled_msa = [
                self.subsample_msa(m, mask, n)
                for m, mask, n in zip(per_batch_msa, per_batch_mask, num_paired_seqs)
            ]

            sampled_msa = [m[0] for m in per_batch_sampled_msa]
            msa_mask = [m[1] for m in per_batch_sampled_msa]
            max_msa_seqs = max([m.shape[-3] for m in sampled_msa])

            def pad_batch_msas(m: torch.Tensor, seq_dim: int) -> torch.Tensor:
                # Pad MSA sequence dimension to max seqs in batch
                non_pad_dims = (0,) * 2 * (abs(seq_dim) - 1)
                return torch.nn.functional.pad(
                    m, (*non_pad_dims, 0, max_msa_seqs - m.shape[seq_dim])
                )

            sampled_msa = torch.stack(
                [pad_batch_msas(m, seq_dim=-3) for m in sampled_msa], dim=0
            )
            msa_mask = torch.stack(
                [pad_batch_msas(m, seq_dim=-2) for m in msa_mask], dim=0
            )

        else:
            sampled_msa, msa_mask = self.subsample_msa(
                msa_feat, msa_mask, batch["num_paired_seqs"]
            )

        m = self.linear_m(sampled_msa)
        m = m + self.linear_s_input(s_input).unsqueeze(-3)
        return m, msa_mask


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
        linear_init_params: ConfigDict = lin_init.preembed_init,
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
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.tf_dim = tf_dim
        self.preembedding_dim = preembedding_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_m = Linear(tf_dim, c_m, **linear_init_params.linear_tf_m)
        self.linear_preemb_m = Linear(
            self.preembedding_dim, c_m, **linear_init_params.linear_preemb_m
        )
        self.linear_preemb_z_i = Linear(
            self.preembedding_dim, c_z, **linear_init_params.linear_preemb_z_i
        )
        self.linear_preemb_z_j = Linear(
            self.preembedding_dim, c_z, **linear_init_params.linear_preemb_z_j
        )

        # Relative Positional Encoding
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(
            self.no_bins, c_z, **linear_init_params.linear_relpos
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tf_m = self.linear_tf_m(tf).unsqueeze(-3)
        preemb_emb = self.linear_preemb_m(preemb[..., None, :, :]) + tf_m
        preemb_emb_i = self.linear_preemb_z_i(preemb)
        preemb_emb_j = self.linear_preemb_z_j(preemb)

        pair_emb = self.relpos(ri.type(preemb_emb_i.dtype))
        pair_emb = add(pair_emb, preemb_emb_i[..., None, :], inplace=inplace_safe)
        pair_emb = add(pair_emb, preemb_emb_j[..., None, :, :], inplace=inplace_safe)

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
        linear_init_params: ConfigDict = lin_init.recycling_emb_init,
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
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = Linear(self.no_bins, self.c_z, **linear_init_params.linear)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        if inplace_safe:
            m.copy_(m_update)
            m_update = m

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)
        if inplace_safe:
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
        squared_bins = bins**2
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
        linear_init_params: ConfigDict = lin_init.extra_msa_emb_init,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear = Linear(self.c_in, self.c_out, **linear_init_params.linear)

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


class FourierEmbedding(nn.Module):
    """
    Implements AF3 Algorithm 22.
    """

    def __init__(self, c: int):
        """
        Args:
            c:
                Embedding dimension
        """
        super().__init__()
        w = torch.empty((c, 1))
        b = torch.empty(c)

        normal_init_(w)
        normal_init_(b)

        self.register_buffer("w", w)
        self.register_buffer("b", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, 1] Input tensor
        Returns:
            [*, c] Embedding
        """
        x = nn.functional.linear(x, self.w, self.b)
        return torch.cos(2 * torch.pi * x)
