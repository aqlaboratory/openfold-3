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

import math

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.layers.sequence_local_atom_attention import (
    AtomAttentionEncoder,
)
from openfold3.core.model.primitives import LayerNorm, Linear, normal_init_
from openfold3.core.utils.relpos import relpos_complex, relpos_monomer, relpos_multimer
from openfold3.core.utils.tensor_utils import add


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
        ri = ri.type(tf_emb_i.dtype)
        relpos_feats = relpos_monomer(ri=ri, relpos_k=self.relpos_k).to(ri.dtype)
        pair_emb = self.linear_relpos(relpos_feats)
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

        relpos_feats = relpos_multimer(
            batch=batch,
            max_relative_idx=self.max_relative_idx,
            use_chain_relative=self.use_chain_relative,
            max_relative_chain=self.max_relative_chain,
        ).to(self.linear_relpos.weight.dtype)
        pair_emb = pair_emb + self.linear_relpos(relpos_feats)

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb


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
        self.max_relative_idx = max_relative_idx
        self.max_relative_chain = max_relative_chain

        self.atom_attn_enc = AtomAttentionEncoder(
            **atom_attn_enc,
            add_noisy_pos=False,
        )

        self.linear_s = Linear(c_s_input, c_s, **linear_init_params.linear_s)
        self.linear_z_i = Linear(c_s_input, c_z, **linear_init_params.linear_z_i)
        self.linear_z_j = Linear(c_s_input, c_z, **linear_init_params.linear_z_j)

        num_rel_pos_bins = 2 * max_relative_idx + 2
        num_rel_token_bins = 2 * max_relative_idx + 2
        num_rel_chain_bins = 2 * max_relative_chain + 2
        num_same_entity_features = 1
        num_relpos_dims = (
            num_rel_pos_bins
            + num_rel_token_bins
            + num_rel_chain_bins
            + num_same_entity_features
        )

        self.linear_relpos = Linear(
            num_relpos_dims, c_z, **linear_init_params.linear_relpos
        )

        # Expecting binary feature "token_bonds" of shape [*, N_token, N_token, 1]
        self.linear_token_bonds = Linear(
            1, c_z, **linear_init_params.linear_token_bonds
        )

    def forward(
        self,
        batch: dict,
        inplace_safe: bool = False,
        use_high_precision_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch:
                Input feature dictionary
            inplace_safe:
                Whether inplace operations can be performed
            use_high_precision_attention:
                Whether to run attention in high precision
        Returns:
            s_input:
                [*, N_token, C_s_input] Single (input) representation
            s:
                [*, N_token, C_s] Single representation
            z:
                [*, N_token, N_token, C_z] Pair representation
        """
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            a, _, _, _ = self.atom_attn_enc(
                batch=batch,
                atom_mask=batch["atom_mask"],
                use_high_precision_attention=use_high_precision_attention,
            )

        a = a.to(dtype=self.linear_s.weight.dtype)

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
        z = s_input_emb_i[..., None, :] + s_input_emb_j[..., None, :, :]

        relpos_feats = relpos_complex(
            batch=batch,
            max_relative_idx=self.max_relative_idx,
            max_relative_chain=self.max_relative_chain,
        ).to(dtype=z.dtype)
        relpos_emb = self.linear_relpos(relpos_feats)
        z = add(z, relpos_emb, inplace=inplace_safe)

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
    def _subsample_main_msa(
        msa_feat: torch.Tensor,
        msa_mask: torch.Tensor,
        num_paired_seqs: torch.Tensor,
        asym_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subsample main MSA (unpaired MSA) features for a single sample in the batch.
        The subsampling is independent per each chain.

        Args:
            msa_feat:
                [N_msa, N_token, c_m_feats] MSA features
            msa_mask:
                [N_msa, N_token] Binary mask indicating valid MSA entries. 
                The MSA is padded to the maximum MSA dimension across chains, so this mask will be all zeros 
                for any chain whose actual MSA dimension is less than the maximum.
            num_paired_seqs:
                [] Number of paired MSA sequences
            asym_id:
                [N_token] Id of the chain each token belongs to
        Returns:
            sampled_msa:
                [N_seq, N_token, c_m_feats] Sampled MSA features
            msa_mask:
                [N_seq, N_token] Binary mask for sampled MSA entries.
                MSA per chain is independently subsampled and re-padded to a shared dimension.
        """
        
        # Set the sequence dimension for the two tensors, the token dimension is this +1
        feat_seq_dim = -3
        mask_seq_dim = -2

        num_paired_seqs = int(num_paired_seqs.item())

        # Separate UniProt paired sequences and main MSA (only the latter is subsampled)
        total_msa_seq = msa_feat.shape[feat_seq_dim]
        num_main_msa_seqs = total_msa_seq - num_paired_seqs

        if num_main_msa_seqs == 0:
            return msa_feat, msa_mask

        split_sections = [num_paired_seqs, num_main_msa_seqs]

        paired_msa_feat, main_msa_feat = torch.split(
            msa_feat, split_sections, dim=feat_seq_dim
        )
        paired_msa_mask, main_msa_mask = torch.split(
            msa_mask, split_sections, dim=mask_seq_dim
        )

        # Get the length of each chain using consecutive unique asym_id
        _, chain_splits = torch.unique_consecutive(asym_id, return_counts=True)

        # Split the tensor obtaining separate tensors for each chain
        per_chain_msa_feat = torch.split(
            main_msa_feat, chain_splits.tolist(), dim=feat_seq_dim + 1
        )
        per_chain_msa_mask = torch.split(
            main_msa_mask, chain_splits.tolist(), dim=mask_seq_dim + 1
        )

        # Get the number of main msa seqs per chain
        # summing the ones in the seq dimension in the mask
        # Use float32 as bf16 precision is not enough to distinguish all 16384 integers
        per_chain_main_msa_dim = [
            int(torch.sum(mask, dim=-2, dtype=torch.float32)[..., 0])
            for mask in per_chain_msa_mask
        ]

        # Max number of sequences across chains
        max_msa_seqs_across_chains = max(per_chain_main_msa_dim)

        # Dimension to subsample all chains to
        seq_subsample_dim = torch.randint(
            low=1,
            high=int(max_msa_seqs_across_chains + 1),
            size=(1,),
            device=msa_feat.device,
        )

        # Get a random permutation of the sequence indexes for each chain
        # Pad it with padding row indexes until max_msa_seqs_across_chains
        chain_index_permutations = [
            torch.cat(
                [
                    torch.randperm(num_seqs, device=msa_feat.device),
                    torch.arange(
                        num_seqs, max_msa_seqs_across_chains, device=msa_feat.device
                    ),
                ]
            )[:seq_subsample_dim]
            for num_seqs in per_chain_main_msa_dim
        ]

        # Apply the permutation and keep seq_subsample_dim sequences
        sampled_chain_feats = [
            feat[..., perm, :, :]
            for feat, perm in zip(per_chain_msa_feat, chain_index_permutations)
        ]
        sampled_chain_masks = [
            mask[..., perm, :]
            for mask, perm in zip(per_chain_msa_mask, chain_index_permutations)
        ]

        # Concatenate the chains back together
        sampled_main_msa_feat = torch.cat(sampled_chain_feats, dim=feat_seq_dim + 1)
        sampled_main_msa_mask = torch.cat(sampled_chain_masks, dim=mask_seq_dim + 1)

        # Stack with the uniprot features and mask
        sampled_msa_feat = torch.cat(
            [paired_msa_feat, sampled_main_msa_feat], dim=feat_seq_dim
        )
        sampled_msa_mask = torch.cat(
            [paired_msa_mask, sampled_main_msa_mask], dim=mask_seq_dim
        )

        return sampled_msa_feat, sampled_msa_mask
    
    @staticmethod
    def _subsample_all_msa(
        msa_feat: torch.Tensor,
        msa_mask: torch.Tensor,
        no_subsampled_all_msa: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Subsample all MSA sequences (paired + main) to a fixed number of sequences,
        prioritizing those with at least one non-masked token.
        
        Args:
            msa_feat:
                [N_msa, N_token, c_m_feats] MSA features
            msa_mask:
                [N_msa, N_token] Binary mask indicating valid MSA entries. 
                The MSA is padded to the maximum MSA dimension across chains, so this mask will be all zeros 
                for any chain whose actual MSA dimension is less than the maximum.
            no_subsampled_all_msa:
                The number of MSA sequences to retain after subsampling.
        Returns:
            sampled_msa:
                [N_seq, N_token, c_m_feats] Sampled MSA features
            msa_mask:
                [N_seq, N_token] Binary mask for sampled MSA entries.
                MSA per chain is independently subsampled and re-padded to a shared dimension.
        """
        
        # Set the sequence dimension for the two tensors, the token dimension is this +1
        feat_seq_dim = -3
        mask_seq_dim = -2
        
        if isinstance(no_subsampled_all_msa, torch.Tensor):
            no_subsampled_all_msa = no_subsampled_all_msa.item()
            
        # If we already have <= desired, nothing to do
        if msa_feat.shape[feat_seq_dim] <= no_subsampled_all_msa:
            return msa_feat, msa_mask

        # Valid msa
        valid_msa = (msa_mask.sum(dim=mask_seq_dim+1) > 0).squeeze() # [N_msa]
        valid_idx = valid_msa.nonzero().squeeze() 
        invalid_idx = (~valid_msa).nonzero().squeeze() 
                
        device = msa_feat.device
        # Pick msa from the valid ones at random
        if valid_idx.numel() >= no_subsampled_all_msa:
            permuted_idx = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
            selected = permuted_idx[:no_subsampled_all_msa]
        else:
            # Take all valid, then fill with random invalid
            take_invalid = no_subsampled_all_msa - valid_idx.numel()
            if invalid_idx.numel() > 0:
                permuted_idx = invalid_idx[torch.randperm(invalid_idx.numel(), device=device)]
                selected = torch.cat([valid_idx, permuted_idx[:take_invalid]], dim=0)
            else:
                selected = valid_idx
        
        feat_sub = msa_feat.index_select(feat_seq_dim, selected)
        mask_sub = msa_mask.index_select(mask_seq_dim, selected)
        return feat_sub, mask_sub
    
    def _apply_subsample_fn_batch(
        fn: callable,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply a MSA subsampling function `fn` independently across a batch.
        Contains extra logic to unbind the batch dim prior to sampling
        and pad/stack the output MSA features and mask.        
        All arguments are unbound and passed to `fn` per-sample.

        Args:
            fn:
                A function that takes per-sample inputs and returns (feat, mask).
            **kwargs:
               Keyword arguments to forward to `fn`. Assumes each input tensor  is batched in dim=0.

        Returns:
            sampled_msa:
                [N_seq, N_token, c_m_feats] Sampled MSA features
            msa_mask:
                [N_seq, N_token] Binary mask for sampled MSA entries.
        """
        
        batch_size = kwargs.values()[0].shape[0]
        per_sample_kwargs_list = [
            {k: v[i] for k, v in kwargs.items()}
            for i in range(batch_size)
        ]
        
        per_sample_subsampled_msa = []
        per_sample_subsampled_msa_mask = []

        for kwarg in per_sample_kwargs_list:
            subsampled_msa, subsampled_mask = fn(**kwarg)
            per_sample_subsampled_msa.append(subsampled_msa)
            per_sample_subsampled_msa_mask.append(subsampled_mask)

        # Number of sequences to pad to for all the batch
        max_msa_seqs_batch = max([m.shape[-3] for m in per_sample_subsampled_msa])

        def pad_sequences_dim(m, max_seqs, seq_dim):
            """Pad the msa to max_seqs along seq_dim to stack them in a batch"""

            # Add zero padding at start and end for all dimensions after seq_dim
            non_pad_dims = (0, 0) * (abs(seq_dim) - 1)

            # Pad the seq_dim to max_msa_seqs length
            pad = non_pad_dims + (0, max_seqs - m.shape[seq_dim])

            return torch.nn.functional.pad(m, pad)

        # Pad the sequences to same seq length and stack them in a batch
        sampled_msa = torch.stack(
            [
                pad_sequences_dim(m, max_msa_seqs_batch, seq_dim=-3)
                for m in per_sample_subsampled_msa
            ],
            dim=0,
        )
        sampled_msa_mask = torch.stack(
            [
                pad_sequences_dim(m, max_msa_seqs_batch, seq_dim=-2)
                for m in per_sample_subsampled_msa_mask
            ],
            dim=0,
        )

        return sampled_msa, sampled_msa_mask

    def forward(
        self, 
        batch: dict, 
        s_input: torch.Tensor, 
        subsample_main_msa: bool, 
        subsample_all_msa: bool, 
        no_subsampled_all_msa: int
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
                    - "asym_id": [*, N_token]
            s_input:
                [*, N_token, C_s_input] single embedding
            subsample_main_msa:
                Whether to subsample only the main MSA to a random depth, following AF3 SI Section 2.2.
            subsample_all_msa:
                Whether to subsample all MSA sequences (paired + main) to a random depth.
            no_subsampled_all_msa:
                If subsample_all_msa is True, this specifies the number of MSA sequences to retain after subsampling.

        Returns:
            m:
                [*, N_seq, N_token, C_m] MSA embedding
            msa_mask:
                [*, N_seq, N_token] MSA mask
        """
        batch_dims = batch["msa"].shape[:-3]

        # [*, N_msa, N_token, 34]
        msa_feat = torch.cat(
            [
                batch["msa"],
                batch["has_deletion"].unsqueeze(-1),
                batch["deletion_value"].unsqueeze(-1),
            ],
            dim=-1,
        )
        
        if subsample_main_msa:
            if math.prod(batch_dims) > 1:
                msa_feat, msa_mask = self._apply_subsample_fn_batch(
                    fn=self._subsample_main_msa,
                    msa_feat=msa_feat,
                    msa_mask=batch["msa_mask"],
                    num_paired_seqs=batch["num_paired_seqs"],
                    asym_id=batch["asym_id"],
                )
            else:
                msa_feat, msa_mask = self._subsample_main_msa(
                    msa_feat=msa_feat,
                    msa_mask=batch["msa_mask"],
                    num_paired_seqs=batch["num_paired_seqs"],
                    asym_id=batch["asym_id"]
                )
                
        if subsample_all_msa:
            if math.prod(batch_dims) > 1:
                msa_feat, msa_mask = self._apply_subsample_fn_batch(
                    fn=self._subsample_all_msa,
                    msa_feat=msa_feat,
                    msa_mask=msa_mask,
                    no_subsampled_all_msa=torch.full(
                        (msa_feat.shape[0],), 
                        no_subsampled_all_msa, 
                        device=msa_feat.device
                    )
                )
            else:
                msa_feat, msa_mask = self._subsample_all_msa(
                    msa_feat=msa_feat,
                    msa_mask=msa_mask,
                    no_subsampled_all_msa=no_subsampled_all_msa
                )

        # [*, N_seq, N_token, C_m]
        m = self.linear_m(msa_feat)
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
