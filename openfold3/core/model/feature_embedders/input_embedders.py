# Copyright 2026 AlQuraishi Laboratory
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
from openfold3.core.model.primitives import Linear
from openfold3.core.utils.relpos import relpos_complex
from openfold3.core.utils.tensor_utils import add


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

        z = self.embed_z(
            batch=batch, s_input=s_input, dtype=s.dtype, inplace_safe=inplace_safe
        )

        return s_input, s, z

    def embed_z(
        self,
        batch: dict,
        s_input: torch.Tensor,
        dtype: torch.dtype,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Build the input pair representation from cached single input features."""
        s_input_emb_i = self.linear_z_i(s_input)
        s_input_emb_j = self.linear_z_j(s_input)

        # [*, N_token, N_token, C_z]
        z = s_input_emb_i[..., None, :] + s_input_emb_j[..., None, :, :]
        del s_input_emb_i, s_input_emb_j

        # Fold in token_bonds early. In inference, linear_token_bonds is a
        # bias-free 1->C projection, so x @ W^T is exactly x * W[0] and avoids
        # materializing a separate pair-sized linear output.
        token_bonds = batch["token_bonds"].to(dtype=dtype)
        if not torch.is_grad_enabled() and self.linear_token_bonds.bias is None:
            z.view(-1, z.shape[-1]).addmm_(
                token_bonds.reshape(-1, 1),
                self.linear_token_bonds.weight[:, 0].unsqueeze(0),
            )
        else:
            token_bonds_emb = self.linear_token_bonds(token_bonds.unsqueeze(-1))
            z = add(z, token_bonds_emb, inplace=inplace_safe)
            del token_bonds_emb
        del token_bonds

        if not torch.is_grad_enabled() and self.linear_relpos.bias is None:
            self._add_relpos_from_weights_(z=z, batch=batch)
        else:
            relpos_feats = relpos_complex(
                batch=batch,
                max_relative_idx=self.max_relative_idx,
                max_relative_chain=self.max_relative_chain,
            ).to(dtype=z.dtype)
            relpos_emb = self.linear_relpos(relpos_feats)
            del relpos_feats
            z = add(z, relpos_emb, inplace=inplace_safe)
            del relpos_emb

        return z

    # by Liang Hong <lhong22@cse.cuhk.edu.hk>: inference-only direct relpos
    # gather-add into z (fused Triton or weight-table path) plus in-place
    # token-bonds projection.
    def _add_relpos_from_weights_(self, z: torch.Tensor, batch: dict) -> None:
        """Add ``linear_relpos(relpos_complex(batch))`` directly into ``z``."""
        res_idx = batch["residue_index"]
        asym_id = batch["asym_id"]
        entity_id = batch["entity_id"]

        same_chain = asym_id[..., None] == asym_id[..., None, :]
        same_res = res_idx[..., None] == res_idx[..., None, :]
        same_entity = entity_id[..., None] == entity_id[..., None, :]

        def relpos_idx(pos: torch.Tensor, condition: torch.Tensor, clip: int):
            offset = pos[..., None] - pos[..., None, :]
            clipped = torch.clamp(offset + clip, min=0, max=2 * clip)
            return torch.where(
                condition,
                clipped,
                (2 * clip + 1) * torch.ones_like(clipped),
            ).long()

        rel_pos_idx = relpos_idx(res_idx, same_chain, self.max_relative_idx)
        rel_token_idx = relpos_idx(
            batch["token_index"],
            same_chain & same_res,
            self.max_relative_idx,
        )
        rel_chain_idx = relpos_idx(
            batch["sym_id"], same_entity, self.max_relative_chain
        )

        w = self.linear_relpos.weight.to(dtype=z.dtype).t().contiguous()
        rel_pos_bins = 2 * self.max_relative_idx + 2
        rel_token_offset = rel_pos_bins
        same_entity_offset = rel_token_offset + rel_pos_bins
        rel_chain_offset = same_entity_offset + 1

        if z.is_cuda and z.is_contiguous():
            from openfold3.core.kernels.triton.fused_relpos_embed import (
                fused_relpos_embed_add_,
            )

            fused_relpos_embed_add_(
                z=z,
                w=w,
                rel_pos_idx=rel_pos_idx,
                rel_token_idx=(rel_token_idx + rel_token_offset),
                rel_chain_idx=(rel_chain_idx + rel_chain_offset),
                same_entity=same_entity,
                same_entity_offset=same_entity_offset,
            )
        else:
            z.add_(w[rel_pos_idx])
            z.add_(w[rel_token_idx + rel_token_offset])
            z.add_(w[rel_chain_idx + rel_chain_offset])
            z.add_(same_entity[..., None].to(dtype=z.dtype) * w[same_entity_offset])

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
        subsample_main_msa: bool,
        subsample_all_msa: bool,
        min_subsampled_all_msa: int,
        max_subsampled_all_msa: int,
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
            subsample_main_msa:
                Whether to subsample only the main MSA to a random depth, following
                AF3 SI Section 2.2.
            subsample_all_msa:
                Whether to subsample all MSA (paired + main) to a random depth.
            min_subsampled_all_msa:
                If subsample_all_msa, this specifies the minimum number of MSA
                sequences to retain after subsampling.
            max_subsampled_all_msa:
                If subsample_all_msa, this specifies the minimum number of MSA
                sequences to retain after subsampling.
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.subsample_main_msa = subsample_main_msa
        self.subsample_all_msa = subsample_all_msa
        self.min_subsampled_all_msa = min_subsampled_all_msa
        self.max_subsampled_all_msa = max_subsampled_all_msa
        self.linear_m = Linear(c_m_feats, c_m, **linear_init_params.linear_m)
        self.linear_s_input = Linear(
            c_s_input, c_m, **linear_init_params.linear_s_input
        )

    # TODO: Move this to the data pipeline
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
                The MSA is padded to the maximum MSA dimension across chains,
                so this mask will be all zeros for any chain whose actual MSA dimension
                is less than the maximum.
            num_paired_seqs:
                [] Number of paired MSA sequences
            asym_id:
                [N_token] Id of the chain each token belongs to
        Returns:
            sampled_msa:
                [N_seq, N_token, c_m_feats] Sampled MSA features
            msa_mask:
                [N_seq, N_token] Binary mask for sampled MSA entries.
                MSA per chain is independently subsampled and re-padded
                to a shared dimension.
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
            for feat, perm in zip(
                per_chain_msa_feat, chain_index_permutations, strict=False
            )
        ]
        sampled_chain_masks = [
            mask[..., perm, :]
            for mask, perm in zip(
                per_chain_msa_mask, chain_index_permutations, strict=False
            )
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
    def _concat_msa_feat(
        msa: torch.Tensor,
        has_deletion: torch.Tensor,
        deletion_value: torch.Tensor,
    ) -> torch.Tensor:
        """Build [*, N_msa, N_token, 34] MSA features from raw inputs."""
        return torch.cat(
            [
                msa,
                has_deletion.unsqueeze(-1),
                deletion_value.unsqueeze(-1),
            ],
            dim=-1,
        )

    @staticmethod
    def _pad_sequences_dim(
        m: torch.Tensor, max_seqs: int, seq_dim: int
    ) -> torch.Tensor:
        """Pad the msa to max_seqs along seq_dim to stack them in a batch."""
        # Add zero padding at start and end for all dimensions after seq_dim
        non_pad_dims = (0, 0) * (abs(seq_dim) - 1)

        # Pad the seq_dim to max_msa_seqs length
        pad = non_pad_dims + (0, max_seqs - m.shape[seq_dim])

        return torch.nn.functional.pad(m, pad)

    @staticmethod
    def _subsample_all_msa_indices(
        msa_mask: torch.Tensor, no_subsampled_all_msa: int
    ) -> torch.Tensor:
        """Select MSA rows for all-MSA subsampling from the mask only."""
        mask_seq_dim = -2

        if isinstance(no_subsampled_all_msa, torch.Tensor):
            no_subsampled_all_msa = no_subsampled_all_msa.item()

        # Valid msa
        valid_msa = (msa_mask.sum(dim=mask_seq_dim + 1) > 0).squeeze()  # [N_msa]

        if valid_msa.ndim == 0:
            valid_msa = valid_msa.unsqueeze(0)

        valid_idx = valid_msa.nonzero().squeeze()
        invalid_idx = (~valid_msa).nonzero().squeeze()

        if valid_idx.ndim == 0:
            valid_idx = valid_idx.unsqueeze(0)
        if invalid_idx.ndim == 0:
            invalid_idx = invalid_idx.unsqueeze(0)

        device = msa_mask.device
        # Pick msa from the valid ones at random
        if valid_idx.numel() >= no_subsampled_all_msa:
            permuted_idx = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
            selected = permuted_idx[:no_subsampled_all_msa]
        else:
            # Take all valid, then fill with random invalid
            take_invalid = no_subsampled_all_msa - valid_idx.numel()
            if invalid_idx.numel() > 0:
                permuted_idx = invalid_idx[
                    torch.randperm(invalid_idx.numel(), device=device)
                ]
                selected = torch.cat([valid_idx, permuted_idx[:take_invalid]], dim=0)
            else:
                selected = valid_idx

        return selected

    @staticmethod
    def _subsample_all_msa(
        msa_feat: torch.Tensor, msa_mask: torch.Tensor, no_subsampled_all_msa: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Subsample all MSA sequences (paired + main) to a fixed number of sequences,
        prioritizing those with at least one non-masked token.

        Args:
            msa_feat:
                [N_msa, N_token, c_m_feats] MSA features
            msa_mask:
                [N_msa, N_token] Binary mask indicating valid MSA entries.
                The MSA is padded to the maximum MSA dimension across chains,
                so this mask will be all zeros for any chain whose actual MSA dimension
                is less than the maximum.
            no_subsampled_all_msa:
                The number of MSA sequences to retain after subsampling.
        Returns:
            sampled_msa:
                [N_seq, N_token, c_m_feats] Sampled MSA features
            msa_mask:
                [N_seq, N_token] Binary mask for sampled MSA entries.
                MSA per chain is independently subsampled and re-padded
                to a shared dimension.
        """

        feat_seq_dim = -3
        mask_seq_dim = -2
        selected = MSAModuleEmbedder._subsample_all_msa_indices(
            msa_mask, no_subsampled_all_msa
        )

        feat_sub = msa_feat.index_select(feat_seq_dim, selected)
        mask_sub = msa_mask.index_select(mask_seq_dim, selected)
        return feat_sub, mask_sub

    @staticmethod
    def _subsample_all_msa_raw_features(
        msa: torch.Tensor,
        has_deletion: torch.Tensor,
        deletion_value: torch.Tensor,
        msa_mask: torch.Tensor,
        no_subsampled_all_msa: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample raw MSA tensors before feature concatenation."""
        selected = MSAModuleEmbedder._subsample_all_msa_indices(
            msa_mask, no_subsampled_all_msa
        )
        feat_seq_dim = -3
        mask_seq_dim = -2
        return (
            msa.index_select(feat_seq_dim, selected),
            has_deletion.index_select(mask_seq_dim, selected),
            deletion_value.index_select(mask_seq_dim, selected),
            msa_mask.index_select(mask_seq_dim, selected),
        )

    def _apply_subsample_fn_batch(
        self, fn: callable, **kwargs
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
               Keyword arguments to forward to `fn`. Assumes each input tensor
               is batched in dim=0.

        Returns:
            sampled_msa:
                [N_seq, N_token, c_m_feats] Sampled MSA features
            msa_mask:
                [N_seq, N_token] Binary mask for sampled MSA entries.
        """

        batch_size = next(iter(kwargs.values())).shape[0]
        per_sample_kwargs_list = [
            {k: v[i] for k, v in kwargs.items()} for i in range(batch_size)
        ]

        per_sample_subsampled_msa = []
        per_sample_subsampled_msa_mask = []

        for kwarg in per_sample_kwargs_list:
            subsampled_msa, subsampled_mask = fn(**kwarg)
            per_sample_subsampled_msa.append(subsampled_msa)
            per_sample_subsampled_msa_mask.append(subsampled_mask)

        # Number of sequences to pad to for all the batch
        max_msa_seqs_batch = max([m.shape[-3] for m in per_sample_subsampled_msa])

        # Pad the sequences to same seq length and stack them in a batch
        sampled_msa = torch.stack(
            [
                self._pad_sequences_dim(m, max_msa_seqs_batch, seq_dim=-3)
                for m in per_sample_subsampled_msa
            ],
            dim=0,
        )
        sampled_msa_mask = torch.stack(
            [
                self._pad_sequences_dim(m, max_msa_seqs_batch, seq_dim=-2)
                for m in per_sample_subsampled_msa_mask
            ],
            dim=0,
        )

        return sampled_msa, sampled_msa_mask

    def _apply_subsample_all_msa_raw_batch(
        self,
        msa: torch.Tensor,
        has_deletion: torch.Tensor,
        deletion_value: torch.Tensor,
        msa_mask: torch.Tensor,
        no_subsampled_all_msa: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample raw MSA tensors per batch element, then pad and stack."""
        batch_size = msa.shape[0]
        sampled = [
            self._subsample_all_msa_raw_features(
                msa=msa[i],
                has_deletion=has_deletion[i],
                deletion_value=deletion_value[i],
                msa_mask=msa_mask[i],
                no_subsampled_all_msa=no_subsampled_all_msa[i],
            )
            for i in range(batch_size)
        ]
        max_msa_seqs_batch = max(item[0].shape[-3] for item in sampled)
        seq_dims = (-3, -2, -2, -2)

        return tuple(
            torch.stack(
                [
                    self._pad_sequences_dim(
                        item[out_idx],
                        max_msa_seqs_batch,
                        seq_dim=seq_dims[out_idx],
                    )
                    for item in sampled
                ],
                dim=0,
            )
            for out_idx in range(4)
        )

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
                    - "asym_id": [*, N_token]
            s_input:
                [*, N_token, C_s_input] single embedding

        Returns:
            m:
                [*, N_seq, N_token, C_m] MSA embedding
            msa_mask:
                [*, N_seq, N_token] MSA mask
        """
        batch_dims = batch["msa"].shape[:-3]

        msa = batch["msa"]
        has_deletion = batch["has_deletion"]
        deletion_value = batch["deletion_value"]
        msa_mask = batch["msa_mask"]

        if self.subsample_main_msa:
            msa_feat = self._concat_msa_feat(msa, has_deletion, deletion_value)
            if math.prod(batch_dims) > 1:
                msa_feat, msa_mask = self._apply_subsample_fn_batch(
                    fn=self._subsample_main_msa,
                    msa_feat=msa_feat,
                    msa_mask=msa_mask,
                    num_paired_seqs=batch["num_paired_seqs"],
                    asym_id=batch["asym_id"],
                )
            else:
                msa_feat, msa_mask = self._subsample_main_msa(
                    msa_feat=msa_feat,
                    msa_mask=msa_mask,
                    num_paired_seqs=batch["num_paired_seqs"],
                    asym_id=batch["asym_id"],
                )
        else:
            if self.subsample_all_msa:
                no_subsampled_all_msa = torch.randint(
                    low=self.min_subsampled_all_msa,
                    high=int(self.max_subsampled_all_msa + 1),
                    size=(1,),
                    device=msa.device,
                ).item()

                if math.prod(batch_dims) > 1:
                    msa, has_deletion, deletion_value, msa_mask = (
                        self._apply_subsample_all_msa_raw_batch(
                            msa=msa,
                            has_deletion=has_deletion,
                            deletion_value=deletion_value,
                            msa_mask=msa_mask,
                            no_subsampled_all_msa=torch.full(
                                (msa.shape[0],),
                                no_subsampled_all_msa,
                                device=msa.device,
                            ),
                        )
                    )
                else:
                    msa, has_deletion, deletion_value, msa_mask = (
                        self._subsample_all_msa_raw_features(
                            msa=msa,
                            has_deletion=has_deletion,
                            deletion_value=deletion_value,
                            msa_mask=msa_mask,
                            no_subsampled_all_msa=no_subsampled_all_msa,
                        )
                    )

            msa_feat = self._concat_msa_feat(msa, has_deletion, deletion_value)

        # [*, N_seq, N_token, C_m]
        m = self.linear_m(msa_feat)
        s_input_emb = self.linear_s_input(s_input).unsqueeze(-3)
        if not torch.is_grad_enabled():
            m.add_(s_input_emb)
        else:
            m = m + s_input_emb
        del s_input_emb

        return m, msa_mask


class FourierEmbedding(nn.Module):
    """
    Implements AF3 Algorithm 22.
    """

    def __init__(self, c: int, seed: int = 42):
        """
        Args:
            c:
                Embedding dimension
            seed:
                Random seed for initialization
        """
        super().__init__()
        generator = torch.Generator()
        generator.manual_seed(seed)

        w = torch.empty(c)
        b = torch.empty(c)

        torch.nn.init.normal_(w, generator=generator)
        torch.nn.init.uniform_(b, generator=generator)

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
        x = x * self.w + self.b
        return torch.cos(2 * torch.pi * x)
