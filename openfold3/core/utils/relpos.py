# Copyright 2026 AlQuraishi Laboratory
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

from openfold3.core.utils.tensor_utils import binned_one_hot


def cyclic_offset(residue_index: torch.Tensor) -> torch.Tensor:
    """Calculate the offsets between tokens of a cyclic chain of residues.

    Each offset is the linear offset wrapped to the shorter way around the ring
    of residues where the ring length is determined based on the difference
    between the maximum and minimum residue indices. The antipode residue
    directly opposite in an even ring keeps the sign of the linear offset, so
    the offsets are antisymmetric.

    Args:
        residue_index:
            [N_token] Residue index of each token of one cyclic chain

    Returns:
        cyclic_offset_array:
            [N_token, N_token] int32 offsets in [-(L // 2), L // 2], congruent
            to residue_index[i] - residue_index[j] modulo the ring length L

    Example:
        >>> cyclic_offset(torch.arange(6))
        tensor([[ 0, -1, -2, -3,  2,  1],
                [ 1,  0, -1, -2, -3,  2],
                [ 2,  1,  0, -1, -2, -3],
                [ 3,  2,  1,  0, -1, -2],
                [-2,  3,  2,  1,  0, -1],
                [-1, -2,  3,  2,  1,  0]], dtype=torch.int32)
    """
    ring_length = int(residue_index.max() - residue_index.min()) + 1
    offset = residue_index[:, None] - residue_index[None, :]
    # At antipodes offset / ring_length is exactly +-1/2, so torch.round RNE rounds to 0
    return (offset - ring_length * torch.round(offset / ring_length)).type(torch.int)


def apply_cyclic_offsets(
    offset: torch.Tensor,
    pos: torch.Tensor,
    cyclic_mask: torch.Tensor | None,
    asym_id: torch.Tensor,
) -> torch.Tensor:
    """Replaces the linear offsets of cyclic chains with wrapped cyclic offsets.

    Wrapping is applied per chain, so that a complex may mix cyclic and linear
    chains. Token pairs that do not lie within the same cyclic chain -- including
    cross-chain pairs -- keep their linear offset.

    Args:
        offset:
            [*, N_token, N_token] Linear token offsets, i.e. pos[i] - pos[j]
        pos:
            [*, N_token] Token index the offsets were computed from
        cyclic_mask:
            [*, N_token] Boolean tensor for cyclic residues, or None if the feature
            is absent, in which case every chain is treated as linear
        asym_id:
            [*, N_token] Chain index per token

    Returns:
        [*, N_token, N_token] Offsets, cyclic within each cyclic chain
    """
    if cyclic_mask is None or not cyclic_mask.any():
        return offset

    n_token = cyclic_mask.shape[-1]

    # Chain membership is per sample, so flatten the leading dims and index each
    # sample separately rather than deriving one set of token indices for the batch.
    flat_mask = cyclic_mask.reshape(-1, n_token)
    flat_asym = asym_id.expand_as(cyclic_mask).reshape(-1, n_token)
    flat_pos = pos.expand_as(cyclic_mask).reshape(-1, n_token)

    cyclic_offsets = offset.clone()
    flat_offsets = cyclic_offsets.reshape(-1, n_token, n_token)

    for sample_idx in range(flat_mask.shape[0]):
        sample_mask = flat_mask[sample_idx]
        sample_asym = flat_asym[sample_idx]

        for chain_id in torch.unique(sample_asym[sample_mask]):
            cyc_indices = torch.where(sample_mask & (sample_asym == chain_id))[0]
            cyc_off = cyclic_offset(flat_pos[sample_idx][cyc_indices])

            flat_offsets[sample_idx][cyc_indices[:, None], cyc_indices[None, :]] = (
                cyc_off.to(dtype=offset.dtype)
            )

    return flat_offsets.reshape(offset.shape)


def relpos_complex(
    batch: dict, max_relative_idx: int, max_relative_chain: int
) -> torch.Tensor:
    """
    Args:
        batch:
            Input feature dictionary
        max_relative_idx:
            Maximum relative position and token indices clipped
        max_relative_chain:
            Maximum relative chain indices clipped

    Returns:
        [*, N_token, N_token, C_z] Relative position embedding
    """
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    cyclic_mask = batch.get("cyclic_mask")
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]

    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def relpos(
        pos: torch.Tensor,
        condition: torch.BoolTensor,
        rel_clip_idx: int,
        cyclic_mask: torch.Tensor | None,
        asym_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pos:
                [*, N_token] Token index
            condition:
                [*, N_token, N_token] Condition for clipping
            rel_clip_idx:
                Max idx for clipping (max_relative_idx or max_relative_chain)
            cyclic_mask:
                [*, N_token] Boolean tensor for cyclic residues, or None if the
                feature is absent from the batch
            asym_id:
                [*, N_token] Used by cyclic mask for multi-chain cyclic
        Returns:
            rel_pos:
                [*, N_token, N_token, 2 * rel_clip_idx + 2] Relative position embedding
        """
        offset = pos[..., None] - pos[..., None, :]
        offset = apply_cyclic_offsets(
            offset=offset, pos=pos, cyclic_mask=cyclic_mask, asym_id=asym_id
        )

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

    rel_pos = relpos(
        pos=res_idx,
        condition=same_chain,
        rel_clip_idx=max_relative_idx,
        cyclic_mask=cyclic_mask,
        asym_id=asym_id,
    )

    # No cyclic_mask. Only the residue offset goes around the ring. The token
    # offset is only used within a residue, and all tokens of a chain share one
    # sym_id.
    rel_token = relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
        cyclic_mask=None,
        asym_id=asym_id,
    )
    rel_chain = relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
        cyclic_mask=None,
        asym_id=asym_id,
    )

    same_entity = same_entity[..., None].to(dtype=rel_pos.dtype)

    rel_feat = torch.cat([rel_pos, rel_token, same_entity, rel_chain], dim=-1)

    return rel_feat
