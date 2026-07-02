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
    """Calculate the cyclic offset for the given residue index.
    Args:
        residue_index:
            [*, N_token] Token index

    Returns:
        cyclic_offset_array:
            [N_token, N_token] token by token index distances

    Example:
        >>> import torch
        >>> residue_index = torch.tensor([0,1,2,3,4,5,6])
        >>> cyclic_offset_array = cyclic_offset(residue_index)
        >>> cyclic_offset_array:
            tensor([[ 0, -1, -2, -3,  2,  1],
                    [ 1,  0, -1, -2, -3,  2],
                    [ 2,  1,  0, -1, -2, -3],
                    [-3,  2,  1,  0, -1, -2],
                    [-2, -3,  2,  1,  0, -1],
                    [-1, -2, -3,  2,  1,  0]], device='cuda:0', dtype=torch.int32)

    """
    peptide_length = residue_index.shape[0]
    cyclic_offset_array = torch.zeros((peptide_length, peptide_length))
    cyc_row = torch.arange(0, -peptide_length, -1)
    pc = int(torch.round(torch.tensor(peptide_length / 2)))  # Get centre
    cyc_row[pc + 1 :] = torch.arange(len(cyc_row[pc + 1 :]), 0, -1)
    for i in range(len(cyclic_offset_array)):
        cyclic_offset_array[i] = torch.roll(cyc_row, i)
    return cyclic_offset_array.type(torch.int).to(residue_index.device)


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
    cyclic_mask = batch["cyclic_mask"]
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]

    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

    def relpos(
        pos: torch.Tensor,
        condition: torch.BoolTensor,
        rel_clip_idx: int,
        cyclic_mask: torch.Tensor,
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
                [*, N_token] Boolean tensor for cyclic residues
            asym_id:
                [*, N_token] Used by cyclic mask for multi-chain cyclic
        Returns:
            rel_pos:
                [*, N_token, N_token, 2 * rel_clip_idx + 2] Relative position embedding
        """
        offset = pos[..., None] - pos[..., None, :]
        if cyclic_mask is not None and cyclic_mask.any():
            for chain_id in torch.unique(asym_id):
                chain_cyclic_mask = cyclic_mask & (asym_id == chain_id)
                pair_cyclic = (
                    chain_cyclic_mask[..., None] & chain_cyclic_mask[..., None, :]
                )

                if not pair_cyclic.any():
                    continue
                cyc_mask_1d = cyclic_mask.view(-1, cyclic_mask.shape[-1])[0]
                cyc_indices = torch.where(
                    cyc_mask_1d & (asym_id.squeeze(0) == chain_id)
                )[0]
                cyc_pos = pos.view(-1)[cyc_indices]
                cyc_off = cyclic_offset(cyc_pos).to(dtype=offset.dtype)
                full_cyc_off = offset.new_zeros(offset.shape)
                full_cyc_off[..., cyc_indices[:, None], cyc_indices[None, :]] = cyc_off
                offset = torch.where(pair_cyclic, full_cyc_off, offset)

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

    rel_token = relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
        cyclic_mask=cyclic_mask,
        asym_id=asym_id,
    )
    rel_chain = relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
        cyclic_mask=cyclic_mask,
        asym_id=asym_id,
    )

    same_entity = same_entity[..., None].to(dtype=rel_pos.dtype)

    rel_feat = torch.cat([rel_pos, rel_token, same_entity, rel_chain], dim=-1)

    return rel_feat
