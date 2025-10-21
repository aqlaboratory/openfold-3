# Copyright 2025 AlQuraishi Laboratory
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
    entity_id = batch["entity_id"]
    same_chain = asym_id[..., None] == asym_id[..., None, :]
    same_res = res_idx[..., None] == res_idx[..., None, :]
    same_entity = entity_id[..., None] == entity_id[..., None, :]

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

    rel_pos = relpos(pos=res_idx, condition=same_chain, rel_clip_idx=max_relative_idx)
    rel_token = relpos(
        pos=batch["token_index"],
        condition=same_chain & same_res,
        rel_clip_idx=max_relative_idx,
    )
    rel_chain = relpos(
        pos=batch["sym_id"],
        condition=same_entity,
        rel_clip_idx=max_relative_chain,
    )

    same_entity = same_entity[..., None].to(dtype=rel_pos.dtype)

    rel_feat = torch.cat([rel_pos, rel_token, same_entity, rel_chain], dim=-1)

    return rel_feat
