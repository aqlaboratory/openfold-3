import torch
import torch.nn as nn

from openfold3.core.utils.tensor_utils import binned_one_hot


def relpos_monomer(ri: torch.Tensor, relpos_k: int) -> torch.Tensor:
    """
    Computes relative positional encodings

    Implements AF2 Algorithm 4.

    Args:
        ri:
            "residue_index" features of shape [*, N]
        relpos_k:
            Window size used in relative positional encoding
    """
    d = ri[..., None] - ri[..., None, :]
    boundaries = torch.arange(start=-relpos_k, end=relpos_k + 1, device=d.device)
    reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
    d = d[..., None] - reshaped_bins
    d = torch.abs(d)
    d = torch.argmin(d, dim=-1)
    d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()

    return d


def relpos_multimer(
    batch: dict,
    max_relative_idx: int,
    use_chain_relative: bool,
    max_relative_chain: int,
) -> torch.Tensor:
    """
    Args:
        batch:
            Input feature dictionary
        max_relative_idx:
            Maximum relative position and token indices clipped
        use_chain_relative:
            Whether to add relative chain encoding
        max_relative_chain:
            Maximum relative chain indices clipped
    """
    pos = batch["residue_index"]
    asym_id = batch["asym_id"]
    asym_id_same = asym_id[..., None] == asym_id[..., None, :]
    offset = pos[..., None] - pos[..., None, :]

    clipped_offset = torch.clamp(offset + max_relative_idx, 0, 2 * max_relative_idx)

    rel_feats = []
    if use_chain_relative:
        final_offset = torch.where(
            asym_id_same,
            clipped_offset,
            (2 * max_relative_idx + 1) * torch.ones_like(clipped_offset),
        )
        boundaries = torch.arange(
            start=0, end=2 * max_relative_idx + 2, device=final_offset.device
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

        max_rel_chain = max_relative_chain
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
            start=0, end=2 * max_relative_idx + 1, device=clipped_offset.device
        )
        rel_pos = binned_one_hot(
            clipped_offset,
            boundaries,
        )
        rel_feats.append(rel_pos)

    rel_feat = torch.cat(rel_feats, dim=-1)

    return rel_feat


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
