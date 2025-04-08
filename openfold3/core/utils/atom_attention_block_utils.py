import math
from typing import Optional

import torch


def convert_to_blocks_1d(
    x: torch.Tensor,
    num_blocks: int,
    block_len: int,
    shift_interval: int,
    dim: int,
) -> torch.Tensor:
    """
    Convert an input tensor to be of shape [num_blocks, block_len] at dimension dim.
    The blocks are created by sliding a window of size block_len by shift_interval.

    Args:
        x:
            Input tensor
        num_blocks:
            Number of blocks to create
        block_len:
            Length of each block
        shift_interval:
            Sliding window size for each block
        dim:
            Dimensions to create blocks along

    Returns:
        Tensor with blocks of shape [num_blocks, block_len] at dimension dim
    """

    # Chunk if there are no overlapping blocks
    if shift_interval == block_len:
        blocks = torch.chunk(x, num_blocks, dim=dim)
    else:
        blocks = [
            x.narrow(dim, shift_interval * i, block_len) for i in range(num_blocks)
        ]
    return torch.stack(blocks, dim=dim - 1)


def convert_to_blocks_2d(
    x: torch.Tensor,
    num_blocks: int,
    block_lens: list,
    shift_interval: int,
    dims: list,
) -> torch.Tensor:
    """
    Convert an input tensor to be of shape [num_blocks, block_lens[0], block_lens[1]]
    at the corresponding dims. The specified block lens are shifted by shift_interval.

    Args:
        x:
            Input tensor
        num_blocks:
            Number of blocks to create
        block_lens:
            List of block lengths at each dimension
        shift_interval:
            Sliding window size for each block
        dims:
            List of dimensions to create blocks along

    Returns:
        Tensor with 2D blocks of shape [num_blocks, block_lens[0], block_lens[1]]
        at the specified dims.
    """
    blocks = [
        x.narrow(dims[0], shift_interval * i, block_lens[0]).narrow(
            dims[1], shift_interval * i, block_lens[1]
        )
        for i in range(num_blocks)
    ]
    return torch.stack(blocks, dim=min(dims) - 1)


def get_subset_center_padding(
    n_atom: int, n_query: int, n_key: int
) -> tuple[int, int, int]:
    """
    Calculate padding for a structure with n_atoms such that the block centers
    match the subset centers in Alg. 7 and the q/k dimensions are divisible by
    n_query and n_key respectively.

    Args:
        n_atom:
            Number of atoms
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)

    Returns:
        pad_len_right_q:
            Padding for the query seqlen dim so that it is divisible by n_query.
            No left padding is needed since the first block center is at
            n_query // 2.
        pad_len_left_k:
            Left padding for the key seqlen dim. Because the subset centers start
            at n_query // 2, padding is needed for even block sizes of length n_key.
            This is an issue for the first two blocks, which would have lengths 80
            and 112 if the default block sizes are used.
        pad_len_right_k:
            Right padding for the key seqlen dim. Because the subset centers are
            shifted by n_query, padding is needed for even block sizes of
            length n_key. This addresses uneven block sizes in the ending blocks.
    """
    offset = n_query // 2
    num_blocks = math.ceil(n_atom / n_query)

    subset_centers = offset + torch.arange(num_blocks) * n_query

    # Calculate padding for rows of plm to be divisible by n_query
    pad_len_right_q = (n_query - n_atom % n_query) % n_query

    # Calculate padding for columns of plm to be divisible by n_key
    # Pad left and right to ensure that the block centers match the
    # subset_centers in Alg. 7
    pad_len_right_k = subset_centers[-1] + n_key // 2 - n_atom
    pad_len_left_k = n_key // 2 - subset_centers[0]

    return pad_len_right_q, pad_len_left_k, pad_len_right_k


def convert_pair_rep_to_blocks(
    plm: torch.Tensor,
    n_query: int,
    n_key: int,
) -> torch.Tensor:
    """Convert pair atom representation to blocks for attention.

    Args:
        plm:
            [*, N_atom, N_atom, c_atom_pair] Atom pair conditioning
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)

    Returns:
        plm:
            [*, N_blocks, N_query, N_key, c_atom_pair] Atom pair conditioning
    """
    n_atom = plm.shape[-2]
    num_blocks = math.ceil(n_atom / n_query)
    pad_len_right_q, pad_len_left_k, pad_len_right_k = get_subset_center_padding(
        n_atom=n_atom, n_query=n_query, n_key=n_key
    )

    # Pad and convert plm to blocks of width n_query and length n_key
    # [*, N_atom, N_atom, c_atom_pair] -> [*, N_blocks, N_query, N_key, c_atom_pair]
    plm = torch.nn.functional.pad(
        plm, (0, 0, pad_len_left_k, pad_len_right_k, 0, pad_len_right_q), value=0.0
    )
    plm = convert_to_blocks_2d(
        x=plm,
        num_blocks=num_blocks,
        block_lens=[n_query, n_key],
        shift_interval=n_query,
        dims=[-3, -2],
    )

    return plm


def convert_single_rep_to_blocks(
    ql: torch.Tensor,
    n_query: int,
    n_key: int,
    atom_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert single atom representation to q/k blocks for attention.
    Optionally convert the atom mask to a 2D mask to account for the padding on the
    first and last blocks.

    Args:
        ql:
            [*, N_atom, c_atom] Atom single representation
        n_query:
            Number of queries (block height)
        n_key:
            Number of keys (block width)
        atom_mask:
            [*, N_atom] Mask for token or atom-level embedding (Optional)

    Returns:
        ql_query:
            [*, N_blocks, N_query, c_atom] Atom single representation
        ql_key:
            [*, N_blocks, N_key, c_atom] Atom single representation
        mask:
            [*, N_blocks, N_query, N_key] 2D mask for atom-level embedding
    """
    batch_dims = ql.shape[:-2]
    n_atom, n_dim = ql.shape[-2:]

    num_blocks = math.ceil(n_atom / n_query)
    pad_len_right_q, pad_len_left_k, pad_len_right_k = get_subset_center_padding(
        n_atom=n_atom, n_query=n_query, n_key=n_key
    )

    # Pad and convert ql to blocks of width n_query
    # [*, N_atom, c_atom] -> [*, N_blocks, N_query, c_atom]
    ql_query = torch.nn.functional.pad(ql, (0, 0, 0, pad_len_right_q), value=0.0)
    ql_query = ql_query.reshape((*batch_dims, num_blocks, n_query, n_dim))

    # Pad and convert ql to blocks of length n_key
    # [*, N_atom, c_atom] -> [*, N_blocks, N_key, c_atom]
    ql_key = torch.nn.functional.pad(
        ql, (0, 0, pad_len_left_k, pad_len_right_k), value=0.0
    )
    ql_key = convert_to_blocks_1d(
        x=ql_key,
        num_blocks=num_blocks,
        block_len=n_key,
        shift_interval=n_query,
        dim=-2,
    )

    atom_pair_mask = None
    if atom_mask is not None:
        # Pad and convert atom mask to blocks of width n_query
        # [*, N_atom] -> [*, N_blocks, N_query]
        atom_mask_q = torch.nn.functional.pad(
            atom_mask, (0, pad_len_right_q), value=0.0
        )
        atom_mask_q = atom_mask_q.reshape((*atom_mask.shape[:-1], num_blocks, n_query))

        # Pad and convert atom mask to blocks of length n_key
        # [*, N_atom] -> [*, N_blocks, N_key]
        atom_mask_k = torch.nn.functional.pad(
            atom_mask, (pad_len_left_k, pad_len_right_k), value=0.0
        )
        atom_mask_k = convert_to_blocks_1d(
            x=atom_mask_k,
            num_blocks=num_blocks,
            block_len=n_key,
            shift_interval=n_query,
            dim=-1,
        )

        # Create pair mask
        # [*, N_blocks, N_query, N_key]
        atom_pair_mask = atom_mask_q[..., None] * atom_mask_k[..., None, :]

    return ql_query, ql_key, atom_pair_mask
