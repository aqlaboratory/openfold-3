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

import math
from typing import Literal

import torch

from openfold3.core.data.resources.residues import (
    STANDARD_PROTEIN_RESIDUES_ORDER,
    STANDARD_RESIDUES_WITH_GAP_3,
)
from openfold3.core.data.resources.token_atom_constants import (
    atom_name_to_index_by_restype,
)


def broadcast_token_feat_to_atoms(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor,
    token_feat: torch.Tensor,
    token_dim: int | None = -1,
    max_num_atoms_per_token: int | None = None,
):
    """
    Broadcast token-level features to atom-level features.

    Args:
        token_mask:
            [*, N_token] Token mask
        num_atoms_per_token:
            [*, N_token] Number of atoms per token
        token_feat:
            [*, N_token] Token-level feature
        token_dim:
            Token dimension
        max_num_atoms_per_token:
            Maximum number of atoms per token
    Returns:
        atom_feat:
            [*, N_atom] Broadcasted atom-level feature (if max_num_atoms_per_token
            is provided, the output would be [*, N_token * max_num_atoms_per_token])
    """
    n_token = token_mask.shape[-1]
    batch_dims = token_mask.shape[:-1]
    feat_batch_dims = token_feat.shape[:token_dim]
    feat_dims = token_feat.shape[token_dim:][1:]

    # Apply token mask
    num_atoms_per_token = num_atoms_per_token * token_mask.int()
    token_feat = token_feat * token_mask.reshape(
        (*batch_dims, n_token, *((1,) * len(feat_dims)))
    )

    # Pad atoms at token level
    if max_num_atoms_per_token is not None:
        num_atoms_per_token = torch.stack(
            [num_atoms_per_token, max_num_atoms_per_token - num_atoms_per_token], dim=-1
        ).reshape((*batch_dims, 2 * n_token))
        token_feat = torch.stack(
            [token_feat, torch.zeros_like(token_feat)], dim=token_dim
        ).reshape((*batch_dims, 2 * n_token, *feat_dims))

    # Pad token features
    # Flatten batch and token dimensions
    max_num_atoms = torch.max(torch.sum(num_atoms_per_token, dim=-1)).int()
    padded_token_feat = torch.concat(
        [
            token_feat,
            torch.zeros(
                (*feat_batch_dims, 1, *feat_dims),
                dtype=token_feat.dtype,
                device=token_feat.device,
            ),
        ],
        dim=token_dim,
    ).reshape(-1, *feat_dims)

    # Pad number of atoms per token
    # Flatten batch and token dimensions
    padded_num_atoms_per_token = torch.concat(
        [
            num_atoms_per_token,
            max_num_atoms - torch.sum(num_atoms_per_token, dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    if batch_dims != feat_batch_dims:
        batch_n_repeat = feat_batch_dims[-1]
        padded_num_atoms_per_token = padded_num_atoms_per_token.repeat(
            *((1,) * len(batch_dims[:-1]) + (batch_n_repeat,) + (1,))
        )
    padded_num_atoms_per_token = padded_num_atoms_per_token.reshape(-1).int()

    # Create atom-level features
    atom_feat = torch.repeat_interleave(
        input=padded_token_feat, repeats=padded_num_atoms_per_token, dim=0
    )

    # Unflatten batch and token dimensions
    atom_feat = atom_feat.reshape((*feat_batch_dims, max_num_atoms, *feat_dims))

    return atom_feat


def aggregate_atom_feat_to_tokens(
    token_mask: torch.Tensor,
    atom_to_token_index: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_feat: torch.Tensor,
    atom_dim: int | None = -1,
    aggregate_fn: Literal["mean", "sum"] = "mean",
    eps: float = 1e-9,
):
    """
    Aggregates atom-level features to token-level features with mean or sum aggregation.

    Args:
        token_mask:
            [*, N_token] Token mask
        atom_to_token_index:
            [*, N_atom] Mapping from atom to its token index
        atom_mask:
            [*, N_atom] Atom mask
        atom_feat:
            [*, N_atom, *feat_dims] Atom-level features
        atom_dim:
            Atom dimension
        aggregate_fn:
            Function to aggregate atom features into tokens. Possible values are
            "mean" and "sum", where mean is the default.
        eps:
            Small float for numerical stability
    Returns:
        token_feat:
            [*, N_token, *feat_dims] Token-level features
    """
    n_token = token_mask.shape[-1]
    batch_dims = token_mask.shape[:-1]
    feat_batch_dims = atom_feat.shape[:atom_dim]
    feat_dims = atom_feat.shape[atom_dim:][1:]
    atom_feat = atom_feat * atom_mask.reshape(atom_mask.shape + (1,) * len(feat_dims))

    # Mask out atoms that are not part of the structure
    # Padding value must be greater than the largest index so that it
    # is properly excluded from the aggregation
    atom_to_token_index = (
        atom_to_token_index * atom_mask.int()
        + n_token * torch.ones_like(atom_to_token_index) * (1 - atom_mask.int())
    )

    # Prepare atom to token index for aggregation
    # Check for broadcasting and repeat accordingly
    if batch_dims == feat_batch_dims:
        repeated_atom_to_token_index = atom_to_token_index.reshape(
            *atom_to_token_index.shape + (1,) * len(feat_dims)
        ).repeat(*((1,) * (len(batch_dims) + 1) + feat_dims))
    else:
        batch_n_repeat = feat_batch_dims[-1]
        repeated_atom_to_token_index = atom_to_token_index.reshape(
            *atom_to_token_index.shape + (1,) * len(feat_dims)
        ).repeat(*((1,) * (len(batch_dims) - 1) + (batch_n_repeat,) + (1,) + feat_dims))

    if aggregate_fn not in ["mean", "sum"]:
        raise ValueError(f"Invalid aggregation function: {aggregate_fn}")

    # Compute summed token-level feature
    token_feat = torch.zeros(
        (*feat_batch_dims, n_token + 1, *feat_dims),
        device=atom_feat.device,
        dtype=atom_feat.dtype,
    ).scatter_add_(
        index=repeated_atom_to_token_index.long(), src=atom_feat, dim=atom_dim
    )
    token_feat = token_feat.reshape((*feat_batch_dims, n_token + 1, -1))[
        ..., :n_token, :
    ].reshape((*feat_batch_dims, n_token, *feat_dims))

    # Compute mean token-level feature
    if aggregate_fn == "mean":
        # Compute number of atoms (non-masked) per token
        token_num_atoms = torch.zeros(
            (*batch_dims, n_token + 1), device=atom_feat.device, dtype=atom_feat.dtype
        ).scatter_add_(
            index=atom_to_token_index.long(),
            src=atom_mask.to(dtype=atom_feat.dtype),
            dim=-1,
        )[..., :n_token]

        token_feat = token_feat / (
            token_num_atoms.reshape(token_num_atoms.shape + (1,) * len(feat_dims)) + eps
        )

    return token_feat


def aggregate_atom_feat_to_tokens_nd(
    atom_feat: torch.Tensor,
    atom_dims: list[int],
    atom_to_token_index_list: list[torch.Tensor],
    *,
    token_sizes: list[int] | None = None,
    atom_masks: list[torch.Tensor | None] | None = None,
    token_masks: list[torch.Tensor | None] | None = None,
    aggregate_fn: Literal["sum", "any", "mean"] = "sum",
    eps: float = 1e-9,
) -> torch.Tensor:
    """Aggregates atom-level to token-level features along an arbitrary number of axes.

    Reduces one axis at a time from largest to smallest using
    aggregate_atom_feat_to_tokens with 'sum' internally.

    Modes:
        - 'sum' uses 'sum'
        - 'any' uses (sum > 0)
        - 'mean' divides the final sum by the product of per-axis atom counts per token

    Shapes:
        - atom_feat: [*, N_atom_0, *, N_atom_i, *, N_atom_k, *]
        - atom_to_token_index_list[i]: [N_atom_0, ..., N_atom_i, ...]
        - atom_masks[i]: [N_atom_0, ..., N_atom_i, ...]
        - token_masks[i]: [N_token_0, ..., N_token_i, ...]
        - returns:  [*, N_token_0, *, N_token_i, *, N_atom_k, *]

    Args:
        atom_feat (torch.Tensor):
            Atom-level features, with k+1 atom-wise dimensions [*, N_atom_0, *,
            N_atom_i, *, N_atom_k, *].
        atom_dims (list[int]):
            The atom-wise dimensions along which to do the atom -> token pooling.
        atom_to_token_index_list (list[torch.Tensor]):
            Atom-wise tensors of token indices, one for each dimension in atom_dims
            indicating which token each atom belongs to along a specific dimension
            [N_atom_0, ..., N_atom_i, ...].
        token_sizes (list[int] | None, optional):
            Number of tokens alogn each dimension to be reduced. If None, inferred as
            the max + 1 of the tensor from atom_to_token_index_list for the
            corresponding dimension.
        atom_masks (list[torch.Tensor  |  None] | None, optional):
            Atom-wise masks, one per dimension, for selecting atoms that contribute to
            the aggregation [N_atom_0, ..., N_atom_i, ...].
        token_masks (list[torch.Tensor  |  None] | None, optional):
            Token padding masks, on per dimension [N_token_0, ..., N_token_i, ...].
        aggregate_fn (Literal['sum', 'any', 'mean'], optional):
            Mode of the aggregation. Defaults to "sum".
        eps (float, optional):
            Constant for numerical stability. Defaults to 1e-9.

    Raises:
        ValueError:
            If there is atom_dims <> atom_to_token_index_list mismatch.
        ValueError:
            If a mapping (atom_to_token_index_list[i]) or an atom mask (atom_masks[i])
            cannot be broadcast to the reduction step shape (*Bshape, A).
        ValueError:
            If a token mask (token_masks[i]) is provided and its last dimension does not
            equal the token size R for that axis.
        ValueError:
            If the aggregation function is not supported.

    Returns:
        torch.Tensor:
            The reduced tensor [*, N_token_0, *, N_token_i, *, N_atom_k, *]. Note that
            atom-wise dimensions not specified in atom_dims are NOT reduced.
    """
    if len(atom_dims) != len(atom_to_token_index_list):
        raise ValueError(
            "`atom_dims` and `atom_to_token_index_list` must have equal length."
        )
    K = len(atom_dims)
    if token_sizes is None:
        token_sizes = [None] * K
    if atom_masks is None:
        atom_masks = [None] * K
    if token_masks is None:
        token_masks = [None] * K

    # Normalize dims to positive and sort from largest to smallest
    D = atom_feat.dim()
    atom_dims_pos = [(d if d >= 0 else D + d) for d in atom_dims]
    order = sorted(range(K), key=lambda i: atom_dims_pos[i], reverse=True)

    y = atom_feat
    per_axis_counts = [None] * K

    def _broadcast_to_other_dim_atom_dim(
        t: torch.Tensor, Bshape: torch.Size, A: int
    ) -> torch.Tensor:
        """
        Broadcasts t to shape (*Bshape, A) by inserting singleton dims before last axis.
        """
        # Ensure last is A
        if t.shape[-1] != A:
            raise ValueError(f"Last dim mismatch: expected {A}, got {t.shape[-1]}")
        # Add singleton dims before the last axis until ranks match
        while t.dim() < (len(Bshape) + 1):
            t = t.unsqueeze(-2)
        # Expand across Bshape
        return t.expand(*Bshape, A)

    for idx in order:
        # Get the original position of this axis
        d = atom_dims_pos[idx]
        # Move to the end so we can reduce along -1
        perm = [i for i in range(y.dim()) if i != d] + [d]
        y_perm = y.permute(*perm)
        bs = y_perm.shape[:-1]
        n_atom = y_perm.shape[-1]

        # Gather per-axis inputs
        idx_map = atom_to_token_index_list[idx].to(y.device).long()
        atom_mask_i = atom_masks[idx]
        token_mask_i = token_masks[idx]
        n_token = token_sizes[idx]
        # Infer from mapping
        if n_token is None:
            n_token = int(idx_map.max().item()) + 1

        # Broadcast atom mask to (*bs, n_atom)
        idx_map_bs_atom = _broadcast_to_other_dim_atom_dim(idx_map, bs, n_atom)
        if atom_mask_i is None:
            atom_mask_bs_atom = torch.ones(
                *bs, n_atom, dtype=torch.bool, device=y.device
            )
        else:
            atom_mask_bs_atom = _broadcast_to_other_dim_atom_dim(
                atom_mask_i.to(torch.bool), bs, n_atom
            )

        # Broadcast token mask to (*bs, n_token)
        if token_mask_i is None:
            token_mask_bs_token = torch.ones(
                *bs, n_token, dtype=torch.bool, device=y.device
            )
        else:
            t = token_mask_i.to(torch.bool)
            if t.shape[-1] != n_token:
                raise ValueError(
                    f"token_mask last dim mismatch: expected {n_token}, got "
                    f"{t.shape[-1]}"
                )
            while t.dim() < (len(bs) + 1):
                t = t.unsqueeze(-2)
            token_mask_bs_token = t.expand(*bs, n_token)

        # Aggregate along last axis [*, N_atom] -> [*, N_token]
        y_perm = aggregate_atom_feat_to_tokens(
            token_mask=token_mask_bs_token,
            atom_to_token_index=idx_map_bs_atom,
            atom_mask=atom_mask_bs_atom,
            atom_feat=y_perm,
            atom_dim=-1,
            aggregate_fn="sum",
            eps=eps,
        )

        # Counts per token on this axis for mean
        if aggregate_fn == "mean":
            counts = torch.zeros(*bs, n_token, device=y.device, dtype=y_perm.dtype)
            counts = counts.scatter_add_(
                -1, idx_map_bs_atom, atom_mask_bs_atom.to(y_perm.dtype)
            )
            per_axis_counts[idx] = counts  # [..., R]

        # Put reduced axis back to d
        axes = list(range(y_perm.dim()))
        axes.pop(-1)
        axes.insert(d, y_perm.dim() - 1)
        y = y_perm.permute(*axes)

    # Apply aggregation fn
    if aggregate_fn == "sum":
        return y
    if aggregate_fn == "any":
        return (y > 0) if y.dtype != torch.bool else y
    if aggregate_fn == "mean":
        # Product of per-axis counts (aligned to output) → joint denominator
        denom = None
        for i in range(K):
            counts = per_axis_counts[i]
            if counts is None:
                continue
            # Counts is [..., N_token_i] with shape 'bs' used when reducing axis i
            c = counts
            # Align to final y by inserting singleton dims
            while c.dim() < y.dim():
                c = c.unsqueeze(-1)
            axes = list(range(c.dim()))
            axes.pop(-1)
            axes.insert(atom_dims_pos[i], c.dim() - 1)
            # Move last axis to atom_dims_pos[i]
            c = c.permute(*axes)
            denom = c if denom is None else denom * c
        if denom is None:
            return y
        return y / (denom.clamp_min(1.0) + eps)

    raise ValueError(f"Invalid aggregate_fn: {aggregate_fn}")


def get_atom_to_onehot_token_index(
    token_mask: torch.Tensor, num_atoms_per_token: torch.Tensor
):
    """
    Get a mapping from atoms to their corresponding one-hot token index.

    Args:
        token_mask:
            [*, N_token] Token mask
        num_atoms_per_token:
            [*, N_token] Number of atoms per token
    Returns:
        atom_to_onehot_token_index:
            [*, N_atom, N_token] Mapping from atom to its one-hot token index
    """
    n_token = token_mask.shape[-1]
    batch_dims = token_mask.shape[:-1]
    token_index = (
        torch.arange(
            n_token, device=num_atoms_per_token.device, dtype=num_atoms_per_token.dtype
        )
        .reshape((*((1,) * len(batch_dims)), n_token))
        .repeat((*batch_dims, 1))
    )
    atom_to_token_index = broadcast_token_feat_to_atoms(
        token_mask=token_mask,
        num_atoms_per_token=num_atoms_per_token,
        token_feat=token_index,
    ).long()
    atom_mask = broadcast_token_feat_to_atoms(
        token_mask=token_mask,
        num_atoms_per_token=num_atoms_per_token,
        token_feat=token_mask,
    )
    atom_to_onehot_token_index = (
        torch.nn.functional.one_hot(
            atom_to_token_index,
            num_classes=n_token,
        ).to(num_atoms_per_token.dtype)
        * atom_mask[..., None]
    )
    return atom_to_onehot_token_index


def max_atom_per_token_masked_select(
    atom_feat: torch.Tensor,
    max_atom_per_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Select atoms from features padded to max atoms per token.

    Args:
        atom_feat
            [*, N_token * max_atoms_per_token, c_out] Atom features padded to
            max atoms per token
        max_atom_per_token_mask:
            [*, N_token * max_atoms_per_token] Mask denoting valid atoms
    Returns:
        atom_feat:
            [*, N_atom, c_out] Selected valid atom features
    """
    batch_dims = atom_feat.shape[:-2]
    c_out = atom_feat.shape[-1]
    max_atoms_in_batch = torch.max(torch.sum(max_atom_per_token_mask.int(), dim=-1))

    # Flatten batch dims
    max_atom_per_token_mask = max_atom_per_token_mask.reshape(
        -1, max_atom_per_token_mask.shape[-1]
    )
    atom_feat = atom_feat.view(-1, *atom_feat.shape[-2:])

    def select_atoms(l: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Select atoms from max-atom padded feature based on max_atoms_in_batch.
        Add padding to max number of atoms in the batch.
        """
        out = torch.masked_select(l, mask[..., None].bool()).reshape(-1, c_out)
        out_padded = torch.nn.functional.pad(
            out, (0, 0, 0, max_atoms_in_batch - out.shape[-2])
        )
        return out_padded

    # Unbind batch dim if it exists, and select atom feats per batch
    if len(batch_dims) > 0:
        per_batch_logits = torch.unbind(atom_feat, dim=0)
        per_batch_mask = torch.unbind(max_atom_per_token_mask, dim=0)

        atom_feat = torch.stack(
            [
                select_atoms(l, m)
                for l, m in zip(per_batch_logits, per_batch_mask, strict=True)
            ],
            dim=0,
        )
    else:
        atom_feat = select_atoms(atom_feat, max_atom_per_token_mask)

    # Expand flattened batch dims
    atom_feat = atom_feat.reshape(*batch_dims, -1, c_out)
    return atom_feat


def get_token_atom_index_offset(atom_name: str, restype: torch.Tensor):
    """
    Get index of a given atom (within its residue) in each residues.

    Args:
        atom_name:
            Atom name to get indices
        restype:
            [*, N_token, 32] One-hot residue types. Must be float dtype.
    Returns:
        token_atom_index_offset:
            [*, N_token] Atom indices (within their residues) of the given atom name
        token_atom_mask:
            [*, N_token] Atom mask to indicate missing atoms
    """
    token_atom_index_offset = torch.einsum(
        "...k,k->...",
        restype.float(),
        torch.tensor(
            atom_name_to_index_by_restype[atom_name]["index"],
            device=restype.device,
        ).float(),
    ).long()
    token_atom_mask = torch.einsum(
        "...k,k->...",
        restype.float(),
        torch.tensor(
            atom_name_to_index_by_restype[atom_name]["mask"],
            device=restype.device,
        ).float(),
    ).long()
    return token_atom_index_offset, token_atom_mask


def get_token_center_atoms(batch: dict, x: torch.Tensor, atom_mask: torch.Tensor):
    """
    Extract center atoms per token, which returns
        -   Ca for standard amino acid residue
        -   C1' for standard nucleotide residue
        -   the first and only atom for modified amino acid or nucleotide residues and
            all ligands (which are tokenized per-atom)

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Atom positions
        atom_mask:
            [*, N_atom] Atom mask
    Returns:
        center_x:
            [*, N_token, 3] Center atom positions
        center_atom_mask:
            [*, N_token] Center atom mask
    """
    # Get index of center atoms
    # [*, N_token]
    start_atom_index = batch["start_atom_index"].long()
    is_standard_protein = batch["is_protein"] * (1 - batch["is_atomized"])
    is_standard_nucleotide = (batch["is_dna"] + batch["is_rna"]) * (
        1 - batch["is_atomized"]
    )

    restype = batch["restype"]
    protein_token_atom_index_offset, protein_token_atom_mask = (
        get_token_atom_index_offset(atom_name="CA", restype=restype)
    )
    nucleotide_token_atom_index_offset, nucleotide_token_atom_mask = (
        get_token_atom_index_offset(atom_name="C1'", restype=restype)
    )
    center_index = (
        (start_atom_index + protein_token_atom_index_offset) * is_standard_protein
        + (start_atom_index + nucleotide_token_atom_index_offset)
        * is_standard_nucleotide
        + start_atom_index * batch["is_atomized"]
    )
    token_atom_mask = (
        protein_token_atom_mask * is_standard_protein
        + nucleotide_token_atom_mask * is_standard_nucleotide
        + batch["is_atomized"]
    )

    # Get coordinates of center atoms
    # [*, N_token, 3]
    center_x = torch.gather(
        x,
        dim=-2,
        index=center_index.unsqueeze(-1)
        .expand(*(x.shape[:-2] + (center_index.shape[-1], 3)))
        .long(),
    )

    # Get center atom mask
    # [*, N_token]
    center_atom_mask = (
        torch.gather(
            atom_mask,
            dim=-1,
            index=center_index.expand(
                *(atom_mask.shape[:-1] + (center_index.shape[-1],))
            ).long(),
        )
        * batch["token_mask"]
    ) * token_atom_mask

    return center_x, center_atom_mask


def get_token_representative_atoms(
    batch: dict, x: torch.Tensor, atom_mask: torch.Tensor
):
    """
    Extract representative atoms per token, which returns
        -   Cb for standard amino acid residues (Ca for glycines)
        -   C4 for purines
        -   C2 for pyrimidines
        -   the first and only atom for modified amino acid or nucleotide residues and
            all ligands (which are tokenized per-atom)

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Atom positions
        atom_mask:
            [*, N_atom] Atom mask
    Returns:
        rep_x:
            [*, N_token, 3] Representative atom positions
        rep_atom_mask:
            [*, N_token] Representative atom mask
    """
    # Create masks for standard amino acid residues
    is_standard_protein = batch["is_protein"] * (1 - batch["is_atomized"])
    is_standard_glycine = (
        is_standard_protein
        * batch["restype"][..., STANDARD_PROTEIN_RESIDUES_ORDER["G"]]
    )

    # Create masks for purines and pyrimadines
    is_standard_dna = batch["is_dna"] * (1 - batch["is_atomized"])
    is_standard_rna = batch["is_rna"] * (1 - batch["is_atomized"])
    is_standard_purine = is_standard_dna * (
        batch["restype"][..., STANDARD_RESIDUES_WITH_GAP_3.index("DA")]
        + batch["restype"][..., STANDARD_RESIDUES_WITH_GAP_3.index("DG")]
    ) + is_standard_rna * (
        batch["restype"][..., STANDARD_RESIDUES_WITH_GAP_3.index("A")]
        + batch["restype"][..., STANDARD_RESIDUES_WITH_GAP_3.index("G")]
    )
    is_standard_pyrimidine = is_standard_dna * (
        batch["restype"][..., STANDARD_RESIDUES_WITH_GAP_3.index("DC")]
        + batch["restype"][..., STANDARD_RESIDUES_WITH_GAP_3.index("DT")]
    ) + is_standard_rna * (
        batch["restype"][..., STANDARD_RESIDUES_WITH_GAP_3.index("C")]
        + batch["restype"][..., STANDARD_RESIDUES_WITH_GAP_3.index("U")]
    )

    # Get index of representative atoms
    restype = batch["restype"]
    start_atom_index = batch["start_atom_index"].long()
    cb_atom_index_offset, cb_atom_mask = get_token_atom_index_offset(
        atom_name="CB", restype=restype
    )
    ca_atom_index_offset, ca_atom_mask = get_token_atom_index_offset(
        atom_name="CA", restype=restype
    )
    c4_atom_index_offset, c4_atom_mask = get_token_atom_index_offset(
        atom_name="C4", restype=restype
    )
    c2_atom_index_offset, c2_atom_mask = get_token_atom_index_offset(
        atom_name="C2", restype=restype
    )
    rep_index = (
        (
            (start_atom_index + cb_atom_index_offset)
            * is_standard_protein
            * (1 - is_standard_glycine)
        )
        + (start_atom_index + ca_atom_index_offset) * is_standard_glycine
        + (start_atom_index + c4_atom_index_offset) * is_standard_purine
        + (start_atom_index + c2_atom_index_offset) * is_standard_pyrimidine
        + start_atom_index * batch["is_atomized"]
    )
    token_atom_mask = (
        cb_atom_mask * is_standard_protein * (1 - is_standard_glycine)
        + ca_atom_mask * is_standard_glycine
        + c4_atom_mask * is_standard_purine
        + c2_atom_mask * is_standard_pyrimidine
        + batch["is_atomized"]
    )

    # Get coordinates of representative atoms
    # [*, N_token, 3]
    rep_x = torch.gather(
        x,
        dim=-2,
        index=rep_index.unsqueeze(-1)
        .expand(*(x.shape[:-2] + (rep_index.shape[-1], 3)))
        .long(),
    )

    # Get representative atom mask
    # [*, N_token]
    rep_atom_mask = (
        torch.gather(
            atom_mask,
            dim=-1,
            index=rep_index.expand(
                *(atom_mask.shape[:-1] + (rep_index.shape[-1],))
            ).long(),
        )
        * batch["token_mask"]
    ) * token_atom_mask

    return rep_x, rep_atom_mask


def get_token_frame_atoms(
    batch: dict,
    x: torch.Tensor,
    atom_mask: torch.Tensor,
    angle_threshold: float = 25.0,
    eps: float = 1e-8,
    inf: float = 1e9,
):
    """
    Extract frame atoms per token, which returns
        -   (N, Ca, C) for standard amino acid residues
        -   (C3', C1', C4') for standard nucleotide residues
        -   closest neighbors for atomized tokens (modified residues and ligands),
            subject to additional angle and chain constraints from Subsection 4.3.2

    Args:
        batch:
            Feature dictionary
        x:
            [*, N_atom, 3] Atom positions
        atom_mask:
            [*, N_atom] Atom mask
        angle_threshold:
            Angle threshold imposed on frame atom selections for atomized tokens
        eps:
            Small constant for numerical stability
        inf:
            Large constant for numerical stability
    Returns:
        phi:
            ([*, N_token, 3], [*, N_token, 3], [*, N_token, 3])
            Tuple of three frame atoms
        valid_frame_mask:
            [*, N_token] Mask denoting valid frames
    """
    # Create pairwise atom mask
    pair_mask = atom_mask[..., None] * atom_mask[..., None, :]

    # Update pairwise atom mask
    # Restrict to atoms within the same chain
    atom_asym_id = broadcast_token_feat_to_atoms(
        token_mask=batch["token_mask"],
        num_atoms_per_token=batch["num_atoms_per_token"],
        token_feat=batch["asym_id"],
    )
    atom_asym_id_mask = atom_asym_id[..., None] == atom_asym_id[..., None, :]
    pair_mask = pair_mask * atom_asym_id_mask

    # Compute distance matrix
    # [*, N_atom, N_atom]
    d = torch.sum(eps + (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1) ** 0.5
    d = d * pair_mask + inf * (1 - pair_mask)

    # Find indices of two closest atoms for start atoms
    # [*, N_token]
    start_atom_index = batch["start_atom_index"].long()
    start_atom_index = start_atom_index.expand(
        *x.shape[:-2], start_atom_index.shape[-1]
    )
    _, closest_atom_index = torch.topk(d, k=3, dim=-1, largest=False)
    a_index = torch.gather(closest_atom_index[..., 1], dim=-1, index=start_atom_index)
    c_index = torch.gather(closest_atom_index[..., 2], dim=-1, index=start_atom_index)

    # Construct indices of atoms used for frame construction
    # [*, N_token]
    is_standard_protein = batch["is_protein"] * (1 - batch["is_atomized"])
    is_standard_nucleotide = (batch["is_dna"] + batch["is_rna"]) * (
        1 - batch["is_atomized"]
    )

    restype = batch["restype"]
    n_atom_index_offset, n_atom_mask = get_token_atom_index_offset(
        atom_name="N", restype=restype
    )
    ca_atom_index_offset, ca_atom_mask = get_token_atom_index_offset(
        atom_name="CA", restype=restype
    )
    c_atom_index_offset, c_atom_mask = get_token_atom_index_offset(
        atom_name="C", restype=restype
    )
    c3p_atom_index_offset, c3p_atom_mask = get_token_atom_index_offset(
        atom_name="C3'", restype=restype
    )
    c1p_atom_index_offset, c1p_atom_mask = get_token_atom_index_offset(
        atom_name="C1'", restype=restype
    )
    c4p_atom_index_offset, c4p_atom_mask = get_token_atom_index_offset(
        atom_name="C4'", restype=restype
    )
    frame_atoms = {
        "a": {
            "index": (
                a_index * batch["is_atomized"]
                + (start_atom_index + n_atom_index_offset) * is_standard_protein
                + (start_atom_index + c3p_atom_index_offset) * is_standard_nucleotide
            ),
            "token_atom_mask": (
                batch["is_atomized"]
                + n_atom_mask * is_standard_protein
                + c3p_atom_mask * is_standard_nucleotide
            ),
        },
        "b": {
            "index": (
                start_atom_index * batch["is_atomized"]
                + (start_atom_index + ca_atom_index_offset) * is_standard_protein
                + (start_atom_index + c1p_atom_index_offset) * is_standard_nucleotide
            ),
            "token_atom_mask": (
                batch["is_atomized"]
                + ca_atom_mask * is_standard_protein
                + c1p_atom_mask * is_standard_nucleotide
            ),
        },
        "c": {
            "index": (
                c_index * batch["is_atomized"]
                + (start_atom_index + c_atom_index_offset) * is_standard_protein
                + (start_atom_index + c4p_atom_index_offset) * is_standard_nucleotide
            ),
            "token_atom_mask": (
                batch["is_atomized"]
                + c_atom_mask * is_standard_protein
                + c4p_atom_mask * is_standard_nucleotide
            ),
        },
    }

    # Extract coordinates
    for key in frame_atoms:
        frame_atoms[key].update(
            {
                "atom_positions": torch.gather(
                    x,
                    dim=-2,
                    index=frame_atoms[key]["index"]
                    .unsqueeze(-1)
                    .expand(*(x.shape[:-2] + (frame_atoms[key]["index"].shape[-1], 3)))
                    .long(),
                ),
                "asym_id": torch.gather(
                    atom_asym_id.expand(*x.shape[:-2], atom_asym_id.shape[-1]),
                    dim=-1,
                    index=frame_atoms[key]["index"].long(),
                ),
                "atom_mask": torch.gather(
                    atom_mask.expand(*x.shape[:-2], atom_mask.shape[-1]),
                    dim=-1,
                    index=frame_atoms[key]["index"].long(),
                )
                * batch["token_mask"]
                * frame_atoms[key]["token_atom_mask"],
            }
        )

    # Compute cosine of angles
    u = frame_atoms["a"]["atom_positions"] - frame_atoms["b"]["atom_positions"]
    v = frame_atoms["c"]["atom_positions"] - frame_atoms["b"]["atom_positions"]
    uv = torch.einsum("...i,...i->...", u, v)
    u_norm = (eps + torch.sum(u**2, dim=-1)) ** 0.5
    v_norm = (eps + torch.sum(v**2, dim=-1)) ** 0.5
    cos_angle = uv / (u_norm * v_norm)

    # Compute valid frame mask from angle constraints
    # (for ligand and non-standard residues)
    cos_angle_min_bound = math.cos((180 - angle_threshold) * math.pi / 180)
    cos_angle_max_bound = math.cos(angle_threshold * math.pi / 180)
    valid_frame_mask_angle = (cos_angle < cos_angle_max_bound) * (
        cos_angle > cos_angle_min_bound
    )
    valid_frame_mask_angle = (
        valid_frame_mask_angle * batch["is_atomized"]
        + torch.ones_like(valid_frame_mask_angle) * (1 - batch["is_atomized"])
    ) * batch["token_mask"]

    # Compute valid frame mask from atom mask constraints
    valid_frame_mask_atom = (
        frame_atoms["a"]["atom_mask"]
        * frame_atoms["b"]["atom_mask"]
        * frame_atoms["c"]["atom_mask"]
    )

    # Compute valid frame mask from chain constraints
    valid_frame_mask_asym_id = (
        frame_atoms["a"]["asym_id"] == frame_atoms["b"]["asym_id"]
    ) * (frame_atoms["b"]["asym_id"] == frame_atoms["c"]["asym_id"])

    # Compute final valid frame mask
    valid_frame_mask = (
        valid_frame_mask_angle * valid_frame_mask_atom * valid_frame_mask_asym_id
    )
    phi = (
        frame_atoms["a"]["atom_positions"],
        frame_atoms["b"]["atom_positions"],
        frame_atoms["c"]["atom_positions"],
    )

    return phi, valid_frame_mask
