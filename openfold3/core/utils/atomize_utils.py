import math
from typing import Dict, Optional

import torch

from openfold3.core.np.residue_constants import restype_order
from openfold3.core.np.token_atom_constants import (
    TOKEN_TYPES_WITH_GAP,
    atom_name_to_index_by_restype,
)


def broadcast_token_feat_to_atoms(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor,
    token_feat: torch.Tensor,
    token_dim: Optional[int] = -1,
    max_num_atoms_per_token: Optional[int] = None,
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
    num_atoms_per_token = num_atoms_per_token * token_mask
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
    max_num_atoms = torch.max(torch.sum(num_atoms_per_token, dim=-1))
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
    atom_feat = atom_feat.reshape((*feat_batch_dims, max_num_atoms.int(), *feat_dims))

    return atom_feat


def aggregate_atom_feat_to_tokens(
    token_mask: torch.Tensor,
    num_atoms_per_token: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_feat: torch.Tensor,
    atom_dim: Optional[int] = -1,
    eps: float = 1e-9,
):
    """
    Aggregate atom-level features to token-level features with mean aggregation.

    Args:
        token_mask:
            [*, N_token] Token mask
        num_atoms_per_token:
            [*, N_token] Number of atoms per token
        atom_mask:
            [*, N_atom] Atom mask
        atom_feat:
            [*, N_atom, *feat_dims] Atom-level features
        atom_dim:
            Atom dimension
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

    # Construct atom to token index
    assert atom_mask.shape[:-1] == batch_dims
    token_index = (
        torch.arange(n_token, device=token_mask.device, dtype=token_mask.dtype)
        .reshape((*((1,) * len(batch_dims)), n_token))
        .repeat((*batch_dims, 1))
    )
    atom_to_token_index = broadcast_token_feat_to_atoms(
        token_mask=token_mask,
        num_atoms_per_token=num_atoms_per_token,
        token_feat=token_index,
    )
    atom_to_token_index = atom_to_token_index * atom_mask + n_token * torch.ones_like(
        atom_to_token_index
    ) * (1 - atom_mask)

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

    # Compute number of atoms (non-masked) per token
    token_num_atoms = torch.zeros(
        (*batch_dims, n_token + 1), device=atom_feat.device, dtype=atom_feat.dtype
    ).scatter_add_(index=atom_to_token_index.long(), src=atom_mask, dim=-1)[
        ..., :n_token
    ]

    # Compute mean token-level feature
    token_feat = token_feat / (
        token_num_atoms.reshape(token_num_atoms.shape + (1,) * len(feat_dims)) + eps
    )

    return token_feat


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


def get_token_atom_index_offset(atom_name: str, restype: torch.Tensor):
    """
    Get index of a given atom (within its residue) in each residues.

    Args:
        atom_name:
            Atom name to get indices
        restype:
            [*, N_token, 32] One-hot residue types
    Returns:
        token_atom_index_offset:
            [*, N_token] Atom indices (within their residues) of the given atom name
        token_atom_mask:
            [*, N_token] Atom mask to indicate missing atoms
    """
    token_atom_index_offset = torch.einsum(
        "...k,k->...",
        restype,
        torch.Tensor(
            atom_name_to_index_by_restype[atom_name]["index"], device=restype.device
        ).to(restype.dtype),
    )
    token_atom_mask = torch.einsum(
        "...k,k->...",
        restype,
        torch.Tensor(
            atom_name_to_index_by_restype[atom_name]["mask"], device=restype.device
        ).to(restype.dtype),
    )
    return token_atom_index_offset, token_atom_mask


def get_token_center_atoms(batch: Dict, x: torch.Tensor, atom_mask: torch.Tensor):
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
    protein_token_atom_index_offset, protein_token_atom_mask = (
        get_token_atom_index_offset(atom_name="CA", restype=batch["restype"])
    )
    nucleotide_token_atom_index_offset, nucleotide_token_atom_mask = (
        get_token_atom_index_offset(atom_name="C1'", restype=batch["restype"])
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
        .repeat(*((1,) * len(center_index.shape) + (3,)))
        .long(),
    )

    # Get center atom mask
    # [*, N_token]
    center_atom_mask = (
        torch.gather(atom_mask, dim=-1, index=center_index.long()) * batch["token_mask"]
    ) * token_atom_mask

    return center_x, center_atom_mask


def get_token_representative_atoms(
    batch: Dict, x: torch.Tensor, atom_mask: torch.Tensor
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
        is_standard_protein * batch["restype"][..., restype_order["G"]]
    )

    # Create masks for purines and pyrimadines
    is_standard_dna = batch["is_dna"] * (1 - batch["is_atomized"])
    is_standard_rna = batch["is_rna"] * (1 - batch["is_atomized"])
    is_standard_purine = is_standard_dna * (
        batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("DA")]
        + batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("DG")]
    ) + is_standard_rna * (
        batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("A")]
        + batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("G")]
    )
    is_standard_pyrimidine = is_standard_dna * (
        batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("DC")]
        + batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("DT")]
    ) + is_standard_rna * (
        batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("C")]
        + batch["restype"][..., TOKEN_TYPES_WITH_GAP.index("U")]
    )

    # Get index of representative atoms
    start_atom_index = batch["start_atom_index"].long()
    cb_atom_index_offset, cb_atom_mask = get_token_atom_index_offset(
        atom_name="CB", restype=batch["restype"]
    )
    ca_atom_index_offset, ca_atom_mask = get_token_atom_index_offset(
        atom_name="CA", restype=batch["restype"]
    )
    c4_atom_index_offset, c4_atom_mask = get_token_atom_index_offset(
        atom_name="C4", restype=batch["restype"]
    )
    c2_atom_index_offset, c2_atom_mask = get_token_atom_index_offset(
        atom_name="C2", restype=batch["restype"]
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
        .repeat(*((1,) * len(rep_index.shape) + (3,)))
        .long(),
    )

    # Get representative atom mask
    # [*, N_token]
    rep_atom_mask = (
        torch.gather(atom_mask, dim=-1, index=rep_index.long()) * batch["token_mask"]
    ) * token_atom_mask

    return rep_x, rep_atom_mask


def get_token_frame_atoms(
    batch: Dict,
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
    _, closest_atom_index = torch.topk(d, k=3, dim=-1, largest=False)
    a_index = torch.gather(closest_atom_index[..., 1], dim=-1, index=start_atom_index)
    c_index = torch.gather(closest_atom_index[..., 2], dim=-1, index=start_atom_index)

    # Construct indices of atoms used for frame construction
    # [*, N_token]
    is_standard_protein = batch["is_protein"] * (1 - batch["is_atomized"])
    is_standard_nucleotide = (batch["is_dna"] + batch["is_rna"]) * (
        1 - batch["is_atomized"]
    )
    n_atom_index_offset, n_atom_mask = get_token_atom_index_offset(
        atom_name="N", restype=batch["restype"]
    )
    ca_atom_index_offset, ca_atom_mask = get_token_atom_index_offset(
        atom_name="CA", restype=batch["restype"]
    )
    c_atom_index_offset, c_atom_mask = get_token_atom_index_offset(
        atom_name="C", restype=batch["restype"]
    )
    c3p_atom_index_offset, c3p_atom_mask = get_token_atom_index_offset(
        atom_name="C3'", restype=batch["restype"]
    )
    c1p_atom_index_offset, c1p_atom_mask = get_token_atom_index_offset(
        atom_name="C1'", restype=batch["restype"]
    )
    c4p_atom_index_offset, c4p_atom_mask = get_token_atom_index_offset(
        atom_name="C4'", restype=batch["restype"]
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
                    .repeat(*((1,) * len(frame_atoms[key]["index"].shape) + (3,)))
                    .long(),
                ),
                "asym_id": torch.gather(
                    atom_asym_id, dim=-1, index=frame_atoms[key]["index"].long()
                ),
                "atom_mask": torch.gather(
                    atom_mask, dim=-1, index=frame_atoms[key]["index"].long()
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
