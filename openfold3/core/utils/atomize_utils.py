import math
from typing import Dict, Optional

import torch

from openfold3.core.np.residue_constants import restype_order
from openfold3.core.np.token_atom_constants import (
    atom_name_to_index_by_restype,
    restypes,
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
            [*, N_atom] Broadcasted atom-level feature
    """
    n_token = token_mask.shape[-1]
    batch_dims = token_mask.shape[:-1]
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

    # Pad atoms at the end
    # Flatten batch and token dimensions
    max_num_atoms = torch.max(torch.sum(num_atoms_per_token, dim=-1))
    padded_num_atoms_per_token = (
        torch.concat(
            [
                num_atoms_per_token,
                max_num_atoms - torch.sum(num_atoms_per_token, dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        .reshape(-1)
        .int()
    )
    padded_token_feat = torch.concat(
        [
            token_feat,
            torch.zeros((*batch_dims, 1, *feat_dims), device=token_feat.device),
        ],
        dim=token_dim,
    ).reshape(-1, *feat_dims)

    # Create atom-level features
    atom_feat = torch.repeat_interleave(
        input=padded_token_feat, repeats=padded_num_atoms_per_token, dim=0
    )

    # Unflatten batch and token dimensions
    atom_feat = atom_feat.reshape((*batch_dims, max_num_atoms.int(), *feat_dims))

    return atom_feat


def get_token_atom_index_offset(atom_name: str, restype: torch.Tensor):
    """
    Get index of a given atom (within its residue) in each residues.

    Args:
        atom_name:
            Atom name to get indices
        restype:
            [*, N_token, 32] One-hot residue types
    Returns:
        [*, N_token] Atom indices (with their residues) of the given atom name
    """
    return torch.einsum(
        "...k,k->...",
        restype,
        torch.Tensor(atom_name_to_index_by_restype[atom_name], device=restype.device),
    )


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
    center_index = (
        (
            start_atom_index
            + get_token_atom_index_offset(atom_name="CA", restype=batch["restype"])
        )
        * is_standard_protein
        + (
            start_atom_index
            + get_token_atom_index_offset(atom_name="C1'", restype=batch["restype"])
        )
        * is_standard_nucleotide
        + start_atom_index * batch["is_atomized"]
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
    )

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
        batch["restype"][..., restypes.index("DA")]
        + batch["restype"][..., restypes.index("DG")]
    ) + is_standard_rna * (
        batch["restype"][..., restypes.index("A")]
        + batch["restype"][..., restypes.index("G")]
    )
    is_standard_pyrimidine = is_standard_dna * (
        batch["restype"][..., restypes.index("DC")]
        + batch["restype"][..., restypes.index("DT")]
    ) + is_standard_rna * (
        batch["restype"][..., restypes.index("C")]
        + batch["restype"][..., restypes.index("U")]
    )

    # Get index of representative atoms
    start_atom_index = batch["start_atom_index"].long()
    rep_index = (
        (
            start_atom_index
            + get_token_atom_index_offset(atom_name="CB", restype=batch["restype"])
        )
        * is_standard_protein
        * (1 - is_standard_glycine)
        + (
            start_atom_index
            + get_token_atom_index_offset(atom_name="CA", restype=batch["restype"])
        )
        * is_standard_glycine
        + (
            start_atom_index
            + get_token_atom_index_offset(atom_name="C4", restype=batch["restype"])
        )
        * is_standard_purine
        + (
            start_atom_index
            + get_token_atom_index_offset(atom_name="C2", restype=batch["restype"])
        )
        * is_standard_pyrimidine
        + start_atom_index * batch["is_atomized"]
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
    )

    return rep_x, rep_atom_mask


def get_token_frame_atoms(
    batch: Dict,
    x: torch.Tensor,
    atom_mask: torch.Tensor,
    angle_threshold: float,
    eps: float,
    inf: float,
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
    frame_atoms = {
        "a": {
            "index": (
                a_index * batch["is_atomized"]
                + (
                    start_atom_index
                    + get_token_atom_index_offset(
                        atom_name="N", restype=batch["restype"]
                    )
                )
                * is_standard_protein
                + (
                    start_atom_index
                    + get_token_atom_index_offset(
                        atom_name="C3'", restype=batch["restype"]
                    )
                )
                * is_standard_nucleotide
            )
        },
        "b": {
            "index": (
                start_atom_index * batch["is_atomized"]
                + (
                    start_atom_index
                    + get_token_atom_index_offset(
                        atom_name="CA", restype=batch["restype"]
                    )
                )
                * is_standard_protein
                + (
                    start_atom_index
                    + get_token_atom_index_offset(
                        atom_name="C1'", restype=batch["restype"]
                    )
                )
                * is_standard_nucleotide
            )
        },
        "c": {
            "index": (
                c_index * batch["is_atomized"]
                + (
                    start_atom_index
                    + get_token_atom_index_offset(
                        atom_name="C", restype=batch["restype"]
                    )
                )
                * is_standard_protein
                + (
                    start_atom_index
                    + get_token_atom_index_offset(
                        atom_name="C4'", restype=batch["restype"]
                    )
                )
                * is_standard_nucleotide
            )
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
                * batch["token_mask"],
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
