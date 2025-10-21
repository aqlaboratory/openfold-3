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

"""
This module contains building blocks for target and ground truth structure feature
generation.
"""

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure import AtomArray

from openfold3.core.data.primitives.structure.cleanup import filter_fully_atomized_bonds
from openfold3.core.data.primitives.structure.labels import get_token_starts
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms


def encode_one_hot(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Encodes a tensor of indices as a one-hot tensor.

    Args:
        x (torch.Tensor):
            Tensor of numerically encoded residues.
        num_classes (int):
            Number of classes to encode.

    Returns:
        torch.Tensor:
            One-hot encoded tensor.
    """
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


def create_sym_id(
    entity_ids_per_chain: np.ndarray, atom_array: AtomArray, token_starts: np.ndarray
) -> np.ndarray:
    """Creates sym_id feature as outlined in AF3 SI Table 5.

    Args:
        entity_ids (np.array):
            Entity ids of the target or ground truth structure.

    Returns:
        np.ndarray:
            Array of sym_ids mapped to each token.
    """
    sym_id_per_chain = np.zeros_like(entity_ids_per_chain)

    unique_entity_ids = np.unique(entity_ids_per_chain)
    entity_id_to_counter = {entity_id: 0 for entity_id in unique_entity_ids}

    for idx, i in enumerate(entity_ids_per_chain):
        entity_id_to_counter[i] += 1
        sym_id_per_chain[idx] = entity_id_to_counter[i]

    sym_id_per_atom = struc.spread_chain_wise(atom_array, sym_id_per_chain)
    return sym_id_per_atom[token_starts]


def extract_starts_entities(atom_array: AtomArray) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the residue starts and entity ids from an AtomArray.

    Args:
        atom_array (AtomArray):
            AtomArray of the target or ground truth structure.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Residue starts and entity ids for each chain.
    """
    token_starts_with_stop = get_token_starts(atom_array, add_exclusive_stop=True)
    chain_starts = struc.get_chain_starts(atom_array)
    entity_ids = atom_array.entity_id[chain_starts]
    return token_starts_with_stop, entity_ids


def create_token_bonds(atom_array: AtomArray, token_index: np.ndarray) -> torch.Tensor:
    """Creates token_bonds feature as outlined in AF3 SI Table 5.

    Args:
        atom_array (AtomArray):
            AtomArray of the target or ground truth structure.
        token_index (np.ndarray):
            Indices of tokens in the cropped AtomArray.

    Returns:
        torch.Tensor:
            token_bonds feature.
    """
    # Fully subset bonds with strict defintion to only the ones in AF3 SI Table 5
    # "token_bonds"
    atom_array = filter_fully_atomized_bonds(
        atom_array,
    )

    # Initialize N_token x N_token bond matrix
    token_bonds = np.zeros([len(token_index), len(token_index)])

    # Get bonded atoms
    bond_partners = atom_array.bonds.as_array()[:, :2]

    if bond_partners.size > 0:
        # Map atom indices to token indices to token-in-crop index
        token_to_token_in_crop = {t: tic for tic, t in enumerate(token_index)}

        get_absolute_token_id = np.vectorize(token_to_token_in_crop.get)

        bond_partner_absolute_token_ids = get_absolute_token_id(
            atom_array.token_id[bond_partners]
        )

        # Unmask corresponding bonds
        token_bonds[
            (
                bond_partner_absolute_token_ids[:, 0],
                bond_partner_absolute_token_ids[:, 1],
            ),
            (
                bond_partner_absolute_token_ids[:, 1],
                bond_partner_absolute_token_ids[:, 0],
            ),
        ] = True

    return torch.tensor(token_bonds, dtype=torch.int32)


def create_atom_to_token_index(
    token_mask: torch.Tensor, num_atoms_per_token: torch.Tensor
) -> torch.Tensor:
    """
    Creates mapping from atom to its corresponding token. Note that this is the
    consecutive token index starting from 0 and not the index in the actual structure.

    Args:
        token_mask:
            [*, N_token] Token mask
        num_atoms_per_token:
            [*, N_token] Number of atoms per token

    Returns:
        atom_to_token_index:
            [*, N_atom] Mapping from atom to its token index
    """
    n_token = token_mask.shape[-1]
    batch_dims = token_mask.shape[:-1]

    # Construct token index to broadcast to atoms
    token_index = (
        torch.arange(n_token, device=token_mask.device, dtype=token_mask.dtype)
        .reshape((*((1,) * len(batch_dims)), n_token))
        .repeat((*batch_dims, 1))
    )

    atom_to_token_index = broadcast_token_feat_to_atoms(
        token_mask=token_mask,
        num_atoms_per_token=num_atoms_per_token,
        token_feat=token_index,
    ).to(dtype=torch.int32)

    return atom_to_token_index
