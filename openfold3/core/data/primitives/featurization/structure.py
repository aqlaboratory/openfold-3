"""
This module contains building blocks for target and ground truth structure feature
generation.
"""

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure import AtomArray


def get_token_starts(
    atom_array: AtomArray, add_exclusive_stop: bool = False
) -> np.ndarray:
    """Gets the indices of the first atom of each token.

    Args:
        atom_array (AtomArray):
            AtomArray of the target or ground truth structure
        add_exclusive_stop (bool, optional):
            Whether to add append an int with the size of the input atom array at the
            end of returned indices. Defaults to False.

    Returns:
        np.ndarray: _description_
    """
    token_id_diffs = np.diff(atom_array.token_id)
    token_starts = np.where(token_id_diffs != 0)[0] + 1
    token_starts = np.append(0, token_starts)
    if add_exclusive_stop:
        token_starts = np.append(token_starts, len(atom_array))
    return token_starts


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


def create_sym_id(entity_ids: np.ndarray) -> np.ndarray:
    """Creates sym_id feature as outlined in AF3 SI Table 5.

    Args:
        entity_ids (np.array):
            Entity ids of the target or ground truth structure.

    Returns:
        np.ndarray:
            Array of sym_ids.
    """
    output_array = np.zeros_like(entity_ids)
    counter = 0

    for i in range(1, len(entity_ids)):
        if entity_ids[i] == entity_ids[i - 1]:
            counter += 1
        else:
            counter = 0
        output_array[i] = counter

    return output_array


def extract_starts_entities(atom_array: AtomArray) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the residue starts and entity ids from an AtomArray.

    Args:
        atom_array (AtomArray):
            AtomArray of the target or ground truth structure.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Residue starts and entity ids.
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
    # Get bonds from whole cropped array
    bonds = atom_array.bonds.as_array()
    token_bonds = np.zeros([len(token_index), len(token_index)])

    # Get covalent subset - exclude coordinate bonds
    # TODO implement

    # Get subset connecting at least one atomized token
    atom_ids_atomized_tokens = np.where(atom_array.is_atomized)[0]
    if atom_ids_atomized_tokens.size > 0:
        bonds_atomized_tokens = bonds[
            (np.isin(bonds[:, 0], atom_ids_atomized_tokens))
            & np.isin(bonds[:, 1], atom_ids_atomized_tokens),
            :,
        ]

        # Get subset < 2.4A
        bonds_atomized_tokens = bonds_atomized_tokens[
            struc.distance(
                atom_array.coord[bonds_atomized_tokens[:, 0]],
                atom_array.coord[bonds_atomized_tokens[:, 1]],
            )
            < 2.4
        ]

        if bonds_atomized_tokens.size > 0:
            # Map atom indices to token indices to token-in-crop index
            token_to_token_in_crop = {t: tic for tic, t in enumerate(token_index)}
            bonds_atomized_token_ids = np.stack(
                [
                    np.vectorize(token_to_token_in_crop.get)(
                        atom_array.token_id[bonds_atomized_tokens[:, i]]
                    )
                    for i in [0, 1]
                ]
            )

            # Unmask corresponding bonds
            token_bonds[
                bonds_atomized_token_ids[0],
                bonds_atomized_token_ids[1],
            ] = True

    return torch.tensor(token_bonds, dtype=torch.int32)


def create_token_mask(token_allocated: int, token_budget: int) -> torch.Tensor:
    """Creates token_mask feature.

    Args:
        token_allocated (int):
            Number of tokens allocated.
        token_budget (int):
            Crop size.

    Returns:
        torch.Tensor:
            token_mask that can be used as a padding mask along the token dim.
    """
    token_mask = torch.zeros(token_budget, dtype=torch.float32)
    token_mask[:token_allocated] = True
    return token_mask
