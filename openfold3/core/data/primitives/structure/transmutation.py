"""Primitives for artificial atom array transformations."""

import numpy as np
from biotite.structure import AtomArray


def replace_coordinates(
    target_atom_array: AtomArray,
    source_atom_array: AtomArray,
    chain_map: dict[str, str],
    target_atom_mask: np.ndarray | None,
    source_atom_mask: np.ndarray | None,
) -> AtomArray:
    """Replaces the coordinates of atoms in the target with those in the source.

    Args:
        target_atom_array (AtomArray):
            The atom array to replace the coordinates in.
        source_atom_array (AtomArray):
            The atom array providing the coordinates for replacement.
        chain_map (dict[str, str]):
            A dictionary mapping chain IDs from the source to chain IDs in the target.
        target_atom_mask (np.ndarray | None):
            A boolean mask to select atoms in the target atom array to replace
            coordinates with.
        source_atom_mask (np.ndarray | None):
            A boolean mask to select atoms in the source atom array to replace
            coordinates for.

    Returns:
        AtomArray:
            The source atom array with the coordinates of the target atom array.
    """

    # Perform checks

    if target_atom_mask is None:
        target_atom_mask = np.full(len(target_atom_array), True)
    if source_atom_mask is None:
        source_atom_mask = np.full(len(source_atom_array), True)

    source_atom_array_slice = source_atom_array[source_atom_mask].copy()

    # Add replaced coordinates back to the full atom array

    return chimera_atom_array


def check_atom_array_compatibility(
    source_atom_array: AtomArray, target_atom_array: AtomArray
) -> None:
    """Checks if two atom arrays are compatible for coordinate replacement.

    Args:
        source_atom_array (AtomArray):
            The first atom array to check.
        target_atom_array (AtomArray):
            The second atom array to check.

    Raises:
        ValueError:
            If the atom arrays are not compatible.
    """

    # TODO: list checks here

    # - same number of atoms
    # - same number of chains
    # - same number of residues and residues within chain are in same order
    # - same number of atoms and atoms in each residue are in same order
