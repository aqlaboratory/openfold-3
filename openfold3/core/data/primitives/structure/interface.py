# TODO: add module docstring

import logging
from typing import Generator

import numpy as np
from biotite.structure import AtomArray
from scipy.spatial import KDTree
from typing_extensions import Self

logger = logging.getLogger(__name__)


class NaNRobustKDTree:
    """Utility wrapper around scipy.spatial.KDTree that can handle NaNs.

    NaN support is handled by running the function on the non-NaN coordinates, and
    mapping the returned indices back to the corresponding indices in the original
    array. This class only implements the `query` and `query_ball_tree` methods.
    """

    def __init__(self, data, *args, **kwargs):
        self._orig_data = data

        # Remove rows with NaNs
        self._nan_mask = np.any(np.isnan(self._orig_data), axis=1)
        valid_data = self._orig_data[~self._nan_mask]

        # Create a mapping from the new data index to the original data index
        self._orig_data_index = np.arange(self._orig_data.shape[0])
        self._data_index_map = self._orig_data_index[~self._nan_mask]

        self._kdtree = KDTree(valid_data, *args, **kwargs)

    def query_pairs(self, r: float, p: float = 2.0, eps: float = 0.0) -> np.ndarray:
        """NaN-robust version of KDTree.query_pairs.

        Runs KDTree.query_pairs but re-indexes the results to be consistent with the
        original array. See scipy.spatial.KDTree.query_pairs for more information.
        """
        valid_data_results = self._kdtree.query_pairs(r, p, eps, output_type="ndarray")
        orig_data_results = self._data_index_map[valid_data_results]

        return orig_data_results

    def query_ball_tree(
        self, other: Self, r: float, p: float = 2.0, eps: float = 0.0
    ) -> list[list]:
        """NaN-robust version of KDTree.query_ball_tree.

        Runs KDTree.query_ball_tree but re-indexes the results to be consistent with the
        original array. See scipy.spatial.KDTree.query_ball_tree for more information.
        """
        if not isinstance(other, NaNRobustKDTree):
            raise ValueError("The other tree must be of type NaNRobustKDTree")

        other_kdtree = other._kdtree
        valid_data_results = self._kdtree.query_ball_tree(other_kdtree, r, p, eps)

        orig_data_results = [[] for _ in range(self._orig_data.shape[0])]

        for idx, result in enumerate(valid_data_results):
            for i, value in enumerate(result):
                result[i] = other._data_index_map[value]

            orig_data_idx = self._data_index_map[idx]
            orig_data_results[orig_data_idx].extend(result)

        return orig_data_results


def get_query_interface_atom_pair_idxs(
    query_atom_array: AtomArray,
    target_atom_array: AtomArray,
    distance_threshold: float = 15.0,
    return_chain_pairs: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Returns interface atom pair indices of the query based on the target.

    Takes in a set of query and target atoms and will return all pairs between query
    and target atoms that have different chain IDs and are within a given distance
    threshold of each other. Optionally, it can also return the chain IDs of the matched
    atom pairs.

    Uses a KDTree internally which will speed up the search for larger structures.

    Args:
        query_atom_array:
            AtomArray containing the first set of atoms
        target_atom_array:
            AtomArray containing the second set of atoms
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 15.0.
        return_chain_pairs:
            Whether to return the chain IDs of the matched atom pairs. Defaults to
            False.

    Returns:
        atom_pairs: np.ndarray
            Array of atom pair indices with query atoms in the first column and target
            atoms in the second column. Pairs will be sorted by the query atom index.
        chain_pairs: np.ndarray
            Array of chain ID pairs corresponding to the atom pairs. Only returned if
            `return_chain_pairs` is True.
    """
    kdtree_query = NaNRobustKDTree(query_atom_array.coord)
    kdtree_target = NaNRobustKDTree(target_atom_array.coord)
    search_result = kdtree_query.query_ball_tree(kdtree_target, distance_threshold)

    # Get to same output format as kdtree.query_pairs
    atom_pair_idxs = np.array(
        [(i, j) for i, j_list in enumerate(search_result) for j in j_list]
    )

    # Pair the chain IDs
    chain_pairs = np.column_stack(
        (
            query_atom_array.chain_id[atom_pair_idxs[:, 0]],
            target_atom_array.chain_id[atom_pair_idxs[:, 1]],  # change to target
        )
    )

    # Get only cross-chain contacts
    cross_chain_mask = chain_pairs[:, 0] != chain_pairs[:, 1]
    atom_pair_idxs = atom_pair_idxs[cross_chain_mask]

    # Optionally also return the matched chain IDs for the atom pairs
    if return_chain_pairs:
        chain_pairs = chain_pairs[cross_chain_mask]
        return atom_pair_idxs, chain_pairs
    else:
        return atom_pair_idxs


def get_interface_atom_pair_idxs(
    atom_array: AtomArray,
    distance_threshold: float = 15.0,
    return_chain_pairs: bool = False,
    sort_by_chain: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Returns interface atom pair indices within a structure.

    Takes in an AtomArray and will return all pairs of atoms that have different chain
    IDs and are within a given distance threshold of each other. Optionally, it can also
    return the chain IDs of the matched atom pairs.

    Uses a KDTree internally which will speed up the search for larger structures.

    Args:
        atom_array:
            AtomArray containing the structure to find interface atom pairs in.
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 15.0.
        return_chain_pairs:
            Whether to return the chain IDs of the matched atom pairs. Defaults to
            False.
        sort_by_chain:
            If True, will sort each individual atom pair so that the corresponding chain
            IDs are in ascending order. Otherwise, atom pairs will be ordered in a way
            that the first index is always smaller than the second index. Defaults to
            False.

    Returns:
        atom_pairs: np.ndarray
            Array of atom pair indices with the first atom in the first column and the
            second atom in the second column. If `sort_by_chain` is False (default),
            pairs will be stored non-redundantly, so that i < j for any pair (i, j). If
            `sort_by_chain` is True, the atom indices in each pair will be sorted such
            that the corresponding chain IDs are in ascending order within each pair,
            which may result in pairs where j < i.
    """
    kdtree = NaNRobustKDTree(atom_array.coord)

    atom_pair_idxs = kdtree.query_pairs(distance_threshold)

    # Pair the chain IDs
    chain_pairs = atom_array.chain_id[atom_pair_idxs]

    if sort_by_chain:
        # Sort by chain within-pair to canonicalize
        # (e.g. [(1, 2), (2, 1)] -> [(1, 2), (1, 2)])
        chain_sort_idx = np.argsort(chain_pairs, axis=1)
        chain_pairs = np.take_along_axis(chain_pairs, chain_sort_idx, axis=1)
        atom_pair_idxs = np.take_along_axis(atom_pair_idxs, chain_sort_idx, axis=1)

    # Get only cross-chain contacts
    cross_chain_mask = chain_pairs[:, 0] != chain_pairs[:, 1]
    atom_pair_idxs = atom_pair_idxs[cross_chain_mask]

    # Optionally also return the matched chain IDs for the atom pairs
    if return_chain_pairs:
        chain_pairs = chain_pairs[cross_chain_mask]
        return atom_pair_idxs, chain_pairs
    else:
        return atom_pair_idxs


def get_query_interface_atoms(
    query_atom_array: AtomArray,
    target_atom_array: AtomArray,
    distance_threshold: float = 15.0,
) -> AtomArray:
    """Returns interface atoms in the query based on the target

    This will find atoms in the query that are within a given distance threshold of any
    atom with a different chain in the target.

    Args:
        query_atom_array:
            AtomArray containing the structure to find interface atoms in.
        target_atom_array:
            AtomArray containing the structure to compare against.
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 15.0.

    Returns:
        Subset of the query AtomArray just containing interface atoms.
    """
    # Get all interface atom pairs of query-target
    interface_atom_pairs = get_query_interface_atom_pair_idxs(
        query_atom_array, target_atom_array, distance_threshold
    )
    # Subset to just unique (sorted) atoms of the query
    query_interface_atoms = query_atom_array[np.unique(interface_atom_pairs[:, 0])]

    return query_interface_atoms


def get_interface_atoms(
    atom_array: AtomArray,
    distance_threshold: float = 15.0,
) -> AtomArray:
    """Returns interface atoms in a structure.

    This will find atoms in a structure that are within a given distance threshold of
    any atom with a different chain in the same structure.

    Args:
        atom_array:
            AtomArray containing the structure to find interface atoms in.
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 15.0.

    Returns:
        AtomArray with interface atoms.
    """
    # Get all pairs of atoms within the distance threshold
    interface_atom_pair_idxs = get_interface_atom_pair_idxs(
        atom_array, distance_threshold
    )

    # Return all atoms participating in any of the pairs
    return atom_array[np.unique(interface_atom_pair_idxs.flatten())]


def get_interface_chain_id_pairs(
    atom_array: AtomArray, distance_threshold: float = 15.0
) -> np.ndarray:
    """Returns chain pairings with interface atoms based on a distance threshold

    This will find all pairs of chains in the AtomArray that have at least one atom
    within a given distance threshold of each other.

    Args:
        atom_array:
            AtomArray containing the structure to find interface chain pairings in.
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 15.0.

    Returns:
        List of tuples with unique chain pairings that have interface atoms, so that
        chain_id_1 < chain_id_2 for a pair (chain_id_1, chain_id_2).
    """
    _, chain_pairs = get_interface_atom_pair_idxs(
        atom_array,
        distance_threshold=distance_threshold,
        return_chain_pairs=True,
        sort_by_chain=True,
    )

    return np.unique(chain_pairs, axis=0)


def chain_paired_interface_atom_iter(
    atom_array: AtomArray,
    distance_threshold: float = 15.0,
    ignore_covalent: bool = False,
) -> Generator[tuple[tuple[int, int], np.ndarray[np.integer, np.integer]], None, None]:
    """Yields interface atom pairs grouped by unique chain pairs.

    Interface atoms are defined as atoms that are within a given distance threshold of
    each other and have different chain IDs. For each unique pair of chains that have at
    least one interface contact, this function will yield the corresponding chain IDs as
    well as all interface atoms between this pair of chains.

    Args:
        atom_array:
            AtomArray containing the structure to find interface atom pairs in.
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 15.0.
        ignore_covalent:
            Whether to ignore pairs corresponding to covalently bonded atoms. Defaults
            to False.

    Yields:
        chain_ids:
            Tuple of chain IDs of the pair (sorted so that chain_id_1 < chain_id_2)
        atom_pair_idxs:
            Array of atom pair indices corresponding to the chain pairing with the first
            chain's atom in the first column and the second chain's atom in the second
            column.
    """
    # Get all interface atom pairs and their corresponding chain ID pairs
    atom_pair_idxs, chain_pairs = get_interface_atom_pair_idxs(
        atom_array,
        distance_threshold=distance_threshold,
        return_chain_pairs=True,
        sort_by_chain=True,
    )

    if atom_pair_idxs.size == 0:
        return

    # Optionally remove pairs corresponding to covalently bonded atoms
    if ignore_covalent:
        # Subset the atom_array to only the relevant atoms to avoid computing huge
        # adjacency matrix
        unique_atoms, unique_atom_idx = np.unique(atom_pair_idxs, return_inverse=True)
        subset_atom_array = atom_array[unique_atoms]

        # Maps the original atom pairs to their indices in the unique atom list
        unique_atom_idx = unique_atom_idx.reshape(atom_pair_idxs.shape)

        subset_adjmat = subset_atom_array.bonds.adjacency_matrix()
        covalent_pair_mask = subset_adjmat[unique_atom_idx[:, 0], unique_atom_idx[:, 1]]

        # Remove the covalent atom pairs
        atom_pair_idxs = atom_pair_idxs[~covalent_pair_mask]
        chain_pairs = chain_pairs[~covalent_pair_mask]

        # If all pairs were covalent, return
        if atom_pair_idxs.size == 0:
            return

    # Subsequent code is only necessary if there are multiple pairs
    if atom_pair_idxs.shape[0] == 1:
        yield tuple(chain_pairs[0]), atom_pair_idxs
        return

    # Sort to group together occurrences of the same pair
    # (e.g. [(0, 1), (0, 2), (0, 1)] -> [(0, 1), (0, 1), (0, 2)])
    group_sort_idx = np.lexsort((chain_pairs[:, 1], chain_pairs[:, 0]))
    chain_pairs_grouped = chain_pairs[group_sort_idx]
    atom_pairs_grouped = atom_pair_idxs[group_sort_idx]

    # Get indices of the first occurrence of each pair
    changes = (
        np.roll(chain_pairs_grouped, shift=1, axis=0) != chain_pairs_grouped
    ).any(axis=1)
    group_start_idx = np.nonzero(changes)[0]

    if len(group_start_idx) == 0:
        # If there is only one group, yield it directly
        yield tuple(chain_pairs_grouped[0]), atom_pairs_grouped
        return

    # Get indices where a new group of chain pairs starts
    group_end_idx = np.roll(group_start_idx, shift=-1)
    group_end_idx[-1] = len(chain_pairs_grouped)

    for start_idx, end_idx in zip(group_start_idx, group_end_idx):
        yield (
            tuple(chain_pairs_grouped[start_idx]),
            atom_pairs_grouped[start_idx:end_idx],
        )


def get_interface_token_center_atoms(
    atom_array: AtomArray,
    distance_threshold: float = 15.0,
) -> AtomArray:
    """Gets interface token center atoms within a structure.

    This will find token center atoms that are within a given distance threshold of
    any token center atom with a different chain in the same structure.

    For example used in 2.5.4 of the AlphaFold3 SI (subsetting of large bioassemblies)

    Args:
        atom_array:
            AtomArray containing the structure to find interface token center atoms in.
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 15.0.

    Returns:
        AtomArray with interface token center atoms.
    """

    if "token_center_atom" not in atom_array.get_annotation_categories():
        raise ValueError(
            "Token center atoms not found in atom array, run "
            "tokenize_atom_array first"
        )

    token_center_atoms = atom_array[atom_array.token_center_atom]

    return get_interface_atoms(token_center_atoms, distance_threshold)


def get_query_interface_token_center_atoms(
    query_atom_array: AtomArray,
    target_atom_array: AtomArray,
    distance_threshold: float = 15.0,
) -> AtomArray:
    """Gets interface token center atoms in the query based on the target

    This will find token center atoms in the query that are within a given distance
    threshold of any token center atom with a different chain in the target.

    For example used in 2.5.4 of the AlphaFold3 SI (subsetting of large bioassemblies)

    Args:
        query_atom_array:
            AtomArray containing the structure to find interface token center atoms in.
        target_atom_array:
            AtomArray containing the structure to compare against.
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 15.0.

    Returns:
        AtomArray with interface token center atoms.
    """
    if "token_center_atom" not in query_atom_array.get_annotation_categories():
        raise ValueError(
            "Token center atoms not found in query atom array, run "
            "tokenize_atom_array first"
        )
    elif "token_center_atom" not in target_atom_array.get_annotation_categories():
        raise ValueError(
            "Token center atoms not found in target atom array, run "
            "tokenize_atom_array first"
        )

    query_token_centers = query_atom_array[query_atom_array.token_center_atom]
    target_token_centers = target_atom_array[target_atom_array.token_center_atom]

    return get_query_interface_atoms(
        query_token_centers, target_token_centers, distance_threshold
    )
