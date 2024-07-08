import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from scipy.spatial.distance import cdist


def extend_with_unresolved(atom_array: AtomArray) -> None:
    """Adds placeholder atoms for unresolved residues.

    Args:
        atom_array (AtomArray):
            Biotite atom array of a PDB entry.
    """


def get_interface_atoms(
    query_atom_array: struc.AtomArray,
    target_atom_array: struc.AtomArray,
    distance_threshold: float = 15.0,
) -> struc.AtomArray:
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
        AtomArray with interface atoms.
    """
    pairwise_dists = cdist(query_atom_array.coord, target_atom_array.coord)

    # All unique chains in the query
    query_chain_ids = set(query_atom_array.chain_id_renumbered)

    # Set masks to avoid intra-chain comparisons
    for chain in query_chain_ids:
        query_chain_mask = query_atom_array.chain_id_renumbered == chain
        target_chain_mask = target_atom_array.chain_id_renumbered == chain

        query_chain_block_mask = np.ix_(query_chain_mask, target_chain_mask)
        pairwise_dists[query_chain_block_mask] = np.inf

    interface_atom_mask = np.any(pairwise_dists < distance_threshold, axis=1)

    return query_atom_array[interface_atom_mask]
