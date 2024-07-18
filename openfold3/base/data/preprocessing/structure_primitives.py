import biotite.structure as struc
import numpy as np
from scipy.spatial.distance import cdist

from .tables import (
    MOLECULE_TYPE_ID_DNA,
    MOLECULE_TYPE_ID_LIGAND,
    MOLECULE_TYPE_ID_PROTEIN,
    MOLECULE_TYPE_ID_RNA,
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES,
    STANDARD_RNA_RESIDUES,
)


def assign_renumbered_chain_ids(atom_array: struc.AtomArray) -> None:
    """Adds a renumbered chain index to the AtomArray

    Iterates through all chains in the atom array and assigns unique numerical chain IDs
    starting with 0 to each chain in the "chain_id_renumbered" field. This is useful for
    bioassembly parsing where chain IDs can be duplicated after the assembly is
    expanded.

    Args:
        atom_array:
            AtomArray containing the structure to assign renumbered chain IDs to.
    """
    chain_start_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # Assign numerical chain IDs
    chain_id_n_repeats = np.diff(chain_start_idxs)
    chain_ids_per_atom = np.repeat(
        np.arange(len(chain_id_n_repeats)), chain_id_n_repeats
    )
    atom_array.set_annotation("chain_id_renumbered", chain_ids_per_atom)


def assign_atom_indices(atom_array: struc.AtomArray) -> None:
    """Assigns atom indices to the AtomArray

    Atom indices are a simple range from 0 to the number of atoms in the AtomArray which
    is used as a convenience feature.

    Args:
        atom_array:
            AtomArray containing the structure to assign atom indices to.
    """
    atom_array.set_annotation("atom_idx", range(len(atom_array)))


def assign_entity_ids(atom_array: struc.AtomArray) -> None:
    """Assigns entity IDs to the AtomArray

    Entity IDs are assigned to each chain in the AtomArray based on the
    "label_entity_id" field. The entity ID is stored in the "entity_id" field of the
    AtomArray.

    Args:
        atom_array:
            AtomArray containing the structure to assign entity IDs to.
    """
    # Cast entity IDs from string to int and shorten name
    atom_array.set_annotation("entity_id", atom_array.label_entity_id.astype(int))
    atom_array.del_annotation("label_entity_id")


def assign_molecule_types(atom_array: struc.AtomArray) -> None:
    """Assigns molecule types to the AtomArray

    Assigns molecule types to each chain based on its residue names. Possible molecule
    types are protein, RNA, DNA, and ligand. The molecule type is stored in the
    "molecule_type" field of the AtomArray.

    Args:
        atom_array:
            AtomArray containing the structure to assign molecule types to.
    """
    chain_start_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # Create molecule type annotation
    molecule_types = np.zeros(len(atom_array))

    # Zip together chain starts and ends
    for chain_start, chain_end in zip(chain_start_idxs[:-1], chain_start_idxs[1:]):
        residues_in_chain = set(atom_array[chain_start:chain_end].res_name)

        # Assign protein if any standard protein residue is present
        if residues_in_chain & set(STANDARD_PROTEIN_RESIDUES):
            molecule_types[chain_start:chain_end] = MOLECULE_TYPE_ID_PROTEIN

        # Assign RNA if any standard RNA residue is present
        elif residues_in_chain & set(STANDARD_RNA_RESIDUES):
            molecule_types[chain_start:chain_end] = MOLECULE_TYPE_ID_RNA

        # Assign DNA if any standard DNA residue is present
        elif residues_in_chain & set(STANDARD_DNA_RESIDUES):
            molecule_types[chain_start:chain_end] = MOLECULE_TYPE_ID_DNA

        # Assign ligand otherwise
        else:
            molecule_types[chain_start:chain_end] = MOLECULE_TYPE_ID_LIGAND

    atom_array.set_annotation("molecule_type", molecule_types)


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


def get_interface_token_center_atoms(
    query_atom_array: struc.AtomArray,
    target_atom_array: struc.AtomArray,
    distance_threshold: float = 15.0,
) -> struc.AtomArray:
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
    query_token_centers = query_atom_array[query_atom_array.af3_token_center_atom]
    target_token_centers = target_atom_array[target_atom_array.af3_token_center_atom]

    return get_interface_atoms(
        query_token_centers, target_token_centers, distance_threshold
    )
