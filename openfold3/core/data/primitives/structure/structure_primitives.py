# TODO: rename this file to something more descriptive
# TODO: add module docstring

import logging
from typing import Generator, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFBlock, CIFFile
from scipy.spatial import KDTree

from openfold3.core.data.resources.tables import (
    NUCLEIC_ACID_PHOSPHATE_ATOMS,
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES,
    STANDARD_RNA_RESIDUES,
    MoleculeType,
)

from .metadata_extraction import get_entity_to_three_letter_codes_dict


def assign_renumbered_chain_ids(atom_array: AtomArray) -> None:
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


def assign_atom_indices(atom_array: AtomArray) -> None:
    """Assigns atom indices to the AtomArray

    Atom indices are a simple range from 0 to the number of atoms in the AtomArray which
    is used as a convenience feature. They are stored in the "_atom_idx" field of the
    AtomArray and meant to be used only temporarily within functions. Should be combined
    with `remove_atom_indices`.

    Args:
        atom_array:
            AtomArray containing the structure to assign atom indices to.
    """
    atom_array.set_annotation("_atom_idx", range(len(atom_array)))


def remove_atom_indices(atom_array: AtomArray) -> None:
    """Removes atom indices from the AtomArray

    Deletes the "_atom_idx" field from the AtomArray. This is meant to be used after
    temporary atom indices are no longer needed. Also see `assign_atom_indices`.

    Args:
        atom_array:
            AtomArray containing the structure to remove atom indices from.
    """
    atom_array.del_annotation("_atom_idx")


def update_author_to_pdb_labels(
    atom_array: AtomArray,
    use_author_res_id_if_missing: bool = True,
    auth_label_annotations: bool = True,
) -> None:
    """Changes labels in an author-assigned PDB structure to PDB-assigned labels.

    This assumes that the AtomArray contains author-assigned labels (e.g. auth_asym_id,
    ...) in the standard fields chain_id, res_id, res_name, and atom_name, and will
    replace them with the PDB-assigned label_asym_id, label_seq_id, label_comp_id, and
    label_atom_id.

    Args:
        atom_array:
            AtomArray containing the structure to change labels in.
        keep_res_id_if_nan:
            Whether to keep the author-assigned residue IDs if they are NaN in
            the PDB labels. This is important for correct bond record parsing/writing in
            Biotite. Defaults to True.
        auth_label_annotations:
            Whether to keep the original author-assigned labels as annotations in the
            AtomArray.

    """
    if auth_label_annotations:
        atom_array.set_annotation("auth_asym_id", atom_array.chain_id)
        atom_array.set_annotation("auth_seq_id", atom_array.res_id)
        atom_array.set_annotation("auth_comp_id", atom_array.res_name)
        atom_array.set_annotation("auth_atom_id", atom_array.atom_name)

    # Replace author-assigned IDs with PDB-assigned IDs
    atom_array.chain_id = atom_array.label_asym_id
    atom_array.res_name = atom_array.label_comp_id
    atom_array.atom_name = atom_array.label_atom_id

    # Set residue IDs to PDB-assigned IDs but fallback to author-assigned IDs if they
    # are not assigned (important for correct bond record parsing/writing)
    if use_author_res_id_if_missing:
        author_res_ids = atom_array.res_id
        pdb_res_ids = atom_array.label_seq_id
        merged_res_ids = np.where(
            pdb_res_ids == ".", author_res_ids, pdb_res_ids
        ).astype(int)
        atom_array.res_id = merged_res_ids


def assign_entity_ids(atom_array: AtomArray) -> None:
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


def assign_molecule_type_ids(atom_array: AtomArray) -> None:
    """Assigns molecule types to the AtomArray

    Assigns molecule type IDs to each chain based on its residue names. Possible
    molecule types are protein, RNA, DNA, and ligand. The molecule type is stored in the
    "molecule_type_id" field of the AtomArray.

    Args:
        atom_array:
            AtomArray containing the structure to assign molecule types to.
    """
    chain_start_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # Create molecule type annotation
    molecule_type_ids = np.zeros(len(atom_array), dtype=int)

    # Zip together chain starts and ends
    for chain_start, chain_end in zip(chain_start_idxs[:-1], chain_start_idxs[1:]):
        chain_array = atom_array[chain_start:chain_end]
        is_polymeric = struc.get_residue_count(chain_array) > 1
        residues_in_chain = set(chain_array.res_name)

        # Assign protein if polymeric and any standard protein residue is present
        if (residues_in_chain & set(STANDARD_PROTEIN_RESIDUES)) and is_polymeric:
            molecule_type_ids[chain_start:chain_end] = MoleculeType.PROTEIN

        # Assign RNA if polymeric and any standard RNA residue is present
        elif (residues_in_chain & set(STANDARD_RNA_RESIDUES)) and is_polymeric:
            molecule_type_ids[chain_start:chain_end] = MoleculeType.RNA

        # Assign DNA if polymeric and any standard DNA residue is present
        elif (residues_in_chain & set(STANDARD_DNA_RESIDUES)) and is_polymeric:
            molecule_type_ids[chain_start:chain_end] = MoleculeType.DNA

        # Assign ligand otherwise
        else:
            molecule_type_ids[chain_start:chain_end] = MoleculeType.LIGAND

    atom_array.set_annotation("molecule_type_id", molecule_type_ids)


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
    kdtree_query = KDTree(query_atom_array.coord)
    kdtree_target = KDTree(target_atom_array.coord)
    search_result = kdtree_query.query_ball_tree(kdtree_target, distance_threshold)

    # Get to same output format as kdtree.query_pairs
    atom_pair_idxs = np.array(
        [(i, j) for i, j_list in enumerate(search_result) for j in j_list]
    )

    # Pair the chain IDs
    chain_pairs = np.column_stack(
        (
            query_atom_array.chain_id_renumbered[atom_pair_idxs[:, 0]],
            target_atom_array.chain_id_renumbered[
                atom_pair_idxs[:, 1]
            ],  # change to target
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
    kdtree = KDTree(atom_array.coord)

    atom_pair_idxs = kdtree.query_pairs(distance_threshold, output_type="ndarray")

    # Pair the chain IDs
    chain_pairs = atom_array.chain_id_renumbered[atom_pair_idxs]

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

    # Get indices where the pair changes
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

    if "af3_token_center_atom" not in atom_array.get_annotation_categories():
        raise ValueError(
            "Token center atoms not found in atom array, run "
            "tokenize_atom_array first"
        )

    token_center_atoms = atom_array[atom_array.af3_token_center_atom]

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
    if "af3_token_center_atom" not in query_atom_array.get_annotation_categories():
        raise ValueError(
            "Token center atoms not found in query atom array, run "
            "tokenize_atom_array first"
        )
    elif "af3_token_center_atom" not in target_atom_array.get_annotation_categories():
        raise ValueError(
            "Token center atoms not found in target atom array, run "
            "tokenize_atom_array first"
        )

    query_token_centers = query_atom_array[query_atom_array.af3_token_center_atom]
    target_token_centers = target_atom_array[target_atom_array.af3_token_center_atom]

    return get_query_interface_atoms(
        query_token_centers, target_token_centers, distance_threshold
    )


def connect_residues(
    atom_array: AtomArray,
    res_id_1: int,
    res_id_2: int,
    chain: int,
    bond_type: Literal["peptide", "phosphate"],
) -> None:
    """Connects two residues in the AtomArray

    This function adds an appropriate bond between two residues in the AtomArray based
    on the canonical way of connecting residues in the given polymer type (peptide or
    nucleic acid).

    Args:
        atom_array:
            AtomArray containing the structure to add the bond to.
        res_id_1:
            Residue ID of the first (upstream) residue to connect. (e.g. will contribute
            the carboxylic carbon in a peptide bond)
        res_id_2:
            Residue ID of the second (downstream) residue to connect. (e.g. will
            contribute the amino nitrogen in a peptide bond)
    """
    # General masks
    chain_mask = atom_array.chain_id_renumbered == chain
    res_id_1_mask = (atom_array.res_id == res_id_1) & chain_mask
    res_id_2_mask = (atom_array.res_id == res_id_2) & chain_mask

    if bond_type == "peptide":
        # Get the index of the C atom in the first residue
        (res_id_1_atom_idx,) = np.where(res_id_1_mask & (atom_array.atom_name == "C"))[
            0
        ]

        # Equivalently get the index of the N atom in the second residue
        (res_id_2_atom_idx,) = np.where(res_id_2_mask & (atom_array.atom_name == "N"))[
            0
        ]

    # Analogously select nucleic acid atoms
    elif bond_type == "phosphate":
        (res_id_1_atom_idx,) = np.where(
            res_id_1_mask & (atom_array.atom_name == "O3'")
        )[0]

        (res_id_2_atom_idx,) = np.where(res_id_2_mask & (atom_array.atom_name == "P"))[
            0
        ]

    atom_array.bonds.add_bond(
        res_id_1_atom_idx, res_id_2_atom_idx, struc.BondType.SINGLE
    )


def build_unresolved_polymer_segment(
    residue_codes: list[str],
    ccd: CIFFile,
    polymer_type: Literal["protein", "nucleic_acid"],
    default_annotations: dict,
    terminal_start: bool = False,
    terminal_end: bool = False,
) -> AtomArray:
    """Builds a polymeric segment with unresolved residues

    This function builds a polymeric segment with unresolved residues based on a list of
    residue 3-letter codes that have matching entries in the Chemical Component
    Dictionary (CCD). The segment is built by adding all atoms of the unresolved
    residues to the AtomArray with dummy coordinates. The BondList of the resulting
    AtomArray is filled appropriately to contain both intra-residue and inter-residue
    bonds.

    Args:
        residue_codes:
            List of 3-letter residue codes of the unresolved residues.
        ccd:
            Parsed Chemical Component Dictionary (CCD) containing the residue
            information.
        polymer_type:
            Type of the polymer segment. Can be either "protein" or "nucleic_acid".
        default_annotations:
            Any annotations and custom annotations to be added to the atoms in the
            output AtomArray. All annotations are kept constant for all atoms in the
            segment except for res_id and _atom_idx (if present) which are incrementally
            increased appropriately (with the first atom/residue index matching the one
            in the default annotations), and occupancy which is set to 0.0.
        terminal_start:
            Whether the segment is the start of the overall sequence. For proteins this
            will have no effect but for nucleic acids the otherwise overhanging 5'
            phosphate is pruned.
        terminal_end:
            Whether the segment is the end of the overall sequence. For proteins this
            will keep the terminal oxygen, for nucleic acids the otherwise overhanging
            3' phosphate is pruned.
    """
    if polymer_type == "nucleic_acid":
        logging.info("Building unresolved nucleic acid segment!")  # dev-only: del later

    atom_idx_is_present = "_atom_idx" in default_annotations

    # Set occupancy to 0.0
    default_annotations["occupancy"] = 0.0

    atom_list = []

    last_residue_idx = len(residue_codes) - 1

    added_atoms = 0

    for added_residues, residue_code in enumerate(residue_codes):
        atom_names = ccd[residue_code]["chem_comp_atom"]["atom_id"].as_array()
        atom_elements = ccd[residue_code]["chem_comp_atom"]["type_symbol"].as_array()

        # Exclude hydrogens
        hydrogen_mask = atom_elements != "H"

        if polymer_type == "nucleic_acid":
            # Prune 5' phosphate if segment is nucleic acid and segment-start is overall
            # sequence start to prevent overhanging phosphates
            if terminal_start and added_residues == 0:
                phosphate_mask = ~np.isin(atom_names, NUCLEIC_ACID_PHOSPHATE_ATOMS)
                atom_mask = hydrogen_mask & phosphate_mask

            # Always prune terminal phosphate-oxygens because they would only need to be
            # kept in overhanging phosphates which we remove above (in-sequence
            # connecting phosphate-oxygens are annotated as O3' and kept)
            else:
                po3_mask = ~np.isin(atom_names, ["OP3", "O3P"])
                atom_mask = hydrogen_mask & po3_mask

        elif polymer_type == "protein":
            # If segment is protein and segment-end is overall sequence end, do not
            # exclude terminal oxygen
            if terminal_end and added_residues == last_residue_idx:
                atom_mask = hydrogen_mask
            # For any other protein residue, exclude terminal oxygen
            else:
                oxt_mask = atom_names != "OXT"
                atom_mask = hydrogen_mask & oxt_mask

        atom_names = atom_names[atom_mask]
        atom_elements = atom_elements[atom_mask]

        # Add atoms for all unresolved residues
        for atom, element in zip(atom_names, atom_elements):
            atom_annotations = default_annotations.copy()
            atom_annotations["atom_name"] = atom
            atom_annotations["element"] = element
            atom_annotations["res_name"] = residue_code

            base_res_id = default_annotations["res_id"]
            atom_annotations["res_id"] = base_res_id + added_residues

            # Avoid error if _atom_idx is not set
            if atom_idx_is_present:
                base_atom_idx = default_annotations["_atom_idx"]
                atom_annotations["_atom_idx"] = base_atom_idx + added_atoms

            # Append unresolved atom explicitly but with dummy coordinates
            atom_list.append(struc.Atom([np.nan, np.nan, np.nan], **atom_annotations))

            added_atoms += 1

    segment_atom_array = struc.array(atom_list)

    # build standard connectivities
    bond_list = struc.connect_via_residue_names(segment_atom_array)
    segment_atom_array.bonds = bond_list

    return segment_atom_array


def shift_up_atom_indices(
    atom_array: AtomArray, shift: int, idx_threshold: int
) -> None:
    """Shifts all atom indices higher than a threshold by a certain amount

    Args:
        atom_array:
            AtomArray containing the structure to shift atom indices in.
        shift:
            Amount by which to shift the atom indices.
        idx_threshold:
            Threshold index above which to shift the atom indices.
    """
    # Update atom indices for all atoms greater than the given atom index
    update_mask = atom_array._atom_idx > idx_threshold
    atom_array._atom_idx[update_mask] += shift


def add_unresolved_polymer_residues(
    atom_array: AtomArray,
    cif_data: CIFBlock,
    ccd: CIFFile,
) -> AtomArray:
    """Adds all missing polymer residues to the AtomArray

    Missing residues are added to the AtomArray explicitly with dummy NaN coordinates
    and the full atom annotations and bonding patterns. This is useful for contiguous
    cropping or inferring the whole sequence of a polymer chain.

    Args:
        atom_array:
            AtomArray containing the structure to add missing residues to.
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see
            `metadata_extraction.get_cif_block`)
        ccd:
            Parsed Chemical Component Dictionary (CCD) containing the residue
            information.
    """
    # Three-letter residue codes of all monomers in the entire sequence
    entity_id_to_3l_seq = get_entity_to_three_letter_codes_dict(cif_data)

    # This will be extended with unresolved residues
    extended_atom_array = atom_array.copy()

    # Assign temporary atom indices
    assign_atom_indices(extended_atom_array)

    # Iterate through all chains and fill missing residues. To not disrupt the bondlist
    # by slicing the atom_array for an insertion operation, this appends all the missing
    # residue atoms at the end of the atom_array but keeps appropriate bookkeeping of
    # the atom indices. That way the entire array can be sorted in a single
    # reindexing at the end without losing any bond information.
    chain_starts = struc.get_chain_starts(extended_atom_array, add_exclusive_stop=True)
    for chain_start, chain_end in zip(chain_starts[:-1], chain_starts[1:]):
        # Infer some chain-wise properties from first atom (could use any atom)
        first_atom = extended_atom_array[chain_start]
        chain_type = first_atom.molecule_type_id
        chain_entity_id = first_atom.entity_id
        chain_id = first_atom.chain_id_renumbered

        # Only interested in polymer chains
        if chain_type == MoleculeType.LIGAND:
            continue
        else:
            if chain_type == MoleculeType.PROTEIN:
                polymer_type = "protein"
                bond_type = "peptide"
            elif chain_type in (MoleculeType.RNA, MoleculeType.DNA):
                polymer_type = "nucleic_acid"
                bond_type = "phosphate"
            else:
                raise ValueError(f"Unknown molecule type: {chain_type}")

        # Three-letter residue codes of the full chain
        chain_3l_seq = entity_id_to_3l_seq[chain_entity_id]

        # Atom annotations of first atom (as template for unresolved residue
        # annotations while updating res_id and atom_idx)
        template_annotations = first_atom._annot.copy()
        template_annotations["ins_code"] = ""

        ## Fill missing residues at chain start
        if first_atom.res_id > 1:
            n_missing_residues = first_atom.res_id - 1

            default_annotations = template_annotations.copy()
            default_annotations["res_id"] = 1
            default_annotations["_atom_idx"] = first_atom._atom_idx

            segment = build_unresolved_polymer_segment(
                residue_codes=chain_3l_seq[:n_missing_residues],
                ccd=ccd,
                polymer_type=polymer_type,
                default_annotations=default_annotations,
                terminal_start=True,
                terminal_end=False,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            shift_up_atom_indices(
                extended_atom_array,
                n_added_atoms,
                idx_threshold=first_atom._atom_idx - 1,
            )
            extended_atom_array += segment

            # Connect residues in BondList
            connect_residues(
                extended_atom_array,
                first_atom.res_id - 1,
                first_atom.res_id,
                chain=chain_id,
                bond_type=bond_type,
            )

        ## Fill missing residues at chain end
        last_atom = extended_atom_array[chain_end]
        full_seq_length = len(entity_id_to_3l_seq[chain_entity_id])

        if last_atom.res_id < full_seq_length:
            n_missing_residues = full_seq_length - last_atom.res_id

            default_annotations = template_annotations.copy()
            default_annotations["res_id"] = last_atom.res_id + 1
            default_annotations["_atom_idx"] = last_atom._atom_idx + 1

            segment = build_unresolved_polymer_segment(
                residue_codes=chain_3l_seq[-n_missing_residues:],
                ccd=ccd,
                polymer_type=polymer_type,
                default_annotations=default_annotations,
                terminal_start=False,
                terminal_end=True,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            shift_up_atom_indices(
                extended_atom_array, n_added_atoms, idx_threshold=last_atom._atom_idx
            )
            extended_atom_array += segment

            # Connect residues in BondList
            connect_residues(
                extended_atom_array,
                last_atom.res_id,
                last_atom.res_id + 1,
                chain=chain_id,
                bond_type=bond_type,
            )

        # Gaps between consecutive residues
        res_id_gaps = np.diff(extended_atom_array.res_id[chain_start : chain_end + 1])
        chain_break_start_idxs = np.where(res_id_gaps > 1)[0] + chain_start

        ## Fill missing residues within the chain
        for chain_break_start_idx in chain_break_start_idxs:
            chain_break_end_idx = chain_break_start_idx + 1
            break_start_atom = extended_atom_array[chain_break_start_idx]
            break_end_atom = extended_atom_array[chain_break_end_idx]

            n_missing_residues = break_end_atom.res_id - break_start_atom.res_id - 1

            default_annotations = template_annotations.copy()
            default_annotations["res_id"] = break_start_atom.res_id + 1
            default_annotations["_atom_idx"] = break_start_atom._atom_idx + 1

            # Residue IDs start with 1 so the indices are offset
            segment_residue_codes = chain_3l_seq[
                break_start_atom.res_id : break_end_atom.res_id - 1
            ]

            segment = build_unresolved_polymer_segment(
                residue_codes=segment_residue_codes,
                ccd=ccd,
                polymer_type=polymer_type,
                default_annotations=default_annotations,
                terminal_start=False,
                terminal_end=False,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            shift_up_atom_indices(
                extended_atom_array,
                n_added_atoms,
                idx_threshold=break_end_atom._atom_idx - 1,
            )
            extended_atom_array += segment

            # Connect residues at start and end of added segment
            connect_residues(
                extended_atom_array,
                break_start_atom.res_id,
                break_start_atom.res_id + 1,
                chain=chain_id,
                bond_type=bond_type,
            )
            connect_residues(
                extended_atom_array,
                break_end_atom.res_id - 1,
                break_end_atom.res_id,
                chain=chain_id,
                bond_type=bond_type,
            )

    # Finally reorder the array so that the atom indices are in order
    extended_atom_array = extended_atom_array[np.argsort(extended_atom_array._atom_idx)]

    # dev-only: TODO remove
    assert np.array_equal(
        extended_atom_array._atom_idx, np.arange(len(extended_atom_array))
    )

    # Remove temporary atom indices
    remove_atom_indices(extended_atom_array)

    return extended_atom_array


# TODO: not sure yet if this function is actually necessary
def add_canonical_one_letter_codes(
    atom_array: AtomArray,
    cif_data: CIFBlock,
) -> None:
    # Get canonical sequences for each entity
    polymer_entities = cif_data["entity_poly"]["entity_id"].as_array(dtype=int)
    polymer_canonical_seqs = cif_data["entity_poly"][
        "pdbx_seq_one_letter_code_can"
    ].as_array()
    polymer_canonical_seqs = np.char.replace(polymer_canonical_seqs, "\n", "")

    entity_id_to_seq = dict(zip(polymer_entities, polymer_canonical_seqs))

    # Add new annotation category for one-letter codes
    atom_array.set_annotation(
        "one_letter_code_can", np.empty(len(atom_array), dtype="U1")
    )

    # Set one-letter codes for each residue
    for entity_id, seq in entity_id_to_seq.items():
        entity_mask = atom_array.entity_id == entity_id
        entity_array = atom_array[entity_mask]

        n_seq_repetitions = len(np.unique(entity_array.chain_id_renumbered))
        seqs_repeated = seq * n_seq_repetitions

        if len(seqs_repeated) != struc.get_residue_count(entity_array):
            raise ValueError(
                "Sequence length does not match the number of residues in the entity. "
                "Make sure to run add_unresolved_polymer_residues first if the "
                "structure contains unresolved residues."
            )

        atom_wise_seqs = struc.spread_residue_wise(
            atom_array[entity_mask], np.array(list(seq * n_seq_repetitions), dtype="U1")
        )

        atom_array.one_letter_code_can[entity_mask] = atom_wise_seqs

    return atom_array
