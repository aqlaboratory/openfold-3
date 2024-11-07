import hashlib
import itertools
import logging
from collections import defaultdict

import networkx as nx
import numpy as np
from biotite.structure import AtomArray, chain_iter, molecule_iter

from openfold3.core.data.primitives.structure.component import component_iter
from openfold3.core.data.primitives.structure.labels import (
    assign_atom_indices,
    remove_atom_indices,
)

logger = logging.getLogger(__name__)


# TODO: put this function somewhere else for better discoverability?
def get_cross_chain_bonds(atom_array):
    """Gets all bonds between different chains in the atom array."""
    all_bonds = atom_array.bonds.as_array()

    chain_ids_atom_1 = atom_array.chain_id[all_bonds[:, 0]]
    chain_ids_atom_2 = atom_array.chain_id[all_bonds[:, 1]]
    cross_chain_selector = chain_ids_atom_1 != chain_ids_atom_2

    return all_bonds[cross_chain_selector]


def hash_bytes(input_bytes: bytes) -> str:
    hash_object = hashlib.sha256()
    hash_object.update(input_bytes)

    return hash_object.hexdigest()


# TODO: make this consistent with single_comparison
def construct_coarse_molecule_graph(atom_array):
    g = nx.Graph()

    # Use non-standard _atom_idx label to avoid collisions
    assign_atom_indices(atom_array, label="_atom_idx_g")

    # Construct nodes for each chain
    for chain in chain_iter(atom_array):
        # Fully identify a chain by its atom names, residue names, entity ID, and
        # cross-chain bonds. The later-defined symmetry finder will only consider
        # molecules as equivalent to each other if they match in all these features.
        chain_repr = hash_bytes(
            chain.atom_name.tobytes()
            + chain.res_name.tobytes()
            + chain.entity_id.tobytes()
            + get_cross_chain_bonds(chain).tobytes()
        )

        chain_id = chain.chain_id[0]
        g.add_node(f"chain: {chain_id}", node_repr=chain_repr)

    all_bonds = atom_array.bonds.as_array()

    chain_ids_atom_1 = atom_array.chain_id[all_bonds[:, 0]]
    chain_ids_atom_2 = atom_array.chain_id[all_bonds[:, 1]]
    cross_chain_selector = chain_ids_atom_1 != chain_ids_atom_2

    cross_chain_bonds = all_bonds[cross_chain_selector]

    # Construct explicit labeled nodes for each bonded atom, because vf2pp doesn't
    # support edge-labels
    for bond in cross_chain_bonds:
        atom_1_idx, atom_2_idx, _ = bond
        chain_id_1 = atom_array.chain_id[atom_1_idx]
        chain_id_2 = atom_array.chain_id[atom_2_idx]

        # Make atom indices relative to chain, not whole atom array
        chain_1_first_index = atom_array._atom_idx_g[atom_array.chain_id == chain_id_1][
            0
        ]
        chain_2_first_index = atom_array._atom_idx_g[atom_array.chain_id == chain_id_2][
            0
        ]

        atom_1_idx_rel = atom_1_idx - chain_1_first_index
        atom_2_idx_rel = atom_2_idx - chain_2_first_index

        # Add atom-nodes to both chains
        atom_node_ids = []

        for atom_idx_rel, chain in (
            (atom_1_idx_rel, chain_id_1),
            (atom_2_idx_rel, chain_id_2),
        ):
            node_id = f"link: {chain}_{atom_idx_rel}"
            atom_node_ids.append(node_id)

            if node_id not in g.nodes:
                g.add_node(node_id, node_repr=atom_idx_rel)

            g.add_edge(f"chain: {chain}", node_id)

        # Add edge between atom-nodes
        g.add_edge(atom_node_ids[0], atom_node_ids[1])

    atom_array.del_annotation("_atom_idx_g")

    return g


def single_comparison(mol1, mol2, with_breakpoint=False):
    """Matches the atom names, residue names and the bond-list."""

    entity_id_atom_names_1 = np.stack(
        [
            mol1.entity_id,
            mol1.atom_name,
            mol1.res_name,
        ],
    )

    entity_id_atom_names_2 = np.stack(
        [
            mol2.entity_id,
            mol2.atom_name,
            mol2.res_name,
        ],
    )

    if not np.array_equal(entity_id_atom_names_1, entity_id_atom_names_2):
        return False

    # Explicitly compare cross-chain bonds only
    cross_chain_bond_list_1 = get_cross_chain_bonds(mol1)
    cross_chain_bond_list_2 = get_cross_chain_bonds(mol2)

    return np.array_equal(cross_chain_bond_list_1, cross_chain_bond_list_2)


def get_mol_groups(atom_array):
    """Groups molecules in the atom array by entity IDs and length.

    Args:
        atom_array (AtomArray):
            The atom array to group.

    Returns:
        dict: A dictionary mapping ((entity_ids), len) to
            corresponding molecule atom_array slices.
    """
    mol_groups = defaultdict(list)

    for mol in molecule_iter(atom_array):
        entity_ids = tuple(np.unique(mol.entity_id))
        mol_len = len(mol)
        group_id = (entity_ids, mol_len)

        mol_groups[group_id].append(mol)

    return mol_groups


def assign_mol_symmetry_ids(atom_array):
    """This assigns an entity ID that takes the whole molecule into account.

    In contrast to normal PDB entity IDs, this parses entire molecules (= covalently
    connected components) in the structure and assigns the same identifier if the entire
    molecule is the same. For example, this will group together all covalently linked
    glycans with their receptor chains under one single identifier.
    """

    # Set atom-wise indices to keep track of the original order
    assign_atom_indices(atom_array)

    # Create a new annotation for the perm_entity_id
    atom_array.set_annotation(
        "perm_entity_id", np.zeros(atom_array.array_length(), dtype=int)
    )
    atom_array.set_annotation(
        "perm_sym_id", np.zeros(atom_array.array_length(), dtype=int)
    )

    # Dict mapping ((entity_ids), len) to the molecule atom_array slices. All molecules
    # in these groups share the same set of PDB entity IDs and have the same length, and
    # are therefore very likely to belong to the same molecule
    mol_groups = get_mol_groups(atom_array)

    # Subset to only groups with at least one molecule in the crop
    if "crop_mask" not in atom_array.get_annotation_categories():
        raise ValueError("AtomArray does not have a crop_mask attribute.")

    for group, mols in mol_groups.items():
        if not any(mol.crop_mask.any() for mol in mols):
            mol_groups.pop(group)

    # Counter for the permutation entity IDs
    perm_id_counter = itertools.count(1)

    # Builds up a sort operation at the end of the function so that chains in symmetric
    # entities are ordered the same way
    resort_index = atom_array._atom_idx.copy()

    # Iterate through the groups and verify which molecules are truly the same
    for grouped_mols in mol_groups.values():
        logger.debug("Processing group...")
        first_mol = grouped_mols[0]

        # Generate a graph capturing the layout of the molecule
        first_mol_graph = construct_coarse_molecule_graph(first_mol)

        # Mols representing unique perm_entities, formatted as
        # (mol, nx_graph, perm_entity_id, total count of identical mols)
        first_mol_perm_entity_id = next(perm_id_counter)
        repr_mol_info = [[first_mol, first_mol_graph, first_mol_perm_entity_id, 1]]

        # Set IDs for first molecule
        atom_array.perm_entity_id[first_mol._atom_idx] = first_mol_perm_entity_id
        atom_array.perm_sym_id[first_mol._atom_idx] = 1  # always first instance

        for mol in grouped_mols[1:]:
            # Get graph capturing layout of query molecule
            mol_graph = construct_coarse_molecule_graph(mol)

            for repr_info in repr_mol_info:
                repr_mol, repr_graph, first_mol_perm_entity_id, count = repr_info

                # Try identity mapping
                if single_comparison(repr_mol, mol):
                    logger.debug("Found match.")
                    # Found a match
                    atom_array.perm_entity_id[mol._atom_idx] = first_mol_perm_entity_id
                    atom_array.perm_sym_id[mol._atom_idx] = count
                    repr_info[2] += 1
                    break

                # Attempt permuting the molecule to find a match
                else:
                    logger.debug("Did not find direct match. Attempting permutation...")

                    # Check if two molecules are identical after permutation
                    mapping = nx.algorithms.isomorphism.vf2pp_isomorphism(
                        repr_graph, mol_graph, node_label="node_repr"
                    )

                    if mapping is not None:
                        logger.debug("Found match after permutation.")

                        # TODO: add code to rearrange atom indices
                        for node_1, node_2 in mapping.items():
                            if node_1.startswith("chain:") and node_2.startswith(
                                "chain:"
                            ):
                                node_1_chain_id = node_1.replace("chain: ", "")
                                node_2_chain_id = node_2.replace("chain: ", "")

                                # Append atom indices to the resort index with the
                                # chains swapped, so that the resorting operation will
                                # effectively permute their order
                                node_1_atom_indices = atom_array._atom_idx[
                                    atom_array.chain_id == node_1_chain_id
                                ]
                                node_2_atom_indices = atom_array._atom_idx[
                                    atom_array.chain_id == node_2_chain_id
                                ]

                                resort_index[node_1_atom_indices] = node_2_atom_indices
                                resort_index[node_2_atom_indices] = node_1_atom_indices

                        atom_array.perm_entity_id[mol._atom_idx] = (
                            first_mol_perm_entity_id
                        )
                        atom_array.perm_sym_id[mol._atom_idx] = count
                        repr_info[2] += 1

                        break

            else:
                logger.debug("No match found after permutation, adding as new entity.")

                # If there is no symmetry group to map to, add this molecule as a new
                # representative with a distinct perm_entity_id
                new_perm_entity_id = next(perm_id_counter)
                repr_mol_info.append([mol, mol_graph, new_perm_entity_id, 1])
                atom_array.perm_entity_id[mol._atom_idx] = new_perm_entity_id
                atom_array.perm_sym_id[mol._atom_idx] = 1  # always first instance

    # Reorder the entire atom_array so that all symmetric molecules share the same exact
    # order of atoms
    atom_array = atom_array[resort_index]

    remove_atom_indices(atom_array)

    assert np.all(atom_array.perm_entity_id != 0)
    assert np.all(atom_array.perm_sym_id != 0)


def perm_sym_id_mol_iter(atom_array: AtomArray):
    """
    Returns atom_array slices corresponding to every (perm_entity_id, perm_sym_id) pair.
    """
    entity_ids = np.unique(atom_array.perm_entity_id)

    for entity_id in entity_ids:
        sym_ids = np.unique(
            atom_array.perm_sym_id[atom_array.perm_entity_id == entity_id]
        )

        for sym_id in sym_ids:
            yield atom_array[
                (atom_array.perm_entity_id == entity_id)
                & (atom_array.perm_sym_id == sym_id)
            ]


def assign_perm_sym_atom_indices(atom_array: AtomArray):
    assign_atom_indices(atom_array)
    atom_array.add_annotation("perm_sym_atom_idx", -np.ones_like(atom_array._atom_idx))

    for mol in perm_sym_id_mol_iter(atom_array):
        atom_index = np.arange(len(mol))
        atom_array.perm_sym_atom_idx[mol._atom_idx] = atom_index

    remove_atom_indices(atom_array)

    assert np.all(atom_array.perm_sym_atom_idx != -1)


def assign_perm_sym_conformer_ids(atom_array: AtomArray):
    assign_atom_indices(atom_array)
    atom_array.add_annotation(
        "perm_sym_conformer_id", -np.ones_like(atom_array._atom_idx)
    )

    for mol_array in perm_sym_id_mol_iter(atom_array):
        for id, component in enumerate(component_iter(mol_array), start=1):
            atom_array.perm_sym_conformer_id[component._atom_idx] = id

    remove_atom_indices(atom_array)

    assert np.all(atom_array.perm_sym_conformer_id != -1)


def assign_mol_permutation_ids_and_subset(atom_array: AtomArray) -> AtomArray:
    atom_array = atom_array.copy()

    # Add the perm_entity_id and perm_sym_id annotations
    assign_mol_symmetry_ids(atom_array)

    # Subset the atom array to only contain molecules symmetric to the cropped molecules
    in_crop_entity_ids = np.unique(atom_array.perm_entity_id[atom_array.crop_mask])
    assert 0 not in in_crop_entity_ids, "0 is not a valid perm_entity_id"
    atom_array = atom_array[np.isin(atom_array.perm_entity_id, in_crop_entity_ids)]

    # Assign perm_sym_atom_idx attribute (renumbered atom indices for every molecule
    # instance
    assign_perm_sym_atom_indices(atom_array)

    # Assign perm_sym_conformer_id attribute (renumbered conformer IDs (ref_space_uids)
    # for every molecule instance)
    assign_perm_sym_conformer_ids(atom_array)

    return atom_array
