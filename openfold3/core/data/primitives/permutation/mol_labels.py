import hashlib
import itertools
import logging
from collections import defaultdict

import networkx as nx
import numpy as np
from biotite.structure import AtomArray, chain_iter, molecule_iter

from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.conformer import renumber_permutations
from openfold3.core.data.primitives.structure.labels import (
    assign_atom_indices,
    component_iter,
    get_token_starts,
    remove_atom_indices,
)

logger = logging.getLogger(__name__)


# TODO: Is this used anywhere? Remove if not
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


def single_comparison(mol1, mol2):
    """Matches the atom names, residue names and entity IDs."""

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


# TODO: Make repr_mol_info and mol_group some special typed datastructure
def assign_mol_symmetry_ids(atom_array) -> AtomArray:
    """This assigns an entity ID that takes the whole molecule into account.

    In contrast to normal PDB entity IDs, this parses entire molecules (= covalently
    connected components) in the structure and assigns the same identifier if the entire
    molecule is the same. For example, this will group together all covalently linked
    glycans with their receptor chains under one single identifier.
    """

    atom_array = atom_array.copy()

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

                # TODO: If I'm already creating the graph, why not check networkx graph
                # equivalence here as well and get rid of the single_comparison
                # function?
                # Try identity mapping
                if single_comparison(repr_mol, mol):
                    logger.debug("Found match.")
                    # Increment count
                    sym_id = count + 1
                    repr_info[3] = sym_id

                    # Set IDs
                    atom_array.perm_entity_id[mol._atom_idx] = first_mol_perm_entity_id
                    atom_array.perm_sym_id[mol._atom_idx] = sym_id
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

                        # Get the chains in the reference molecule in their order of
                        # appearance
                        unique_repr_chain_idx = np.unique(
                            repr_mol.chain_id, return_index=True
                        )[1]
                        unique_repr_chains = repr_mol.chain_id[
                            np.sort(unique_repr_chain_idx)
                        ]

                        # Extract the chain mappings from the vf2pp mapping
                        chain_mappings = {
                            node_1.replace("chain: ", ""): node_2.replace("chain: ", "")
                            for node_1, node_2 in mapping.items()
                            if node_1.startswith("chain:")
                            and node_2.startswith("chain:")
                        }

                        # Will build up an index that sorts the chains within the mol to
                        # the order matching the reference mol
                        within_mol_resort_index = []

                        for chain_id_repr in unique_repr_chains:
                            # Get the corresponding chain of the current mol
                            chain_id_mol = chain_mappings[chain_id_repr]

                            # Append its atom indices to the resort index
                            within_mol_resort_index.extend(
                                mol._atom_idx[mol.chain_id == chain_id_mol].tolist()
                            )

                        # Add the reordering operations for this mol to the global
                        # resort index
                        resort_index[mol._atom_idx] = within_mol_resort_index

                        # Increment count
                        sym_id = count + 1
                        repr_info[3] = sym_id

                        # Set IDs
                        atom_array.perm_entity_id[mol._atom_idx] = (
                            first_mol_perm_entity_id
                        )
                        atom_array.perm_sym_id[mol._atom_idx] = sym_id
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

    return atom_array


def perm_sym_entity_iter(atom_array: AtomArray):
    """
    Returns atom_array slices corresponding to every perm_entity_id.

    WARNING: The order with which the slices are returned may be different from the
    order of first appearance in the AtomArray.
    """
    entity_ids = np.unique(atom_array.perm_entity_id)

    for entity_id in entity_ids:
        yield atom_array[atom_array.perm_entity_id == entity_id]


def perm_sym_mol_iter(atom_array: AtomArray):
    """
    Returns atom_array slices corresponding to every (perm_entity_id, perm_sym_id) pair.

    WARNING: The order with which the slices are returned may be different from the
    order of first appearance in the AtomArray.
    """
    for entity in perm_sym_entity_iter(atom_array):
        sym_ids = np.unique(entity.perm_sym_id)

        for sym_id in sym_ids:
            yield entity[entity.perm_sym_id == sym_id]


def assign_perm_sym_token_indices(atom_array: AtomArray):
    assign_atom_indices(atom_array)

    atom_array.set_annotation(
        "perm_sym_token_index", -np.ones(len(atom_array), dtype=int)
    )

    # Go through each symmetric mol and renumber the token indices from 1
    for mol in perm_sym_mol_iter(atom_array):
        token_starts = get_token_starts(mol, add_exclusive_stop=True)
        token_id_repeats = np.diff(token_starts)
        token_indices_renumbered = np.repeat(
            np.arange(len(token_id_repeats)), token_id_repeats
        )
        atom_array.perm_sym_token_index[mol._atom_idx] = token_indices_renumbered

    remove_atom_indices(atom_array)

    assert np.all(atom_array.perm_sym_token_index != -1)


def assign_perm_sym_conformer_ids(atom_array: AtomArray):
    assign_atom_indices(atom_array)

    atom_array.set_annotation(
        "perm_sym_conformer_id", -np.ones(len(atom_array), dtype=int)
    )

    for mol_array in perm_sym_mol_iter(atom_array):
        for id, component in enumerate(component_iter(mol_array), start=1):
            atom_array.perm_sym_conformer_id[component._atom_idx] = id

    remove_atom_indices(atom_array)

    assert np.all(atom_array.perm_sym_conformer_id != -1)


@log_runtime_memory(runtime_dict_key="")
def assign_mol_permutation_ids(atom_array: AtomArray) -> AtomArray:
    atom_array = atom_array.copy()

    # Add the perm_entity_id and perm_sym_id annotations
    atom_array = assign_mol_symmetry_ids(atom_array)

    # Assign perm_sym_token_index attribute (renumbered token indices for every molecule
    # instance)
    assign_perm_sym_token_indices(atom_array)

    # Assign perm_sym_conformer_id attribute (renumbered conformer IDs (ref_space_uids)
    # for every molecule instance)
    assign_perm_sym_conformer_ids(atom_array)

    return atom_array


def separate_cropped_and_gt(
    atom_array_gt: AtomArray,
    crop_strategy: str,
    processed_ref_mol_list: list[ProcessedReferenceMolecule],
) -> tuple[AtomArray, AtomArray]:
    if not all(
        annotation in atom_array_gt.get_annotation_categories()
        for annotation in [
            "perm_entity_id",
            "perm_sym_id",
            "perm_sym_conformer_id",
        ]
    ):
        raise ValueError("Permutation labels not found in atom array.")

    if "crop_mask" not in atom_array_gt.get_annotation_categories():
        raise ValueError("AtomArray does not have a crop_mask attribute.")

    # Store all atom indices that are symmetry-related to atoms in the crop
    keep_atom_indices = set()

    assign_atom_indices(atom_array_gt)

    # Apply the crop
    atom_array_cropped = atom_array_gt[atom_array_gt.crop_mask].copy()

    # Map component indices of components in the crop to relevant permutations
    component_id_to_permutations = {
        processed_mol.component_id: processed_mol.permutations
        for processed_mol in processed_ref_mol_list
    }

    # Defines the set of sym IDs to which symmetry-equivalence is restricted
    entity_to_valid_sym_ids = defaultdict(list)

    for mol in perm_sym_mol_iter(atom_array_gt):
        # For spatial crops, restrict sym IDs to the in-crop ones
        if crop_strategy in ["spatial", "spatial_interface"]:
            if not mol.crop_mask.any():
                continue
        # For contiguous crops, allow all sym IDs
        elif crop_strategy in ["contiguous", "whole"]:
            pass
        else:
            logger.warning(
                f"Unknown crop strategy: {crop_strategy}, expanding ground-truth to all"
                " symmetric molecules (equivalent to contiguous crop)."
            )

        entity_id = mol.perm_entity_id[0]
        sym_id = mol.perm_sym_id[0]
        entity_to_valid_sym_ids[entity_id].append(sym_id)

    # Keep exactly the sections of the ground-truth that are symmetry-related to
    # sections in the crop
    for cropped_entities in perm_sym_entity_iter(atom_array_cropped):
        entity_id = cropped_entities.perm_entity_id[0]

        # Get the exact symmetry-equivalent atom sets per component
        sym_component_id_to_required_gt_atoms = defaultdict(set)
        for sym_mol in perm_sym_mol_iter(cropped_entities):
            for component in component_iter(sym_mol):
                absolute_component_id = component.component_id[0]
                sym_component_id = component.perm_sym_conformer_id[0]

                permutations = component_id_to_permutations[absolute_component_id]
                required_gt_atom_indices = np.unique(permutations)

                sym_component_id_to_required_gt_atoms[sym_component_id].update(
                    required_gt_atom_indices.tolist()
                )

        # Get the valid symmetry-equivalent GT molecules
        same_entity_gt_mols = atom_array_gt[atom_array_gt.perm_entity_id == entity_id]
        valid_sym_ids = entity_to_valid_sym_ids[entity_id]
        sym_equivalent_gt_mols = same_entity_gt_mols[
            np.isin(same_entity_gt_mols.perm_sym_id, valid_sym_ids)
        ]

        # Get an arbitrary symmetry-equivalent GT molecule to construct the resulting
        # mask that can be equivalently applied to all symmetry-equivalent GT molecules
        sym_equivalent_gt_mol = next(perm_sym_mol_iter(sym_equivalent_gt_mols))
        gt_mol_keep_atom_mask = []

        for gt_component in component_iter(sym_equivalent_gt_mol):
            gt_sym_component_id = gt_component.perm_sym_conformer_id[0]

            if gt_sym_component_id not in sym_component_id_to_required_gt_atoms:
                gt_mol_keep_atom_mask.extend(np.zeros(len(gt_component), dtype=bool))
                continue

            # All atoms from this component that are required for symmetry permutations
            required_gt_atom_indices = np.array(
                list(sym_component_id_to_required_gt_atoms[gt_sym_component_id])
            )

            # All atoms in the component
            gt_component_relative_atom_indices = np.arange(len(gt_component))

            # Subset to only required atoms
            relative_keep_atom_mask = np.isin(
                gt_component_relative_atom_indices, required_gt_atom_indices
            )

            gt_mol_keep_atom_mask.extend(relative_keep_atom_mask.tolist())

        # Apply the mask to every symmetry-equivalent GT molecule to get the final atom
        # indices that need to be kept
        for mol in perm_sym_mol_iter(sym_equivalent_gt_mols):
            keep_atom_indices.update(mol._atom_idx[gt_mol_keep_atom_mask])

    # Renumber the permutations, as the ground-truth atoms are now a subset of the
    # previous ones so the indices need to be re-mapped
    for processed_mol in processed_ref_mol_list:
        processed_mol.permutations = renumber_permutations(processed_mol.permutations)

    # Construct the final atom array
    remove_atom_indices(atom_array_gt)
    atom_array_gt = atom_array_gt[sorted(keep_atom_indices)]

    remove_atom_indices(atom_array_cropped)

    return atom_array_cropped, atom_array_gt
