import hashlib
import itertools
import logging
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from typing import NamedTuple

import networkx as nx
import numpy as np
from biotite.structure import AtomArray, chain_iter, get_chain_starts

import openfold3.core.data.resources.patches as patch
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
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array

logger = logging.getLogger(__name__)


def hash_bytes(input_bytes: bytes) -> str:
    """Hashes a byte string using SHA-256.

    Args:
        input_bytes (bytes):
            byte string to hash.

    Returns:
        str:
            SHA-256 hash of the input byte string.
    """
    hash_object = hashlib.sha256()
    hash_object.update(input_bytes)

    return hash_object.hexdigest()


def construct_coarse_molecule_graph(atom_array: AtomArray) -> nx.Graph:
    """Constructs a high-level graph representation of the molecule.

    This function creates a graph that summarizes the overall layout of a molecule using
    the following strategy:
        - Each chain is represented by a node. Each node gets an internal representation
          which is equivalent to an efficient SHA-256 hash of its internal atom names,
          residue names, and entity ID.
        - Each bond between two chains is represented by two nodes, one for each atom in
          the bond. These special bond-atom nodes get an internal representation
          equivalent to their atom index relative to the chain.

    Following the above strategy, each symmetry-equivalent covalently connected set of
    chains (e.g. a protein chain bound to its glycans and another symmetry-equivalent
    protein chain bound to equivalent glycans) will have an equivalent set of bond-atom
    nodes and chain nodes, and appear isomorphic in this graph representation.

    Args:
        atom_array (AtomArray):
            The atom array to encode as a graph.

    Returns:
        nx.Graph:
            A networkx graph object representing the molecule. Internal node_reprs
            follow the strategy described above. Chain-level nodes get a string node
            label formatted like "chain: {chain_id}", and bond-atom nodes get a string
            node label formatted like "link: {chain_id}_{atom_idx_relative}". The node
            labels are only for inspection convenience and should not be used for graph
            matching (as they differ between symmetric molecules), where the node_repr
            values should solely be used instead.
    """

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


class PrecursorMolGroupID(NamedTuple):
    """Unique precursor molecule group identifier.

    Attributes:
        entity_ids (tuple[int]):
            Unique sorted entity IDs in the group.
        mol_len (int):
            The atom count of the molecules in the group.
    """

    entity_ids: tuple[int]
    mol_len: int


def get_precursor_mol_groups(
    atom_array: AtomArray,
) -> dict[PrecursorMolGroupID, list[AtomArray]]:
    """Groups molecules in the atom array by entity IDs and length.

    This function groups molecules in the atom array by their entity IDs and length.
    This is a precursor step to assigning molecular symmetry IDs, as molecules
    containing the same set of entity IDs and having the same number of atoms are very
    likely to be symmetry-equivalent.

    Args:
        atom_array (AtomArray):
            The atom array to group.

    Returns:
        dict:
            A dictionary mapping ((entity_ids), len) to corresponding molecule
            atom_array slices.
    """
    mol_groups = defaultdict(list)

    for mol in patch.molecule_iter(atom_array):
        entity_ids = tuple(np.unique(mol.entity_id))
        mol_len = len(mol)
        group_id = PrecursorMolGroupID(entity_ids, mol_len)

        mol_groups[group_id].append(mol)

    return mol_groups


def naive_graph_match(g1: nx.Graph, g2: nx.Graph) -> bool:
    """Matches the graphs directly without any isomorphic graph matching.
    
    The checks here are specific to the high-level graph representation of molecules,
    and we only check the equivalence of the node_reprs (respecting the order of the
    nodes), as well as the adjacency matrix of the graphs.

    Args:
        g1 (nx.Graph):
            The first graph to match.
        g2 (nx.Graph):
            The second graph to match.

    Returns:
        bool:
            True if the graphs are equal, False otherwise.
    """
    # Check node_repr equivalence
    node_reprs_g1 = [g1.nodes[node]["node_repr"] for node in g1.nodes]
    node_reprs_g2 = [g2.nodes[node]["node_repr"] for node in g2.nodes]
    
    if node_reprs_g1 != node_reprs_g2:
        return False

    # Check adjacency matrix equivalence
    adj_matrix_g1 = nx.to_numpy_array(g1)
    adj_matrix_g2 = nx.to_numpy_array(g2)
    
    return np.array_equal(adj_matrix_g1, adj_matrix_g2)


@dataclass
class SymmetricMolGroup:
    """Representation of information for a group of symmetry-equivalent molecules.

    Attributes:
        repr_graph (nx.Graph):
            A representative networkx graph capturing the layout of the molecule.
        repr_chain_order (tuple[str]):
            A representative chain order for the molecule (corresponding to the first
            molecule). All other molecules in the group will be reordered to match this
            order.
        mol_entity_id (int):
            A unique identifier for the symmetric molecule group, referred to as
            molecular entity ID. This is different from the PDB entity ID in that it
            groups together all covalently connected components in the structure into a
            single molecule.
        n_symmetric_instances (int):
            The number of symmetry-equivalent instances of this molecule group.
    """

    repr_graph: nx.Graph
    repr_chain_order: tuple[str]
    mol_entity_id: int
    n_symmetric_instances: int = 1


def assign_mol_symmetry_ids(atom_array: AtomArray) -> AtomArray:
    """Assings molecular entity IDs and symmetry IDs to the atom array.

    Following 4.2 of the AF3 SI, covalently connected components must be treated as a
    single entity for the purpose of the chain permutation alignment. This raises the
    problem of identifying symmetry-equivalent connected components in the atom array.
    This function solves this by using isomorphic graph matching between coarse-grained
    graph representations of covalently connected molecules in the atom array. The graph
    representation is constructed by the construct_coarse_molecule_graph function.

    Args:
        atom_array (AtomArray):
            The atom array to assign molecular symmetry IDs to.

    Returns:
        AtomArray:
            The same atom array with two new annotations:
                - mol_entity_id:
                    A unique identifier for a group of symmetry-equivalent molecules,
                    referred to as molecular entity ID. All molecules with the same
                    molecular entity ID are fully symmetry-equivalent to each other.
                - mol_sym_id:
                    A unique identifier for each individual symmetry-equivalent molecule
                    within a symmetric group, starting from 1.
    """
    atom_array = atom_array.copy()

    # Set atom-wise indices to keep track of the original order
    assign_atom_indices(atom_array)

    # Create a new annotation for the mol_entity_id
    atom_array.set_annotation(
        "mol_entity_id", np.zeros(atom_array.array_length(), dtype=int)
    )
    atom_array.set_annotation(
        "mol_sym_id", np.zeros(atom_array.array_length(), dtype=int)
    )

    # Dict mapping ((entity_ids), len) to the molecule atom_array slices. All molecules
    # in these groups share the same set of PDB entity IDs and have the same length, and
    # are therefore very likely to belong to the same molecule
    mol_groups = get_precursor_mol_groups(atom_array)

    # Counter for the molecular entity IDs
    mol_entity_counter = itertools.count(start=1)

    # Builds up a sort operation at the end of the function so that chains in symmetric
    # entities are ordered the same way
    resort_index = atom_array._atom_idx.copy()

    # Iterate through the groups and verify which molecules are truly the same
    for grouped_mols in mol_groups.values():
        # Will group truly symmetric molecules together
        symmetric_mol_groups = []

        logger.debug("Processing group...")

        # Start with the first molecule to build an initial group of symmetric molecules
        # ------------------------------------------------
        first_mol = grouped_mols[0]

        # Generate a graph capturing the layout of the molecule
        first_mol_graph = construct_coarse_molecule_graph(first_mol)

        first_mol_entity_id = next(mol_entity_counter)
        first_mol_chain_order = tuple(first_mol.chain_id[get_chain_starts(first_mol)])

        symmetric_mol_groups.append(
            SymmetricMolGroup(
                repr_graph=first_mol_graph,
                repr_chain_order=first_mol_chain_order,
                mol_entity_id=first_mol_entity_id,
                n_symmetric_instances=1,
            )
        )

        # Set symmetry IDs for first molecule
        atom_array.mol_entity_id[first_mol._atom_idx] = first_mol_entity_id
        atom_array.mol_sym_id[first_mol._atom_idx] = 1  # always first instance
        # ------------------------------------------------

        # Iterate through all other molecules in this group, adding them to already
        # existing symmetry groups if they are symmetry-equivalent to the
        # representative, or creating new groups if they are not
        if len(grouped_mols) == 1:
            logger.debug("Only one molecule in group, advancing to next.")
            continue

        for mol in grouped_mols[1:]:
            # Get graph capturing layout of query molecule
            mol_graph = construct_coarse_molecule_graph(mol)

            # Check all existing symmetry groups for a match
            for symm_group in symmetric_mol_groups:
                # Try identity mapping
                if naive_graph_match(symm_group.repr_graph, mol_graph):
                    logger.debug("Found match.")

                    # Increment count
                    mol_sym_id = symm_group.n_symmetric_instances + 1
                    symm_group.n_symmetric_instances = mol_sym_id

                    # Set IDs
                    mol_entity_id = symm_group.mol_entity_id
                    atom_array.mol_entity_id[mol._atom_idx] = mol_entity_id
                    atom_array.mol_sym_id[mol._atom_idx] = mol_sym_id
                    break

                # Attempt permuting the molecule to find a match
                else:
                    logger.debug("Did not find direct match. Attempting isomorphism...")

                    # Check if two molecules are identical after permutation
                    mapping = nx.algorithms.isomorphism.vf2pp_isomorphism(
                        symm_group.repr_graph, mol_graph, node_label="node_repr"
                    )

                    # If isomorphism was found, add to group of same mol_entities and
                    # reorder the atoms so that their features match the representative
                    if mapping is not None:
                        logger.debug("Found match after permutation.")

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

                        for chain_id_repr in symm_group.repr_chain_order:
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
                        mol_sym_id = symm_group.n_symmetric_instances + 1
                        symm_group.n_symmetric_instances = mol_sym_id

                        # Set IDs
                        mol_entity_id = symm_group.mol_entity_id
                        atom_array.mol_entity_id[mol._atom_idx] = mol_entity_id
                        atom_array.mol_sym_id[mol._atom_idx] = mol_sym_id
                        break

            else:
                logger.debug("No match found after permutation, adding as new entity.")

                # If there is no symmetry group to map to, add this molecule as a new
                # representative with a distinct mol_entity_id
                new_mol_entity_id = next(mol_entity_counter)

                symmetric_mol_groups.append(
                    SymmetricMolGroup(
                        repr_graph=mol_graph,
                        repr_chain_order=tuple(mol.chain_id[get_chain_starts(mol)]),
                        mol_entity_id=new_mol_entity_id,
                        n_symmetric_instances=1,
                    )
                )

                atom_array.mol_entity_id[mol._atom_idx] = new_mol_entity_id
                atom_array.mol_sym_id[mol._atom_idx] = 1  # always first instance

    # Reorder the entire atom_array so that all symmetric molecules share the same exact
    # order of atoms
    atom_array = atom_array[resort_index]

    remove_atom_indices(atom_array)

    assert np.all(atom_array.mol_entity_id != 0)
    assert np.all(atom_array.mol_sym_id != 0)

    return atom_array


def mol_entity_iter(atom_array: AtomArray) -> Generator[AtomArray, None, None]:
    """Returns atom_array slices corresponding to every mol_entity_id.

    WARNING: The order with which the slices are returned may be different from the
    order of first appearance in the AtomArray.

    Args:
        atom_array (AtomArray):
            The atom array to iterate over.

    Yields:
        AtomArray:
            AtomArray slice corresponding to a unique mol_entity_id.
    """
    entity_ids = np.unique(atom_array.mol_entity_id)

    for entity_id in entity_ids:
        yield atom_array[atom_array.mol_entity_id == entity_id]


def mol_unique_instance_iter(
    atom_array: AtomArray,
) -> Generator[AtomArray, None, None]:
    """Returns atom_array slices corresponding to every (mol_entity_id, mol_sym_id).

    Similar in concept to Biotite's molecule_iter, but logic is based on the previously
    assigned molecular symmetry IDs and does not read the internal AtomArray's bond
    list.

    WARNING: The order with which the slices are returned may be different from the
    order of first appearance in the AtomArray.

    Args:
        atom_array (AtomArray):
            The atom array to iterate over.

    Yields:
        AtomArray:
            AtomArray slice corresponding to a unique (mol_entity_id, mol_sym_id).
    """
    for entity in mol_entity_iter(atom_array):
        sym_ids = np.unique(entity.mol_sym_id)

        for sym_id in sym_ids:
            yield entity[entity.mol_sym_id == sym_id]


def assign_mol_sym_token_index(atom_array: AtomArray) -> None:
    """Assigns renumbered token indices for every molecule instance.

    Renumbers the token indices for every unique molecule (identified as a
    (mol_entity_id, mol_sym_id) pair) in the atom array.

    Args:
        atom_array (AtomArray):
            The atom array to assign the token indices to.

    Returns:
        None, the atom array is modified in-place.
    """

    assign_atom_indices(atom_array)

    atom_array.set_annotation(
        "mol_sym_token_index", -np.ones(len(atom_array), dtype=int)
    )

    # Go through each symmetric mol and renumber the token indices from 1
    for mol in mol_unique_instance_iter(atom_array):
        token_starts = get_token_starts(mol, add_exclusive_stop=True)
        token_id_repeats = np.diff(token_starts)
        token_indices_renumbered = np.repeat(
            np.arange(len(token_id_repeats)), token_id_repeats
        )
        atom_array.mol_sym_token_index[mol._atom_idx] = token_indices_renumbered

    remove_atom_indices(atom_array)

    assert np.all(atom_array.mol_sym_token_index != -1)


def assign_mol_sym_component_ids(atom_array: AtomArray):
    """Assigns renumbered component IDs for every molecule instance.

    Renumbers the component IDs for every unique molecule (identified as a
    (mol_entity_id, mol_sym_id) pair) in the atom array.

    Args:
        atom_array (AtomArray):
            The atom array to assign the component IDs to.

    Returns:
        None, the atom array is modified in-place.
    """
    assign_atom_indices(atom_array)

    atom_array.set_annotation(
        "mol_sym_component_id", -np.ones(len(atom_array), dtype=int)
    )

    for mol_array in mol_unique_instance_iter(atom_array):
        for id, component in enumerate(component_iter(mol_array), start=1):
            atom_array.mol_sym_component_id[component._atom_idx] = id

    remove_atom_indices(atom_array)

    assert np.all(atom_array.mol_sym_component_id != -1)


@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc-permutation-labels")
def assign_mol_permutation_ids(
    atom_array: AtomArray, retokenize: bool = True
) -> AtomArray:
    """Assigns all permutation-related annotations to the atom array.

    This function detects symmetry-equivalent covalently connected components (=
    molecules) in the atom array and assigns symmetry-related annotations, required for
    the permutation alignment. Additionally it reorders the chains within
    symmetry-equivalent molecules to a consistent order if necessary, so that their
    features match each other.

    Args:
        atom_array (AtomArray):
            The atom array to assign the permutation labels to.
        retokenize (bool):
            If True, the atom array's token indices are reassigned after the operation
            that can change the internal chain order.

    Returns:
        AtomArray:
            The same atom array with the following new annotations:
                - mol_entity_id:
                    A unique identifier for a group of symmetry-equivalent molecules,
                    referred to as molecular entity ID. All molecules with the same
                    molecular entity ID are fully symmetry-equivalent to each other.
                - mol_sym_id:
                    A unique identifier for each individual symmetry-equivalent molecule
                    within a symmetric group, starting from 1.
                - mol_sym_token_index:
                    Renumbered token indices for every molecule instance.
                - mol_sym_component_id:
                    Renumbered component IDs for every molecule instance.
    """

    atom_array = atom_array.copy()

    # Add the mol_entity_id and mol_sym_id annotations (and potentially change internal
    # chain order of atom array if necessary)
    atom_array = assign_mol_symmetry_ids(atom_array)

    # Reassign token indices to match the new chain order
    if retokenize:
        tokenize_atom_array(atom_array)

    # Assign mol_sym_token_index attribute (renumbered token indices for every molecule
    # instance)
    assign_mol_sym_token_index(atom_array)

    # Assign mol_sym_component_id attribute (renumbered component IDs (ref_space_uids)
    # for every molecule instance)
    assign_mol_sym_component_ids(atom_array)

    return atom_array


class SeparatedTargetStructure(NamedTuple):
    """Separated target structure for a single molecule.

    Attributes:
        cropped (AtomArray):
            The cropped atom array.
        gt (AtomArray):
            The ground-truth atom array, subset to only the atoms that are
            symmetry-related to the cropped atom array.
    """

    cropped: AtomArray
    gt: AtomArray


def separate_cropped_and_gt(
    atom_array_gt: AtomArray,
    crop_strategy: str,
    processed_ref_mol_list: list[ProcessedReferenceMolecule],
) -> tuple[AtomArray, AtomArray]:
    """Separates the cropped and ground-truth atom arrays.

    Splits the preprocessed atom array into the actual cropped subset and ground-truth
    atoms. For efficiency reasons, only the atoms that are symmetry-related to the
    cropped subset are kept in the ground-truth atom array.

    Additionally, deviating slightly from AF2-Multimer/AF3, the way the ground-truth is
    returned is dependent on the crop_strategy:
        - spatial/spatial-interface:
            The ground-truth atoms are restricted to molecules (= covalently connected
            components) that have at least one atom in the crop. This means that the
            chains included in the spatial crop can be permuted within themselves, but
            not with chains outside the crop, to keep spatial proximity.
        - contiguous/whole/other:
            The ground-truth is expanded to atom slices in every symmetry-equivalent
            molecule in the entire structure that is symmetry-equivalent to atoms in the
            crop, no matter if they themselves are in the crop or not.

    Args:
        atom_array_gt (AtomArray):
            The atom array to separate.
        crop_strategy (str):
            The crop strategy used to generate the crop. Should be one of "spatial",
            "spatial_interface", "contiguous", or "whole". If the strategy is not
            recognized, the ground-truth is expanded to atoms in all symmetric molecules
            following the contiguous/whole strategy.
        processed_ref_mol_list (list[ProcessedReferenceMolecule]):
            List of processed reference molecules.

    Returns:
        tuple[AtomArray, AtomArray]:
            A tuple containing the cropped and subset ground-truth atom arrays.
    """

    if not all(
        annotation in atom_array_gt.get_annotation_categories()
        for annotation in [
            "mol_entity_id",
            "mol_sym_id",
            "mol_sym_component_id",
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

    for mol in mol_unique_instance_iter(atom_array_gt):
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

        entity_id = mol.mol_entity_id[0]
        sym_id = mol.mol_sym_id[0]
        entity_to_valid_sym_ids[entity_id].append(sym_id)

    # This keeps track of which total ground-truth atoms are required for this
    # particular component. This is important later, as we need to edit the indices of
    # this component's permutations to match the new subset of the ground-truth atom
    # array.
    absolute_component_id_to_required_gt_atoms = defaultdict(set)

    # Keep exactly the sections of the ground-truth that are symmetry-related to
    # sections in the crop
    for cropped_entities in mol_entity_iter(atom_array_cropped):
        entity_id = cropped_entities.mol_entity_id[0]

        sym_component_id_to_absolute_conformer_id = defaultdict(set)
        for atom in cropped_entities:
            sym_component_id_to_absolute_conformer_id[atom.mol_sym_component_id].add(
                atom.component_id
            )

        # Get the exact symmetry-equivalent atom sets per component
        sym_component_id_to_required_gt_atoms = defaultdict(set)
        for sym_mol in mol_unique_instance_iter(cropped_entities):
            for component in component_iter(sym_mol):
                absolute_component_id = component.component_id[0]
                sym_component_id = component.mol_sym_component_id[0]

                # Symmetry-equivalent permutations from which the necessary ground-truth
                # atoms can be concluded
                permutations = component_id_to_permutations[absolute_component_id]
                required_gt_atom_indices = np.unique(permutations)

                required_gt_atom_indices_list = required_gt_atom_indices.tolist()
                sym_component_id_to_required_gt_atoms[sym_component_id].update(
                    required_gt_atom_indices_list
                )

                # Update all related absolute conformer IDs
                for absolute_conformer_id in sym_component_id_to_absolute_conformer_id[
                    sym_component_id
                ]:
                    absolute_component_id_to_required_gt_atoms[
                        absolute_conformer_id
                    ].update(required_gt_atom_indices_list)

        # Get the valid symmetry-equivalent GT molecules
        same_entity_gt_mols = atom_array_gt[atom_array_gt.mol_entity_id == entity_id]
        valid_sym_ids = entity_to_valid_sym_ids[entity_id]
        sym_equivalent_gt_mols = same_entity_gt_mols[
            np.isin(same_entity_gt_mols.mol_sym_id, valid_sym_ids)
        ]

        # Get an arbitrary symmetry-equivalent GT molecule to construct the resulting
        # mask that can be equivalently applied to all symmetry-equivalent GT molecules
        sym_equivalent_gt_mol = next(mol_unique_instance_iter(sym_equivalent_gt_mols))
        gt_mol_keep_atom_mask = []

        for gt_component in component_iter(sym_equivalent_gt_mol):
            gt_sym_component_id = gt_component.mol_sym_component_id[0]

            # If component is not in the crop at all, append all-False mask
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
        for mol in mol_unique_instance_iter(sym_equivalent_gt_mols):
            keep_atom_indices.update(mol._atom_idx[gt_mol_keep_atom_mask])

    # Renumber the permutations, as the ground-truth atoms are now a subset of the
    # previous ones so the indices need to be re-mapped
    for processed_mol in processed_ref_mol_list:
        required_gt_atoms = absolute_component_id_to_required_gt_atoms[
            processed_mol.component_id
        ]
        processed_mol.permutations = renumber_permutations(
            processed_mol.permutations, required_gt_atoms
        )

    # Construct the final atom array
    remove_atom_indices(atom_array_gt)
    atom_array_gt = atom_array_gt[sorted(keep_atom_indices)]

    remove_atom_indices(atom_array_cropped)

    return atom_array_cropped, atom_array_gt
