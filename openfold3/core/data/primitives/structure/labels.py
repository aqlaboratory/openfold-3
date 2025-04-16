import logging
from collections import defaultdict
from collections.abc import Generator
from typing import Any, Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.resources.residues import (
    CHEM_COMP_TYPE_TO_MOLECULE_TYPE,
    STANDARD_NUCLEIC_ACID_RESIDUES,
    STANDARD_PROTEIN_RESIDUES_3,
    MoleculeType,
)

logger = logging.getLogger(__name__)


def get_chain_to_entity_dict(atom_array: struc.AtomArray) -> dict[int, int]:
    """Get a dictionary mapping chain IDs to their entity IDs.

    Args:
        atom_array:
            AtomArray containing the chain IDs and entity IDs.

    Returns:
        A dictionary mapping chain IDs to their entity IDs.
    """
    chain_starts = struc.get_chain_starts(atom_array)

    return dict(
        zip(
            atom_array[chain_starts].chain_id.tolist(),
            atom_array[chain_starts].entity_id.tolist(),
        )
    )


def get_chain_to_author_chain_dict(atom_array: struc.AtomArray) -> dict[int, str]:
    """Get a dictionary mapping chain IDs to their author chain IDs.

    Args:
        atom_array:
            AtomArray containing the chain IDs and author chain IDs.

    Returns:
        A dictionary mapping chain IDs to their author chain IDs.
    """
    if "auth_asym_id" not in atom_array.get_annotation_categories():
        raise ValueError(
            "The AtomArray does not contain author chain IDs. "
            "Make sure to load the 'auth_asym_id' field when parsing the structure."
        )

    chain_starts = struc.get_chain_starts(atom_array)

    return dict(
        zip(
            atom_array[chain_starts].chain_id.tolist(),
            atom_array[chain_starts].auth_asym_id.tolist(),
        )
    )


def get_chain_to_pdb_chain_dict(atom_array: struc.AtomArray) -> dict[int, str]:
    """Get a dictionary mapping chain IDs to their PDB chain IDs.

    Args:
        atom_array:
            AtomArray containing the chain IDs and PDB chain IDs.

    Returns:
        A dictionary mapping chain IDs to their PDB chain IDs.
    """
    chain_starts = struc.get_chain_starts(atom_array)

    return dict(
        zip(
            atom_array[chain_starts].chain_id.tolist(),
            atom_array[chain_starts].label_asym_id.tolist(),
        )
    )


def get_chain_to_molecule_type_id_dict(atom_array: struc.AtomArray) -> dict[int, int]:
    """Get a dictionary mapping chain IDs to their molecule type IDs.

    Args:
        atom_array:
            AtomArray containing the chain IDs and molecule type IDs.

    Returns:
        A dictionary mapping chain IDs to their molecule type IDs.
    """
    chain_starts = struc.get_chain_starts(atom_array)

    return dict(
        zip(
            atom_array[chain_starts].chain_id.tolist(),
            atom_array[chain_starts].molecule_type_id.tolist(),
        )
    )


def get_chain_to_molecule_type_dict(atom_array: struc.AtomArray) -> dict[int, str]:
    """Get a dictionary mapping chain IDs to their molecule types.

    Args:
        atom_array:
            AtomArray containing the chain IDs and molecule type IDs.

    Returns:
        A dictionary mapping chain IDs to their molecule types (as strings instead of
        IDs).
    """
    chain_to_molecule_type_id = get_chain_to_molecule_type_id_dict(atom_array)

    return {
        chain: MoleculeType(molecule_type_id).name
        for chain, molecule_type_id in chain_to_molecule_type_id.items()
    }


def get_residue_tuples(
    atom_array: struc.AtomArray,
    include_resname: bool = False,
) -> list[tuple[str, int]]:
    """Get a list of (chain_id, res_id) tuples for all residues in an AtomArray.

    Args:
        atom_array:
            AtomArray containing the residues to get unique tuples for.
        include_resname:
            Whether to add the residue name to the tuple. Defaults to False. If True,
            the tuple will be (chain_id, res_id, res_name).

    Returns:
        A list of (chain_id, res_id) tuples for unique residues in the AtomArray.
    """
    attrs_to_include = ["chain_id", "res_id"]

    # Add other attributes if specified
    if include_resname:
        attrs_to_include.append("res_name")

    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=False)

    # Construct the list of residue tuples
    residue_tuples = [
        tuple(getattr(atom_array, attr)[residue_start] for attr in attrs_to_include)
        for residue_start in residue_starts
    ]

    return residue_tuples


def get_differing_chain_ids(
    atom_array_1: AtomArray, atom_array_2: AtomArray
) -> list[str]:
    """Get a list of chain IDs that differ between two AtomArrays.

    Computes the symmetric difference between the chain IDs of the two AtomArrays. E.g.:

        chain_ids_1 = ["A", "B", "C"]
        chain_ids_2 = ["A", "D", "E"]
        get_differing_chain_ids(chain_ids_1, chain_ids_2) -> ["B", "C", "D", "E"]

    Args:
        atom_array_1:
            First AtomArray to compare.
        atom_array_2:
            Second AtomArray to compare.

    Returns:
        A list of chain IDs that differ between the two AtomArrays.
    """
    differing_chain_ids = np.setxor1d(
        atom_array_1.chain_id,
        atom_array_2.chain_id,
    )

    # Chain IDs in this codebase are often numerical so sort them nicely.
    return sorted(differing_chain_ids, key=lambda x: x.rjust(5, "0"))


def assign_renumbered_chain_ids(
    atom_array: AtomArray, store_original_as: str | None = None
) -> None:
    """Renumbers the chain IDs in the AtomArray starting from 1

    Iterates through all chains in the atom array and assigns unique numerical chain IDs
    starting with 0 to each chain. This is useful for bioassembly parsing where chain
    IDs can be duplicated after the assembly is expanded.

    Args:
        atom_array:
            AtomArray containing the structure to assign renumbered chain IDs to.
        store_original_as:
            If set, the original chain IDs are stored in the specified field of the
            AtomArray. If None, the original chain IDs are discarded. Defaults to None.
    """
    chain_start_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # Assign numerical chain IDs
    chain_id_n_repeats = np.diff(chain_start_idxs)
    chain_ids_per_atom = np.repeat(
        np.arange(1, len(chain_id_n_repeats) + 1), chain_id_n_repeats
    )

    if store_original_as is not None:
        atom_array.set_annotation(store_original_as, atom_array.chain_id)

    atom_array.chain_id = chain_ids_per_atom


def assign_atom_indices(
    atom_array: AtomArray, label: str = "_atom_idx", overwrite: bool = False
) -> None:
    """Assigns atom indices to the AtomArray

    Atom indices are a simple range from 0 to the number of atoms in the AtomArray which
    is used as a convenience feature. They are stored in the "_atom_idx" field of the
    AtomArray and meant to be used only temporarily within functions. Should be combined
    with `remove_atom_indices`.

    Args:
        atom_array:
            AtomArray containing the structure to assign atom indices to.
        label:
            Name of the annotation field to store the atom indices in. Defaults to
            "_atom_idx". It is a good practice to set this to something custom per
            function, as this avoids attribute clashes if the function is called within
            a parent function that has already set an _atom_idx.
        overwrite:
            Whether to overwrite an existing annotation field with the same name.
            Defaults to False.
    """
    if label in atom_array.get_annotation_categories() and not overwrite:
        raise ValueError(f"Annotation field '{label}' already exists in AtomArray.")
    else:
        atom_array.set_annotation(label, range(len(atom_array)))


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
    create_auth_label_annotations: bool = True,
    atom_array_source_format: Literal["cif", "pdb", "pdb_af2"] = "cif",
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
            Whether to keep the author-assigned residue IDs if they are NaN in the PDB
            labels. This is important for correct bond record parsing/writing in
            Biotite. Defaults to True.
        auth_label_annotations:
            Whether to keep the original author-assigned labels as annotations in the
            AtomArray.
        atom_array_source_format:
            Whether the AtomArray was parsed from a CIF file, PDB file or a PDB file
            predicted by AF2, required due to some fields missing from the PDB format.

    """
    if create_auth_label_annotations:
        atom_array.set_annotation("auth_asym_id", atom_array.chain_id)
        atom_array.set_annotation("auth_seq_id", atom_array.res_id)
        atom_array.set_annotation("auth_comp_id", atom_array.res_name)
        atom_array.set_annotation("auth_atom_id", atom_array.atom_name)

    if atom_array_source_format == "cif":
        # Replace author-assigned IDs with PDB-assigned IDs if cif
        atom_array.chain_id = atom_array.label_asym_id
        atom_array.res_name = atom_array.label_comp_id
        atom_array.atom_name = atom_array.label_atom_id
    elif atom_array_source_format == "pdb":
        raise NotImplementedError(
            "PDB format is not yet supported for updating author-assigned labels."
        )
    elif atom_array_source_format == "pdb_af2":
        # Add "label_" IDs if pdb
        atom_array.set_annotation("label_asym_id", atom_array.chain_id)
        atom_array.set_annotation("label_comp_id", atom_array.res_name)
        atom_array.set_annotation("label_atom_id", atom_array.atom_name)
    else:
        raise ValueError(
            "The AtomArray source format must be either 'cif', 'pdb' or 'pdb_af2'."
        )

    # Set residue IDs to PDB-assigned IDs but fallback to author-assigned IDs if they
    # are not assigned (important for correct bond record parsing/writing)
    # TODO: check if this is necessary for 'pdb' format
    if use_author_res_id_if_missing & (atom_array_source_format == "cif"):
        author_res_ids = atom_array.res_id
        pdb_res_ids = atom_array.label_seq_id
        merged_res_ids = np.where(
            pdb_res_ids == ".", author_res_ids, pdb_res_ids
        ).astype(int)
        atom_array.res_id = merged_res_ids


def assign_entity_ids(
    atom_array: AtomArray,
    atom_array_source_format: Literal["cif", "pdb", "pdb_af2"] = "cif",
) -> None:
    """Assigns entity IDs to the AtomArray

    Entity IDs are assigned to each chain in the AtomArray based on the
    "label_entity_id" field. The entity ID is stored in the "entity_id" field of the
    AtomArray.

    Args:
        atom_array:
            AtomArray containing the structure to assign entity IDs to.
        atom_array_source_format:
            Whether the AtomArray was parsed from a CIF file, PDB file or a PDB file
            predicted by AF2, required due to some fields missing from the PDB format.
    """
    # Cast entity IDs from string to int and shorten name
    if atom_array_source_format == "cif":
        atom_array.set_annotation("entity_id", atom_array.label_entity_id.astype(int))
        atom_array.del_annotation("label_entity_id")
    elif atom_array_source_format == "pdb":
        raise NotImplementedError(
            "PDB format is not yet supported for assigning entity IDs."
        )
    elif atom_array_source_format == "pdb_af2":
        pass


def assign_molecule_type_ids(atom_array: AtomArray, cif_file: pdbx.CIFFile) -> None:
    """Assigns molecule types to the AtomArray

    Assigns molecule type IDs to each chain based on its residue names. Possible
    molecule types are protein, RNA, DNA, and ligand. The molecule type is stored in the
    "molecule_type_id" field of the AtomArray.

    Args:
        atom_array:
            AtomArray containing the structure to assign molecule types to.
    """
    # Get chemical component-to-type mapping
    # All type values are mapped to upper case to ensure case-insensitive matching
    chem_comp_ids = cif_file.block["chem_comp"]["id"].as_array()
    chem_comp_types = cif_file.block["chem_comp"]["type"].as_array()
    try:
        chem_comp_id_to_type = {
            k: CHEM_COMP_TYPE_TO_MOLECULE_TYPE[v.upper()]
            for k, v in zip(chem_comp_ids, chem_comp_types)
        }
    except KeyError:
        missing_types = chem_comp_types[
            ~np.isin(
                chem_comp_types, np.array(list(CHEM_COMP_TYPE_TO_MOLECULE_TYPE.keys()))
            )
        ]

        logger.error(
            "Found chemical component types that are missing from the "
            "component type-to-molecule type map. Mapping the following "
            f'types to "OTHER" i.e. MoleculeType.LIGAND: {missing_types}'
        )
        chem_comp_id_to_type = {
            k: CHEM_COMP_TYPE_TO_MOLECULE_TYPE.get(
                v.upper(), CHEM_COMP_TYPE_TO_MOLECULE_TYPE["OTHER"]
            )
            for k, v in zip(chem_comp_ids, chem_comp_types)
        }

    @np.vectorize
    def get_mol_types(key: str) -> MoleculeType:
        return chem_comp_id_to_type.get(key, MoleculeType.LIGAND)

    chain_start_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # Create molecule type annotation
    molecule_type_ids = np.zeros(len(atom_array), dtype=int)

    # Zip together chain starts
    for chain_start, next_chain_start in zip(
        chain_start_idxs[:-1], chain_start_idxs[1:]
    ):
        chain_array = atom_array[chain_start:next_chain_start]
        is_polymeric = struc.get_residue_count(chain_array) > 1
        atom_mol_types = get_mol_types(chain_array.res_name)

        # Non-polymeric chains are always ligands
        # TODO fix edge case where only one residue is resolved for a polymeric chain
        if not is_polymeric:
            molecule_type_ids[chain_start:next_chain_start] = MoleculeType.LIGAND
        # Assign a single molecule type to all atoms in the chain based on the majority
        # vote of molecule types of all atomis in the chain
        else:
            molecule_type_ids[chain_start:next_chain_start] = np.argmax(
                np.bincount(atom_mol_types)
            )

    atom_array.set_annotation("molecule_type_id", molecule_type_ids)


def uniquify_ids(ids: list[str]) -> list[str]:
    """
    Uniquify a list of string IDs by appending occurrence count.

    This function takes a list of string IDs and returns a new list where each ID is
    made unique by appending an underscore followed by its occurrence count.

    Args:
        ids (list[str]):
            A list of string IDs, which may contain duplicates.

    Returns:
        list[str]:
            A list of uniquified IDs, where each ID is appended with its occurrence
            count (e.g., "id_1", "id_2").
    """

    id_counter = defaultdict(lambda: 0)
    uniquified_ids = []

    for id in ids:
        id_counter[id] += 1
        uniquified_ids.append(f"{id}_{id_counter[id]}")

    return uniquified_ids


@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc-unqual-atoms")
def assign_uniquified_atom_names(atom_array: AtomArray) -> None:
    """Assigns unique atom indices to symmetric mols."""
    assign_atom_indices(atom_array, label="_atom_idx_unqf_atoms")
    atom_array.set_annotation(
        "atom_name_unique", np.full(len(atom_array), fill_value="-", dtype=object)
    )

    for component_view in component_view_iter(atom_array):
        atom_names = component_view.atom_name
        atom_names_uniquified = uniquify_ids(atom_names)

        atom_array.atom_name_unique[component_view._atom_idx_unqf_atoms] = (
            atom_names_uniquified
        )

    atom_array.del_annotation("_atom_idx_unqf_atoms")

    assert np.all(atom_array.atom_name_unique != "-")

    return atom_array


def get_id_starts(
    atom_array: AtomArray, id_field: str, add_exclusive_stop: bool = False
) -> np.ndarray:
    """Gets the indices of the first atom of each ID.

    Args:
        atom_array:
            AtomArray of the target or ground truth structure
        id_field:
            Name of an annotation field containing consecutive IDs.

    Returns:
        np.ndarray:
            Array of indices of the first atom of each ID.
    """
    id_diffs = np.diff(getattr(atom_array, id_field))
    id_starts = np.where(id_diffs != 0)[0] + 1
    id_starts = np.append(0, id_starts)

    if add_exclusive_stop:
        id_starts = np.append(id_starts, len(atom_array))

    return id_starts


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
    return get_id_starts(atom_array, "token_id", add_exclusive_stop)


def get_component_starts(
    atom_array: AtomArray, add_exclusive_stop: bool = False
) -> np.ndarray:
    """Gets the indices of the first atom of each component.

    Args:
        atom_array (AtomArray):
            AtomArray of the target or ground truth structure
        add_exclusive_stop (bool, optional):
            Whether to append an int with the size of the input atom array at the
            end of returned indices. Defaults to False.

    Returns:
        np.ndarray:
            Array of indices of the first atom of each component.
    """
    return get_id_starts(atom_array, "component_id", add_exclusive_stop)


class AtomArrayView:
    """Container to access underlying arrays holding AtomArray attributes."""

    def __init__(self, atom_array: AtomArray, indices: np.ndarray | slice):
        if not isinstance(indices, (np.ndarray, slice)):
            raise ValueError(
                "The indices argument must be a NumPy array or a slice object."
            )

        self.atom_array = atom_array
        self.indices = indices

    def __getattr__(self, attr):
        """Returns the result of NumPy-indexing to the attribute of the AtomArray.

        This will be a view for slice-like indexing and a copy for advanced indexing
        (following NumPy rules). Note that "atom_array" and "indices" are special
        attributes that are required for this class and should not exist in the parent
        AtomArray.
        """
        if attr == "atom_array":
            return self.atom_array
        elif attr == "indices":
            return self.indices
        else:
            return self.atom_array.__getattr__(attr)[self.indices]

    def materialize(self) -> AtomArray:
        """Creates a new AtomArray from the view.

        Returns:
            AtomArray:
                AtomArray containing the data of the view.
        """
        return self.atom_array[self.indices]

    def __len__(self):
        if isinstance(self.indices, slice):
            start, stop, step = self.indices.indices(len(self.atom_array))
            return len(range(start, stop, step))
        else:
            if self.indices.dtype == bool:
                return np.count_nonzero(self.indices)
            else:
                return len(self.indices)


def component_view_iter(atom_array: AtomArray) -> Generator[AtomArrayView, None, None]:
    """Iterates through components in an AtomArray.

    Args:
        atom_array (AtomArray):
            AtomArray to return components for.

    Yields:
        AtomArrayView:
            AtomArrayView for a single component.
    """
    component_starts = get_component_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(component_starts[:-1], component_starts[1:]):
        yield AtomArrayView(atom_array, slice(start, stop))


def residue_view_iter(atom_array: AtomArray) -> Generator[AtomArrayView, None, None]:
    """Iterates through residues in an AtomArray.

    Args:
        atom_array (AtomArray):
            AtomArray to return residues for.

    Yields:
        AtomArrayView:
            AtomArrayView for a single residue.
    """
    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
        yield AtomArrayView(atom_array, slice(start, stop))


def chain_view_iter(atom_array: AtomArray) -> Generator[AtomArrayView, None, None]:
    """Iterates through chains in an AtomArray.

    Args:
        atom_array (AtomArray):
            AtomArray to return chains for.

    Yields:
        AtomArrayView:
            AtomArrayView for a single chain.
    """
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
    for start, stop in zip(chain_starts[:-1], chain_starts[1:]):
        yield AtomArrayView(atom_array, slice(start, stop))


def set_residue_hetero_values(atom_array: AtomArray) -> None:
    """Sets the "hetero" annotation in the AtomArray based on the residue names.

    This function sets the "hetero" annotation in the AtomArray based on the residue
    names. If the residue name is in the list of standard residues for the respective
    molecule type, the "hetero" annotation is set to False, otherwise it is set to True.

    Args:
        atom_array:
            AtomArray containing the structure to set the "hetero" annotation for.

    Returns:
        None, the "hetero" annotation is modified in-place.
    """
    protein_mask = atom_array.molecule_type_id == MoleculeType.PROTEIN
    if protein_mask.any():
        in_standard_protein_residues = np.isin(
            atom_array.res_name, STANDARD_PROTEIN_RESIDUES_3
        )
    else:
        in_standard_protein_residues = np.zeros(len(atom_array), dtype=bool)

    rna_mask = atom_array.molecule_type_id == MoleculeType.RNA
    if rna_mask.any():
        in_standard_rna_residues = np.isin(
            atom_array.res_name, STANDARD_NUCLEIC_ACID_RESIDUES
        )
    else:
        in_standard_rna_residues = np.zeros(len(atom_array), dtype=bool)

    dna_mask = atom_array.molecule_type_id == MoleculeType.DNA
    if dna_mask.any():
        in_standard_dna_residues = np.isin(
            atom_array.res_name, STANDARD_NUCLEIC_ACID_RESIDUES
        )
    else:
        in_standard_dna_residues = np.zeros(len(atom_array), dtype=bool)

    atom_array.hetero[:] = True
    atom_array.hetero[
        (protein_mask & in_standard_protein_residues)
        | (rna_mask & in_standard_rna_residues)
        | (dna_mask & in_standard_dna_residues)
    ] = False


def remove_transfer_annotations(
    target_atom_array: AtomArray,
    source_atom_array: AtomArray,
    chain_map: dict[str, str],
    transfer_annot_dict: dict[str, Any],
    delete_annot_list: list[str],
) -> AtomArray:
    """Removes and transfers annotations between AtomArrays with identical chains and
    residues.

    IMPORTANT: This function currently only supports residue- and chain-level annotation
    transfers. The source-to-target chain mapping is assumed to match the label_asym_id
    fields of the source (chain_map key) and target (chain_map value) AtomArrays.

    Args:
        target_atom_array (AtomArray):
            Atom array from which to remove annotations and to which to transfer
            annotations.
        source_atom_array (AtomArray):
            Atom array from which to transfer annotations.
        chain_map (dict[str, str]):
            A dictionary mapping source chain IDs to target chain IDs.
        transfer_annot_dict (dict[str, Any]):
            Dict of annotations to transfer from the source AtomArray to the target
            AtomArray, mapping to default values to be used if the source AtomArray does
            not contain the annotation.
        delete_annot_list (list[str]):
            List of annotations to remove from the target AtomArray.

    Returns:
        AtomArray:
            AtomArray with annotations removed and transferred from the source
            AtomArray.
    """
    target_atom_array_annotated = target_atom_array.copy()

    # Remove annotations
    for annot in target_atom_array_annotated.get_annotation_categories():
        if annot in delete_annot_list:
            target_atom_array_annotated.del_annotation(annot)

    # Add missing annotations that need transferring with defaults
    for annot, default_value in transfer_annot_dict.items():
        if annot not in target_atom_array_annotated.get_annotation_categories():
            target_atom_array_annotated.set_annotation(
                annot, np.full(len(target_atom_array_annotated), default_value)
            )

    # Iterate over chains
    for source_chain_id, target_chain_id in chain_map.items():
        # Get current chain slice in copy and associated mask
        target_chain_mask = target_atom_array.label_asym_id == target_chain_id
        target_chain = target_atom_array_annotated[target_chain_mask]

        # Get source chain slice and residue starts
        source_chain = source_atom_array[source_atom_array.chain_id == source_chain_id]
        source_chain_res_starts = struc.get_residue_starts(source_chain)

        # Transfer annotations
        for annot in transfer_annot_dict:
            # Get residue-wise source annotations
            source_annot_per_res = struc.spread_residue_wise(
                target_chain, getattr(source_chain[source_chain_res_starts], annot)
            )
            getattr(target_atom_array_annotated, annot)[target_chain_mask] = (
                source_annot_per_res
            )

    return target_atom_array_annotated
