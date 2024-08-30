import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.resources.residues import (
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES_3,
    STANDARD_RNA_RESIDUES,
    MoleculeType,
)


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
        chain: MoleculeType(molecule_type_id).name.lower()
        for chain, molecule_type_id in chain_to_molecule_type_id.items()
    }


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

    # Have to set via _annot to override the <U4 chain_id default dtype
    atom_array._annot["chain_id"] = chain_ids_per_atom


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
    create_auth_label_annotations: bool = True,
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
    if create_auth_label_annotations:
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

    std_protein_residues = set(STANDARD_PROTEIN_RESIDUES_3)
    std_dna_residues = set(STANDARD_DNA_RESIDUES)
    std_rna_residues = set(STANDARD_RNA_RESIDUES)

    # Create molecule type annotation
    molecule_type_ids = np.zeros(len(atom_array), dtype=int)

    # Zip together chain starts
    for chain_start, next_chain_start in zip(
        chain_start_idxs[:-1], chain_start_idxs[1:]
    ):
        chain_array = atom_array[chain_start:next_chain_start]
        is_polymeric = struc.get_residue_count(chain_array) > 1
        residues_in_chain = set(chain_array.res_name)

        # Assign protein if polymeric and any standard protein residue is present
        if (residues_in_chain & std_protein_residues) and is_polymeric:
            molecule_type_ids[chain_start:next_chain_start] = MoleculeType.PROTEIN

        # Assign RNA if polymeric and any standard RNA residue is present
        elif (residues_in_chain & std_rna_residues) and is_polymeric:
            molecule_type_ids[chain_start:next_chain_start] = MoleculeType.RNA

        # Assign DNA if polymeric and any standard DNA residue is present
        elif (residues_in_chain & std_dna_residues) and is_polymeric:
            molecule_type_ids[chain_start:next_chain_start] = MoleculeType.DNA

        # Assign ligand otherwise
        else:
            molecule_type_ids[chain_start:next_chain_start] = MoleculeType.LIGAND

    atom_array.set_annotation("molecule_type_id", molecule_type_ids)
