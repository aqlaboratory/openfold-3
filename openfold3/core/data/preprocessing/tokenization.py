# TODO add license

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from .structure_primitives import assign_atom_indices, remove_atom_indices
from .tables import (
    MOLECULE_TYPE_ID_DNA,
    MOLECULE_TYPE_ID_PROTEIN,
    MOLECULE_TYPE_ID_RNA,
    NUCLEIC_ACID_MAIN_CHAIN_ATOMS,
    PROTEIN_MAIN_CHAIN_ATOMS,
    STANDARD_RESIDUES,
    TOKEN_CENTER_ATOMS,
)


# NOTE for LJ changes:
# - removed af3_atom_id in favor of _atom_idx with assign/remove functions
# - removed assign_chains
# - changed af3_chain_id to chain_id_renumbered
# - changed af3_molecule_type to molecule_type_id
# TODO for nits/potential changes:
# - change af3 in all annotations, e.g. af3_aux_residue_id to aux_residue_id and
#   af3_token_id to token_id
def tokenize_atom_array(atom_array: AtomArray):
    """Generates token ids, token center atom annotations and atom id annotations for a
    biotite atom_array.

    Tokenizes the input atom array according to section 2.6. in the AF3 SI. The
    tokenization is added to the input atom array as 'af3_token_id' annotation alongside
    'af3_token_center_atom' and 'af3_atom_id' annotations.

    Args:
        atom_array (AtomArray): biotite atom array of the first bioassembly of a PDB
        entry

    Returns:
        None
    """
    # Create temporary atom indices
    assign_atom_indices(atom_array)

    # Create auxiliary residue id annotation
    # The auxiliary residue id is used to tokenize covalently modified residues
    # per atom and is removed afterwards
    atom_array.set_annotation(
        "af3_aux_residue_id",
        struc.spread_residue_wise(
            atom_array, np.arange(struc.get_residue_count(atom_array))
        ),
    )

    # Get standard residues
    n_atoms = len(atom_array)
    residue_ids, residue_names = struc.get_residues(atom_array)
    standard_residue_ids = residue_ids[np.isin(residue_names, STANDARD_RESIDUES)]

    # Get ids where residue-tokens start
    is_standard_residue_atom = np.isin(atom_array.res_id, standard_residue_ids)
    standard_residue_atom_ids = atom_array._atom_idx[is_standard_residue_atom]
    residue_token_start_ids = np.unique(
        struc.get_residue_starts_for(atom_array, standard_residue_atom_ids)
    )

    # Tokenize modified residues per atom
    # Get bonds
    bondlist = atom_array.bonds.as_array()

    # Find bonds with at least one standard residue atom (i.e. non-heteroatom)
    bondlist = bondlist[
        np.isin(bondlist[:, 0], standard_residue_atom_ids)
        | np.isin(bondlist[:, 1], standard_residue_atom_ids)
    ]
    # Find bonds which contain two standard residue atoms
    bondlist_standard = bondlist[
        np.isin(bondlist[:, 0], standard_residue_atom_ids)
        & np.isin(bondlist[:, 1], standard_residue_atom_ids)
    ]

    # Find bonds which contain
    # - exactly one heteroatom
    #   (standard residues with covalent ligands)
    is_heteoratom = atom_array.hetero
    is_one_heteroatom = is_heteoratom[bondlist[:, 0]] ^ is_heteoratom[bondlist[:, 1]]
    # - two non-heteroatoms in different chains
    #   (standard residues covalently linking different chains)
    chain_ids = atom_array.chain_id_renumbered
    is_different_chain = chain_ids[bondlist[:, 0]] != chain_ids[bondlist[:, 1]]
    # - two non-heteroatoms in the same chain but side chains of different residues
    #   (standard residues covalently linking non-consecutive residues in the same
    #   chain)
    atom_names = atom_array.atom_name
    molecule_types = atom_array.molecule_type_id
    # Find atoms connecting residues in the same chain via side chains
    is_side_chain = (
        ~np.isin(atom_names, NUCLEIC_ACID_MAIN_CHAIN_ATOMS)
        & np.isin(molecule_types, [MOLECULE_TYPE_ID_RNA, MOLECULE_TYPE_ID_DNA])
    ) | (
        ~np.isin(atom_names, PROTEIN_MAIN_CHAIN_ATOMS)
        & (molecule_types == MOLECULE_TYPE_ID_PROTEIN)
    )
    is_same_chain = (
        chain_ids[bondlist_standard[:, 0]] == chain_ids[bondlist_standard[:, 1]]
    )
    is_both_side_chain = (
        is_side_chain[bondlist_standard[:, 0]] & is_side_chain[bondlist_standard[:, 1]]
    )
    is_different_residue = (
        atom_array.af3_aux_residue_id[bondlist_standard[:, 0]]
        != atom_array.af3_aux_residue_id[bondlist_standard[:, 1]]
    )
    is_same_chain_diff_sidechain = (
        is_same_chain & is_both_side_chain & is_different_residue
    )

    # Combine
    bondlist_covalent_modification = np.concatenate(
        (
            bondlist[is_one_heteroatom | is_different_chain],
            bondlist_standard[is_same_chain_diff_sidechain],
        ),
        axis=0,
    )

    # Get corresponding non-heteroatoms
    atom_ids_covalent_modification = np.unique(
        bondlist_covalent_modification[:, :2].flatten()
    )
    nonhetero_atoms_in_covalent_modification = atom_array[
        atom_ids_covalent_modification
    ][~atom_array[atom_ids_covalent_modification].hetero]

    # Get the set of all atoms in corresponding residues
    atomized_residue_token_start_ids = atom_array[
        np.isin(
            atom_array.af3_aux_residue_id,
            nonhetero_atoms_in_covalent_modification.af3_aux_residue_id,
        )
    ]._atom_idx

    # Remove the corresponding residue token start ids
    modified_residue_token_start_ids = np.unique(
        struc.get_residue_starts_for(atom_array, atomized_residue_token_start_ids)
    )
    residue_token_start_ids = residue_token_start_ids[
        ~np.isin(residue_token_start_ids, modified_residue_token_start_ids)
    ]

    # Get atom-token ids
    atom_token_start_ids = atom_array._atom_idx[~is_standard_residue_atom]

    # Combine all token start ids
    all_token_start_ids = np.sort(
        np.concatenate(
            [
                residue_token_start_ids,
                atom_token_start_ids,
                atomized_residue_token_start_ids,
            ]
        )
    )

    # Create token index
    token_id_repeats = np.diff(np.append(all_token_start_ids, n_atoms))
    token_ids_per_atom = np.repeat(np.arange(len(token_id_repeats)), token_id_repeats)
    atom_array.set_annotation("af3_token_id", token_ids_per_atom)

    # Create token center atom annotation
    af3_token_center_atoms = np.repeat(True, n_atoms)
    af3_token_center_atoms[is_standard_residue_atom] = np.isin(
        atom_array[is_standard_residue_atom].atom_name, TOKEN_CENTER_ATOMS
    )
    # Edit token center atoms for covalently modified residues
    af3_token_center_atoms[atomized_residue_token_start_ids] = True
    atom_array.set_annotation("af3_token_center_atom", af3_token_center_atoms)

    # Remove temporary atom & residue indices
    remove_atom_indices(atom_array)
    atom_array.del_annotation("af3_aux_residue_id")

    return None
