# TODO add license

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.primitives.featurization.structure import get_token_starts
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.labels import (
    assign_atom_indices,
    assign_residue_indices,
    remove_atom_indices,
    remove_residue_indices,
)
from openfold3.core.data.resources.residues import (
    NUCLEIC_ACID_MAIN_CHAIN_ATOMS,
    PROTEIN_MAIN_CHAIN_ATOMS,
    STANDARD_RESIDUES_3,
    TOKEN_CENTER_ATOMS,
    MoleculeType,
)


@log_runtime_memory(runtime_dict_key="runtime-add-token-pos")
def add_token_positions(atom_array: AtomArray) -> None:
    """Adds token_position annotation to the input atom array.

    Args:
        atom_array (AtomArray):
            AtomArray of the input assembly.
    """
    # Create token ID to token position mapping
    token_starts = get_token_starts(atom_array)
    token_positions_map = {
        token: position
        for position, token in enumerate(atom_array[token_starts].token_id)
    }

    # Map token ID to token position for all atoms and add annotation
    token_positions = np.vectorize(token_positions_map.get)(atom_array.token_id)
    atom_array.set_annotation("token_position", token_positions)


@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc-token")
def tokenize_atom_array(atom_array: AtomArray):
    """Creates token id, token center atom, and is_atomized annotations for atom array.

    Tokenizes the input atom array according to section 2.6. in the AF3 SI. The
    tokenization is added to the input atom array as a 'token_id' annotation alongside
    'token_center_atom' and 'is_atomized' annotations.

    High-level logic of the tokenizer:
        1. Get atoms in canonical residues in polymers
        2. Get atoms in small molecule ligands, non-canonical residues in polymers and
        amino acid or nucleotide small molecule ligands
        3. Get atoms in canonical residues in polymer with covalent modifications or
        non-canonical bonding pattern, i.e., those that are connected to:
            3.1. a small molecule ligand
            3.2. any atom of a non-canonical residue via a side chain atom
            3.3. any OTHER residue via a bond involving a main-chain-
            -non-main-chain atom pair
            3.4. another canonical residue of a different molecule type
            via a bond involving a main-chain--main-chain atom pair
        4. Tokenize
            - 2. per atom
            - 3. per atom
            - the difference of sets 1.-3. per residue

    Args:
        atom_array (AtomArray):
            biotite atom array of the first bioassembly of a PDB entry

    Returns:
        None
    """
    assign_atom_indices(atom_array)
    assign_residue_indices(atom_array)

    # 1. Find token start IDs of canonical residues in-polymer (CRP), excluding amino
    # acid and nucleotide small molecule ligands
    is_l_atom = atom_array.molecule_type_id == MoleculeType.LIGAND
    is_cr_atom = np.isin(atom_array.res_name, STANDARD_RESIDUES_3)
    is_crp_atom = ~is_l_atom & is_cr_atom
    crp_atom_ids = atom_array._atom_idx[is_crp_atom]
    crp_token_start_ids = np.unique(
        struc.get_residue_starts_for(atom_array, crp_atom_ids)
    )

    # 2. Find atom-token start ids: includes small molecule ligands, non-canonical
    # residues and amino acid or nucleotide small molecule ligands
    atom_token_start_ids = atom_array._atom_idx[~is_crp_atom]

    # 3. Find covalently modified canonical residues in-polymer (CRP)
    bonds = atom_array.bonds.as_array()[:, :2]

    # 3.1. Connected to a small molecule ligand
    is_atom_1_crp_atom = np.isin(bonds[:, 0], crp_atom_ids)
    is_atom_2_crp_atom = np.isin(bonds[:, 1], crp_atom_ids)
    has_one_crp_atom = is_atom_1_crp_atom ^ is_atom_2_crp_atom
    has_l_atom = np.any(np.isin(bonds, atom_array._atom_idx[is_l_atom]), axis=1)
    v1_crp_bonds = bonds[has_one_crp_atom & has_l_atom]

    # 3.2. Connected to any atom of a non-canonical residue via a side chain atom
    # > find side chain atoms in the bond list
    atom_names = atom_array.atom_name
    molecule_types = atom_array.molecule_type_id
    is_side_chain_atom = (
        ~np.isin(atom_names, NUCLEIC_ACID_MAIN_CHAIN_ATOMS)
        & np.isin(molecule_types, [MoleculeType.RNA, MoleculeType.DNA])
    ) | (
        ~np.isin(atom_names, PROTEIN_MAIN_CHAIN_ATOMS)
        & (molecule_types == MoleculeType.PROTEIN)
    )
    side_chain_atom_ids = atom_array._atom_idx[is_side_chain_atom]
    is_atom_1_side_chain_atom = np.isin(bonds[:, 0], side_chain_atom_ids)
    is_atom_2_side_chain_atom = np.isin(bonds[:, 1], side_chain_atom_ids)
    # > find non-canonical residue in-polymer atoms in the bond list
    is_ncrp_atom = ~is_l_atom & ~is_cr_atom
    ncrp_atom_ids = atom_array._atom_idx[is_ncrp_atom]
    is_atom_1_ncrp_atom = np.isin(bonds[:, 0], ncrp_atom_ids)
    is_atom_2_ncrp_atom = np.isin(bonds[:, 1], ncrp_atom_ids)
    # > find bonds between
    # an atom in the side chain of a canonical residue in a polymer and
    # an atom of a non-canonical residue in a polymer
    v2_crp_bonds = bonds[
        ((is_atom_1_crp_atom & is_atom_1_side_chain_atom) & is_atom_2_ncrp_atom)
        | (is_atom_2_crp_atom & is_atom_2_side_chain_atom) & is_atom_1_ncrp_atom
    ]

    # 3.3. Connected to any OTHER residue via a bond involving a
    # main-chain--non-main-chain atom pair
    # > find main chain atoms in the bond list
    is_main_chain_atom = (
        np.isin(atom_names, NUCLEIC_ACID_MAIN_CHAIN_ATOMS)
        & np.isin(molecule_types, [MoleculeType.RNA, MoleculeType.DNA])
    ) | (
        np.isin(atom_names, PROTEIN_MAIN_CHAIN_ATOMS)
        & (molecule_types == MoleculeType.PROTEIN)
    )
    main_chain_atom_ids = atom_array._atom_idx[is_main_chain_atom]
    is_atom_1_main_chain_atom = np.isin(bonds[:, 0], main_chain_atom_ids)
    is_atom_2_main_chain_atom = np.isin(bonds[:, 1], main_chain_atom_ids)
    has_different_chain_id = (
        atom_array.chain_id[bonds[:, 0]] != atom_array.chain_id[bonds[:, 1]]
    )
    has_different_res_id = (
        atom_array.res_id[bonds[:, 0]] != atom_array.res_id[bonds[:, 1]]
    )
    # > find bonds between
    # an atom in the main chain of a canonical residue in a polymer and
    # an atom not in the main chain of a residue in a polymer
    # st. the two connected residues have different residue ids or chain ids
    v3_crp_bonds = bonds[
        (has_different_chain_id | has_different_res_id)
        & (
            (
                (is_atom_1_main_chain_atom & is_atom_1_crp_atom)
                & ~is_atom_2_main_chain_atom
            )
            | (
                (is_atom_2_main_chain_atom & is_atom_2_crp_atom)
                & ~is_atom_1_main_chain_atom
            )
        )
    ]

    # 3.4. Connected to another canonical residue of a different molecule type via a
    # main-chain-- main-chain atom pair
    has_different_molecule_type = (
        atom_array.molecule_type_id[bonds[:, 0]]
        != atom_array.molecule_type_id[bonds[:, 1]]
    )
    v4_crp_bonds = bonds[
        has_different_molecule_type
        & (
            (is_atom_1_main_chain_atom & is_atom_1_crp_atom)
            & (is_atom_2_main_chain_atom & is_atom_2_crp_atom)
        )
    ]

    # Combine into all canonical residues in-polymer with a covalent modification
    mod_crp_bonds = np.concatenate(
        (
            v1_crp_bonds,
            v2_crp_bonds,
            v3_crp_bonds,
            v4_crp_bonds,
        ),
        axis=0,
    )

    # Get corresponding canonical residue atoms
    mod_crp_atom_ids = np.unique(mod_crp_bonds[:, :2].flatten())
    mod_crp_atoms = atom_array[np.isin(atom_array._atom_idx, mod_crp_atom_ids)]

    # Get corresponding canonical residue token starts
    atomized_crp_token_start_ids = atom_array[
        np.isin(
            atom_array._residue_idx,
            mod_crp_atoms._residue_idx,
        )
    ]._atom_idx

    # Remove the corresponding residue token start ids
    mod_crp_token_start_ids = np.unique(
        struc.get_residue_starts_for(atom_array, atomized_crp_token_start_ids)
    )
    crp_token_start_ids = crp_token_start_ids[
        ~np.isin(crp_token_start_ids, mod_crp_token_start_ids)
    ]

    # Combine all token start ids
    all_token_start_ids = np.sort(
        np.concatenate(
            [
                crp_token_start_ids,
                atom_token_start_ids,
                atomized_crp_token_start_ids,
            ]
        )
    )

    # Add is_atomized annotation
    n_atoms = len(atom_array)
    is_atomized = np.repeat(False, n_atoms)
    is_atomized[
        np.concatenate([atom_token_start_ids, atomized_crp_token_start_ids])
    ] = True
    atom_array.set_annotation("is_atomized", is_atomized)

    # Create token index
    token_id_repeats = np.diff(np.append(all_token_start_ids, n_atoms))
    token_ids_per_atom = np.repeat(np.arange(len(token_id_repeats)), token_id_repeats)
    atom_array.set_annotation("token_id", token_ids_per_atom)

    # Create token center atom annotation
    token_center_atoms = np.repeat(True, n_atoms)
    token_center_atoms[is_crp_atom] = np.isin(
        atom_array[is_crp_atom].atom_name, TOKEN_CENTER_ATOMS
    )
    # Edit token center atoms for covalently modified residues
    token_center_atoms[atomized_crp_token_start_ids] = True
    atom_array.set_annotation("token_center_atom", token_center_atoms)

    # Remove temporary atom & residue indices
    remove_atom_indices(atom_array)
    remove_residue_indices(atom_array)

    # Add token_position annotation
    add_token_positions(atom_array)


def get_token_count(atom_array: AtomArray) -> int:
    """Get the number of tokens in the input atom array.

    If the input atom array is not yet tokenized, the function will tokenize it.

    Args:
        atom_array (AtomArray):
            AtomArray of the input assembly.

    Returns:
        int: Number of tokens in the input atom array.
    """
    if "token_id" not in atom_array.get_annotation_categories():
        tokenize_atom_array(atom_array)

    return len(np.unique(atom_array.token_id))
