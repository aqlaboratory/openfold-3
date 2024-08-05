"""This module contains building blocks for tokenization and chain assignment."""

# TODO add license

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.resources.tables import (
    MOLECULE_TYPE_ID_DNA,
    MOLECULE_TYPE_ID_LIGAND,
    MOLECULE_TYPE_ID_PROTEIN,
    MOLECULE_TYPE_ID_RNA,
    NUCLEIC_ACID_MAIN_CHAIN_ATOMS,
    PROTEIN_MAIN_CHAIN_ATOMS,
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES,
    STANDARD_RESIDUES,
    STANDARD_RNA_RESIDUES,
    TOKEN_CENTER_ATOMS,
)


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

    # Create atom id annotation and auxiliary residue id annotation
    # The auxiliary residue id is used to tokenize covalently modified residues
    # per atom and is removed afterwards
    atom_array.set_annotation("af3_atom_id", np.arange(len(atom_array)))
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
    standard_residue_atom_ids = atom_array.af3_atom_id[is_standard_residue_atom]
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
    chain_ids = atom_array.af3_chain_id
    is_different_chain = chain_ids[bondlist[:, 0]] != chain_ids[bondlist[:, 1]]
    # - two non-heteroatoms in the same chain but side chains of different residues
    #   (standard residues covalently linking non-consecutive residues in the same
    #   chain)
    atom_names = atom_array.atom_name
    molecule_types = atom_array.af3_molecule_type
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
    ].af3_atom_id

    # Remove the corresponding residue token start ids
    modified_residue_token_start_ids = np.unique(
        struc.get_residue_starts_for(atom_array, atomized_residue_token_start_ids)
    )
    residue_token_start_ids = residue_token_start_ids[
        ~np.isin(residue_token_start_ids, modified_residue_token_start_ids)
    ]
    # Remove auxiliary residue id annotation
    atom_array.del_annotation("af3_aux_residue_id")

    # Get atom-token ids
    atom_token_start_ids = atom_array.af3_atom_id[~is_standard_residue_atom]

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

    return None


def assign_chains(atom_array: AtomArray):
    """Generates chain ids and molecule types for a biotite atom_array.

    Separate chain ids are given to each protein chain, nucleic acid chain and
    non-covalent ligands including lipids, glycans and small molecules. For ligands
    covalently bound to polymers, we follow the PDB auto-assigned chain ids: small PTMs
    and ligands are assigned to the same chain as the polymer they are bound to, whereas
    glycans are assigned to a separate chain. Note: chain assignment needs to be ran
    before tokenization to create certain features used by the tokenizer.

    Updates the input biotite AtomArray with added 'af3_chain_id' and
    'af3_molecule_type' annotations and a 'chain_id_map' class attribute in-place.

    Args:
        atom_array (AtomArray): biotite atom array of the first bioassembly of a PDB
        entry

    Returns:
        None
    """

    # Get chain id start indices
    chain_start_ids = struc.get_chain_starts(atom_array)

    # Create chain ids
    # This is necessary to do because some PDB-assigned homomeric chain
    # IDs are not unique
    chain_id_repeats = np.diff(np.append(chain_start_ids, len(atom_array)))
    chain_ids_per_atom = np.repeat(np.arange(len(chain_id_repeats)), chain_id_repeats)
    atom_array.set_annotation("af3_chain_id", chain_ids_per_atom)

    # Create chain id map from our chain IDs to auto-assigned PDB chain IDs
    atom_array.chain_id_map = {
        k: v for (k, v) in list(set(zip(atom_array.af3_chain_id, atom_array.chain_id)))
    }

    # Create molecule type annotation
    molecule_types = np.zeros(len(atom_array))
    for start_id, end_id in zip(chain_start_ids, chain_start_ids + chain_id_repeats):
        residues_in_chain = set(atom_array[start_id:end_id].res_name)
        # Assign protein
        if residues_in_chain & set(STANDARD_PROTEIN_RESIDUES):
            molecule_types[start_id:end_id] = MOLECULE_TYPE_ID_PROTEIN
        # Assign RNA
        elif residues_in_chain & set(STANDARD_RNA_RESIDUES):
            molecule_types[start_id:end_id] = MOLECULE_TYPE_ID_RNA
        # Assign DNA
        elif residues_in_chain & set(STANDARD_DNA_RESIDUES):
            molecule_types[start_id:end_id] = MOLECULE_TYPE_ID_DNA
        # Assign ligand
        else:
            molecule_types[start_id:end_id] = MOLECULE_TYPE_ID_LIGAND

    atom_array.set_annotation("af3_molecule_type", molecule_types)

    return None
