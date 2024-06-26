import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.preprocessing.tables import (
    STANDARD_RNA_RESIDUES,
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES,
    STANDARD_RESIDUES,
    TOKEN_CENTER_ATOMS,
)


def tokenize_atom_array(atom_array: AtomArray):
    """Generates token ids and token center atom annotations for a biotite atom_array.

    Updates the input biotite atom array with added 'af3_token_id' and 'af3_token_center_atom'
    and 'af3_atom_id' annotations in-place.

    Args:
        atom_array (AtomArray): biotite atom array of the first bioassembly of a PDB entry

    Returns:
        None
    """

    # Create atom id annotation
    atom_array.set_annotation("af3_atom_id", np.arange(len(atom_array)))

    # Get standard residues
    n_atoms = len(atom_array)
    atomidx = np.arange(n_atoms)
    residue_ids, residue_names = struc.get_residues(atom_array)
    standard_residue_ids = residue_ids[np.isin(residue_names, STANDARD_RESIDUES)]

    # Get ids where residue-tokens start
    is_standard_residue_atom = np.isin(atom_array.res_id, standard_residue_ids)
    standard_residue_atom_ids = atomidx[is_standard_residue_atom]
    residue_token_start_ids = np.unique(
        struc.get_residue_starts_for(atom_array, standard_residue_atom_ids)
    )

    # TODO this needs to be adapted to covalently modified standard residues !!!
    # !!! Misses unannotated bonds
    # Filter out cooordinate bonds
    # Get atom-token ids
    atom_token_start_ids = atomidx[is_standard_residue_atom == False]

    # Combine for all token start ids
    all_token_start_ids = np.sort(
        np.concatenate([residue_token_start_ids, atom_token_start_ids])
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
    atom_array.set_annotation("af3_token_center_atom", af3_token_center_atoms)

    return None


def assign_chains(atom_array: AtomArray):
    """Generates chain ids and molecule types for a biotite atom_array.

    Separate chain ids are given to each protein chain, nucleic acid chain and
    non-covalent ligands including lipids, glycans and small molecules. TODO Covalently
    bound ligands are assigned the id of the chain they are bound to if they have
    less than n atoms, otherwise they are assigned a separate chain id.

    Updates the input biotite AtomArray with added 'af3_chain_id' and 'af3_molecule_type' 
    annotations and a 'chain_id_map' class attribute in-place.

    Args:
        atom_array (AtomArray): biotite atom array of the first bioassembly of a PDB entry

    Returns:
        None
    """

    # TODO For now, only consider separate chains as assigned by biotite (disregard covalent ligands etc)
    # Get chain id start indices
    n_atoms = len(atom_array)
    chain_start_ids = struc.get_chain_starts(atom_array)

    # Create chain ids
    # This is necessary to do because some PDB-assigned homomeric chain IDs are not unique
    chain_id_repeats = np.diff(np.append(chain_start_ids, n_atoms))
    chain_ids_per_atom = np.repeat(np.arange(len(chain_id_repeats)), chain_id_repeats)
    atom_array.set_annotation("af3_chain_id", chain_ids_per_atom)

    # Create chain id map from our chain IDs to auto-assigned PDB chain IDs
    atom_array.chain_id_map = {k: v for (k, v) in list(set(zip(atom_array.af3_chain_id, atom_array.chain_id)))}

    # Create molecule type annotation
    molecule_types = np.zeros(len(atom_array))
    for start_id, end_id in zip(chain_start_ids, chain_start_ids + chain_id_repeats):
        residues_in_chain = set(atom_array[start_id:end_id].res_name)
        # Assign protein
        if residues_in_chain & set(STANDARD_PROTEIN_RESIDUES):
            molecule_types[start_id:end_id] = 0
        # Assign RNA
        elif residues_in_chain & set(STANDARD_RNA_RESIDUES):
            molecule_types[start_id:end_id] = 1
        # Assign DNA
        elif residues_in_chain & set(STANDARD_DNA_RESIDUES):
            molecule_types[start_id:end_id] = 2
        # Assign ligand
        else:
            molecule_types[start_id:end_id] = 3
    # TODO need to add annotation for covalently modified residues as they are tokenized atomically
    atom_array.set_annotation("af3_molecule_type", molecule_types)

    return None
