import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

# Standard residues as defined in AF3 SI, Table 13
STANDARD_PROTEIN_RESIDUES = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
]
STANDARD_NUCLEIC_ACID_RESIDUES = [
    "A",
    "G",
    "C",
    "U",
    "DA",
    "DG",
    "DC",
    "DT",
    "N",
    "DN",
]
STANDARD_RESIDUES = STANDARD_PROTEIN_RESIDUES + STANDARD_NUCLEIC_ACID_RESIDUES

# Token center atoms as defined in AF3 SI, Section 2.6.
TOKEN_CENTER_ATOMS = ["CA", "C1'"]


def tokenize_atomarray(atomarray: AtomArray):
    """Generate token ids and token center atom annotations for a biotite AtomArray.

    Args:
        atomarray (AtomArray): biotite atom array of the first bioassembly of a PDB entry

    Returns:
        AtomArray: biotite atom array with added 'af3_token_id' and 'af3_token_center_atom' annotations
    """

    # Get standard residues
    n_atoms = len(atomarray)
    atomidx = np.arange(n_atoms)
    residue_ids, residue_names = struc.get_residues(atomarray)
    standard_residue_ids = residue_ids[np.isin(residue_names, STANDARD_RESIDUES)]

    # Get ids where residue-tokens start
    is_standard_residue_atom = np.isin(atomarray.res_id, standard_residue_ids)
    standard_residue_atom_ids = atomidx[is_standard_residue_atom]
    residue_token_start_ids = np.unique(
        struc.get_residue_starts_for(atomarray, standard_residue_atom_ids)
    )

    # Get atom-token ids
    atom_token_start_ids = atomidx[is_standard_residue_atom == False]

    # Combine for all token start ids
    all_token_start_ids = np.sort(
        np.concatenate([residue_token_start_ids, atom_token_start_ids])
    )

    # Create token index
    token_id_repeats = np.diff(np.append(all_token_start_ids, n_atoms))
    token_ids_per_atom = np.repeat(np.arange(len(token_id_repeats)), token_id_repeats)
    atomarray.set_annotation("af3_token_id", token_ids_per_atom)

    # Create token center atom annotation
    af3_token_center_atoms = np.repeat(True, n_atoms)
    af3_token_center_atoms[is_standard_residue_atom] = np.isin(
        atomarray[is_standard_residue_atom].atom_name, TOKEN_CENTER_ATOMS
    )
    atomarray.set_annotation("af3_token_center_atom", af3_token_center_atoms)

    return atomarray


def chain_assignment(atomarray: AtomArray):
    """Generate chain ids and molecule types for a biotite AtomArray.

    Separate chain ids are given to each protein chain, nucleic acid chain and
    non-covalent ligands including lipids, glycans and small molecules. TODO Covalently
    bound ligands are assigned the id of the chain they are bound to if they have
    less than n atoms, otherwise they are assigned a separate chain id.

    Args:
        atomarray (AtomArray): biotite atom array of the first bioassembly of a PDB entry

    Returns:
        AtomArray: biotite atom array with added 'af3_chain_id' and 'af3_molecule_type' annotations
    """

    # TODO For now, only consider separate chains as assigned by biotite (disregard covalent ligands etc)
    # Get chain id start indices
    n_atoms = len(atomarray)
    chain_start_ids = struc.get_chain_starts(atomarray)

    # Create chain ids
    chain_id_repeats = np.diff(np.append(chain_start_ids, n_atoms))
    chain_ids_per_atom = np.repeat(np.arange(len(chain_id_repeats)), chain_id_repeats)
    atomarray.set_annotation("af3_chain_id", chain_ids_per_atom)

    # Create molecule type annotation
    molecule_types = np.zeros(len(atomarray))
    for start_id, end_id in zip(chain_start_ids, chain_start_ids + chain_id_repeats):
        residues_in_chain = set(atomarray[start_id:end_id].res_name)
        # Assign protein
        if residues_in_chain & set(STANDARD_PROTEIN_RESIDUES):
            molecule_types[start_id:end_id] = 0
        # Assign nucleic acid
        elif residues_in_chain & set(STANDARD_NUCLEIC_ACID_RESIDUES):
            molecule_types[start_id:end_id] = 1
        # Assign ligand
        else:
            molecule_types[start_id:end_id] = 2
    # TODO need to add annotation for covalently modified residues as they are tokenized atomically
    atomarray.set_annotation("af3_molecule_type", molecule_types)
    return atomarray
