import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

# Standard residues as defined in AF3 SI, Table 13
STANDARD_RESIDUES = [
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

TOKEN_CENTER_ATOM_NAMES = ["CA", "C1'"]


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
        atomarray[is_standard_residue_atom].atom_name, TOKEN_CENTER_ATOM_NAMES
    )
    atomarray.set_annotation("af3_token_center_atom", af3_token_center_atoms)

    return atomarray
