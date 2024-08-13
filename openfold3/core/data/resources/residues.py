"""Residue constants."""

from enum import IntEnum

import numpy as np


# Molecule type used in tokenization
class MoleculeType(IntEnum):
    PROTEIN = 0
    RNA = 1
    DNA = 2
    LIGAND = 3


# Standard residues as defined in AF3 SI, Table 13
STANDARD_PROTEIN_RESIDUES_3 = [
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

STANDARD_RNA_RESIDUES = ["A", "G", "C", "U", "N"]
STANDARD_DNA_RESIDUES = ["DA", "DG", "DC", "DT", "DN"]
STANDARD_NUCLEIC_ACID_RESIDUES = STANDARD_RNA_RESIDUES + STANDARD_DNA_RESIDUES
STANDARD_RESIDUES_3 = STANDARD_PROTEIN_RESIDUES_3 + STANDARD_NUCLEIC_ACID_RESIDUES
STANDARD_RESIDUES_WITH_GAP_3 = STANDARD_RESIDUES_3 + ["GAP"]

STANDARD_PROTEIN_RESIDUES_1 = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
]
STANDARD_RESIDUES_1 = STANDARD_PROTEIN_RESIDUES_1 + STANDARD_NUCLEIC_ACID_RESIDUES
STANDARD_RESIDUES_WITH_GAP_1 = STANDARD_RESIDUES_1 + ["-"]

# Atom names constituting the phosphate in nucleic acids (including alt_atom_ids which
# can't hurt)
NUCLEIC_ACID_PHOSPHATE_ATOMS = ["P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P"]

# Token center atoms as defined in AF3 SI, Section 2.6.
TOKEN_CENTER_ATOMS = ["CA", "C1'"]

# Main chain atoms - needed for modified residue tokenization
NUCLEIC_ACID_MAIN_CHAIN_ATOMS = ["C3'", "C4'", "C5'", "O3'", "O5'", "P"]
PROTEIN_MAIN_CHAIN_ATOMS = ["N", "C", "CA", "O"]

# Protein residue maps
RESTYPE_1TO3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
RESTPYE_3TO1 = {v: k for k, v in RESTYPE_1TO3.items()}

# One-hot residue mappings
RESTYPE_INDEX_3 = {k: v for v, k in enumerate(STANDARD_RESIDUES_WITH_GAP_3)}
RESTYPE_INDEX_1 = {k: v for v, k in enumerate(STANDARD_RESIDUES_WITH_GAP_1)}


@np.vectorize
def get_with_unknown_3(key: str) -> int:
    """Wraps a RESTYPE_INDEX_3 dictionary lookup with a default value of "UNK".

    Args:
        key (str):
            Key to look up in the dictionary.

    Returns:
        int:
            Index of residue type.
    """
    return RESTYPE_INDEX_3.get(key, RESTYPE_INDEX_3["UNK"])


@np.vectorize
def get_with_unknown_1(key: str) -> int:
    """Wraps a RESTYPE_INDEX_1 dictionary lookup with a default value of "UNK".

    Args:
        key (str):
            Key to look up in the dictionary.

    Returns:
        int:
            Index of residue type.
    """
    return RESTYPE_INDEX_1.get(key, RESTYPE_INDEX_1["X"])
