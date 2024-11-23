"""Residue constants."""

from enum import IntEnum

import numpy as np


# Molecule type used in tokenization
class MoleculeType(IntEnum):
    PROTEIN = 0
    RNA = 1
    DNA = 2
    LIGAND = 3


# _chem_comp.type to molecule type mapping
# see https://mmcif.wwpdb.org/dictionaries/mmcif_std.dic/Items/_chem_comp.type.html

CHEM_COMP_TYPE_TO_MOLECULE_TYPE = {
    "PEPTIDE LINKING": MoleculeType.PROTEIN,
    "PEPTIDE-LIKE": MoleculeType.PROTEIN,
    "D-PEPTIDE LINKING": MoleculeType.PROTEIN,
    "L-PEPTIDE LINKING": MoleculeType.PROTEIN,
    "D-BETA-PEPTIDE, C-GAMMA LINKING": MoleculeType.PROTEIN,
    "D-GAMMA-PEPTIDE, C-DELTA LINKING": MoleculeType.PROTEIN,
    "L-BETA-PEPTIDE, C-GAMMA LINKING": MoleculeType.PROTEIN,
    "L-GAMMA-PEPTIDE, C-DELTA LINKING": MoleculeType.PROTEIN,
    "D-PEPTIDE NH3 AMINO TERMINUS": MoleculeType.PROTEIN,
    "D-PEPTIDE COOH CARBOXY TERMINUS": MoleculeType.PROTEIN,
    "L-PEPTIDE NH3 AMINO TERMINUS": MoleculeType.PROTEIN,
    "L-PEPTIDE COOH CARBOXY TERMINUS": MoleculeType.PROTEIN,
    "RNA LINKING": MoleculeType.RNA,
    "L-RNA LINKING": MoleculeType.RNA,
    "RNA OH 5 PRIME TERMINUS": MoleculeType.RNA,
    "RNA OH 3 PRIME TERMINUS": MoleculeType.RNA,
    "DNA LINKING": MoleculeType.DNA,
    "L-DNA LINKING": MoleculeType.DNA,
    "DNA OH 5 PRIME TERMINUS": MoleculeType.DNA,
    "DNA OH 3 PRIME TERMINUS": MoleculeType.DNA,
    "SACCHARIDE": MoleculeType.LIGAND,
    "L-SACCHARIDE": MoleculeType.LIGAND,
    "D-SACCHARIDE": MoleculeType.LIGAND,
    "L-SACCHARIDE, ALPHA LINKING": MoleculeType.LIGAND,
    "L-SACCHARIDE, BETA LINKING": MoleculeType.LIGAND,
    "D-SACCHARIDE, ALPHA LINKING": MoleculeType.LIGAND,
    "D-SACCHARIDE, BETA LINKING": MoleculeType.LIGAND,
    "NON-POLYMER": MoleculeType.LIGAND,
    "OTHER": MoleculeType.LIGAND,
}


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
NUCLEIC_ACID_PHOSPHATE_OXYGENS = ["OP1", "OP2", "OP3", "O1P", "O2P", "O3P"]

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

# Molecule-type to residue mappings
MOLECULE_TYPE_TO_RESIDUES_3 = {
    MoleculeType.PROTEIN: np.array(STANDARD_PROTEIN_RESIDUES_3 + ["GAP"]),
    MoleculeType.RNA: np.array(STANDARD_RNA_RESIDUES + ["GAP"]),
    MoleculeType.DNA: np.array(STANDARD_DNA_RESIDUES + ["GAP"]),
    MoleculeType.LIGAND: np.array(["UNK", "GAP"]),
}
MOLECULE_TYPE_TO_RESIDUES_1 = {
    MoleculeType.PROTEIN: np.array(STANDARD_PROTEIN_RESIDUES_1 + ["-"]),
    MoleculeType.RNA: np.array(STANDARD_RNA_RESIDUES + ["-"]),
    MoleculeType.DNA: np.array(STANDARD_DNA_RESIDUES + ["-"]),
    MoleculeType.LIGAND: np.array(["X", "-"]),
}


def get_mol_residue_index_mappings() -> tuple[dict, dict, dict]:
    """Get mappings from molecule type to residue indices.

    Returns:
        tuple[dict, dict, dict]:
            Tuple containing
                - Mapping for each molecule type from the molecule alphabet to the full
                  shared alphabet.
                - Mapping for each molecule type from the 3-letter molecule alphabet the
                  sorted 3-letter molecule alphabet.
                - Mapping for each molecule type from the 1-letter molecule alphabet the
                  sorted 1-letter molecule alphabet.
    """
    _prot_a_len = len(STANDARD_PROTEIN_RESIDUES_1)
    _rna_a_len = len(STANDARD_RNA_RESIDUES)
    _dna_a_len = len(STANDARD_DNA_RESIDUES)
    _gap_pos = len(STANDARD_RESIDUES_WITH_GAP_1) - 1
    molecule_type_to_residues_pos = {
        MoleculeType.PROTEIN: np.concatenate(
            [
                np.arange(0, _prot_a_len),
                np.array([_gap_pos]),
            ]
        ),
        MoleculeType.RNA: np.concatenate(
            [
                np.arange(
                    _prot_a_len,
                    _prot_a_len + _rna_a_len,
                ),
                np.array([_gap_pos]),
            ]
        ),
        MoleculeType.DNA: np.concatenate(
            [
                np.arange(
                    _prot_a_len + _rna_a_len,
                    _prot_a_len + _rna_a_len + _dna_a_len,
                ),
                np.array([_gap_pos]),
            ]
        ),
        MoleculeType.LIGAND: np.concatenate(
            [
                np.where(np.array(STANDARD_PROTEIN_RESIDUES_1) == "X")[0],
                np.array([_gap_pos]),
            ]
        ),
    }

    molecule_type_to_argsort_residues_3 = {
        k: np.argsort(v) for k, v in MOLECULE_TYPE_TO_RESIDUES_3.items()
    }
    molecule_type_to_argsort_residues_1 = {
        k: np.argsort(v) for k, v in MOLECULE_TYPE_TO_RESIDUES_1.items()
    }

    return (
        molecule_type_to_residues_pos,
        molecule_type_to_argsort_residues_3,
        molecule_type_to_argsort_residues_1,
    )


(
    MOLECULE_TYPE_TO_RESIDUES_POS,
    MOLECULE_TYPE_TO_ARGSORT_RESIDUES_3,
    MOLECULE_TYPE_TO_ARGSORT_RESIDUES_1,
) = get_mol_residue_index_mappings()
MOLECULE_TYPE_TO_UKNOWN_RESIDUES_3 = {
    MoleculeType.PROTEIN: "UNK",
    MoleculeType.RNA: "N",
    MoleculeType.DNA: "DN",
    MoleculeType.LIGAND: "UNK",
}
MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1 = {
    MoleculeType.PROTEIN: "X",
    MoleculeType.RNA: "N",
    MoleculeType.DNA: "DN",
    MoleculeType.LIGAND: "X",
}


@np.vectorize
def get_with_unknown_3_to_idx(key: str) -> int:
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
def get_with_unknown_1_to_idx(key: str) -> int:
    """Wraps a RESTYPE_INDEX_1 dictionary lookup with a default value of "UNK".

    Args:
        key (str):
            Key to look up in the dictionary.

    Returns:
        int:
            Index of residue type.
    """
    return RESTYPE_INDEX_1.get(key, RESTYPE_INDEX_1["X"])


@np.vectorize
def get_with_unknown_3_to_1(key: str) -> str:
    """Maps a 3-letter residue array to 1-letter residue array.

    Args:
        key (np.ndarray):
            3-letter residue array.

    Returns:
        np.ndarray:
            1-letter residue array.
    """
    return RESTPYE_3TO1.get(key, RESTPYE_3TO1["UNK"])


@np.vectorize
def get_with_unknown_1_to_3(key: str) -> str:
    """Maps a 3-letter residue array to 1-letter residue array.

    Args:
        key (np.ndarray):
            3-letter residue array.

    Returns:
        np.ndarray:
            1-letter residue array.
    """
    return RESTYPE_1TO3.get(key, RESTYPE_1TO3["X"])
