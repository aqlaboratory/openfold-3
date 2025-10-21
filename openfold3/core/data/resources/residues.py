# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Residue constants."""

from enum import IntEnum

import numpy as np


# TODO: remove unused variables from this module
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
STANDARD_PROTEIN_RESIDUES_ORDER = {
    res: i for i, res in enumerate(STANDARD_PROTEIN_RESIDUES_1)
}
STANDARD_RESIDUES_1 = STANDARD_PROTEIN_RESIDUES_1 + STANDARD_NUCLEIC_ACID_RESIDUES
STANDARD_RESIDUES_WITH_GAP_1 = STANDARD_RESIDUES_1 + ["-"]

# Atom names constituting the phosphate in nucleic acids (including alt_atom_ids which
# can't hurt)
NUCLEIC_ACID_PHOSPHATE_OXYGENS = ["OP1", "OP2", "OP3", "O1P", "O2P", "O3P"]

# Token center atoms as defined in AF3 SI, Section 2.6.
TOKEN_CENTER_ATOMS = ["CA", "C1'"]

# Main chain atoms - needed for modified residue tokenization
NUCLEIC_ACID_MAIN_CHAIN_ATOMS = [
    "C1'",
    "C2'",
    "C3'",
    "C4'",
    "C5'",
    "O3'",
    "O4'",
    "O5'",
    "P",
    "OP1",
    "OP2",
]
PROTEIN_MAIN_CHAIN_ATOMS = ["N", "C", "CA", "O"]
PHOSPHODIESTER_BOND_ATOMS = ["P", "O3'", "O5'"]
PEPTIDE_BOND_ATOMS = ["N", "C"]

# Which atoms are considered leaving atoms (displaced by canonical bond formation)
MOLECULE_TYPE_TO_LEAVING_ATOMS = {
    MoleculeType.PROTEIN: ["OXT"],
    MoleculeType.DNA: ["OP3", "O3P"],
    MoleculeType.RNA: ["OP3", "O3P"],
}

# Protein residue maps
PROTEIN_RESTYPE_1TO3 = {
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
PROTEIN_RESTYPE_3TO1 = {v: k for k, v in PROTEIN_RESTYPE_1TO3.items()}

DNA_RESTYPE_1TO3 = {
    "A": "DA",
    "G": "DG",
    "C": "DC",
    "T": "DT",
    "N": "DN",
}
DNA_RESTYPE_3TO1 = {v: k for k, v in DNA_RESTYPE_1TO3.items()}

RNA_RESTYPE_1TO3 = {
    "A": "A",
    "G": "G",
    "C": "C",
    "U": "U",
    "N": "N",
}
RNA_RESTYPE_3TO1 = {v: k for k, v in RNA_RESTYPE_1TO3.items()}

# One-hot residue mappings
RESTYPE_INDEX_3 = {k: v for v, k in enumerate(STANDARD_RESIDUES_WITH_GAP_3)}
# No RESTYPE_INDEX_1 as we need to differentiate between overlapping 1-letter codes
# for different molecule types

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


def get_mol_residue_index_mappings() -> tuple[dict, dict, dict, dict]:
    """Get mappings from molecule type to residue indices.

    Returns:
        tuple[dict, dict, dict, dict]:
            Tuple containing
                - Mapping for each molecule type from the molecule alphabet to the full
                  shared alphabet.
                - Mapping for each molecule type from the 3-letter molecule alphabet the
                  sorted 3-letter molecule alphabet.
                - Mapping for each molecule type from the 1-letter molecule alphabet the
                  sorted 1-letter molecule alphabet.
                - Mapping for each molecule type from the molecule alphabet to the full
                    shared alphabet.
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
    molecule_type_to_residues_pos_map = {}
    for moltype in MoleculeType:
        residue_pos_map = {}
        for residue, residue_idx in zip(
            MOLECULE_TYPE_TO_RESIDUES_1[moltype],
            molecule_type_to_residues_pos[moltype],
            strict=False,
        ):
            residue_pos_map[residue] = residue_idx
        molecule_type_to_residues_pos_map[moltype] = residue_pos_map
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
        molecule_type_to_residues_pos_map,
    )


(
    MOLECULE_TYPE_TO_RESIDUES_POS,
    MOLECULE_TYPE_TO_ARGSORT_RESIDUES_3,
    MOLECULE_TYPE_TO_ARGSORT_RESIDUES_1,
    MOLECULE_TYPE_TO_RESIDUES_POS_MAP,
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


# TODO: make a reusable primitive type for this function
def map_str_array_to_idx_array(
    msa_array: np.ndarray[str], molecule_type: MoleculeType
) -> np.ndarray[int]:
    """Creates an integer MSA array from a 1-character string MSA array.

    The mapping is done onto the global molecule alphabet of all molecule types,
    STANDARD_RESIDUES_WITH_GAP_1. Given that some characters in this alphabet are
    repeated for different molecule types (A, C, G, N shared by RNA and PROTEIN),
    the molecule_type argument is used to determine the correct mapping.

    Args:
        msa_array (np.ndarray[str]):
            String MSA array.
        molecule_type (MoleculeType):
            The molecule type of the MSA.

    Returns:
        np.ndarray[int]:
            Integer MSA array.
    """
    # Create container with full gap indices
    msa_idx = np.full(
        msa_array.shape,
        np.where(np.array(STANDARD_RESIDUES_WITH_GAP_1) == "-")[0].item(),
        dtype=MOLECULE_TYPE_TO_RESIDUES_POS[molecule_type].dtype,
    )
    # For each residue in the molecule type's alphabet, replace the corresponding
    # positions with their index
    for residue, idx in MOLECULE_TYPE_TO_RESIDUES_POS_MAP[molecule_type].items():
        # Replace all but the gap positions
        if residue != "-":
            msa_idx[msa_array == residue] = idx
    # Replace positions not in the molecule type's alphabet with the unknown index
    msa_idx[~np.isin(msa_array, MOLECULE_TYPE_TO_RESIDUES_1[molecule_type])] = (
        MOLECULE_TYPE_TO_RESIDUES_POS_MAP[molecule_type][
            MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1[molecule_type]
        ]
    )

    return msa_idx


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
    return PROTEIN_RESTYPE_3TO1.get(key, PROTEIN_RESTYPE_3TO1["UNK"])


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
    return PROTEIN_RESTYPE_1TO3.get(key, PROTEIN_RESTYPE_1TO3["X"])


# Maximum accesible surface area for residues
# The values are taken from https://github.com/biopython/biopython/blob/master/Bio/Data/PDBData.py
RESIDUE_SASA_SCALES = {
    # Ahmad: Ahmad et al. 2003 https://doi.org/10.1002/prot.10328
    "Ahmad": {
        "ALA": 110.2,
        "ARG": 229.0,
        "ASN": 146.4,
        "ASP": 144.1,
        "CYS": 140.4,
        "GLN": 178.6,
        "GLU": 174.7,
        "GLY": 78.7,
        "HIS": 181.9,
        "ILE": 183.1,
        "LEU": 164.0,
        "LYS": 205.7,
        "MET": 200.1,
        "PHE": 200.7,
        "PRO": 141.9,
        "SER": 117.2,
        "THR": 138.7,
        "TRP": 240.5,
        "TYR": 213.7,
        "VAL": 153.7,
    },
    # Miller max acc: Miller et al. 1987 https://doi.org/10.1016/0022-2836(87)90038-6
    "Miller": {
        "ALA": 113.0,
        "ARG": 241.0,
        "ASN": 158.0,
        "ASP": 151.0,
        "CYS": 140.0,
        "GLN": 189.0,
        "GLU": 183.0,
        "GLY": 85.0,
        "HIS": 194.0,
        "ILE": 182.0,
        "LEU": 180.0,
        "LYS": 211.0,
        "MET": 204.0,
        "PHE": 218.0,
        "PRO": 143.0,
        "SER": 122.0,
        "THR": 146.0,
        "TRP": 259.0,
        "TYR": 229.0,
        "VAL": 160.0,
    },
    # Sander: Sander & Rost 1994 https://doi.org/10.1002/prot.340200303
    "Sander": {
        "ALA": 106.0,
        "ARG": 248.0,
        "ASN": 157.0,
        "ASP": 163.0,
        "CYS": 135.0,
        "GLN": 198.0,
        "GLU": 194.0,
        "GLY": 84.0,
        "HIS": 184.0,
        "ILE": 169.0,
        "LEU": 164.0,
        "LYS": 205.0,
        "MET": 188.0,
        "PHE": 197.0,
        "PRO": 136.0,
        "SER": 130.0,
        "THR": 142.0,
        "TRP": 227.0,
        "TYR": 222.0,
        "VAL": 142.0,
    },
    # Wilke: Tien et al. 2013 https://doi.org/10.1371/journal.pone.0080635
    "Wilke": {
        "ALA": 129.0,
        "ARG": 274.0,
        "ASN": 195.0,
        "ASP": 193.0,
        "CYS": 167.0,
        "GLN": 225.0,
        "GLU": 223.0,
        "GLY": 104.0,
        "HIS": 224.0,
        "ILE": 197.0,
        "LEU": 201.0,
        "LYS": 236.0,
        "MET": 224.0,
        "PHE": 240.0,
        "PRO": 159.0,
        "SER": 155.0,
        "THR": 172.0,
        "TRP": 285.0,
        "TYR": 263.0,
        "VAL": 174.0,
    },
}
