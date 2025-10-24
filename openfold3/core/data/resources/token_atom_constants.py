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

PROTEIN_RESTYPES = [
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

RNA_NUCLEOTIDE_TYPES = [
    "A",
    "G",
    "C",
    "U",
    "N",
]

DNA_NUCLEOTIDE_TYPES = [
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",
]

TOKEN_TYPES = PROTEIN_RESTYPES + RNA_NUCLEOTIDE_TYPES + DNA_NUCLEOTIDE_TYPES
TOKEN_TYPES_WITH_GAP = TOKEN_TYPES + ["GAP"]

AA_NAME_TO_ATOM_NAMES = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
    ],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "UNK": ["N", "CA", "C", "O", "CB", "CG"],
}

RNA_BACKBONE_ATOMS = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
]

DNA_BACKBONE_ATOMS = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "C1'",
]

NUCLEOTIDE_ATOMS = {
    "A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
    "DA": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
    "DC": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "DG": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
    "DT": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],
    "N": [],
    "DN": [],
}

NUCLEOTIDE_NAME_TO_ATOM_NAMES = {
    n: (RNA_BACKBONE_ATOMS + a if n in RNA_NUCLEOTIDE_TYPES else DNA_BACKBONE_ATOMS + a)
    for n, a in NUCLEOTIDE_ATOMS.items()
}


TOKEN_NAME_TO_ATOM_NAMES = {
    **AA_NAME_TO_ATOM_NAMES,
    **NUCLEOTIDE_NAME_TO_ATOM_NAMES,
    "GAP": [],
}


def get_atom_name_to_index(atom_name):
    """
    Get atom index (with mask) based on residue type.
    """
    indices = []
    mask = []
    for name in TOKEN_TYPES_WITH_GAP:
        try:
            indices.append(TOKEN_NAME_TO_ATOM_NAMES[name].index(atom_name))
            mask.append(1)
        except (KeyError, ValueError):
            indices.append(-1)
            mask.append(0)
    return {"index": indices, "mask": mask}


atom_name_to_index_by_restype = {
    atom_name: get_atom_name_to_index(atom_name=atom_name)
    for atom_name in ["N", "CA", "C", "CB", "C1'", "C3'", "C4'", "C2", "C4"]
}
