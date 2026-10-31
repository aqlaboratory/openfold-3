import random
from collections import Counter

import pytest
from biotite.interface.rdkit import to_mol
from rdkit import Chem

from openfold3.core.data.primitives.structure.query import (
    atom_array_from_ccd_code,
    structure_with_ref_mol_from_ccd_code,
    structure_with_ref_mol_from_smiles,
    structure_with_ref_mols_from_sequence,
)
from openfold3.core.data.resources.residues import (
    DNA_RESTYPE_1TO3,
    PROTEIN_RESTYPE_1TO3,
    RNA_RESTYPE_1TO3,
    MoleculeType,
)


@pytest.mark.parametrize(
    "smiles, ccd_code",
    [
        # Simple test cases
        ("CCO", "EOH"),  # Ethanol
        # Pat Walter's CYP substrates
        ("CC(C)(C)C(=O)Nc1cc(ccc1n2ccnc2)C(F)(F)F", "A1ASV"),  # cyp3a4_9bv5
        ("Cc1nc(cs1)c2ccc(cc2)n3c(cnn3)c4ccc(cc4)OC", "A1ASU"),  # cyp3a4_9bv6
        ("c1ccc(c(c1)CCC(=O)NC[C@@H]2Cc3cccc(c3O2)c4cccnc4)Cl", "A1AST"),  # cyp3a4_9bv7
        ("c1cc(c(cc1C(F)(F)F)NC(=O)C2CCC2)n3ccnc3", "A1ASS"),  # cyp3a4_9bv8
        ("c1ccc(c(c1)NC(=O)Nc2cc(ccc2n3ccnc3)C(F)(F)F)Cl", "A1ASR"),  # cyp3a4_9bv9
        ("c1ccc(c(c1)CCCl)NC(=O)Nc2cc(ccc2n3ccnc3)C(F)(F)F", "A1ASQ"),  # cyp3a4_9bva
        ("c1ccc(c(c1)CC(=O)Nc2cc(ccc2n3ccnc3)C(F)(F)F)Cl", "A1ASP"),  # cyp3a4_9bvb
        (
            "c1ccc(c(c1)N(Cc2cccc(c2)O)C(=O)Nc3cc(ccc3n4ccnc4)C(F)(F)F)Cl",
            "A1ASO",
        ),  # cyp3a4_9bvc
        (
            "c1ccc(c(c1)NC(=O)N(Cc2cccc(c2)O)c3cc(ccc3n4ccnc4)C(F)(F)F)Cl",
            "A1BNX",
        ),  # cyp3a4_9ms1
        (
            "c1ccc(cc1)C(c2ccccc2)([C@@H]3CCN(C3)CCc4ccc5c(c4)CCO5)C(=O)N",
            "A1CIW",
        ),  # cyp3a4_9plk
    ],
)
def test_consistent_structure_from_smiles_and_ccd_code(smiles, ccd_code):
    struct_from_smiles = structure_with_ref_mol_from_smiles(smiles, chain_id="X")
    struct_from_ccd = structure_with_ref_mol_from_ccd_code(ccd_code, chain_id="X")

    # Ideally, one day, we'll be able to do just this
    # from openfold3.tests.utils import custom_assert_utils
    # custom_assert_utils.assert_atomarray_equal(struct_from_smiles.atom_array, struct_from_ccd.atom_array)

    assert len(struct_from_smiles.atom_array) == len(struct_from_ccd.atom_array)

    assert Counter(struct_from_smiles.atom_array.element) == Counter(
        struct_from_ccd.atom_array.element
    )


def _stereocenter_orientations(mol: Chem.Mol) -> list[str]:
    return sorted(
        orientation
        for _, orientation in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    )


def _stereo_bond_orientations(mol: Chem.Mol) -> list[Chem.BondStereo]:
    return sorted(
        (
            bond.GetStereo()
            for bond in mol.GetBonds()
            if bond.GetStereo() != Chem.BondStereo.STEREONONE
        ),
        key=int,
    )


def test_ccd_tetrahedral_stereochemistry_matches_smiles():
    ccd_structure = structure_with_ref_mol_from_ccd_code("DAS", chain_id="X")
    smiles_structure = structure_with_ref_mol_from_smiles(
        "N[C@H](CC(O)=O)C(O)=O", chain_id="X"
    )

    ccd_orientations = _stereocenter_orientations(
        ccd_structure.processed_reference_mols[0].mol
    )
    smiles_orientations = _stereocenter_orientations(
        smiles_structure.processed_reference_mols[0].mol
    )

    assert smiles_orientations
    assert ccd_orientations == smiles_orientations


def test_ccd_double_bond_stereochemistry_matches_smiles():
    ccd_structure = structure_with_ref_mol_from_ccd_code("FUM", chain_id="X")
    smiles_structure = structure_with_ref_mol_from_smiles(
        "O=C(O)/C=C/C(=O)O", chain_id="X"
    )

    ccd_orientations = _stereo_bond_orientations(
        ccd_structure.processed_reference_mols[0].mol
    )
    smiles_orientations = _stereo_bond_orientations(
        smiles_structure.processed_reference_mols[0].mol
    )

    assert smiles_orientations
    assert ccd_orientations == smiles_orientations


def _named_stereocenters_from_3d(mol: Chem.Mol, atom_names: list[str]) -> dict:
    """CIP label of each stereocenter as set by the 3D coordinates, keyed by name."""
    mol = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return {
        atom_names[idx]: label
        for idx, label in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    }


_STANDARD_POLYMER_RESIDUES = [
    pytest.param(poly_type, res_1, res_3, id=f"{poly_type.name}-{res_3}")
    for poly_type, mapping in [
        (MoleculeType.PROTEIN, PROTEIN_RESTYPE_1TO3),
        (MoleculeType.RNA, RNA_RESTYPE_1TO3),
        (MoleculeType.DNA, DNA_RESTYPE_1TO3),
    ]
    for res_1, res_3 in mapping.items()
    if res_3 not in ("UNK", "N", "DN")
]


@pytest.mark.parametrize("poly_type, res_1, res_3", _STANDARD_POLYMER_RESIDUES)
def test_sequence_residue_conformer_keeps_ccd_stereochemistry(poly_type, res_1, res_3):
    """Reference conformers of residues built from a sequence must keep the CCD
    stereochemistry (e.g. L-amino acids, D-ribose), independent of the seed.

    Previously stereochemistry was only assigned from the CCD coordinates for
    ligands, so polymer residue conformers were regenerated with random chirality.
    """
    ccd_array = atom_array_from_ccd_code(res_3, chain_id="A", molecule_type=poly_type)
    ccd_mol = to_mol(ccd_array, kekulize=True)
    Chem.SanitizeMol(ccd_mol)
    expected = _named_stereocenters_from_3d(ccd_mol, list(ccd_array.atom_name))

    for seed in range(3):
        structure = structure_with_ref_mols_from_sequence(
            res_1, poly_type, chain_id="A", rng=random.Random(seed)
        )
        mol = structure.processed_reference_mols[0].mol
        names = [atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()]
        assert _named_stereocenters_from_3d(mol, names) == expected


def test_sequence_alanine_is_l_alanine():
    structure = structure_with_ref_mols_from_sequence(
        "A", MoleculeType.PROTEIN, chain_id="A", rng=random.Random(0)
    )
    mol = structure.processed_reference_mols[0].mol
    names = [atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()]
    assert _named_stereocenters_from_3d(mol, names) == {"CA": "S"}
