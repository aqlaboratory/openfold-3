from collections import Counter

import pytest

from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mol_from_ccd_code,
    structure_with_ref_mol_from_smiles,
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
    # from openfold3.tests import custom_assert_utils
    # custom_assert_utils.assert_atomarray_equal(struct_from_smiles.atom_array, struct_from_ccd.atom_array)

    assert len(struct_from_smiles.atom_array) == len(struct_from_ccd.atom_array)

    assert Counter(struct_from_smiles.atom_array.element) == Counter(
        struct_from_ccd.atom_array.element
    )
