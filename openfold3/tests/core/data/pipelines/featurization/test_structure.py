import pytest
from biotite.structure import BondList

from openfold3.core.data.pipelines.featurization.structure import (
    featurize_structure_of3,
)
from openfold3.core.data.resources.residues import MoleculeType

BASE_EXPECTED_KEYS = [
    "token_index",
    "restype",
    "is_protein",
    "is_rna",
    "is_dna",
    "is_ligand",
    "is_atomized",
    "token_mask",
    "num_atoms_per_token",
    "start_atom_index",
    "atom_mask",
]

NOT_GT_EXPECTED_KEYS = [
    "residue_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "token_bonds",
    "atom_to_token_index",
]

GT_EXPECTED_KEYS = [
    "atom_positions",
    "atom_resolved_mask",
]

CYCLIC_EXPECTED_KEYS = ["cyclic_mask"]


# Test case for permutation features not yet added - would need
# to run `assign_mol_permutation_ids`
@pytest.mark.parametrize(
    "is_gt, has_cyclic, expected_keys",
    [
        pytest.param(
            False,
            False,
            BASE_EXPECTED_KEYS + NOT_GT_EXPECTED_KEYS,
            id="inference",
        ),
        pytest.param(
            True,
            False,
            BASE_EXPECTED_KEYS + GT_EXPECTED_KEYS,
            id="ground_truth",
        ),
        pytest.param(
            False,
            True,
            BASE_EXPECTED_KEYS + NOT_GT_EXPECTED_KEYS + CYCLIC_EXPECTED_KEYS,
            id="inference_cyclic",
        ),
    ],
)
def test_featurize_structure(_make_atom_array, is_gt, has_cyclic, expected_keys):
    # Create a dummy AtomArray with necessary attributes
    n_tokens = 10
    atom_array = _make_atom_array(
        n_tokens,
        entity_id=1,
        molecule_type_id=MoleculeType.PROTEIN,
        is_atomized=[False] * n_tokens,
        is_cyclic=[has_cyclic] * n_tokens,
        bonds=BondList(n_tokens),
    )

    # Call the featurization function
    features = featurize_structure_of3(
        atom_array, n_tokens, is_gt=is_gt, add_perm_features=False
    )

    # Check that the features dictionary contains expected keys
    assert set(expected_keys).issubset(set(features.keys()))
