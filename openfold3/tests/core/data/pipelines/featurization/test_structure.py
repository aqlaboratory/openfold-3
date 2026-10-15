import pytest
import torch
from biotite.structure import BondList

from openfold3.core.data.pipelines.featurization.structure import (
    featurize_structure_of3,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.core.utils.relpos import relpos_complex

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
    "cyclic_mask",
]

GT_EXPECTED_KEYS = [
    "atom_positions",
    "atom_resolved_mask",
]


def _featurize(
    _make_atom_array, n_tokens, is_gt=False, is_cyclic=None, token_budget=None
):
    """Featurize a single-chain protein AtomArray of `n_tokens` single-atom tokens.

    `is_cyclic=None` leaves the annotation off the AtomArray entirely, which is what
    every structure parsed from mmCIF looks like -- only queries built from the
    inference JSON set it.
    """
    atom_array = _make_atom_array(
        n_tokens,
        entity_id=1,
        molecule_type_id=MoleculeType.PROTEIN,
        is_atomized=[False] * n_tokens,
        is_cyclic=None if is_cyclic is None else [is_cyclic] * n_tokens,
        bonds=BondList(n_tokens),
    )
    return featurize_structure_of3(
        atom_array,
        token_budget if token_budget is not None else n_tokens,
        is_gt=is_gt,
        add_perm_features=False,
    )


# Test case for permutation features not yet added - would need
# to run `assign_mol_permutation_ids`
@pytest.mark.parametrize(
    "is_gt, is_cyclic, expected_keys",
    [
        pytest.param(
            False,
            None,
            BASE_EXPECTED_KEYS + NOT_GT_EXPECTED_KEYS,
            id="inference_unannotated",
        ),
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
            BASE_EXPECTED_KEYS + NOT_GT_EXPECTED_KEYS,
            id="inference_cyclic",
        ),
    ],
)
def test_featurize_structure(_make_atom_array, is_gt, is_cyclic, expected_keys):
    features = _featurize(_make_atom_array, 10, is_gt=is_gt, is_cyclic=is_cyclic)

    # Check that the features dictionary contains expected keys
    assert set(expected_keys).issubset(set(features.keys()))


@pytest.mark.parametrize("is_cyclic", [None, False, True])
def test_cyclic_mask_always_emitted(_make_atom_array, is_cyclic):
    # The model reads batch["cyclic_mask"] unconditionally, so featurization must
    # emit it even for structures that carry no is_cyclic annotation.
    n_tokens = 10
    features = _featurize(_make_atom_array, n_tokens, is_cyclic=is_cyclic)

    cyclic_mask = features["cyclic_mask"]
    assert cyclic_mask.dtype == torch.bool
    assert cyclic_mask.shape == (n_tokens,)
    assert cyclic_mask.all() if is_cyclic else not cyclic_mask.any()


@pytest.mark.parametrize("is_cyclic", [None, True])
def test_cyclic_mask_padded_to_token_budget(_make_atom_array, is_cyclic):
    # Every other per-token feature is padded up to the crop size; if cyclic_mask is
    # not, it desyncs from asym_id/residue_index as soon as a batch is collated.
    n_tokens, token_budget = 6, 10
    features = _featurize(
        _make_atom_array, n_tokens, is_cyclic=is_cyclic, token_budget=token_budget
    )

    cyclic_mask = features["cyclic_mask"]
    assert cyclic_mask.shape == features["residue_index"].shape == (token_budget,)
    # Padding tokens are never cyclic.
    assert not cyclic_mask[n_tokens:].any()


@pytest.mark.parametrize("is_cyclic", [None, False, True])
def test_featurized_output_feeds_relpos(_make_atom_array, is_cyclic):
    # Contract test between the data pipeline and the model: relpos_complex is the
    # first consumer of cyclic_mask, and a missing key there surfaced only once
    # training reached its first validation step.
    n_tokens = 8
    features = _featurize(_make_atom_array, n_tokens, is_cyclic=is_cyclic)
    batch = {k: v.unsqueeze(0) for k, v in features.items()}

    relpos_feats = relpos_complex(batch, max_relative_idx=32, max_relative_chain=2)

    assert relpos_feats.shape[:3] == (1, n_tokens, n_tokens)
