import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz
from openfold3.core.data.primitives.permutation.mol_labels import (
    assign_mol_permutation_ids,
)
from openfold3.core.data.resources.residues import RESTYPE_INDEX_3
from openfold3.core.utils.permutation_alignment import multi_chain_permutation_alignment
from tests.custom_assert_utils import assert_atomarray_equal

TEST_DIR = Path("tests/test_data/permutation_alignment")


def test_mol_symmetry_id_assignment():
    """Checks that permutation IDs are correctly assigned.

    This checks that the permutation IDs required to detect symmetry-equivalent parts of
    the AtomArray are working as expected.

    The test case here is a challenging structure with covalent ligands and symmetric
    molecules with different chain order that was manually verified.

    The input AtomArray was properly processed beforehand to have the additional IDs
    (like token IDs & component IDs) required for the permutation ID assignment.
    """
    atom_array_in = read_atomarray_from_npz(
        TEST_DIR / "inputs/npz/7pbd/7pbd_subset.npz"
    )
    atom_array_out = read_atomarray_from_npz(
        TEST_DIR / "outputs/npz/7pbd_subset_with-perm-ids.npz"
    )

    atom_array_out_test = assign_mol_permutation_ids(atom_array_in, retokenize=True)

    assert_atomarray_equal(atom_array_out, atom_array_out_test)

    # Assert retokenization
    assert np.array_equal(
        np.unique(np.diff(atom_array_out_test.token_id)), np.array([0, 1])
    )


class PermutableFeatureDict:
    """Wrapper around a feature dict for easy permutations.

    Attributes:
        features:
            The feature dict with all relevant features.
        chain_starts_token_idx:
            Tensor of token-level indices where new chains start, including an exclusive
            stop index.
        chain_starts_atom_idx:
            Tensor of atom-level indices where new chains start, including an exclusive
            stop index.
        chain_entity_ids:
            Symmetry-label for each chain. E.g. for two symmetric chains and one
            asymmetric chain, use [0, 0, 1].
    """

    atom_level_features = [
        "atom_positions",
        "atom_resolved_mask",
    ]

    def __init__(
        self, features, chain_starts_token_idx, chain_starts_atom_idx, chain_entity_ids
    ):
        self.features = deepcopy(features)
        self.chain_starts_token_idx = chain_starts_token_idx
        self.chain_starts_atom_idx = chain_starts_atom_idx
        self.chain_entity_ids = chain_entity_ids

    def __getitem__(self, key):
        return self.features[key]

    def _sample_permutation(self, seed: int = None) -> torch.Tensor:
        """Sample a random permutation of symmetric chains.

        This will only permute chains with the same sym-ID, so for example for the
        following chains & entity IDs:

        chains:     [1, 2, 3]
        entity IDs: [0, 0, 1]

        Only the following permutations are allowed:

        [1, 2, 3]
        [2, 1, 3]
        """
        if seed:
            np.random.seed(seed)

        # Get symmetry groups
        entity_id_to_chain = defaultdict(list)

        for chain_idx, entity_id in enumerate(self.chain_entity_ids.tolist()):
            entity_id_to_chain[entity_id].append(chain_idx)

        # Final permutation
        final_perm = torch.full_like(self.chain_entity_ids, -1)

        for entity_id, chain_indices in entity_id_to_chain.items():
            perm = torch.tensor(np.random.permutation(chain_indices), dtype=torch.int32)
            entity_mask = self.chain_entity_ids == entity_id
            final_perm[entity_mask] = perm

        print(final_perm)

        return final_perm

    def permute_gt_features(self, perm) -> None:
        """Permute atom coords & resolved mask according to chain-level permutation."""
        new_coord_slices = []
        new_mask_slices = []

        for chain_idx in perm.tolist():
            chain_start = self.chain_starts_atom_idx[chain_idx]
            chain_end = self.chain_starts_atom_idx[chain_idx + 1]
            chain_slice = slice(chain_start, chain_end)

            new_coord_slices.append(
                self.features["ground_truth"]["atom_positions"][0, chain_slice]
            )
            new_mask_slices.append(
                self.features["ground_truth"]["atom_resolved_mask"][0, chain_slice]
            )

        new_coords = torch.cat(new_coord_slices, dim=0)
        new_mask = torch.cat(new_mask_slices, dim=0)

        self.features["ground_truth"]["atom_positions"][0, :] = new_coords
        self.features["ground_truth"]["atom_resolved_mask"][0, :] = new_mask

    def get_permuted_gt_copy(self, seed: int = None) -> None:
        """
        Randomly permute ground truth atom-level features.

        This samples a random chain-level permutation from `_sample_permutation`
        and applies it to the relevant ground truth features via `permute_gt_features`.

        Args:
            seed (int, optional):
                Deterministic random seed for reproducible permutations.

        Returns:
            None
        """
        perm = self._sample_permutation(seed)

        new_copy = deepcopy(self)
        new_copy.permute_gt_features(perm)

        return new_copy


# @pytest.fixture
def permutable_feature_dict():
    """Creates a fake feature dict with chain-wise permutation support.

    The feature dict returns by these corresponds to a smalls tructure with three
    symmetric protein chains, two symmetric ligand chains, and one umique protein chain.
    The chain order is ABAABC. Every protein residue is set to have 3 atoms, and the
    ligand residues have 6 atoms.
    """
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # Make a fake feature dict first

    # Features shared between ground-truth and normal features
    shared_features = {
        "atom_mask": [],
        "token_mask": [],
        "token_index": [],
        "num_atoms_per_token": [],
        "is_protein": [],
        "is_ligand": [],
        "is_rna": [],
        "is_dna": [],
        "is_atomized": [],
        "start_atom_index": [],
        "restype": [],
        "mol_entity_id": [],
        "mol_sym_id": [],
        "mol_sym_component_id": [],
        "mol_sym_token_index": [],
    }

    feature_dict = {
        "ground_truth": {
            "atom_positions": [],
            "atom_resolved_mask": [],
        },
        "residue_index": [],
    }

    shared_feature_dtypes = {
        "atom_mask": torch.bfloat16,
        "token_mask": torch.bfloat16,
        "token_index": torch.int32,
        "num_atoms_per_token": torch.int32,
        "is_protein": torch.int32,
        "is_ligand": torch.int32,
        "is_rna": torch.int32,
        "is_dna": torch.int32,
        "is_atomized": torch.int32,
        "start_atom_index": torch.int32,
        "restype": torch.int32,
        "mol_entity_id": torch.int32,
        "mol_sym_id": torch.int32,
        "mol_sym_component_id": torch.int32,
        "mol_sym_token_index": torch.int32,
    }

    feature_dict_dtypes = {
        "ground_truth": {
            "atom_positions": torch.bfloat16,
            "atom_resolved_mask": torch.bfloat16,
        },
        "residue_index": torch.int32,
        "ref_space_uid": torch.int32,
        "ref_space_uid_to_perm": torch.int32,
    }

    feature_dict_dtypes.update(shared_feature_dtypes)
    feature_dict_dtypes["ground_truth"].update(shared_feature_dtypes)

    # Make 3 symmetric protein chains, two symmetric ligand chains, one asymmetric
    # protein chain. Do them in this order, so that they are not consecutive:
    # ABAABC
    # Assume here that every protein token corresponds to 4 atoms and every ligand token
    # corresponds to 1 as usual.
    chain_token_counts = [4, 5, 4, 4, 5, 6]
    chain_mol_entity_ids = [0, 1, 0, 0, 1, 2]
    chain_is_ligand = [False, True, False, False, True, False]

    # Special features that will help with permuting chain-wise features
    chain_starts_token_idx = np.cumsum([0] + chain_token_counts)
    chain_atom_counts = [
        4 * count if not is_ligand else count
        for count, is_ligand in zip(chain_token_counts, chain_is_ligand)
    ]
    chain_starts_atom_idx = np.cumsum([0] + chain_atom_counts)

    # Make some allowed atom-wise permutations for 4-atom residues and the 5-atom
    # ligands. Keep 0s for proteins and 1s for ligands fixed so not everything is
    # allowed to permute.
    protein_atom_perms = [
        [0, 1, 2, 3],
        [0, 2, 1, 3],
        [0, 3, 1, 2],
    ]
    ligand_atom_perms = [
        [0, 1, 2, 3, 4],
        [0, 1, 3, 2, 4],
        [2, 1, 0, 3, 4],
        [4, 1, 2, 3, 0],
    ]

    # Fill the feature dict
    sym_id_counter = defaultdict(int)

    for chain_idx in range(len(chain_token_counts)):
        global_token_index = 0

        # Get sym ID
        mol_entity_id = chain_mol_entity_ids[chain_idx]
        mol_sym_id = sym_id_counter[mol_entity_id]
        sym_id_counter[chain_mol_entity_ids[chain_idx]] += 1

        for token_index in range(chain_token_counts[chain_idx]):
            # TOKEN-WISE FEATURES
            feature_dict["residue_index"].append(token_index + 1)
            shared_features["token_mask"].append(1)  # Don't use padding here
            shared_features["token_index"].append(global_token_index)
            shared_features["mol_entity_id"].append(mol_entity_id)
            shared_features["mol_sym_id"].append(mol_sym_id)
            shared_features["mol_sym_token_index"].append(
                token_index
            )  # Renumbered for every chain

            if chain_is_ligand[chain_idx]:
                num_atoms = 1
                shared_features["num_atoms_per_token"].append(num_atoms)
                shared_features["is_ligand"].append(True)
                shared_features["is_protein"].append(False)
                shared_features["is_rna"].append(False)
                shared_features["is_dna"].append(False)
                shared_features["is_atomized"].append(True)
                shared_features["restype"].append(RESTYPE_INDEX_3["UNK"])
            else:
                num_atoms = 4
                shared_features["num_atoms_per_token"].append(num_atoms)
                shared_features["is_ligand"].append(False)
                shared_features["is_protein"].append(True)
                shared_features["is_rna"].append(False)
                shared_features["is_dna"].append(False)
                shared_features["is_atomized"].append(False)
                shared_features["restype"].append(RESTYPE_INDEX_3["GLY"])

            # ATOM-WISE FEATURES
            # Create random coordinates
            atom_pos = np.random.rand(num_atoms, 3)
            feature_dict["ground_truth"]["atom_positions"].extend(atom_pos)

            # Set one atom to be unresolved randomly
            resolved_mask = np.ones(num_atoms, dtype=bool)
            feature_dict["ground_truth"]["atom_resolved_mask"].extend(resolved_mask)

            shared_features["atom_mask"].extend(np.ones(num_atoms, dtype=bool))

    shared_features["restype"] = torch.nn.functional.one_hot(
        torch.tensor(shared_features["restype"]), num_classes=32
    )
    shared_features["start_atom_index"] = np.concatenate(
        (
            np.zeros((1,)),
            np.cumsum(shared_features["num_atoms_per_token"], axis=-1)[:-1],
        ),
        axis=-1,
    )

    # Set one atom to be unresolved randomly
    feature_dict["ground_truth"]["atom_resolved_mask"][
        np.random.randint(0, len(resolved_mask))
    ] = False

    # Create the ref_space_uid-related features manually because they are a little
    # annoying

    # REF_SPACE_UID
    # Components are 3 protein residues, 1 ligand residue, 3 protein residues, 3 protein
    # residues, 1 ligand residue, 4 protein residues
    protein_A_repeats = [4] * 4  # 4 tokens with 4 atoms
    ligand_B_repeats = [5]  # all 5 tokens have the same ref_space_uid
    protein_C_repeats = [4] * 6  # 6 tokens with 4 atoms

    repeats = (
        protein_A_repeats
        + ligand_B_repeats
        + protein_A_repeats
        + protein_A_repeats
        + ligand_B_repeats
        + protein_C_repeats
    )

    n_components = len(repeats)
    ref_space_uid = np.repeat(np.arange(n_components), repeats).tolist()

    # REF_SPACE_UID_TO_PERM
    ref_space_uid_to_perm = {
        # PROTEIN A
        0: protein_atom_perms,
        1: protein_atom_perms,
        2: protein_atom_perms,
        3: protein_atom_perms,
        # LIGAND B
        4: ligand_atom_perms,
        # PROTEIN A
        5: protein_atom_perms,
        6: protein_atom_perms,
        7: protein_atom_perms,
        8: protein_atom_perms,
        # PROTEIN A
        9: protein_atom_perms,
        10: protein_atom_perms,
        11: protein_atom_perms,
        12: protein_atom_perms,
        # LIGAND B
        13: ligand_atom_perms,
        # PROTEIN C
        14: protein_atom_perms,
        15: protein_atom_perms,
        16: protein_atom_perms,
        17: protein_atom_perms,
        18: protein_atom_perms,
        19: protein_atom_perms,
    }
    feature_dict["ref_space_uid"] = ref_space_uid
    feature_dict["ref_space_uid_to_perm"] = ref_space_uid_to_perm

    mol_sym_component_id = []
    for chain_idx, (chain_start, chain_end) in enumerate(
        zip(chain_starts_token_idx[:-1], chain_starts_token_idx[1:])
    ):
        if not chain_is_ligand[chain_idx]:
            mol_sym_component_id.extend(np.arange(chain_end - chain_start).tolist())
        else:
            mol_sym_component_id.extend([0] * (chain_end - chain_start))
    shared_features["mol_sym_component_id"] = mol_sym_component_id

    # Update feature dicts with shared feature values
    feature_dict.update(shared_features)
    feature_dict["ground_truth"].update(shared_features)

    # Now format the feature dict, casting things to tensors and adding a batch
    # dimension
    feature_dict_fmt = defaultdict(dict)
    for key, values in feature_dict.items():
        # These two features need special handling
        if key == "ground_truth":
            for sub_key, sub_values in values.items():
                dtype = feature_dict_dtypes[key][sub_key]
                feature_dict_fmt[key][sub_key] = torch.tensor(
                    sub_values, dtype=dtype
                ).unsqueeze(0)
        elif key == "ref_space_uid_to_perm":
            for uid, perm_list in values.items():
                feature_dict_fmt[key][uid] = torch.tensor(perm_list, dtype=torch.int32)

            # Batch dimension gets added here in form of a list with a single dict-entry
            # (would be multiple dict-entries for batch size > 1)
            feature_dict_fmt[key] = [feature_dict_fmt[key]]

        # Every other feature
        else:
            dtype = feature_dict_dtypes[key]
            feature_dict_fmt[key] = torch.tensor(values, dtype=dtype).unsqueeze(0)

    return PermutableFeatureDict(
        features=feature_dict_fmt,
        chain_starts_token_idx=torch.tensor(chain_starts_token_idx, dtype=torch.int32),
        chain_starts_atom_idx=torch.tensor(chain_starts_atom_idx, dtype=torch.int32),
        chain_entity_ids=torch.tensor(chain_mol_entity_ids, dtype=torch.int32),
    )


def test_permutation_alignment(permutable_feature_dict, seed: int = 2395872):
    # Make a copy of the unpermuted GT-features
    unpermuted_gt = deepcopy(permutable_feature_dict["ground_truth"])
    unpermuted_positions = unpermuted_gt["atom_positions"]
    unpermuted_mask = unpermuted_gt["atom_resolved_mask"]

    # Randomly permute the ground-truth features
    permuted_feature_dict = permutable_feature_dict.get_permuted_gt_copy(seed)
    permuted_positions = permuted_feature_dict["ground_truth"]["atom_positions"]

    # Assert that the ground-truth features are different
    assert not torch.equal(unpermuted_positions, permuted_positions)

    # Now run permutation alignment, putting in the unpermuted positions as the
    # "prediction"
    permuted_gt = multi_chain_permutation_alignment(
        batch=permuted_feature_dict.features,
        atom_positions_predicted=unpermuted_positions,
    )

    # Assert that the permuted ground-truth features are the same as the unpermuted ones
    assert torch.allclose(permuted_gt["atom_positions"], permuted_positions)
    assert torch.allclose(permuted_gt["atom_resolved_mask"], unpermuted_mask)


test_permutation_alignment(permutable_feature_dict())
