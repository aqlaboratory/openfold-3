"""Asserts for checking the correctness of FeatureDict entries."""

import torch

from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms


def assert_no_nan_inf(features):
    """Checks if any tensor in the features dictionary contains NaNs or infs."""
    for key, entry in features.items():
        if isinstance(entry, dict):
            for subkey, subentry in entry.items():
                assert ~torch.isnan(subentry).any(), f"Tensor '{subkey}' contains NaNs."
                assert ~torch.isinf(subentry).any(), f"Tensor '{subkey}' contains infs."
        else:
            assert ~torch.isnan(entry).any(), f"Tensor '{key}' contains NaNs."
            assert ~torch.isinf(entry).any(), f"Tensor '{key}' contains infs."


def assert_token_atom_sum_match(features):
    """Checks if the sums of numbers of atoms in different tensors match."""

    # Match total sizes of
    sizes = []
    # - broadcast_token_feat_to_atoms
    sizes.append(
        broadcast_token_feat_to_atoms(
            token_mask=features["token_mask"],
            num_atoms_per_token=features["num_atoms_per_token"],
            token_feat=features["token_index"],
        ).shape[0]
    )
    # - ref_* features
    sizes.extend([f.shape[0] for k, f in features.items() if k.startswith("ref_")])
    # - GT N atoms
    sizes.extend(
        [
            f.shape[0]
            for k, f in features["ground_truth"].items()
            if k.startswith("atom_")
        ]
    )
    # - sum of num atoms per token
    sizes.append(features["num_atoms_per_token"].sum().item())

    assert len(set(sizes)) == 1, "Mismatch in total atom counts from different sources."


def assert_num_atom_per_token(features, token_budget):
    """Asserts that casting tokens to atoms results in the correct number of atoms."""

    per_atom_token_index = broadcast_token_feat_to_atoms(
        token_mask=features["token_mask"],
        num_atoms_per_token=features["num_atoms_per_token"],
        token_feat=features["token_index"],
    )
    start_atom_index_from_broadcast = torch.zeros(
        [token_budget], dtype=features["start_atom_index"].dtype
    )
    start_atom_index_from_broadcast_unpadded = torch.cat(
        [torch.zeros(1), torch.where(torch.diff(per_atom_token_index) != 0)[0] + 1]
    )
    start_atom_index_from_broadcast[
        : start_atom_index_from_broadcast_unpadded.shape[0]
    ] = start_atom_index_from_broadcast_unpadded

    assert torch.all(start_atom_index_from_broadcast == features["start_atom_index"]), (
        "Mismatch in number of atoms per token from broadcasting."
    )


def assert_max_23_atoms_per_token(features):
    """Asserts that no token has more than 23 atoms."""
    assert ~(features["num_atoms_per_token"] > 23).any(), (
        "Token with more than 23 atoms found.",
    )


def assert_ligand_atomized(features):
    """Asserts that ligand tokens are atomized."""
    assert (features["is_ligand"] & features["is_atomized"])[
        features["is_ligand"].to(torch.bool)
    ].all(), ("Ligand token found that is not atomized.",)


def assert_atomized_one_atom(features):
    """Asserts that atomized tokens have only one atom."""
    assert (features["is_atomized"] & (features["num_atoms_per_token"] == 1))[
        features["is_atomized"].to(torch.bool)
    ].all(), "Atomized token with more than one atom found."


def assert_resid_asym_refuid_match(features):
    "Asserts that within the same residue_index-asym_id tuple, the ref_uid is the same."
    atom_resids = broadcast_token_feat_to_atoms(
        token_mask=features["token_mask"],
        num_atoms_per_token=features["num_atoms_per_token"],
        token_feat=features["residue_index"],
    )
    atom_asymids = broadcast_token_feat_to_atoms(
        token_mask=features["token_mask"],
        num_atoms_per_token=features["num_atoms_per_token"],
        token_feat=features["asym_id"],
    )

    unique_tuples = {}
    unique_id = 0
    result_ids = []

    # Iterate over elements of both tensors
    for a, b in zip(atom_resids, atom_asymids, strict=False):
        tup = (a.item(), b.item())

        if tup not in unique_tuples:
            unique_tuples[tup] = unique_id
            unique_id += 1

        result_ids.append(unique_tuples[tup])

    # Convert list of ids to a tensor
    result_tensor = torch.tensor(result_ids)

    ref_space_uid_pos = torch.where(torch.diff(features["ref_space_uid"]) > 0)
    result_tensor_pos = torch.where(torch.diff(result_tensor) > 0)

    assert torch.isin(ref_space_uid_pos[0], result_tensor_pos[0]).all(), (
        "Mismatch between changing positions of ref_space_uid and atom-broadcasted "
    )
    "residue_index-asym_id tuples."


def assert_atom_pos_resolved(features):
    """Asserts that there are resolved atoms in the crop."""
    assert (
        ~(features["ground_truth"]["atom_resolved_mask"] == 0).all()
        | ~torch.isnan(features["ground_truth"]["atom_positions"]).all()
        | ~torch.isinf(features["ground_truth"]["atom_positions"]).all()
    ), "Atom positions are all-nan/inf or all atoms are unresolved."


def assert_one_entityid_per_asymid(features):
    """Asserts that there is only one entity_id per asym_id."""

    tups = set()
    for a, b in zip(features["asym_id"], features["entity_id"], strict=False):
        t = (a.item(), b.item())
        if ((t[0] != 0) & (t[1] != 0)) & (t not in tups):
            tups.add(t)

    assert len(tups) == len(
        set(features["asym_id"][features["asym_id"] != 0].tolist())
    ), "Multiple entity_ids per asym_id found."


def assert_no_identical_ref_pos(features):
    """Asserts that no two rows in ref_pos are identical."""

    ref_pos = features["ref_pos"]
    ref_pos_regular = ref_pos[
        ~(
            (ref_pos == 0).all(dim=1)
            | torch.isnan(ref_pos).all(dim=1)
            | torch.isinf(ref_pos).all(dim=1)
        )
    ]
    secondary_sort = ref_pos_regular[ref_pos_regular[:, 1].argsort(dim=0, stable=False)]
    primary_sort = secondary_sort[secondary_sort[:, 0].argsort(dim=0, stable=True)]
    assert torch.all(~(torch.diff(primary_sort, dim=0) == 0).all(dim=1)), (
        "Found identical ref_pos coordinates."
    )


def assert_no_all_zero_idxs(features):
    """Asserts none of the indexing tensors are all-zero."""
    index_keys = [
        "residue_index",
        "token_index",
        "asym_id",
        "entity_id",
        "sym_id",
        "restype",
        "is_protein",
        "is_rna",
        "is_dna",
        "is_ligand",
    ]
    for idx_key in index_keys:
        torch.any(~(features[idx_key] == 0)), f"Tensor '{idx_key}' contains all zeros."
    for idx_key in index_keys:
        (
            torch.any(~(features["ground_truth"][idx_key] == 0)),
            f"GT tensor '{idx_key}' contains all zeros.",
        )


def assert_gt_crop_slice(features):
    """Asserts that slicing the GT features creates matching target features."""
    cropped_token_index = features["token_index"]
    is_gt_in_crop = torch.isin(
        features["ground_truth"]["token_index"], cropped_token_index
    )

    for k in FEATURE_CORE_DTYPES:
        assert (features["ground_truth"][k][is_gt_in_crop] == features[k]).all(), (
            f"GT feature '{k}' does not match cropped feature."
        )


def assert_shape(features, token_budget, n_templates):
    """Asserts the shape of the features."""
    for k, v in FULL_TOKEN_DIM_INDEX_MAP.items():
        for i in v:
            assert features[k].shape[i] == token_budget, (
                f"Token shape mismatch for key '{k}'."
            )
    for k, v in FULL_MSA_DIM_INDEX_MAP.items():
        for i in v:
            assert features[k].shape[i] <= 16384, f"MSA shape '{k}' larger than 16384."
    for k, v in FULL_TEMPLATE_DIM_INDEX_MAP.items():
        for i in v:
            assert features[k].shape[i] == n_templates, (
                f"Template shape '{k}' shape mismatch."
            )
    for k, v in FULL_OTHER_DIM_INDEX_MAP.items():
        for i in v:
            assert features[k].shape[i] == FULL_OTHER_DIM_SIZE_MAP[k][i], (
                f"Other shape '{k}' shape mismatch."
            )


def assert_dtype(features):
    """Asserts the dtype of the features."""
    for k, v in (FEATURE_CORE_DTYPES | FEATURE_OTHER_DTYPES).items():
        assert features[k].dtype == v, f"Cropped feature '{k}' dtype mismatch."
    for k, v in (FEATURE_CORE_DTYPES | FEATURE_GT_DTYPES).items():
        assert features["ground_truth"][k].dtype == v, (
            f"GT feature '{k}' dtype mismatch."
        )
    for k, v in FEATURE_LOSS_DTYPES.items():
        assert features["loss_weights"][k].dtype == v, (
            f"Loss feature '{k}' dtype mismatch."
        )


def assert_resid_same_in_tokenid(features):
    """Asserts that the residue index doesn't change in each token."""
    residue_index_pos = torch.where(torch.diff(features["residue_index"]) > 0)
    token_index_pos = torch.where(torch.diff(features["token_index"]) > 0)

    assert torch.isin(residue_index_pos, token_index_pos).all(), (
        "Found residue indices that change within tokens."
    )


def assert_all_unk_atomized(features):
    """Asserts that all tokens with unknown residue type are atomized."""
    is_unknown = features["restype"][:, 20] == 1
    assert (features["is_atomized"][is_unknown] == 1).all(), (
        "Found unknown residue tokens that are not atomized."
    )
    assert (features["ground_truth"]["is_atomized"][is_unknown] == 1).all(), (
        "Found unknown GT residue tokens that are not atomized."
    )


def assert_token_bonds_atomized(features):
    """Asserts that all tokens with token_bonds are atomized."""
    has_token_bonds_pos = torch.where(
        (
            torch.sum(features["token_bonds"], dim=0)
            + torch.sum(features["token_bonds"], dim=1)
        )
        > 0
    )[0]
    is_atomized_pos = torch.where(features["is_atomized"] == 1)[0]

    assert torch.isin(has_token_bonds_pos, is_atomized_pos).all(), (
        "Found unatomized tokens with token_bonds."
    )


def assert_profile_sum(features):
    """Asserts that the sum of the profiles is 1 or 0."""
    profile_sum = torch.sum(features["profile"], dim=1)
    assert torch.all(
        torch.isclose(profile_sum, torch.tensor(1.0))
        | torch.isclose(profile_sum, torch.tensor(0.0))
    ), "Found profile column sums that are not 1 or 0."


ENSEMBLED_ASSERTS = [
    assert_no_nan_inf,
    assert_token_atom_sum_match,
    assert_num_atom_per_token,
    assert_max_23_atoms_per_token,
    assert_ligand_atomized,
    assert_atomized_one_atom,
    assert_resid_asym_refuid_match,
    assert_atom_pos_resolved,
    assert_one_entityid_per_asymid,
    assert_no_identical_ref_pos,
    assert_no_all_zero_idxs,
    assert_gt_crop_slice,
    assert_shape,
    assert_dtype,
    assert_all_unk_atomized,
    assert_token_bonds_atomized,
    assert_profile_sum,
]

FULL_TOKEN_DIM_INDEX_MAP = {
    "residue_index": [-1],
    "token_index": [-1],
    "asym_id": [-1],
    "entity_id": [-1],
    "sym_id": [-1],
    "restype": [-2],
    "is_protein": [-1],
    "is_rna": [-1],
    "is_dna": [-1],
    "is_ligand": [-1],
    "token_bonds": [-1, -2],
    "num_atoms_per_token": [-1],
    "is_atomized": [-1],
    "start_atom_index": [-1],
    "msa": [-2],
    "has_deletion": [-1],
    "deletion_value": [-1],
    "profile": [-2],
    "deletion_mean": [-1],
    "template_restype": [-2],
    "template_pseudo_beta_mask": [-1],
    "template_backbone_frame_mask": [-1],
    "template_distogram": [-2, -3],
    "template_unit_vector": [-2, -3],
}
FULL_MSA_DIM_INDEX_MAP = {
    "msa": [-3],
    "has_deletion": [-2],
    "deletion_value": [-2],
}
FULL_TEMPLATE_DIM_INDEX_MAP = {
    "template_restype": [-3],
    "template_pseudo_beta_mask": [-2],
    "template_backbone_frame_mask": [-2],
    "template_distogram": [-4],
    "template_unit_vector": [-4],
}
FULL_OTHER_DIM_INDEX_MAP = {
    "restype": [-1],
    "ref_pos": [-1],
    "ref_element": [-1],
    "ref_atom_name_chars": [-1, -2],
    "msa": [-1],
    "template_restype": [-1],
    "template_distogram": [-1],
    "template_unit_vector": [-1],
}
FULL_OTHER_DIM_SIZE_MAP = {
    "restype": [32],
    "ref_pos": [3],
    "ref_element": [119],
    "ref_atom_name_chars": [4, 64],
    "msa": [32],
    "template_restype": [32],
    "template_distogram": [39],
    "template_unit_vector": [3],
}

FEATURE_CORE_DTYPES = {
    "residue_index": torch.int32,
    "token_index": torch.int32,
    "asym_id": torch.int32,
    "entity_id": torch.int32,
    "sym_id": torch.int32,
    "restype": torch.int32,
    "is_protein": torch.int32,
    "is_rna": torch.int32,
    "is_dna": torch.int32,
    "is_ligand": torch.int32,
    "token_bonds": torch.int32,
    "num_atoms_per_token": torch.int32,
    "is_atomized": torch.int32,
    "start_atom_index": torch.int32,
    "token_mask": torch.float32,
}
FEATURE_GT_DTYPES = {
    "atom_positions": torch.float32,
    "atom_resolved_mask": torch.float32,
}
FEATURE_OTHER_DTYPES = {
    "ref_pos": torch.float32,
    "ref_mask": torch.int32,
    "ref_element": torch.int32,
    "ref_charge": torch.float32,
    "ref_atom_name_chars": torch.int32,
    "ref_space_uid": torch.int32,
    "msa": torch.int32,
    "has_deletion": torch.float32,
    "deletion_value": torch.float32,
    "profile": torch.float32,
    "deletion_mean": torch.float32,
    "template_restype": torch.int32,
    "template_pseudo_beta_mask": torch.float32,
    "template_backbone_frame_mask": torch.float32,
    "template_distogram": torch.float32,
    "template_unit_vector": torch.float32,
    "msa_mask": torch.float32,
    "num_paired_seqs": torch.int32,
}
FEATURE_LOSS_DTYPES = {
    "bond": torch.float32,
    "smooth_lddt": torch.float32,
    "mse": torch.float32,
    "distogram": torch.float32,
    "experimentally_resolved": torch.float32,
    "plddt": torch.float32,
    "pae": torch.float32,
    "pde": torch.float32,
}
