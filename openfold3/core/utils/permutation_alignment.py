# NIT: confusing that this needs to be transposed, while the rotation matrix in
# Transformation doesn't
import logging
from collections import Counter
from functools import partial
from typing import overload

import torch

from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_center_atoms,
)
from openfold3.core.utils.geometry.kabsch_alignment import get_optimal_transformation

logger = logging.getLogger(__name__)


def get_pred_anchor_molecule_coords(
    pred_coords: torch.Tensor,
    perm_entity_ids: torch.Tensor,
    perm_sym_ids: torch.Tensor,
    anchor_perm_id: int,
) -> torch.Tensor:
    # Selector for symmetric molecules of the anchor molecule
    anchor_entity_mask = perm_entity_ids == anchor_perm_id

    # Predicted coordinates for all symmetric molecules
    pred_anchor_mols_concat = pred_coords[anchor_entity_mask]

    # Split coordinates by unique molecule instance
    pred_anchor_mol_coords, _ = split_feats_by_id(
        pred_anchor_mols_concat, perm_sym_ids[anchor_entity_mask]
    )

    return torch.stack(pred_anchor_mol_coords)


def get_centroid(coords: torch.Tensor, mask: torch.Tensor):
    # Get centroid of only the unmasked coordinates
    n_observed_atoms = torch.sum(mask, dim=-1, keepdim=True)
    centroid = (
        torch.sum(
            coords * mask[..., None],
            dim=-2,
            keepdim=True,
        )
        / n_observed_atoms[..., None]
    )

    return centroid


@overload
def split_feats_by_id(
    feats: torch.Tensor, id: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor]: ...


def split_feats_by_id(
    feats: list[torch.Tensor], id: torch.Tensor
) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
    unique_ids = torch.unique(id, sort=False)

    split_feats = [
        [feats[id == single_id] for single_id in unique_ids] for feat in feats
    ]

    return split_feats, unique_ids


# TODO: Think again about whether perm_entity_ids should be [N_atom] or [N_token]
# (leaning towards the latter)
def find_greedy_optimal_mol_permutation(
    gt_token_center_positions_transformed: torch.Tensor,
    gt_token_center_resolved_mask: torch.Tensor,
    gt_perm_entity_ids: torch.Tensor,
    gt_perm_sym_ids: torch.Tensor,
    pred_token_center_positions: torch.Tensor,
    pred_perm_entity_ids: torch.Tensor,
    pred_perm_sym_ids: torch.Tensor,
    pred_perm_sym_token_idx: torch.Tensor,
):
    # Get the unique entity IDs
    unique_pred_perm_entity_ids = torch.unique(pred_perm_entity_ids)

    optimal_permutation = None
    optimal_rmsd = torch.inf

    # Iterate through each anchor alignment
    for gt_token_center_coords in gt_token_center_positions_transformed:
        centroid_dists_sq = []
        permutation = {}

        # Resolve permutation for each entity group separately
        for entity_id in unique_pred_perm_entity_ids:
            pred_coords_entity = pred_token_center_positions[
                pred_perm_entity_ids == entity_id
            ]
            pred_sym_ids_entity = pred_perm_sym_ids[pred_perm_entity_ids == entity_id]
            pred_sym_atom_idx_entity = pred_perm_sym_token_idx[
                pred_perm_entity_ids == entity_id
            ]

            gt_coords_entity = gt_token_center_coords[gt_perm_entity_ids == entity_id]
            gt_resolved_mask_entity = gt_token_center_resolved_mask[
                gt_perm_entity_ids == entity_id
            ]
            gt_sym_ids_entity = gt_perm_sym_ids[gt_perm_entity_ids == entity_id]

            (gt_coords_entity, gt_resolved_mask_entity), gt_unique_sym_ids_entity = (
                split_feats_by_id(
                    [gt_coords_entity, gt_resolved_mask_entity], gt_sym_ids_entity
                )
            )
            gt_coords_entity = torch.stack(gt_coords_entity)
            gt_resolved_mask_entity = torch.stack(gt_resolved_mask_entity)

            # Shuffle order of molecules (this is not mentioned in the SI but avoids any
            # bias on the PDB ordering)
            unique_pred_sym_ids_entity = torch.unique(pred_sym_ids_entity)
            unique_pred_sym_ids_entity = torch.randperm(
                unique_pred_sym_ids_entity.shape[0]
            )

            # Track which ground-truth symmetry IDs have already been assigned
            used_gt_sym_ids = []

            # Run greedy assignment for this entity
            for sym_id in unique_pred_sym_ids_entity:
                # Get particular atom subset of the predicted molecule
                pred_sym_atom_idx = pred_sym_atom_idx_entity[
                    pred_sym_ids_entity == sym_id
                ]

                # Match the atom subset of the predicted molecule
                gt_coords_entity_subset = (
                    torch.index_select(  # TODO: torch.where instead?
                        gt_coords_entity, 0, pred_sym_atom_idx
                    )
                )
                gt_resolved_mask_entity_subset = torch.index_select(
                    gt_resolved_mask_entity, 0, pred_sym_atom_idx
                )

                # Get centroids of ground-truth
                gt_coords_entity_centroids = get_centroid(
                    gt_coords_entity_subset, gt_resolved_mask_entity_subset
                )

                # Get centroids of prediction, while matching unresolved atoms on the GT
                # side
                pred_coords_sym = pred_coords_entity[pred_sym_ids_entity == sym_id]
                pred_coords_sym_centroids = get_centroid(
                    pred_coords_sym.unsqueeze(-3), gt_resolved_mask_entity_subset
                )

                # Get squared distances between centroids
                pred_gt_dists_sq = (
                    (pred_coords_sym_centroids - gt_coords_entity_centroids)
                    .pow(2)
                    .sum(dim=-1)
                )

                # Mask already used IDs
                used_gt_sym_ids_mask = torch.isin(
                    gt_unique_sym_ids_entity, torch.tensor(used_gt_sym_ids)
                )
                pred_gt_dists_sq[:, used_gt_sym_ids_mask] = torch.inf

                # Make sure that entirely unresolved ground-truth centroids are picked
                # last (by setting higher than any other dist but lower than inf)
                gt_any_resolved_mask_centroid = gt_resolved_mask_entity_subset.any(
                    dim=0
                )
                pred_gt_dists_sq[:, ~gt_any_resolved_mask_centroid] = torch.finfo(
                    pred_gt_dists_sq.dtype
                ).max

                # Get the best matching ground-truth symmetry ID
                best_gt_index = torch.argmin(pred_gt_dists_sq)
                best_gt_sym_id = gt_unique_sym_ids_entity[best_gt_index]
                best_gt_sym_id_dist_sq = pred_gt_dists_sq[best_gt_index]

                centroid_dists_sq.append(best_gt_sym_id_dist_sq)
                used_gt_sym_ids.append(best_gt_sym_id)

                # Append the mapping between the predicted and ground-truth symmetry and
                # entity IDs to the permutation
                permutation[(entity_id, sym_id)] = (entity_id, best_gt_sym_id)

        # Single RMSD over all centroid distances (this is a slight deviation from the
        # AF2-Multimer SI)
        rmsd = torch.sqrt(torch.tensor(centroid_dists_sq).mean())
        if rmsd < optimal_rmsd:
            optimal_permutation = permutation
            optimal_rmsd = rmsd

    logger.debug(f"Found optimal permutation with RMSD: {optimal_rmsd}")

    return optimal_permutation


def get_gt_atom_permutation(
    gt_positions: torch.Tensor,
    gt_resolved_mask: torch.Tensor,
    gt_perm_entity_id: torch.Tensor,
    gt_perm_sym_id: torch.Tensor,
    gt_perm_sym_conformer_id: torch.Tensor,
    gt_num_atoms_per_token: torch.Tensor,
    pred_positions: torch.Tensor,
    pred_perm_entity_id: torch.Tensor,
    pred_perm_sym_id: torch.Tensor,
    pred_perm_sym_conformer_id: torch.Tensor,
    pred_num_atoms_per_token: torch.Tensor,
    pred_ref_space_uid: torch.Tensor,
    pred_ref_space_uid_to_perm: dict[int, torch.Tensor],
    optimal_mol_permutation: dict[tuple[int, int], tuple[int, int]],
):
    # Will create a final indexing operation into the gt-features (initialize to -1 just
    # for easier assert at the end)
    atom_subset_index = -torch.ones(
        pred_positions.shape[0], dtype=torch.long, device=pred_positions.device
    )

    # Indices of all atoms in the ground-truth
    gt_all_atom_index = torch.arange(gt_positions.shape[0], device=gt_positions.device)

    # Expand features to atom-wise level
    dummy_token_mask = torch.ones_like(pred_perm_entity_id, dtype=torch.bool)
    pred_token_feat_to_atoms = partial(
        broadcast_token_feat_to_atoms,
        token_mask=dummy_token_mask,
        num_atoms_per_token=pred_num_atoms_per_token,
    )
    pred_perm_entity_id_atom = pred_token_feat_to_atoms(pred_perm_entity_id)
    pred_perm_sym_id_atom = pred_token_feat_to_atoms(pred_perm_sym_id)
    pred_sym_conformer_id_atom = pred_token_feat_to_atoms(pred_perm_sym_conformer_id)

    dummy_token_mask = torch.ones_like(gt_perm_entity_id, dtype=torch.bool)
    gt_token_feat_to_atoms = partial(
        broadcast_token_feat_to_atoms,
        token_mask=dummy_token_mask,
        num_atoms_per_token=gt_num_atoms_per_token,
    )
    gt_perm_entity_id_atom = gt_token_feat_to_atoms(gt_perm_entity_id)
    gt_perm_sym_id_atom = gt_token_feat_to_atoms(gt_perm_sym_id)
    gt_sym_conformer_id_atom = gt_token_feat_to_atoms(gt_perm_sym_conformer_id)

    for (pred_entity_id, pred_sym_id), (
        gt_entity_id,
        gt_sym_id,
    ) in optimal_mol_permutation.items():
        # Subset the predicted coords to current entity and symmetry ID
        pred_mask = (pred_perm_entity_id_atom == pred_entity_id) & (
            pred_perm_sym_id_atom == pred_sym_id
        )
        pred_positions_subset = pred_positions[pred_mask]
        pred_ref_space_uids_subset = pred_ref_space_uid[pred_mask]

        pred_sym_conformer_id_atom_subset = pred_sym_conformer_id_atom[pred_mask]
        unique_sym_conformer_ids_subset = torch.unique_consecutive(
            pred_sym_conformer_id_atom_subset
        )

        # Subset ground-truth coords to current entity and symmetry ID
        gt_mask = (gt_perm_entity_id_atom == gt_entity_id) & (
            gt_perm_sym_id_atom == gt_sym_id
        )
        # Subset ground-truth to only the conformer-instances present in the prediction
        # (while still including potential extra symmetry-expanded atoms)
        gt_conformer_mask = torch.isin(
            gt_sym_conformer_id_atom[gt_mask],
            unique_sym_conformer_ids_subset,
        )
        gt_positions_subset = gt_positions[gt_mask][gt_conformer_mask]
        gt_resolved_mask_subset = gt_resolved_mask[gt_mask][gt_conformer_mask]
        gt_atom_idx_subset = gt_all_atom_index[gt_mask][gt_conformer_mask]
        gt_sym_conformer_id_atom_subset = gt_sym_conformer_id_atom[gt_mask][
            gt_conformer_mask
        ]

        # Group predictions by ref-space UID (= absolute conformer ID) - this gets the
        # same split as grouping by symmetry conformer ID but we need the absolute IDs
        # for permutation mapping
        pred_positions_subset_grouped, unique_ref_space_uids_subset = split_feats_by_id(
            pred_positions_subset, pred_ref_space_uids_subset
        )

        split_feats, _ = split_feats_by_id(
            [gt_positions_subset, gt_resolved_mask_subset, gt_atom_idx_subset],
            gt_sym_conformer_id_atom_subset,
        )
        (
            gt_positions_subset_grouped,
            gt_resolved_mask_subset_grouped,
            gt_atom_idx_subset_grouped,
        ) = split_feats

        assert unique_ref_space_uids_subset == len(pred_positions_subset_grouped)

        # Get the optimal permutation for each conformer
        permuted_atom_idxs = []
        for (
            ref_space_uid,
            pred_positions_subset_conf,
            gt_positions_subset_conf,
            gt_resolved_mask_subset_conf,
            gt_atom_idx_subset_conf,
        ) in zip(
            unique_ref_space_uids_subset,
            pred_positions_subset_grouped,
            gt_positions_subset_grouped,
            gt_resolved_mask_subset_grouped,
            gt_atom_idx_subset_grouped,
        ):
            # All possible permutations for this conformer
            permutations = pred_ref_space_uid_to_perm[ref_space_uid]

            # If there is only the identity permutation (which is the case for a lot of
            # residues) skip the remaining computations
            if permutations.shape[0] == 1:
                assert torch.all(permutations[0] == torch.arange(permutations.shape[1]))
                permuted_atom_idxs.extend(gt_atom_idx_subset_conf)
                continue

            # Versions of the ground-truth positions for each permutation
            gt_positions_subset_conf_perm = gt_positions_subset_conf.unsqueeze(0)[
                permutations
            ]

            # Minimize RMSD
            dists_sq = (
                (pred_positions_subset_conf - gt_positions_subset_conf_perm)
                .pow(2)
                .sum(dim=-1)
                .mul(gt_resolved_mask_subset_conf)
            )
            n_resolved_atoms = gt_resolved_mask_subset_conf.sum(dim=-1)
            rmsd = torch.sqrt(dists_sq.sum(dim=-1) / n_resolved_atoms)

            # Get permutation that minimizes RMSD
            best_permutation = permutations[torch.argmin(rmsd)]

            # Append the global atom indices corresponding to this permutation
            permuted_atom_idxs.extend(gt_atom_idx_subset_conf[best_permutation])

        # Add the atom indices to the final atom index that tracks the global
        # permutation. We use pred_mask here to ensure that we directly match the layout
        # of the pred-features, even if the features belonging to a connected molecule
        # are not contiguous.
        atom_subset_index[pred_mask] = permuted_atom_idxs

    assert torch.all(atom_subset_index != -1)

    return atom_subset_index


def atom_subset_index_to_token_subset_index(
    atom_subset_index: torch.Tensor,
    n_token: int,
) -> torch.Tensor:
    # Create an equivalent token subset index from the atom subset index
    gt_token_index = torch.arange(n_token, device=atom_subset_index.device)
    gt_token_index_atom = broadcast_token_feat_to_atoms(gt_token_index)
    gt_token_index_atom = gt_token_index_atom[atom_subset_index]
    token_subset_index = torch.unique_consecutive(gt_token_index_atom)

    return token_subset_index


def permute_gt_features(
    gt_feature_dict: dict[str, torch.Tensor],
    atom_indexes: list[torch.Tensor],
    token_indexes: list[torch.Tensor],
):
    new_feature_dict = {}
    batch_size = gt_feature_dict["residue_index"].shape[0]

    # Extend the indexes to account for padding
    n_tokens = gt_feature_dict["residue_index"].shape[1]
    for i, token_index in enumerate(token_indexes):
        full_token_index = torch.arange(n_tokens, device=token_index.device)
        full_token_index[: token_index.shape[0]] = token_index
        token_indexes[i] = full_token_index

    n_atoms = gt_feature_dict["atom_positions"].shape[1]
    for i, atom_index in enumerate(atom_indexes):
        full_atom_index = torch.arange(n_atoms, device=atom_index.device)
        full_atom_index[: atom_index.shape[0]] = atom_index
        atom_indexes[i] = full_atom_index

    # Permute all features in the feature dict
    for key, value in gt_feature_dict.items():
        if key in ["atom_positions", "atom_resolved_mask"]:
            new_feature_dict[key] = torch.stack(
                [value[i][..., atom_indexes[i]] for i in range(batch_size)]
            )
        elif key == "token_bonds":
            # token_bonds is [*, N_token, N_token]
            new_feature_dict[key] = torch.stack(
                [
                    value[i][..., token_indexes[i]][:, token_indexes[i]]
                    for i in range(batch_size)
                ]
            )
        else:
            new_feature_dict[key] = torch.stack(
                [value[i][..., token_indexes[i]] for i in range(batch_size)]
            )

    return new_feature_dict


def get_gt_anchor_molecule_data(
    gt_token_center_positions: torch.Tensor,
    gt_token_center_resolved_mask: torch.Tensor,
    gt_perm_entity_id: torch.Tensor,
    gt_perm_sym_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    # Figure out least ambiguous stoichiometry (ignoring entirely unresolved molecules)
    entity_sym_id_combinations = torch.cat(
        [
            gt_perm_entity_id[gt_token_center_resolved_mask].unsqueeze(-1),
            gt_perm_sym_id[gt_token_center_resolved_mask].unsqueeze(-1),
        ],
        dim=-1,
    )
    unique_entity_sym_id_combinations = torch.unique(entity_sym_id_combinations, dim=0)
    entity_stoichiometry = Counter(unique_entity_sym_id_combinations[:, 0].tolist())
    min_n_symmetry_mates = min(entity_stoichiometry.values())
    least_ambiguous_entity_ids = [
        entity_id
        for entity_id, count in entity_stoichiometry.items()
        if count == min_n_symmetry_mates
    ]

    # In case of tie, take the entity ID with more tokens per molecule (equivalent to
    # longest chain in SI)
    if len(least_ambiguous_entity_ids) > 1:
        logger.debug("Multiple least ambiguous entities found.")

        n_tokens_per_entity = {}

        for entity_id in least_ambiguous_entity_ids:
            entity_mask = gt_perm_entity_id == entity_id

            # Get first sym ID arbitrarily
            sym_id = gt_perm_sym_id[entity_mask][0].item()

            n_tokens_per_entity[entity_id] = torch.sum(
                entity_mask & (gt_perm_sym_id == sym_id)
            )

        anchor_entity_id = max(n_tokens_per_entity, key=n_tokens_per_entity.get)
    else:
        # Single-element tuple unpacking (will fail if != 1 elements)
        (anchor_entity_id,) = least_ambiguous_entity_ids

    # Select arbitrary instance of the anchor molecule
    anchor_sym_id = torch.multinomial(
        gt_perm_sym_id[gt_perm_entity_id == anchor_entity_id].float(), 1
    )

    # Get the mask for the anchor molecule
    anchor_mask = (gt_perm_entity_id == anchor_entity_id) & (
        gt_perm_sym_id == anchor_sym_id
    )

    anchor_coords = gt_token_center_positions[anchor_mask]
    anchor_resolved_mask = gt_token_center_resolved_mask[anchor_mask]

    logger.debug(
        "Number of resolved atoms in anchor molecule: "
        + f"{torch.sum(anchor_resolved_mask)}"
    )

    return anchor_coords, anchor_resolved_mask, anchor_entity_id


def single_batch_multi_chain_permutation_alignment(
    single_batch: dict, single_predicted_positions: torch.Tensor
):
    # Get relevant features and get rid of padding
    gt_token_pad_mask = single_batch["ground_truth"]["token_mask"].bool()
    gt_atom_pad_mask = single_batch["ground_truth"]["atom_pad_mask"]
    pred_token_pad_mask = single_batch["token_mask"].bool()
    pred_atom_pad_mask = single_batch["atom_pad_mask"]

    gt_coords = single_batch["ground_truth"]["atom_positions"][gt_atom_pad_mask]
    gt_resolved_mask = single_batch["ground_truth"]["resolved_mask"][
        gt_atom_pad_mask
    ].bool()
    gt_perm_entity_id = single_batch["ground_truth"]["perm_entity_id"][
        gt_token_pad_mask
    ]
    gt_perm_sym_id = single_batch["ground_truth"]["perm_sym_id"][gt_token_pad_mask]
    gt_perm_sym_conformer_id = single_batch["ground_truth"]["perm_sym_conformer_id"][
        gt_atom_pad_mask
    ]

    pred_coords = single_predicted_positions[pred_atom_pad_mask]
    pred_perm_entity_id = single_batch["perm_entity_id"][pred_token_pad_mask]
    pred_perm_sym_id = single_batch["perm_sym_id"][pred_token_pad_mask]
    pred_perm_sym_conformer_id = single_batch["perm_sym_conformer_id"][
        pred_atom_pad_mask
    ]
    pred_perm_sym_atom_idx = single_batch["perm_sym_atom_idx"][pred_atom_pad_mask]
    pred_ref_space_uid = single_batch["ref_space_uid"][pred_atom_pad_mask]
    pred_ref_space_uid_to_perm = single_batch["ref_space_uid_to_perm"]

    # Subset coordinates to only token centers (generalization of C-alpha in
    # AF2-Multimer algorithm)
    gt_token_center_coords, gt_token_center_mask = get_token_center_atoms(
        single_batch["ground_truth"], gt_coords, gt_resolved_mask
    )

    # Here everything is "resolved" because the model predicts all coordinates so no
    # need to pass an actual mask
    pred_coords_dummy_mask = torch.ones_like(pred_coords[..., 0], dtype=torch.bool)
    pred_token_center_coords, _ = get_token_center_atoms(
        single_batch, pred_coords, pred_coords_dummy_mask
    )

    # Get the anchor molecule from the ground truth
    gt_anchor_mol_coords, gt_anchor_mol_resolved_mask, gt_anchor_perm_entity_id = (
        get_gt_anchor_molecule_data(
            gt_token_center_coords,
            gt_token_center_mask,
            gt_perm_entity_id,
            gt_perm_sym_id,
        )
    )

    # Get candidate anchor molecule coordinates from the prediction
    pred_anchor_mol_coords = get_pred_anchor_molecule_coords(
        pred_coords,
        pred_perm_entity_id,
        pred_perm_sym_id,
        gt_anchor_perm_entity_id,
    )

    # Get transformations of ground-truth anchor onto candidate anchors
    gt_anchor_mol_coords = gt_anchor_mol_coords.unsqueeze(0)

    transformations = get_optimal_transformation(
        gt_anchor_mol_coords,
        pred_anchor_mol_coords,
        gt_anchor_mol_resolved_mask,
    )

    gt_token_center_positions_transformed = (
        gt_token_center_coords @ transformations.rotation_matrix
        + transformations.translation_vector
    )

    # ASSIGNMENT STAGE
    # Find optimal mapping between symmetry-equivalent molecules
    optimal_mol_permutation = find_greedy_optimal_mol_permutation(
        gt_token_center_positions_transformed,
        gt_token_center_mask,
        gt_perm_entity_id,
        gt_perm_sym_id,
        pred_token_center_coords,
        pred_perm_entity_id,
        pred_perm_sym_id,
        pred_perm_sym_atom_idx,
    )

    # Get the final atom index permutation by also resolving symmetry-equivalent atoms
    gt_atom_index = get_gt_atom_permutation(
        gt_positions=gt_coords,
        gt_resolved_mask=gt_resolved_mask,
        gt_perm_entity_id=gt_perm_entity_id,
        gt_perm_sym_id=gt_perm_sym_id,
        gt_perm_sym_conformer_id=gt_perm_sym_conformer_id,
        pred_positions=pred_coords,
        pred_perm_entity_id=pred_perm_entity_id,
        pred_perm_sym_id=pred_perm_sym_id,
        pred_perm_sym_conformer_id=pred_perm_sym_conformer_id,
        pred_ref_space_uid=pred_ref_space_uid,
        pred_ref_space_uid_to_perm=pred_ref_space_uid_to_perm,
        optimal_mol_permutation=optimal_mol_permutation,
    )

    # Get the equivalent token index permutation
    gt_token_index = atom_subset_index_to_token_subset_index(
        gt_atom_index, n_token=single_batch["residue_index"]
    )

    return gt_atom_index, gt_token_index


# TODO: group all permutation features under separate key?
def multi_chain_permutation_alignment(batch: dict, output: dict):
    batch_size = batch["residue_index"].shape[0]

    # This will store the final per-batch subsetting and reordering of the ground-truth
    # features
    gt_atom_indexes = []
    gt_token_indexes = []

    # Currently has to be run sequentially per batch, so split features batch-wise
    for i in range(batch_size):
        single_batch = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                single_batch[key] = value[i]
            elif isinstance(value, dict):
                single_batch[key] = {k: v[i] for k, v in value.items()}
            else:
                # Ignore other types
                pass

        single_predicted_positions = output["atom_positions_predicted"][i]

        # Get the permutation indices for the current batch that subset and reorder the
        # ground-truth features to match the prediction
        gt_atom_index, gt_token_index = single_batch_multi_chain_permutation_alignment(
            single_batch, single_predicted_positions
        )
        gt_atom_indexes.append(gt_atom_index)
        gt_token_indexes.append(gt_token_index)

    # TODO: Could require a specific input format here in which features are grouped
    # into tensordicts by token-wise and atom-wise with the batch-dims precisely
    # specified
    ground_truth_dict_permuted = permute_gt_features(
        batch["ground_truth"], gt_atom_indexes, gt_token_indexes
    )

    return ground_truth_dict_permuted
