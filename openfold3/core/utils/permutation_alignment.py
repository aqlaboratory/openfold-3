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
from openfold3.core.utils.geometry.kabsch_alignment import (
    Transformation,
    get_optimal_transformation,
)

logger = logging.getLogger(__name__)


def apply_transformation(
    positions: torch.Tensor,
    transformation: Transformation,
) -> torch.Tensor:
    """
    Apply an affine transformation to a set of coordinates.

    Args:
        positions:
            [*, N, 3] the coordinates to transform
        transformation:
            the transformation to apply

    Returns:
        [*, N, 3] the transformed coordinates
    """
    positions = positions @ transformation.rotation_matrix
    positions = positions + transformation.translation_vector

    return positions


@overload
def split_feats_by_id(
    feats: torch.Tensor, id: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor]: ...


def split_feats_by_id(
    feats: list[torch.Tensor],
    id: torch.Tensor,
) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
    unique_ids = torch.unique(id)

    if isinstance(feats, torch.Tensor):
        split_feats = [feats[id == single_id] for single_id in unique_ids]
    else:
        split_feats = [
            [feat[id == single_id] for single_id in unique_ids] for feat in feats
        ]

    return split_feats, unique_ids


# TODO: Call all this stuff mol_entity_id, mol_sym_id, mol_atom_id, mol_conformer_id
# TODO: Revisit this function if we really need all these optional args
def get_gt_segment_mask(
    segment_perm_sym_token_index: torch.Tensor,
    gt_perm_sym_token_index: torch.Tensor,
    segment_perm_sym_id: int | None = None,
    segment_perm_entity_id: int | None = None,
    gt_perm_entity_id: torch.Tensor | None = None,
    gt_perm_sym_id: torch.Tensor | None = None,
) -> torch.Tensor:
    segment_mask = torch.ones_like(gt_perm_sym_token_index, dtype=torch.bool)

    if segment_perm_sym_id is not None:
        if gt_perm_sym_id is None:
            raise ValueError(
                "Need to pass gt_perm_sym_id if segment_perm_sym_id is set"
            )

        segment_mask &= gt_perm_sym_id == segment_perm_sym_id

    if segment_perm_entity_id is not None:
        if gt_perm_entity_id is None:
            raise ValueError(
                "Need to pass gt_perm_entity_id if segment_perm_entity_id is set"
            )

        segment_mask &= gt_perm_entity_id == segment_perm_entity_id

    segment_mask &= torch.isin(gt_perm_sym_token_index, segment_perm_sym_token_index)

    return segment_mask


def get_centroid(coords: torch.Tensor, mask: torch.Tensor):
    # Get centroid of only the unmasked coordinates
    n_observed_atoms = torch.sum(mask, dim=-1, keepdim=True)
    centroid = (
        torch.sum(
            coords * mask[..., None],
            dim=-2,
        )
        / n_observed_atoms
    )

    return centroid


def get_sym_id_with_most_resolved_atoms(
    resolved_mask: torch.Tensor,
    perm_sym_id: torch.Tensor,
) -> tuple[int, int]:
    perm_sym_id = perm_sym_id[resolved_mask]

    unique_sym_ids, n_resolved_atoms = torch.unique(perm_sym_id, return_counts=True)

    # If no resolved atoms, return first sym_id
    if n_resolved_atoms.numel() == 0:
        best_idx = 0
    else:
        best_idx = torch.argmax(n_resolved_atoms)

    best_sym_id = unique_sym_ids[best_idx].item()
    n_resolved = n_resolved_atoms[best_idx].item()

    return best_sym_id, n_resolved


def get_gt_anchor_mask(
    gt_token_center_resolved_mask: torch.Tensor,
    gt_perm_entity_id: torch.Tensor,
    gt_perm_sym_id: torch.Tensor,
    gt_is_ligand: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    gt_token_center_resolved_mask = gt_token_center_resolved_mask.bool()

    # Avoid selecting ligand anchor chains if there are any resolved non-ligand chains
    # TODO: This also ignores covalent ligands for the purpose of selecting the anchor
    # molecule, which isn't wrong but also not really necessary, but was more convenient
    # to implement for now
    if torch.any(~gt_is_ligand[gt_token_center_resolved_mask]):
        gt_token_center_mask = gt_token_center_resolved_mask & ~gt_is_ligand
    else:
        logger.warning(
            "No non-ligand chains found, selecting ligand anchor. This is likely to "
            + "result in an unstable alignment in the current implementation if the "
            + "ligand contains symmetric atoms."
        )
        gt_token_center_mask = gt_token_center_resolved_mask

    # Figure out least ambiguous stoichiometry (ignoring entirely unresolved molecules)
    entity_sym_id_combinations = torch.cat(
        [
            gt_perm_entity_id[gt_token_center_mask].unsqueeze(-1),
            gt_perm_sym_id[gt_token_center_mask].unsqueeze(-1),
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

    # In case of tie, take the entity ID and sym ID with the biggest number of resolved
    # ground-truth atoms (hopefully resulting in the most stable alignments on average)
    if len(least_ambiguous_entity_ids) > 1:
        logger.debug("Multiple least ambiguous entities found.")

        entity_sym_id_to_n_tokens = {}

        for entity_id in least_ambiguous_entity_ids:
            entity_mask = gt_perm_entity_id == entity_id
            best_sym_id, n_resolved = get_sym_id_with_most_resolved_atoms(
                gt_token_center_resolved_mask[entity_mask],
                gt_perm_sym_id[entity_mask],
            )

            entity_sym_id_to_n_tokens[(entity_id, best_sym_id)] = n_resolved

        # Select final anchor entity and sym ID
        anchor_entity_id, anchor_sym_id = max(
            entity_sym_id_to_n_tokens, key=entity_sym_id_to_n_tokens.get
        )

    else:
        # Single-element tuple unpacking (will fail if != 1 elements)
        (anchor_entity_id,) = least_ambiguous_entity_ids

        # Get sym ID with biggest number of resolved ground-truth atoms
        anchor_entity_mask = gt_perm_entity_id == anchor_entity_id
        anchor_sym_id, _ = get_sym_id_with_most_resolved_atoms(
            gt_token_center_resolved_mask[anchor_entity_mask],
            gt_perm_sym_id[anchor_entity_mask],
        )

    # Get the mask for the anchor molecule
    anchor_mask = (gt_perm_entity_id == anchor_entity_id) & (
        gt_perm_sym_id == anchor_sym_id
    )

    # TODO: prevent this from happening instead of just throwing a warning
    anchor_resolved_mask = gt_token_center_resolved_mask[anchor_mask]
    n_resolved_anchor_atoms = anchor_resolved_mask.sum()
    if n_resolved_anchor_atoms < 3:
        logger.warning(
            f"Anchor molecule has less than 3 resolved atoms "
            f"({n_resolved_anchor_atoms}). This will result in an unstable "
            "alignment."
        )

    logger.debug(
        f"Number of resolved atoms in anchor molecule: {n_resolved_anchor_atoms}"
    )

    return anchor_mask


def get_anchor_transformations(
    gt_anchor_coords: torch.Tensor,
    gt_anchor_resolved_mask: torch.Tensor,
    gt_anchor_sym_token_index: torch.Tensor,
    gt_anchor_entity_id: int,
    pred_coords: torch.Tensor,
    pred_perm_entity_id: torch.Tensor,
    pred_perm_sym_id: torch.Tensor,
    pred_perm_sym_token_index: torch.Tensor,
):
    # Get all the molecules in the pred that are equivalent to the anchor
    pred_entity_mask = pred_perm_entity_id == gt_anchor_entity_id
    pred_coords_entity = pred_coords[pred_entity_mask]
    pred_perm_sym_id_entity = pred_perm_sym_id[pred_entity_mask]
    pred_perm_sym_token_index_entity = pred_perm_sym_token_index[pred_entity_mask]

    pred_sym_ids = torch.unique(pred_perm_sym_id_entity)

    transformations = []

    # Transform each equivalent predicted molecule onto the anchor molecule
    for sym_id in pred_sym_ids:
        pred_sym_mask = pred_perm_sym_id_entity == sym_id

        # Get the part of the full anchor chain that matches the in-crop segment
        gt_segment_mask = get_gt_segment_mask(
            pred_perm_sym_token_index_entity[pred_sym_mask],
            gt_anchor_sym_token_index,
        )
        gt_segment_coords = gt_anchor_coords[gt_segment_mask]
        gt_segment_resolved_mask = gt_anchor_resolved_mask[gt_segment_mask]

        pred_coords_sym = pred_coords_entity[pred_sym_mask]

        # Get the optimal transformation
        transformations.append(
            get_optimal_transformation(
                gt_segment_coords,
                pred_coords_sym,
                gt_segment_resolved_mask,
            )
        )

    # Stack the transformations to enable broadcasting
    transformations = Transformation(
        rotation_matrix=torch.stack([t.rotation_matrix for t in transformations]),
        translation_vector=torch.stack([t.translation_vector for t in transformations]),
    )

    return transformations


# TODO: Think again about whether perm_entity_ids should be [N_atom] or [N_token]
# (leaning towards the latter)
def find_greedy_optimal_mol_permutation(
    gt_token_center_positions_transformed: torch.Tensor,
    gt_token_center_resolved_mask: torch.Tensor,
    gt_perm_entity_ids: torch.Tensor,
    gt_perm_sym_ids: torch.Tensor,
    gt_perm_sym_token_index: torch.Tensor,
    pred_token_center_positions: torch.Tensor,
    pred_perm_entity_ids: torch.Tensor,
    pred_perm_sym_ids: torch.Tensor,
    pred_perm_sym_token_index: torch.Tensor,
):
    gt_token_center_resolved_mask = gt_token_center_resolved_mask.bool()

    # Get the unique entity IDs
    unique_pred_perm_entity_ids = torch.unique(pred_perm_entity_ids)

    # optimal_permutation = None
    # optimal_rmsd = torch.inf

    # Keep track of centroid distances and permutations over all anchor alignments
    centroid_dists_sq = []
    permutations = []

    # Iterate through each anchor alignment
    for gt_token_center_coords in gt_token_center_positions_transformed:
        centroid_dists_sq_aln = []
        permutation_aln = {}

        # Resolve permutation for each entity group separately
        for entity_id in unique_pred_perm_entity_ids:
            pred_entity_mask = pred_perm_entity_ids == entity_id
            pred_coords_entity = pred_token_center_positions[pred_entity_mask]
            pred_sym_ids_entity = pred_perm_sym_ids[pred_entity_mask]
            pred_sym_token_index_entity = pred_perm_sym_token_index[pred_entity_mask]

            gt_entity_mask = gt_perm_entity_ids == entity_id
            gt_coords_entity = gt_token_center_coords[gt_entity_mask]
            gt_resolved_mask_entity = gt_token_center_resolved_mask[gt_entity_mask]
            gt_sym_ids_entity = gt_perm_sym_ids[gt_entity_mask]
            gt_sym_token_index_entity = gt_perm_sym_token_index[gt_entity_mask]

            # Split ground-truth features into symmetric instances
            (
                (
                    gt_coords_entity_split,
                    gt_sym_token_index_entity_split,
                    gt_resolved_mask_entity_split,
                ),
                gt_unique_sym_ids_entity,
            ) = split_feats_by_id(
                [gt_coords_entity, gt_sym_token_index_entity, gt_resolved_mask_entity],
                gt_sym_ids_entity,
            )
            gt_coords_entity_split = torch.stack(gt_coords_entity_split)
            gt_resolved_mask_entity_split = torch.stack(gt_resolved_mask_entity_split)

            # TODO: Dev-only, remove later
            assert all(
                torch.equal(sym_token_tensor, gt_sym_token_index_entity_split[0])
                for sym_token_tensor in gt_sym_token_index_entity_split[1:]
            )
            # All these should be equivalent, so we just take the first one
            gt_sym_token_index_segment = gt_sym_token_index_entity_split[0]

            # Shuffle order of molecules (avoids any bias on the PDB ordering)
            unique_pred_sym_ids_entity = torch.unique(pred_sym_ids_entity)
            unique_pred_sym_ids_entity = unique_pred_sym_ids_entity[
                torch.randperm(unique_pred_sym_ids_entity.shape[0])
            ]

            # Track which ground-truth symmetry IDs have already been assigned
            used_gt_sym_ids = []

            # Run greedy assignment for this entity
            for sym_id in unique_pred_sym_ids_entity:
                # Get particular in-crop segment of the predicted molecule
                pred_sym_token_index_segment = pred_sym_token_index_entity[
                    pred_sym_ids_entity == sym_id
                ]

                # Match the segment of the predicted molecule
                gt_segment_mask = get_gt_segment_mask(
                    segment_perm_sym_token_index=pred_sym_token_index_segment,
                    gt_perm_sym_token_index=gt_sym_token_index_segment,
                )
                gt_coords_entity_segment = gt_coords_entity_split[:, gt_segment_mask, :]
                gt_resolved_mask_entity_segment = gt_resolved_mask_entity_split[
                    :, gt_segment_mask
                ]

                # Get centroids of ground-truth
                gt_coords_entity_centroids = get_centroid(
                    gt_coords_entity_segment, gt_resolved_mask_entity_segment
                )

                # Get centroid of prediction, while matching unresolved atoms on the GT
                # side
                pred_coords_sym = pred_coords_entity[pred_sym_ids_entity == sym_id]
                pred_coords_sym_centroid = get_centroid(
                    pred_coords_sym.unsqueeze(0), gt_resolved_mask_entity_segment
                )

                # Get squared distances between centroids
                pred_gt_dists_sq = (
                    (pred_coords_sym_centroid - gt_coords_entity_centroids)
                    .pow(2)
                    .sum(dim=-1)
                )

                # Mask already used IDs
                used_gt_sym_ids_mask = torch.isin(
                    gt_unique_sym_ids_entity, torch.tensor(used_gt_sym_ids)
                )
                pred_gt_dists_sq[used_gt_sym_ids_mask] = torch.inf

                # Make sure that entirely unresolved ground-truth centroids are picked
                # last (by setting higher than any other dist but lower than inf)
                gt_any_resolved_mask_centroid = gt_resolved_mask_entity_segment.any(
                    dim=-1
                )
                # TODO: check this
                pred_gt_dists_sq[~gt_any_resolved_mask_centroid] = torch.finfo(
                    pred_gt_dists_sq.dtype
                ).max

                # Get the best matching ground-truth symmetry ID
                best_gt_index = torch.argmin(pred_gt_dists_sq)
                best_gt_sym_id = gt_unique_sym_ids_entity[best_gt_index]
                best_gt_sym_id_dist_sq = pred_gt_dists_sq[best_gt_index]

                centroid_dists_sq_aln.append(best_gt_sym_id_dist_sq)
                used_gt_sym_ids.append(best_gt_sym_id)

                # Append the mapping between the predicted and ground-truth symmetry and
                # entity IDs to the permutation
                permutation_aln[(entity_id, sym_id)] = (entity_id, best_gt_sym_id)

        centroid_dists_sq.append(centroid_dists_sq_aln)
        permutations.append(permutation_aln)

    centroid_dists_sq = torch.tensor(centroid_dists_sq)

    # Change the max-value placeholder of entirely unresolved chains to be just above
    # the max of any resolved centroid distance for numerical stability
    unresolved_vals = centroid_dists_sq == torch.finfo(centroid_dists_sq.dtype).max
    centroid_dists_sq[unresolved_vals] = centroid_dists_sq[~unresolved_vals].max() + 1

    # Single RMSD over all centroid distances (this is a slight deviation from the
    # AF2-Multimer SI which calculates an RMSD per centroid, which would just be
    # equivalent to summing up absolute distances)
    # [N_align, N_centroid] -> [N_align]
    rmsds = torch.sqrt(centroid_dists_sq.mean(dim=-1))
    optimal_alignment_index = torch.argmin(rmsds)
    optimal_permutation = permutations[optimal_alignment_index]

    optimal_rmsd = rmsds[optimal_alignment_index]
    logger.debug(f"Found optimal permutation with RMSD: {optimal_rmsd}")

    assert optimal_permutation is not None

    return optimal_permutation


def get_permuted_gt_segment_token_index(
    gt_token_center_positions: torch.Tensor,
    gt_token_center_resolved_mask: torch.Tensor,
    gt_perm_entity_ids: torch.Tensor,
    gt_perm_sym_ids: torch.Tensor,
    gt_perm_sym_token_index: torch.Tensor,
    pred_token_center_positions: torch.Tensor,
    pred_perm_entity_ids: torch.Tensor,
    pred_perm_sym_ids: torch.Tensor,
    pred_perm_sym_token_index: torch.Tensor,
    optimal_permutation: dict[tuple[int, int], tuple[int, int]],
):
    gt_token_center_resolved_mask = gt_token_center_resolved_mask.bool()

    # Final index that will rearrange gt-token features to exactly match the prediction
    # (both in terms of matching the in-crop segment and the permuted order)
    token_subset_index = -torch.ones(
        pred_token_center_positions.shape[0],
        dtype=torch.long,
        device=pred_token_center_positions.device,
    )

    gt_token_index = torch.arange(
        gt_token_center_positions.shape[0], device=gt_token_center_positions.device
    )

    for (pred_entity_id, pred_sym_id), (
        gt_entity_id,
        gt_sym_id,
    ) in optimal_permutation.items():
        # The exact section of the prediction feature tensor that corresponds to the
        # current (entity, sym) segment
        pred_segment_mask = (pred_perm_entity_ids == pred_entity_id) & (
            pred_perm_sym_ids == pred_sym_id
        )

        # The exact section of the ground-truth feature tensor that corresponds to the
        # matching (entity, sym) segment as given by the optimal permutation
        gt_segment_mask = get_gt_segment_mask(
            segment_perm_sym_token_index=pred_perm_sym_token_index[pred_segment_mask],
            gt_perm_sym_token_index=gt_perm_sym_token_index,
            segment_perm_entity_id=gt_entity_id,
            segment_perm_sym_id=gt_sym_id,
            gt_perm_entity_id=gt_perm_entity_ids,
            gt_perm_sym_id=gt_perm_sym_ids,
        )

        # Insert the corresponding token index
        token_subset_index[pred_segment_mask] = gt_token_index[gt_segment_mask]

    assert torch.all(token_subset_index != -1)
    assert torch.equal(gt_perm_entity_ids[token_subset_index], pred_perm_entity_ids)

    return token_subset_index


def get_pred_to_permuted_gt_transformation(
    gt_token_center_positions: torch.Tensor,
    gt_token_center_resolved_mask: torch.Tensor,
    gt_perm_entity_ids: torch.Tensor,
    gt_perm_sym_ids: torch.Tensor,
    gt_perm_sym_token_index: torch.Tensor,
    pred_token_center_positions: torch.Tensor,
    pred_perm_entity_ids: torch.Tensor,
    pred_perm_sym_ids: torch.Tensor,
    pred_perm_sym_token_index: torch.Tensor,
    optimal_permutation: dict[tuple[int, int], tuple[int, int]],
) -> torch.Tensor:
    # Rearrange and subset the ground-truth coordinates to match the prediction (this
    # uses an arbitrary assignment of symmetry-equivalent atoms based on the original
    # ordering)
    gt_token_subset_index = get_permuted_gt_segment_token_index(
        gt_token_center_positions,
        gt_token_center_resolved_mask,
        gt_perm_entity_ids,
        gt_perm_sym_ids,
        gt_perm_sym_token_index,
        pred_token_center_positions,
        pred_perm_entity_ids,
        pred_perm_sym_ids,
        pred_perm_sym_token_index,
        optimal_permutation,
    )

    gt_token_center_positions_rearranged = gt_token_center_positions[
        gt_token_subset_index
    ]
    gt_token_center_resolved_mask_rearranged = gt_token_center_resolved_mask[
        gt_token_subset_index
    ]

    transformation = get_optimal_transformation(
        pred_token_center_positions,
        gt_token_center_positions_rearranged,
        gt_token_center_resolved_mask_rearranged,
    )

    return transformation


# TODO: should broacdcasting happen inside this?
# do I want to keep these variables this general if they're unpadded?
def get_final_atom_permutation_index(
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
    gt_resolved_mask = gt_resolved_mask.bool()

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
        dummy_token_mask,
        pred_num_atoms_per_token,
    )
    pred_perm_entity_id_atom = pred_token_feat_to_atoms(token_feat=pred_perm_entity_id)
    pred_perm_sym_id_atom = pred_token_feat_to_atoms(pred_perm_sym_id)
    pred_sym_conformer_id_atom = pred_token_feat_to_atoms(pred_perm_sym_conformer_id)

    dummy_token_mask = torch.ones_like(gt_perm_entity_id, dtype=torch.bool)
    gt_token_feat_to_atoms = partial(
        broadcast_token_feat_to_atoms,
        dummy_token_mask,
        gt_num_atoms_per_token,
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

        (
            (
                gt_positions_subset_grouped,
                gt_resolved_mask_subset_grouped,
                gt_atom_idx_subset_grouped,
            ),
            _,
        ) = split_feats_by_id(
            [gt_positions_subset, gt_resolved_mask_subset, gt_atom_idx_subset],
            gt_sym_conformer_id_atom_subset,
        )

        assert unique_ref_space_uids_subset.shape[0] == len(
            pred_positions_subset_grouped
        )

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
            permutations = pred_ref_space_uid_to_perm[ref_space_uid.item()]

            # If there is only the identity permutation (which is the case for a lot of
            # residues) skip the remaining computations
            if permutations.shape[0] == 1:
                identity_permutation = permutations[0]
                assert torch.all(
                    identity_permutation == torch.arange(permutations.shape[1])
                )
                permuted_atom_idxs.extend(gt_atom_idx_subset_conf[identity_permutation])
                continue

            # Versions of the ground-truth positions for each permutation
            gt_positions_subset_conf_perm = gt_positions_subset_conf[permutations]
            gt_resolved_mask_subset_conf_perm = gt_resolved_mask_subset_conf[
                permutations
            ]

            # Minimize RMSD
            dists_sq = (
                (pred_positions_subset_conf - gt_positions_subset_conf_perm)
                .pow(2)
                .sum(dim=-1)
                .mul(gt_resolved_mask_subset_conf_perm)
            )
            n_resolved_atoms = gt_resolved_mask_subset_conf.sum(dim=-1)
            rmsd = torch.sqrt(dists_sq.sum(dim=-1) / n_resolved_atoms)

            # Get permutation that minimizes RMSD
            best_permutation = permutations[torch.argmin(rmsd)]

            # Append the global atom indices corresponding to this permutation
            permuted_atom_idxs.extend(
                gt_atom_idx_subset_conf[best_permutation].tolist()
            )

        # Add the atom indices to the final atom index that tracks the global
        # permutation. We use pred_mask here to ensure that we directly match the layout
        # of the pred-features, even if the features belonging to a connected molecule
        # are not contiguous.
        atom_subset_index[pred_mask] = torch.tensor(permuted_atom_idxs)

    assert torch.all(atom_subset_index != -1)

    return atom_subset_index


def atom_subset_index_to_token_subset_index(
    atom_subset_index: torch.Tensor,
    gt_atoms_per_token: torch.Tensor,
) -> torch.Tensor:
    # Create an equivalent token subset index from the atom subset index
    gt_token_index = torch.arange(
        gt_atoms_per_token.shape[0], device=atom_subset_index.device
    )
    gt_token_index_atom = broadcast_token_feat_to_atoms(
        torch.ones_like(gt_token_index, dtype=torch.bool),
        gt_atoms_per_token,
        gt_token_index,
    )
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

    # NIT: Token dimensions for the feature dict could be handled in a more principled
    # way across code, e.g. in a tensordict with according batch-dims?
    # Permute all features in the feature dict
    for key, value in gt_feature_dict.items():
        if key in ["atom_positions", "atom_resolved_mask"]:
            new_feature_dict[key] = torch.stack(
                [
                    torch.index_select(value[i], dim=0, index=atom_indexes[i])
                    for i in range(batch_size)
                ]
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
                [
                    torch.index_select(value[i], dim=0, index=token_indexes[i])
                    for i in range(batch_size)
                ]
            )

    return new_feature_dict


def single_batch_multi_chain_permutation_alignment(
    single_batch: dict, single_predicted_positions: torch.Tensor
):
    # TODO: Note down what assumptions this function makes, especially with respect to
    # ordering of the mols

    gt_batch = single_batch["ground_truth"]

    # Get relevant features and get rid of padding
    gt_token_pad_mask = gt_batch["token_mask"].bool()
    gt_atom_pad_mask = gt_batch["atom_pad_mask"]
    pred_token_pad_mask = single_batch["token_mask"].bool()
    pred_atom_pad_mask = single_batch["atom_pad_mask"]

    gt_coords = gt_batch["atom_positions"][gt_atom_pad_mask]
    gt_resolved_mask = gt_batch["atom_resolved_mask"][gt_atom_pad_mask].bool()
    gt_perm_entity_id = gt_batch["perm_entity_id"][gt_token_pad_mask]
    gt_perm_sym_id = gt_batch["perm_sym_id"][gt_token_pad_mask]
    gt_perm_sym_conformer_id = gt_batch["perm_sym_conformer_id"][
        gt_token_pad_mask
    ]  # TODO: change this to atom-wise?
    gt_perm_sym_token_index = gt_batch["perm_sym_token_index"][gt_token_pad_mask]
    gt_num_atoms_per_token = gt_batch["num_atoms_per_token"][gt_token_pad_mask]
    gt_is_ligand = gt_batch["is_ligand"][gt_token_pad_mask]

    pred_coords = single_predicted_positions[pred_atom_pad_mask]
    pred_perm_entity_id = single_batch["perm_entity_id"][pred_token_pad_mask]
    pred_perm_sym_id = single_batch["perm_sym_id"][pred_token_pad_mask]
    pred_perm_sym_conformer_id = single_batch["perm_sym_conformer_id"][
        pred_token_pad_mask
    ]
    pred_perm_sym_token_index = single_batch["perm_sym_token_index"][
        pred_token_pad_mask
    ]
    pred_ref_space_uid = single_batch["ref_space_uid"][pred_atom_pad_mask]
    pred_ref_space_uid_to_perm = single_batch["ref_space_uid_to_perm"]
    pred_num_atoms_per_token = single_batch["num_atoms_per_token"][pred_token_pad_mask]

    # Subset coordinates to only token centers (generalization of C-alpha in
    # AF2-Multimer algorithm)
    gt_token_center_coords, gt_token_center_mask = get_token_center_atoms(
        gt_batch, gt_batch["atom_positions"], gt_batch["atom_resolved_mask"]
    )
    gt_token_center_coords = gt_token_center_coords[gt_token_pad_mask]
    gt_token_center_mask = gt_token_center_mask[gt_token_pad_mask]

    # Here everything is "resolved" because the model predicts all coordinates so no
    # need to pass an actual mask
    pred_coords_dummy_mask = torch.ones_like(
        single_predicted_positions[:, 0], dtype=torch.bool
    )
    pred_token_center_coords, _ = get_token_center_atoms(
        single_batch, single_predicted_positions, pred_coords_dummy_mask
    )
    pred_token_center_coords = pred_token_center_coords[pred_token_pad_mask]

    # Get the anchor molecule from the ground truth
    gt_anchor_mask = get_gt_anchor_mask(
        gt_token_center_mask,
        gt_perm_entity_id,
        gt_perm_sym_id,
        gt_is_ligand,
    )
    gt_anchor_coords = gt_token_center_coords[gt_anchor_mask]
    gt_anchor_resolved_mask = gt_token_center_mask[gt_anchor_mask]
    gt_anchor_perm_entity_id = gt_perm_entity_id[gt_anchor_mask][0].item()
    gt_anchor_perm_sym_token_index = gt_perm_sym_token_index[gt_anchor_mask]

    # Get transformations of ground-truth anchor molecule to predicted anchor molecules
    transformations = get_anchor_transformations(
        gt_anchor_coords,
        gt_anchor_resolved_mask,
        gt_anchor_perm_sym_token_index,
        gt_anchor_perm_entity_id,
        pred_token_center_coords,
        pred_perm_entity_id,
        pred_perm_sym_id,
        pred_perm_sym_token_index,
    )

    gt_token_center_positions_transformed = apply_transformation(
        gt_token_center_coords, transformations
    )

    # ASSIGNMENT STAGE
    # Find optimal mapping between symmetry-equivalent molecules
    optimal_mol_permutation = find_greedy_optimal_mol_permutation(
        gt_token_center_positions_transformed,
        gt_token_center_mask,
        gt_perm_entity_id,
        gt_perm_sym_id,
        gt_perm_sym_token_index,
        pred_token_center_coords,
        pred_perm_entity_id,
        pred_perm_sym_id,
        pred_perm_sym_token_index,
    )

    # Get alignment of predicted token-center coordinates to the ground-truth
    # token-center coordinates before resolving the atom-level symmetry permutations
    pred_to_gt_transformation = get_pred_to_permuted_gt_transformation(
        gt_token_center_coords,
        gt_token_center_mask,
        gt_perm_entity_id,
        gt_perm_sym_id,
        gt_perm_sym_token_index,
        pred_token_center_coords,
        pred_perm_entity_id,
        pred_perm_sym_id,
        pred_perm_sym_token_index,
        optimal_mol_permutation,
    )

    pred_coords_aligned = apply_transformation(pred_coords, pred_to_gt_transformation)

    # Get the final atom index permutation that applies the optimal permutation found
    # earlier, and simultaneously resolves symmetry-equivalent atoms, resulting in a
    # final atom-wise rearrangement of the entire ground-truth feature tensor
    gt_atom_index = get_final_atom_permutation_index(
        gt_positions=gt_coords,
        gt_resolved_mask=gt_resolved_mask,
        gt_perm_entity_id=gt_perm_entity_id,
        gt_perm_sym_id=gt_perm_sym_id,
        gt_perm_sym_conformer_id=gt_perm_sym_conformer_id,
        gt_num_atoms_per_token=gt_num_atoms_per_token,
        pred_positions=pred_coords_aligned,
        pred_perm_entity_id=pred_perm_entity_id,
        pred_perm_sym_id=pred_perm_sym_id,
        pred_perm_sym_conformer_id=pred_perm_sym_conformer_id,
        pred_num_atoms_per_token=pred_num_atoms_per_token,
        pred_ref_space_uid=pred_ref_space_uid,
        pred_ref_space_uid_to_perm=pred_ref_space_uid_to_perm,
        optimal_mol_permutation=optimal_mol_permutation,
    )

    # Get the equivalent token index permutation
    gt_token_index = atom_subset_index_to_token_subset_index(
        gt_atom_index, gt_num_atoms_per_token
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
            elif key == "ref_space_uid_to_perm":
                # This is a special case that provides a separate dict per batch
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
