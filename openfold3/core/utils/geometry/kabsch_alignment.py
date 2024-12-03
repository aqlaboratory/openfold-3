from typing import NamedTuple

import torch


def get_optimal_rotation_matrix(
    mobile_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Finds the optimal rotation matrix to superpose a set of predicted
    coordinates onto a set of target coordinates. Essentially equivalent to the
    Kabsch algorithm but does not perform any centering of the coordinates
    before computing the rotation matrix.

    Also see https://en.wikipedia.org/wiki/Kabsch_algorithm

    Because inputs are of shape [N, 3] instead of [3, N], the predictions need
    to be right-multiplied with the transpose of the returned rotation matrix
    (R @ X.T).T = X @ R.T

    Args:
        mobile_positions:
            [*, N, 3] the coordinates that should be rotated
        target_positions:
            [*, N, 3] the fixed target coordinates
        positions_mask:
            [*, N] mask for coordinates that should not be considered

    Returns:
        [*, 3, 3] the optimal rotation matrix, so that
        mobile_positions @ R.T ~= target_positions
    """
    # Set masked atoms to the origin (which makes the rotation matrix
    # independent of them)
    mobile_positions = mobile_positions * positions_mask[..., None]
    target_positions = target_positions * positions_mask[..., None]

    # Calculate covariance matrix [*, 3, 3]
    H = mobile_positions.transpose(-2, -1) @ target_positions

    # NOTE: cast H to float if this doesn't work with bf16
    # SVD
    U, _, Vt = torch.linalg.svd(H)

    V = Vt.transpose(-2, -1)
    Ut = U.transpose(-2, -1)

    # Determinants for reflection correction (should be either 1 or -1)
    dets = torch.det(V @ Ut)

    # Create correction tensor [*, 3, 3]
    batch_dims = H.shape[:-2]
    D = torch.eye(3, device=mobile_positions.device).tile(*batch_dims, 1, 1)
    D[..., -1, -1] = torch.sign(dets)

    R = V @ D @ Ut

    return R


# NIT: Maybe a bit confusing that there is already a rotation_matrix.py but that one
# comes from OF2 and is way overkill for this purpose
class Transformation(NamedTuple):
    rotation_matrix: torch.Tensor
    translation_vector: torch.Tensor


def get_optimal_transformation(
    mobile_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
) -> NamedTuple:
    """
    Uses the Kabsch algorithm to get the optimal rotation matrix and translation
    vector to align a set of mobile coordinates onto a set of fixed target
    coordinates.

    Args:
        mobile_positions:
            [*, N, 3] the predicted coordinates
        target_positions:
            [*, N, 3] the ground-truth coordinates
        positions_mask:
            [*, N] mask for coordinates that should not be considered

    Returns:
        A named tuple with the optimal rotation matrix [*, 3, 3] and the optimal
        translation vector [*, 3], so that:
        (mobile_positions @ R) + t ~= target_positions
    """
    # Get centroid of only the unmasked coordinates
    n_observed_atoms = torch.sum(positions_mask, dim=-1, keepdim=True)
    centroid_target = (
        torch.sum(
            target_positions * positions_mask[..., None],
            dim=-2,
            keepdim=True,
        )
        / n_observed_atoms[..., None]
    )
    centroid_mobile = (
        torch.sum(
            mobile_positions * positions_mask[..., None],
            dim=-2,
            keepdim=True,
        )
        / n_observed_atoms[..., None]
    )
    # Center coordinates
    mobile_positions_centered = mobile_positions - centroid_mobile
    target_positions_centered = target_positions - centroid_target

    # Calculate rotation matrix
    R = get_optimal_rotation_matrix(
        mobile_positions_centered, target_positions_centered, positions_mask
    )
    Rt = R.transpose(-2, -1)
    t = centroid_target - (centroid_mobile @ Rt)

    return Transformation(rotation_matrix=Rt, translation_vector=t)
