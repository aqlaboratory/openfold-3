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

import logging
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)


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

    batch_dims = H.shape[:-2]
    original_dtype = H.dtype

    # This is necessary for bf16/fp16 training
    with torch.amp.autocast("cuda", dtype=torch.float32):
        try:
            U, _, Vt = torch.linalg.svd(H.float())

            V = Vt.transpose(-2, -1)
            Ut = U.transpose(-2, -1)

            # Determinants for reflection correction (should be either 1 or -1)
            dets = torch.det(V @ Ut)

            # Create correction tensor [*, 3, 3]
            D = torch.eye(3, device=mobile_positions.device, dtype=V.dtype).tile(
                (*batch_dims, 1, 1)
            )
            D[..., -1, -1] = torch.sign(dets)

            R = V @ D @ Ut

        # Fix for rare edge-cases
        except Exception as e:
            logger.warning(
                f"Error in computing rotation matrix."
                f"Matrix:\n{H}\nError: {e}\n"
                "Returning identity matrix instead."
            )
            # Return identity rotation
            R = torch.eye(
                3, device=mobile_positions.device, dtype=mobile_positions.dtype
            ).tile((*batch_dims, 1, 1))

    return R.to(dtype=original_dtype)


# NIT: Maybe a bit confusing that there is already a rotation_matrix.py but that one
# comes from OF2 and is way overkill for this purpose
class Transformation(NamedTuple):
    """Named tuple to store a rotation matrix and translation vector.

    The transformation is stored in a way such that:

    (mobile_positions @ rotation_matrix) + translation_vector ≈ target_positions

    Attributes:
        rotation_matrix (torch.Tensor):
            [*, 3, 3] the rotation matrix (right-multiplied)
        translation_vector (torch.Tensor):
            [*, 3] the translation vector
    """

    rotation_matrix: torch.Tensor
    translation_vector: torch.Tensor


def get_optimal_transformation(
    mobile_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
) -> Transformation:
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
        (mobile_positions @ R) + t ≈ target_positions
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


def apply_transformation(
    positions: torch.Tensor,
    transformation: Transformation,
) -> torch.Tensor:
    """Applies an affine transformation to a set of coordinates.

    The rotation matrix is right-multiplied with the coordinates and the
    translation vector is added afterwards.

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


def kabsch_align(
    mobile_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
):
    """
    Aligns the predicted coordinates to the ground-truth coordinates
    using the Kabsch algorithm.

    Args:
        mobile_positions:
            [*, N, 3] the predicted coordinates
        target_positions:
            [*, N, 3] the ground-truth coordinates
        positions_mask:
            [*, N] mask for coordinates that should not be considered

    Returns:
        [*, N, 3] the mobile positions aligned to the target positions
    """
    transformation = get_optimal_transformation(
        mobile_positions=mobile_positions,
        target_positions=target_positions,
        positions_mask=positions_mask,
    )

    mobile_positions_aligned = apply_transformation(
        positions=mobile_positions, transformation=transformation
    )

    return mobile_positions_aligned
