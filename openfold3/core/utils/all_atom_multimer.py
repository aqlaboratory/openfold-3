# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ops for all atom representations."""

import numpy as np
import torch

from openfold3.core.utils import geometry


def squared_difference(x, y):
    return np.square(x - y)


def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]


def make_transform_from_reference(
    a_xyz: geometry.Vec3Array, b_xyz: geometry.Vec3Array, c_xyz: geometry.Vec3Array
) -> geometry.Rigid3Array:
    """Returns rotation and translation matrices to convert from reference.

    Note that this method does not take care of symmetries. If you provide the
    coordinates in the non-standard way, the A atom will end up in the negative
    y-axis rather than in the positive y-axis. You need to take care of such
    cases in your code.

    Args:
        a_xyz: A Vec3Array.
        b_xyz: A Vec3Array.
        c_xyz: A Vec3Array.

    Returns:
        A Rigid3Array which, when applied to coordinates in a canonicalized
        reference frame, will give coordinates approximately equal
        the original coordinates (in the global frame).
    """
    rotation = geometry.Rot3Array.from_two_vectors(c_xyz - b_xyz, a_xyz - b_xyz)
    return geometry.Rigid3Array(rotation, b_xyz)
