# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
# Copyright 2021 DeepMind Technologies Limited
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

"""
Lightweight geometric augmentation utilities (AF3 Algorithm 19).

These pure-tensor helpers are used both by the model (``diffusion_module``) and
by the data featurization pipeline (``pipelines.featurization.conformer``). They
intentionally live in their own module so that importing them does not pull in
the heavy model stack (e.g. ``cuequivariance``). This keeps DataLoader worker
processes free of those imports, avoiding leaked-semaphore warnings at shutdown
(see issue #268).
"""

import torch

from openfold3.core.utils.rigid_utils import quat_to_rot


def sample_rotations(shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Sample random quaternions"""
    q = torch.randn(*shape, 4, dtype=dtype, device=device)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)

    rots = quat_to_rot(q)

    return rots


def centre_random_augmentation(
    xl: torch.Tensor, atom_mask: torch.Tensor, scale_trans: float = 1.0
) -> torch.Tensor:
    """
    Implements AF3 Algorithm 19.

    Args:
        xl:
            [*, N_atom, 3] Atom positions
        atom_mask:
            [*, N_atom] Atom mask
        scale_trans:
            Translation scaling factor
    Returns:
        Updated atom position with random global rotation and translation
    """
    rots = sample_rotations(shape=xl.shape[:-2], dtype=xl.dtype, device=xl.device)

    trans = scale_trans * torch.randn(
        (*xl.shape[:-2], 3), dtype=xl.dtype, device=xl.device
    )

    mean_xl = torch.sum(
        xl * atom_mask[..., None],
        dim=-2,
        keepdim=True,
    ) / torch.sum(atom_mask[..., None], dim=-2, keepdim=True)

    # center coordinates
    pos_centered = xl - mean_xl
    pos_out = pos_centered @ rots.transpose(-1, -2) + trans[..., None, :]
    pos_out = pos_out * atom_mask[..., None]

    return pos_out
