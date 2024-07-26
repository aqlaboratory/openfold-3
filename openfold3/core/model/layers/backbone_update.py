from typing import Tuple

import torch
from torch import nn as nn

from openfold3.core.config import default_linear_init_config as lin_init
from openfold3.core.model.primitives import Linear
from openfold3.core.utils.geometry.rigid_matrix_vector import Rigid3Array
from openfold3.core.utils.geometry.rotation_matrix import Rot3Array
from openfold3.core.utils.geometry.vector import Vec3Array


class BackboneUpdate(nn.Module):
    """
    Implements part of AF2 Algorithm 23.
    """

    def __init__(self, c_s, linear_init_params=lin_init.bb_update_init):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super().__init__()

        self.c_s = c_s

        self.linear = Linear(self.c_s, 6, **linear_init_params.linear)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class QuatRigidUpdate(nn.Module):
    def __init__(self, c_hidden, full_quat, linear_init_params=lin_init.bb_update_init):
        super().__init__()
        self.full_quat = full_quat
        rigid_dim = 7 if self.full_quat else 6

        self.linear = Linear(
            c_hidden, rigid_dim, precision=torch.float32, **linear_init_params.linear
        )

    def forward(self, activations: torch.Tensor) -> Rigid3Array:
        # NOTE: During training, this needs to be run in higher precision
        rigid_flat = self.linear(activations)

        rigid_flat = torch.unbind(rigid_flat, dim=-1)
        if self.full_quat:
            qw, qx, qy, qz = rigid_flat[:4]
            translation = rigid_flat[4:]
        else:
            qx, qy, qz = rigid_flat[:3]
            qw = torch.ones_like(qx)
            translation = rigid_flat[3:]

        rotation = Rot3Array.from_quaternion(
            qw,
            qx,
            qy,
            qz,
            normalize=True,
        )
        translation = Vec3Array(*translation)
        return Rigid3Array(rotation, translation)
