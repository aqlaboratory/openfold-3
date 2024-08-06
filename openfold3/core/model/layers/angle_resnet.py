from typing import Tuple

import torch
from torch import nn as nn

from openfold3.core.config import default_linear_init_config as lin_init
from openfold3.core.model.primitives import Linear


class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden, linear_init_params=lin_init.angle_resnet_block_init):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super().__init__()

        self.c_hidden = c_hidden

        self.linear_1 = Linear(
            self.c_hidden, self.c_hidden, **linear_init_params.linear_1
        )
        self.linear_2 = Linear(
            self.c_hidden, self.c_hidden, **linear_init_params.linear_2
        )

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)

        return a + s_initial


class AngleResnet(nn.Module):
    """
    Implements AF2 Algorithm 20, lines 11-14
    """

    def __init__(
        self,
        c_in,
        c_hidden,
        no_blocks,
        no_angles,
        epsilon,
        linear_init_params=lin_init.angle_resnet_init,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
            linear_init_params:
                Initialization parameters for linear layers
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = Linear(
            self.c_in, self.c_hidden, **linear_init_params.linear_in
        )
        self.linear_initial = Linear(
            self.c_in, self.c_hidden, **linear_init_params.linear_initial
        )

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(
                c_hidden=self.c_hidden,
                linear_init_params=linear_init_params.angle_resnet_block,
            )
            self.layers.append(layer)

        self.linear_out = Linear(
            self.c_hidden, self.no_angles * 2, **linear_init_params.linear_out
        )

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, no_angles * 2]
        s = self.linear_out(s)

        # [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.eps,
            )
        )
        s = s / norm_denom

        return unnormalized_s, s
