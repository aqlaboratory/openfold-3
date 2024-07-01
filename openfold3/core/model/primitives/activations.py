# Copyright 2021 AlQuraishi Laboratory
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

"""Activation functions."""

import torch
from torch import nn

from .linear import Linear


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def __init__(self, c_in: int, c_out: int, bias: bool = False, init="he_normal"):
        """
        Args:
            c_in: Number of input channels
            c_out: Number of output channels
            bias: Whether to include a bias term in linear layers
            init: Linear layer initialization method
        """
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear_a = Linear(self.c_in, self.c_out, bias=bias, init=init)
        self.linear_b = Linear(self.c_in, self.c_out, bias=bias, init=init)
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish(self.linear_a(x)) * self.linear_b(x)
