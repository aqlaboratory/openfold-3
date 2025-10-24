# Copyright 2025 AlQuraishi Laboratory
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

import importlib

import torch
from ml_collections import ConfigDict
from torch import nn

import openfold3.core.config.default_linear_init_config as lin_init

from .linear import Linear

triton_is_installed = importlib.util.find_spec("triton") is not None
if triton_is_installed:
    from openfold3.core.kernels.triton.swiglu import LigerSiLUMulFunction


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        linear_init_params: ConfigDict = lin_init.swiglu_init,
    ):
        """
        Args:
            c_in: Number of input channels
            c_out: Number of output channels
            linear_init_params: Linear layer initialization parameters
        """
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear_a = Linear(self.c_in, self.c_out, **linear_init_params.linear_a)
        self.linear_b = Linear(self.c_in, self.c_out, **linear_init_params.linear_b)
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if triton_is_installed and x.is_cuda:
            return LigerSiLUMulFunction.apply(self.linear_a(x), self.linear_b(x))

        return self.swish(self.linear_a(x)) * self.linear_b(x)
