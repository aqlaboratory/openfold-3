# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Outpace Bio, Inc.
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

"""Normalization layers. Includes LayerNorm and AdaptiveLayerNorm."""

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.primitives.linear import Linear
from openfold3.core.utils.device_utils import autocast_device_type


class LayerNorm(nn.Module):
    """Basic LayerNorm layer with learnable scale and offset."""

    def __init__(
        self, c_in: int, create_scale: bool = True, create_offset: bool = True, eps=1e-5
    ):
        """
        Args:
            c_in: Number of input channels
            create_scale: Whether to create a learnable scale parameter
            create_offset: Whether to create a learnable offset parameter
            eps: Epsilon value for numerical stability
        """
        super().__init__()

        self.c_in = (c_in,)
        self.eps = eps
        self.weight = None
        self.bias = None

        if create_scale:
            self.weight = nn.Parameter(torch.ones(c_in))

        if create_offset:
            self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x) -> torch.Tensor:
        d = x.dtype

        # LayerNorm should be upcasted to fp32 anyway in torch
        # This enforces it if not running with autocast context
        if d in (torch.bfloat16, torch.float16):
            with torch.amp.autocast(autocast_device_type(x), enabled=False):
                weight = self.weight.float() if self.weight is not None else None
                bias = self.bias.float() if self.bias is not None else None
                out = nn.functional.layer_norm(
                    input=x.float(),
                    normalized_shape=self.c_in,
                    weight=weight,
                    bias=bias,
                    eps=self.eps,
                )
            return out.to(dtype=d)

        return nn.functional.layer_norm(
            input=x,
            normalized_shape=self.c_in,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )


class AdaLN(nn.Module):
    """Adaptive LayerNorm.

    Implements AF3 Algorithm 26.
    """

    def __init__(
        self,
        c_a: int,
        c_s: int,
        eps: float = 1e-5,
        linear_init_params: ConfigDict = lin_init.ada_ln_init,
    ):
        """
        Args:
            c_a: Number of input channels for input tensor
            c_s: Number of input channels for shift/scale tensor
            eps: Epsilon value for numerical stability
            linear_init_params: Linear layer initialization parameters
        """
        super().__init__()

        self.c_a = c_a
        self.c_s = c_s
        self.eps = eps

        self.layer_norm_a = LayerNorm(
            self.c_a, create_scale=False, create_offset=False, eps=self.eps
        )
        self.layer_norm_s = LayerNorm(
            self.c_s, create_scale=True, create_offset=False, eps=self.eps
        )

        self.sigmoid = nn.Sigmoid()
        self.linear_g = Linear(self.c_s, self.c_a, **linear_init_params.linear_g)
        self.linear_s = Linear(self.c_s, self.c_a, **linear_init_params.linear_s)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: Input tensor to be normalized
            s: Input tensor to compute shift/scale

        Returns:
            Normalized tensor
        """
        a = self.layer_norm_a(a)
        s = self.layer_norm_s(s)
        g = self.sigmoid(self.linear_g(s))
        a = g * a + self.linear_s(s)

        return a
