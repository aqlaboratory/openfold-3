# Copyright 2026 AlQuraishi Laboratory
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

import importlib
import math

import torch
import torch.nn as nn
from ml_collections import ConfigDict

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.primitives.linear import Linear
from openfold3.core.utils.chunk_utils import chunk_layer

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed


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
        deepspeed_is_initialized = (
            deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
        )

        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.amp.autocast("cuda", enabled=False):
                weight = self.weight.to(dtype=d) if self.weight is not None else None
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                out = self._layer_norm(x, weight, bias)
        else:
            out = self._layer_norm(x, self.weight, self.bias)

        return out

    # There are at least two bugs in the Torch layer norm kernel for very
    # large inputs at the time of writing:

    # - A: https://github.com/pytorch/pytorch/issues/181555: when numel > 2^32
    #   and the channel dimension is a multiple of 4 (so the vectorized kernel
    #   is used) there is a uint32 overflow in the row offset. Fixed in
    #   https://github.com/pytorch/pytorch/pull/181600, but that is not yet in
    #   the stable release.
    # - B: https://github.com/pytorch/pytorch/issues/184826: when combined batch
    #   dim is > 2^23 and the channel dimension is not a multiple of 4 (so the
    #   non-vectorized kernel is used), there's some failure not yet root
    #   caused, that means that some outputs are never addressed and are filled
    #   with garbage values. At batch dim of *exactly* 2^23, there's actually an
    #   "invalid configuration argument" error instead of bad output. This is
    #   not yet fixed at the time of writing.

    # We chunk over the leading (batch) dims to keep every kernel call below
    # both cliffs: chunk_size < 2^23 AND chunk_size * C < 2^32.
    _SAFE_NUMEL = 2**32 - 1
    _SAFE_BATCH = 2**23 - 1

    def _layer_norm(self, x, weight, bias):
        def _ln(x):
            return nn.functional.layer_norm(
                input=x,
                normalized_shape=self.c_in,
                weight=weight,
                bias=bias,
                eps=self.eps,
            )

        no_batch_dims = x.dim() - len(self.c_in)
        per_slice_numel = math.prod(self.c_in)
        flat_batch = x.numel() // per_slice_numel
        max_chunk = min(self._SAFE_BATCH, self._SAFE_NUMEL // per_slice_numel)
        if flat_batch <= max_chunk:
            return _ln(x)
        return chunk_layer(
            _ln,
            {"x": x},
            chunk_size=max_chunk,
            no_batch_dims=no_batch_dims,
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
