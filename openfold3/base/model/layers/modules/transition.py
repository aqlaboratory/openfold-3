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
from typing import Optional

import torch
import torch.nn as nn

from openfold3.base.model.primitives import AdaLN, LayerNorm, Linear, ReLULayer, SwiGLU
from openfold3.base.utils.chunk_utils import chunk_layer


class ReLUTransitionLayer(nn.Module):
    """
    Feed-forward network applied to activations after attention.
    """
    def __init__(self, num_relu_layers, c_in, n):
        """
        Args:
            num_layers:
                Number of ReluLayers to apply.
            c_in:
                Input channel dimension
            n:
                Factor multiplied to c_in to obtain the hidden channel
                dimension
        """
        super(ReLUTransitionLayer, self).__init__()

        self.c_in = c_in
        self.n= n
        self.num_relu_layers = num_relu_layers

        self.layers = nn.ModuleList([ReLULayer(c_in=self.c_in, c_out=self.n * self.c_in, bias=True, init='relu')
                                     for _ in range(self.num_relu_layers)])
        self.linear_out = Linear(self.n * self.c_in, self.c_in, init="final")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, C_in] Input tensor
            mask:
                [*, N_res, C_in] Tensor mask
        Returns:
            x:
                [*, N_res, C_m] Tensor update
        """
        for l in self.layers:
            x = l(x)

        x = self.linear_out(x) * mask
        return x


class ReLUTransition(nn.Module):
    """
    Feed-forward network applied to MSA and Pair activations after attention.

    Implements AF2 Algorithm 9 and 15
    """
    def __init__(self, c_in, n):
        """
        Args:
            c_in:
                MSA/Pair channel dimension
            n:
                Factor multiplied to c_in to obtain the hidden channel
                dimension
        """
        super(ReLUTransition, self).__init__()

        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(self.c_in)
        self.transition_mlp = ReLUTransitionLayer(num_relu_layers=1, c_in=self.c_in, n=self.n)

    def _transition(self, x, mask):
        x = self.layer_norm(x)
        x = self.transition_mlp(x=x, mask=mask)
        return x

    @torch.jit.ignore
    def _chunk(self,
        x: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
         return chunk_layer(
             self._transition,
             {"x": x, "mask": mask},
             chunk_size=chunk_size,
             no_batch_dims=len(x.shape[:-2]),
         )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_seq/N_res, N_res, C_in] MSA/Pair activation
            mask:
                [*, N_seq/N_res, N_res, C_in] MSA/Pair mask
        Returns:
            m:
                [*, N_seq/N_res, N_res, C_in] MSA/Pair activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the mask here.
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            x = self._chunk(x, mask, chunk_size)
        else:
            x = self._transition(x, mask)

        return x


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers, dropout_rate):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList([ReLUTransitionLayer(num_relu_layers=2, c_in=self.c, n=1)
                                     for _ in range(self.num_layers)])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s, mask=None):
        if mask is None:
            mask = s.new_ones(s.shape[:-1])

        mask = mask.unsqueeze(-1)

        for l in self.layers:
            s = s + l(s, mask)

        s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class SwiGLUTransition(nn.Module):
    """
    Implements AF3 Algorithm 11.
    """
    def __init__(self, c_in: int, n: int = 4):
        """

        Args:
            c_in:
                Input channel dimension
            n:
                Factor by which c_in is multiplied to obtain hidden channel
                dimension
        """
        super(SwiGLUTransition, self).__init__()

        self.c_in = c_in
        self.n = n

        self.layer_norm = LayerNorm(self.c_in)
        self.swiglu = SwiGLU(self.c_in, self.n * self.c_in)
        self.linear_out = Linear(self.n * self.c_in, c_in, bias=False, init="final")

    def _transition(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x:
            mask:

        Returns:

        """
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        mask = mask.unsqueeze(-1)

        # [*, N_res/N_seq, N_res, C_in]
        x = self.layer_norm(x)

        # [*, N_res/N_seq, N_res, C_hidden]
        x = self.swiglu(x)

        # [*, N_res/N_seq, N_res, C_in]
        x = self.linear_out(x)
        x = x * mask

        return x

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """

        Args:
            x:
            mask:
            chunk_size:

        Returns:

        """
        return chunk_layer(
            self._transition,
            {"x": x, "mask": mask},
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """

        Args:
            x:
            mask:
            chunk_size:

        Returns:

        """
        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        # [*, N_res/N_seq, N_res, 1]
        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            x = self._chunk(x, mask, chunk_size)
        else:
            x = self._transition(x=x, mask=mask)

        return x


class ConditionedTransitionBlock(nn.Module):
    """
    Implements AF3 Algorithm 25.
    """

    def __init__(self, c_in: int, n: int = 2):
        """

        Args:
            c_in:
                Input channel dimension
            n:
                Factor by which c_in is multiplied to obtain hidden channel
                dimension
        """
        super(ConditionedTransitionBlock, self).__init__()

        self.c_in = c_in
        self.n = n

        self.layer_norm = AdaLN(c_in=c_in)

        self.swiglu = SwiGLU(self.c_in, self.n * self.c_in)

        self.sigmoid = nn.Sigmoid()
        self.linear_g = Linear(self.c_in, self.c_in, init="gating_ada_zero")
        self.linear_out = Linear(self.n * self.c_in, self.c_in, bias=False, init="final")

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            a:
            s:
            mask:

        Returns:

        """
        if mask is None:
            mask = a.new_ones(a.shape[:-1])

        mask = mask.unsqueeze(-1)

        a = self.layer_norm(a, s)
        b = self.swiglu(a)
        # AdaLN-zero
        a = self.sigmoid(self.linear_g(s)) * self.linear_out(b)
        a = a * mask

        return a
