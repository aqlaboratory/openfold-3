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

"""Outer product mean layer."""

import os
from functools import partial

import torch
import torch.nn as nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.chunk_utils import chunk_layer

_INFERENCE_CHUNK_CAP = 64


def is_inplace_opm_enabled() -> bool:
    """Return whether inference OPM updates should accumulate into z."""
    return os.environ.get("OPENFOLD3_INPLACE_OPM", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class OuterProductMean(nn.Module):
    """
    Implements AF2 Algorithm 10 / AF3 Algorithm 9.
    """

    def __init__(
        self, c_m, c_z, c_hidden, eps=1e-3, linear_init_params=lin_init.opm_init
    ):
        """
        Args:
            c_m:
                MSA embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
            eps:
                Epsilon value for numerical stability
            linear_init_params:
                Linear layer initialization parameters
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.eps = eps

        self.layer_norm = LayerNorm(c_m)
        self.linear_1 = Linear(c_m, c_hidden, **linear_init_params.linear_1)
        self.linear_2 = Linear(c_m, c_hidden, **linear_init_params.linear_2)
        self.linear_out = Linear(c_hidden**2, c_z, **linear_init_params.linear_out)

    def _opm(self, a, b, norm=None):
        # [*, N_res, N_res, C, C]
        outer = torch.einsum("...bac,...dae->...bdce", a, b)

        # [*, N_res, N_res, C * C]
        outer = outer.reshape(outer.shape[:-2] + (-1,))

        # [*, N_res, N_res, C_z]
        outer = self.linear_out(outer)
        if norm is not None:
            outer /= norm

        return outer

    @torch.jit.ignore
    def _chunk(self, a: torch.Tensor, b: torch.Tensor, chunk_size: int) -> torch.Tensor:
        # Since the "batch dim" in this case is not a true batch dimension
        # (in that the shape of the output depends on it), we need to
        # iterate over it ourselves
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        out = []
        for a_prime, b_prime in zip(a_reshape, b_reshape, strict=True):
            outer = chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
            )
            out.append(outer)

        # For some cursed reason making this distinction saves memory
        if len(out) == 1:
            outer = out[0].unsqueeze(0)
        else:
            outer = torch.stack(out, dim=0)

        outer = outer.reshape(a.shape[:-3] + outer.shape[1:])

        return outer

    @torch.jit.ignore
    def _chunk_add(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        norm: torch.Tensor,
        out: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        a_reshape = a.reshape((-1,) + a.shape[-3:])
        b_reshape = b.reshape((-1,) + b.shape[-3:])
        norm_reshape = norm.reshape((-1,) + norm.shape[-3:])
        out_reshape = out.reshape((-1,) + out.shape[-3:])

        for a_prime, b_prime, norm_prime, out_prime in zip(
            a_reshape,
            b_reshape,
            norm_reshape,
            out_reshape,
            strict=True,
        ):
            chunk_layer(
                partial(self._opm, b=b_prime),
                {"a": a_prime, "norm": norm_prime},
                chunk_size=chunk_size,
                no_batch_dims=1,
                _out=out_prime,
                _add_into_out=True,
            )

        return out

    def _forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
        add_to: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            mask:
                [*, N_seq, N_res] MSA mask
            add_to:
                Optional pair tensor updated directly during chunked inference.
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        if add_to is not None:
            expected_shape = m.shape[:-3] + (m.shape[-2], m.shape[-2], self.c_z)
            if (
                self.training
                or torch.is_grad_enabled()
                or not inplace_safe
                or chunk_size is None
                or not add_to.is_contiguous()
                or add_to.shape != expected_shape
                or add_to.device != m.device
                or add_to.dtype != m.dtype
            ):
                raise ValueError(
                    "Direct OPM accumulation requires eval-mode, no-grad, chunked "
                    "inference and a contiguous output matching the pair tensor."
                )

        # [*, N_seq, N_res, C_m]
        ln = self.layer_norm(m)

        # [*, N_seq, N_res, C]
        mask = mask.unsqueeze(-1)
        a = self.linear_1(ln)
        a = a * mask

        b = self.linear_2(ln)
        b = b * mask

        del ln

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)

        if add_to is not None:
            norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
            norm = norm + self.eps
            return self._chunk_add(
                a,
                b,
                norm,
                add_to,
                min(chunk_size, _INFERENCE_CHUNK_CAP),
            )

        if chunk_size is not None:
            outer = self._chunk(a, b, chunk_size)
        else:
            outer = self._opm(a, b)

        # [*, N_res, N_res, 1]
        norm = torch.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        # [*, N_res, N_res, C_z]
        if inplace_safe:
            outer /= norm
        else:
            outer = outer / norm

        return outer

    def forward(
        self,
        m: torch.Tensor,
        mask: torch.Tensor | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
        add_to: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward(m, mask, chunk_size, inplace_safe, add_to)
