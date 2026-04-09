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
Triangle multiplicative update layers. Includes TriangleMultiplicativeUpdate from AF2
and FusedTriangleMultiplicativeUpdate from AF2-Multimer.
"""

import warnings
from abc import ABC, abstractmethod
from functools import partialmethod

import torch
import torch.nn as nn

import openfold3.core.config.default_linear_init_config as lin_init
from openfold3.core.kernels.cueq_utils import is_cuequivariance_available
from openfold3.core.model.primitives import LayerNorm, Linear
from openfold3.core.utils.tensor_utils import permute_final_dims

if is_cuequivariance_available():
    from cuequivariance_torch import triangle_multiplicative_update

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

warnings.filterwarnings("once")

if TRITON_AVAILABLE:

    @triton.jit
    def sigmoid_mul_kernel(
        x_ptr,
        gate_ptr,
        output_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused kernel: output = x * sigmoid(gate)
        """
        pid = tl.program_id(0)

        block_start = pid * BLOCK_SIZE
        if block_start >= N:
            return

        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = (offs >= 0) & (offs < N)

        safe_offs = offs.to(tl.int64)
        x_vals = tl.load(x_ptr + safe_offs, mask=mask, other=0.0)
        gate_vals = tl.load(gate_ptr + safe_offs, mask=mask, other=0.0)

        sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate_vals))
        result = x_vals * sigmoid_gate

        tl.store(output_ptr + safe_offs, result, mask=mask)

    @triton.jit
    def layernorm_kernel(
        x_ptr,
        output_ptr,
        weight_ptr,
        bias_ptr,
        M,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
    ):
        """
        Vectorized LayerNorm kernel with multi-row processing and safe bounds checking.
        Each program processes ROWS_PER_PROGRAM rows to amortize launch overhead.
        Uses vectorized tl.sum reductions instead of scalar Welford loop.
        Normalizes over the last dimension (N).
        """
        row_start = tl.program_id(0) * ROWS_PER_PROGRAM

        # Early exit if out of bounds
        if row_start >= M:
            return

        for row in range(ROWS_PER_PROGRAM):
            row_idx = row_start + row
            if row_idx < M:
                row_offset = row_idx.to(tl.int64) * N
                x_row_ptr = x_ptr + row_offset
                out_row_ptr = output_ptr + row_offset

                # Pass 1: vectorized mean and variance computation
                sum_val = 0.0
                sum_sq = 0.0
                for block_start in range(0, N, BLOCK_SIZE):
                    offs = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = (offs >= 0) & (offs < N)
                    safe_offs = offs.to(tl.int64)
                    x_block = tl.load(x_row_ptr + safe_offs, mask=mask, other=0.0).to(
                        tl.float32
                    )
                    sum_val += tl.sum(x_block, axis=0)
                    sum_sq += tl.sum(x_block * x_block, axis=0)

                mean = sum_val / N
                var = sum_sq / N - mean * mean
                rstd = 1.0 / tl.sqrt(tl.maximum(var, eps))

                # Pass 2: normalize and apply affine transform
                for block_start in range(0, N, BLOCK_SIZE):
                    offs = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = (offs >= 0) & (offs < N)

                    safe_offs = offs.to(tl.int64)
                    x_vals = tl.load(x_row_ptr + safe_offs, mask=mask, other=0.0).to(
                        tl.float32
                    )
                    w_vals = tl.load(weight_ptr + safe_offs, mask=mask, other=1.0).to(
                        tl.float32
                    )
                    b_vals = tl.load(bias_ptr + safe_offs, mask=mask, other=0.0).to(
                        tl.float32
                    )

                    out = (x_vals - mean) * rstd * w_vals + b_vals
                    tl.store(out_row_ptr + safe_offs, out, mask=mask)

    @triton.jit
    def linear_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        output_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Linear layer: output = x @ weight.T + bias
        x: [M, K], weight: [N, K] (PyTorch layout), bias: [N], output: [M, N]
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = rm < M
        n_mask = rn < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            k_mask = rk < K

            x_offsets = (
                rm[:, None].to(tl.int64) * stride_xm
                + rk[None, :].to(tl.int64) * stride_xk
            )

            w_offsets = (
                rn[None, :].to(tl.int64) * stride_wn
                + rk[:, None].to(tl.int64) * stride_wk
            )

            x = tl.load(
                x_ptr + x_offsets,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            w = tl.load(
                w_ptr + w_offsets,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            acc = tl.dot(x, w, acc)

        if HAS_BIAS:
            bias_offsets = rn.to(tl.int64)
            bias = tl.load(bias_ptr + bias_offsets, mask=n_mask, other=0.0)
            acc += bias[None, :]

        out_offsets = (
            rm[:, None].to(tl.int64) * stride_om + rn[None, :].to(tl.int64) * stride_on
        )

        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(
            output_ptr + out_offsets,
            acc,
            mask=out_mask,
        )

    @triton.jit
    def linear_fused_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        other_ptr,
        mask_ptr,
        add_tensor_ptr,
        output_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_om,
        stride_on,
        stride_other_m,
        stride_other_n,
        stride_mask_m,
        stride_mask_n,
        stride_add_m,
        stride_add_n,
        HAS_BIAS: tl.constexpr,
        APPLY_SIGMOID: tl.constexpr,
        APPLY_MUL: tl.constexpr,
        HAS_MASK: tl.constexpr,
        HAS_ADD: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused linear layer with optional sigmoid, elementwise multiply, mask, and add
        Supports combinations: linear [+ sigmoid] [* other] [* mask] [+ add_tensor]
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        m_mask = rm < M
        n_mask = rn < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            k_mask = rk < K

            x_offsets = (
                rm[:, None].to(tl.int64) * stride_xm
                + rk[None, :].to(tl.int64) * stride_xk
            )

            w_offsets = (
                rn[None, :].to(tl.int64) * stride_wn
                + rk[:, None].to(tl.int64) * stride_wk
            )

            x = tl.load(
                x_ptr + x_offsets,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            )

            w = tl.load(
                w_ptr + w_offsets,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            acc = tl.dot(x, w, acc)

        if HAS_BIAS:
            bias_offsets = rn.to(tl.int64)
            bias = tl.load(bias_ptr + bias_offsets, mask=n_mask, other=0.0)
            acc += bias[None, :]

        if APPLY_SIGMOID:
            # Clamp to avoid exp overflow
            acc = tl.where(acc > 20.0, 20.0, acc)
            acc = tl.where(acc < -20.0, -20.0, acc)
            acc = 1.0 / (1.0 + tl.exp(-acc))

        if APPLY_MUL:
            other_offsets = (
                rm[:, None].to(tl.int64) * stride_other_m
                + rn[None, :].to(tl.int64) * stride_other_n
            )
            other_vals = tl.load(
                other_ptr + other_offsets,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = acc * other_vals

        if HAS_MASK:
            mask_offsets = (
                rm[:, None].to(tl.int64) * stride_mask_m
                + rn[None, :].to(tl.int64) * stride_mask_n
            )
            mask_vals = tl.load(
                mask_ptr + mask_offsets,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = acc * mask_vals

        if HAS_ADD:
            add_offsets = (
                rm[:, None].to(tl.int64) * stride_add_m
                + rn[None, :].to(tl.int64) * stride_add_n
            )
            add_vals = tl.load(
                add_tensor_ptr + add_offsets,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc = acc + add_vals

        out_offsets = (
            rm[:, None].to(tl.int64) * stride_om + rn[None, :].to(tl.int64) * stride_on
        )

        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(
            output_ptr + out_offsets,
            acc,
            mask=out_mask,
        )


def triton_sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Fused sigmoid + elementwise multiply: x * sigmoid(gate)
    Supports broadcasting like PyTorch's native operations.

    Args:
        x: Input tensor
        gate: Gate tensor (broadcastable with x)

    Returns:
        x * sigmoid(gate)
    """
    broadcasted_shape = torch.broadcast_shapes(x.shape, gate.shape)
    x_expanded = x.expand(broadcasted_shape)
    gate_expanded = gate.expand(broadcasted_shape)

    x_flat = x_expanded.contiguous().reshape(-1)
    gate_flat = gate_expanded.contiguous().reshape(-1)
    N = x_flat.numel()

    output_flat = torch.empty_like(x_flat)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    sigmoid_mul_kernel[grid](x_flat, gate_flat, output_flat, N, BLOCK_SIZE=BLOCK_SIZE)

    return output_flat.reshape(broadcasted_shape)


def triton_layernorm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Triton LayerNorm with vectorized reductions and multi-row processing.

    Args:
        x: Input tensor [..., N]
        weight: Scale parameters [N]
        bias: Shift parameters [N]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as x
    """
    input_shape = x.shape
    N = input_shape[-1]
    M = x.numel() // N

    x_2d = x.reshape(M, N).contiguous()
    output_2d = torch.empty_like(x_2d)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(N))
    ROWS_PER_PROGRAM = 8
    grid = (triton.cdiv(M, ROWS_PER_PROGRAM),)

    layernorm_kernel[grid](
        x_2d,
        output_2d,
        weight,
        bias,
        M,
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        ROWS_PER_PROGRAM=ROWS_PER_PROGRAM,
    )

    return output_2d.reshape(input_shape)


def triton_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Triton Linear: output = x @ weight.T + bias

    Args:
        x: Input tensor [..., K]
        weight: Weight tensor [N, K] (same layout as nn.Linear.weight)
        bias: Optional bias tensor [N]

    Returns:
        Output tensor [..., N]
    """
    input_shape = x.shape
    K = input_shape[-1]
    M = x.numel() // K
    N = weight.shape[0]

    x_2d = x.reshape(M, K).contiguous()
    output_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = min(128, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_K = min(64, triton.next_power_of_2(K))

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    bias_ptr = bias if bias is not None else weight  # guarded by HAS_BIAS constexpr

    linear_kernel[grid](
        x_2d,
        weight,
        bias_ptr,
        output_2d,
        M,
        K,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        HAS_BIAS=(bias is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    output_shape = input_shape[:-1] + (N,)
    return output_2d.reshape(output_shape)


def triton_linear_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    other: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    add_tensor: torch.Tensor | None = None,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """
    Fused linear layer with optional sigmoid, elementwise multiply, mask, and add.

    Computes: linear(x) [* sigmoid] [* other] [* mask] [+ add_tensor]

    Args:
        x: Input tensor [..., K]
        weight: Weight tensor [N, K] (same layout as nn.Linear.weight)
        bias: Optional bias tensor [N]
        other: Optional tensor to multiply with (same shape as output)
        mask: Optional mask tensor (broadcastable with output)
        add_tensor: Optional tensor to add to result (same shape as output)
        apply_sigmoid: Whether to apply sigmoid to linear result

    Returns:
        Output tensor [..., N]
    """
    input_shape = x.shape
    K = input_shape[-1]
    M = x.numel() // K
    N = weight.shape[0]

    x_2d = x.reshape(M, K).contiguous()
    output_2d = torch.empty((M, N), dtype=x.dtype, device=x.device)

    BLOCK_M = min(128, triton.next_power_of_2(M))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_K = min(64, triton.next_power_of_2(K))

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    if other is not None:
        # other should match the output shape [..., N], so reshape accordingly
        other_M = other.numel() // N
        other_2d = other.reshape(other_M, N).contiguous()
        if other_M != M:
            raise ValueError(
                f"Shape mismatch: tensor expects {other_M} rows but output has {M}"
            )
        other_ptr = other_2d
        other_stride_m, other_stride_n = other_2d.stride(0), other_2d.stride(1)
    else:
        other_ptr = x_2d  # guarded by APPLY_MUL constexpr
        other_stride_m, other_stride_n = 0, 0

    if mask is not None:
        # mask should broadcast to output shape [..., N]
        mask_M = mask.numel() // (mask.shape[-1] if mask.numel() > 0 else 1)
        mask_features = mask.shape[-1] if mask.ndim > 0 else 1
        if mask_features == 1:
            # Broadcast mask to match output
            mask_2d = mask.reshape(mask_M, 1).expand(M, N).contiguous()
        else:
            mask_2d = mask.reshape(mask_M, mask_features).contiguous()
            if mask_M != M or mask_features != N:
                raise ValueError(
                    f"Mask shape mismatch: got {mask_M}x{mask_features}, "
                    f"expected {M}x{N}"
                )
        mask_ptr = mask_2d
        mask_stride_m, mask_stride_n = mask_2d.stride(0), mask_2d.stride(1)
    else:
        mask_ptr = x_2d  # guarded by HAS_MASK constexpr
        mask_stride_m, mask_stride_n = 0, 0

    if add_tensor is not None:
        # add_tensor should match the output shape [..., N]
        add_M = add_tensor.numel() // N
        add_tensor_2d = add_tensor.reshape(add_M, N).contiguous()
        if add_M != M:
            raise ValueError(
                f"Shape mismatch: add_tensor expects {add_M} rows but output has {M}"
            )
        add_tensor_ptr = add_tensor_2d
        add_stride_m, add_stride_n = add_tensor_2d.stride(0), add_tensor_2d.stride(1)
    else:
        add_tensor_ptr = x_2d  # guarded by HAS_ADD constexpr
        add_stride_m, add_stride_n = 0, 0

    bias_ptr = bias if bias is not None else x_2d  # guarded by HAS_BIAS constexpr

    linear_fused_kernel[grid](
        x_2d,
        weight,
        bias_ptr,
        other_ptr,
        mask_ptr,
        add_tensor_ptr,
        output_2d,
        M,
        K,
        N,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        output_2d.stride(0),
        output_2d.stride(1),
        other_stride_m,
        other_stride_n,
        mask_stride_m,
        mask_stride_n,
        add_stride_m,
        add_stride_n,
        HAS_BIAS=(bias is not None),
        APPLY_SIGMOID=apply_sigmoid,
        APPLY_MUL=(other is not None),
        HAS_MASK=(mask is not None),
        HAS_ADD=(add_tensor is not None),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    output_shape = input_shape[:-1] + (N,)
    return output_2d.reshape(output_shape)


class BaseTriangleMultiplicativeUpdate(nn.Module, ABC):
    """
    Common base class for TriangleMultiplicativeUpdate and
    FusedTriangleMultiplicativeUpdate.
    """

    @abstractmethod
    def __init__(
        self, c_z, c_hidden, _outgoing, linear_init_params=lin_init.tri_mul_init
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_g = Linear(self.c_z, self.c_z, **linear_init_params.linear_g)
        self.linear_z = Linear(self.c_hidden, self.c_z, **linear_init_params.linear_z)

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: int | None = None,
    ) -> torch.Tensor:
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        if _inplace_chunk_size is not None:
            # To be replaced by torch vmap
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                a[..., i : i + _inplace_chunk_size, :, :] = torch.einsum(
                    "...ij,...jk->...ik", a_chunk, b_chunk
                )

            p = a
        else:
            p = torch.einsum("...ij,...jk->...ik", a, b)

        return permute_final_dims(p, (1, 2, 0))

    @abstractmethod
    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        pass


class TriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements AF2 Algorithms 11 and 12 / AF3 Algorithms 12 and 13.
    """

    def __init__(
        self, c_z, c_hidden, _outgoing=True, linear_init_params=lin_init.tri_mul_init
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super().__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            _outgoing=_outgoing,
            linear_init_params=linear_init_params,
        )

        self.linear_a_p = Linear(
            self.c_z, self.c_hidden, **linear_init_params.linear_a_p
        )
        self.linear_a_g = Linear(
            self.c_z, self.c_hidden, **linear_init_params.linear_a_g
        )
        self.linear_b_p = Linear(
            self.c_z, self.c_hidden, **linear_init_params.linear_b_p
        )
        self.linear_b_g = Linear(
            self.c_z, self.c_hidden, **linear_init_params.linear_b_g
        )

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_chunk_size: int | None = None,
        with_add: bool = True,
        use_triton_triangle_kernels: bool = False,
    ):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            inplace_chunk_size:
                Size of chunks used in the main computation. Increase to trade
                memory for speed.
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
        Returns:
            A reference to the overwritten z

        More memory-efficient, inference-only version of the forward function.
        Uses in-place operations, fusion of the addition that happens after
        this module in the Evoformer, a smidge of recomputation, and
        a cache of overwritten values to lower peak memory consumption of this
        module from 5x the size of the input tensor z to 2.5x its size. Useful
        for inference on extremely long sequences.

        It works as follows. We will make reference to variables used in the
        default forward implementation below. Naively, triangle multiplication
        attention requires the manifestation of 5 tensors the size of z:
        1) z, the "square" input tensor, 2) a, the first projection of z,
        3) b, the second projection of b, 4) g, a z-sized mask, and 5) a
        z-sized tensor for intermediate computations. For large N, this is
        prohibitively expensive; for N=4000, for example, z is more than 8GB
        alone. To avoid this problem, we compute b, g, and all intermediate
        tensors in small chunks, noting that the chunks required to compute a
        chunk of the output depend only on the tensor a and corresponding
        vertical and horizontal chunks of z. This suggests an algorithm that
        loops over pairs of chunks of z: hereafter "columns" and "rows" of
        z, even though each "column" and "row" in fact contains
        inplace_chunk_size contiguous true columns and rows of z. Writing
        output chunks to a new tensor would bring total memory consumption
        down to 3x the size of z. However, more memory can be saved by writing
        output chunks directly to z in-place. WLOG, we choose to write output
        chunks vertically, overwriting the ith "column" of z at the end of
        the ith iteration of the main loop. Despite this overwriting, the
        ith column is always one column ahead of previously overwritten columns
        and can be recovered directly from z. After the first iteration,
        however, the ith row of z is always at least partially overwritten. For
        this reason, we introduce the z-cache, a tensor one-half the size of
        z. The z-cache initially contains the left half (2nd and 3rd quadrants)
        of z. For 0 < i < N/2, the missing left part of the ith row of z is
        recovered from this cache at the beginning of the ith iteration. Once i
        exceeds n/2, the cache is "reoriented" to encompass the 3rd and 4th
        quadrants of z instead. Though the 3rd quadrant of the original z is
        entirely overwritten at this point, it can be recovered from the z-cache
        itself. Thereafter, the ith row of z can be recovered in its entirety
        from the reoriented z-cache. After the final iteration, z has been
        completely overwritten and contains the triangular multiplicative
        update. If with_add is True, it instead contains the sum of z and the
        triangular multiplicative update. In either case, peak memory
        consumption is just 2.5x the size of z, disregarding memory used for
        chunks and other small variables.
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        def compute_projection_helper(pair, mask, a=True, use_triton=False):
            if a:
                linear_g = self.linear_a_g
                linear_p = self.linear_a_p
            else:
                linear_g = self.linear_b_g
                linear_p = self.linear_b_p

            if use_triton:
                pair = triton_layernorm(
                    pair,
                    self.layer_norm_in.weight,
                    self.layer_norm_in.bias,
                    self.layer_norm_in.eps,
                )
                # Fused: sigmoid(linear_g(pair)) * linear_p(pair) * mask
                p_g = triton_linear_fused(
                    pair, linear_g.weight, linear_g.bias, apply_sigmoid=True
                )
                p = triton_linear_fused(
                    pair, linear_p.weight, linear_p.bias, other=p_g, mask=mask
                )
            else:
                pair = self.layer_norm_in(pair)
                p = linear_g(pair)
                p.sigmoid_()
                p *= linear_p(pair)
                p *= mask
            p = permute_final_dims(p, (2, 0, 1))
            return p

        def compute_projection(pair, mask, a=True, chunked=True, use_triton=False):
            need_transpose = self._outgoing ^ a
            if not chunked:
                p = compute_projection_helper(pair, mask, a, use_triton)
                if need_transpose:
                    p = p.transpose(-1, -2)
            else:
                # This computation is chunked so as not to exceed our 2.5x
                # budget with a large intermediate tensor
                linear_g = self.linear_a_g if a else self.linear_b_g
                c = linear_g.weight.shape[-2]
                out_shape = pair.shape[:-3] + (c,) + pair.shape[-3:-1]
                p = pair.new_zeros(out_shape)
                for i in range(0, pair.shape[-3], inplace_chunk_size):
                    pair_chunk = compute_projection_helper(
                        pair[..., i : i + inplace_chunk_size, :, :],
                        mask[..., i : i + inplace_chunk_size, :, :],
                        a,
                        use_triton,
                    )
                    if need_transpose:
                        pair_chunk = pair_chunk.transpose(-1, -2)
                        p[..., i : i + inplace_chunk_size] = pair_chunk
                    else:
                        p[..., i : i + inplace_chunk_size, :] = pair_chunk

                    del pair_chunk

            return p

        # We start by fully manifesting a. In addition to the input, this
        # brings total memory consumption to 2x z (disregarding size of chunks)
        # [*, N, N, c]
        a = compute_projection(
            z, mask, True, chunked=True, use_triton=use_triton_triangle_kernels
        )

        if inplace_chunk_size is not None:
            n = a.shape[-1]
            half_n = n // 2 + n % 2
            row_dim = -3
            col_dim = -2
            b_chunk_dim = row_dim if self._outgoing else col_dim

            def empty_slicer(t):
                return [slice(None) for _ in t.shape]

            def slice_tensor(t, start, end, dim):
                # Slices start:end from the dim dimension of t
                s = empty_slicer(t)
                s[dim] = slice(start, end)
                return t[tuple(s)]

            def flip_z_cache_(z_cache, z):
                # "Reorient" the z_cache (see below), filling it with quadrants
                # 3---recovered from the z_cache---and 4---recovered from z---
                # of the input tensor z.
                quadrant_3 = slice_tensor(z_cache, half_n, None, row_dim)
                z_cache = z_cache.transpose(row_dim, col_dim)

                # If n is odd, we need to shrink the z_cache by one row
                z_cache = z_cache[..., : (n // 2), :, :]

                # Move the 3rd quadrant of z into the
                first_half_slicer = empty_slicer(z_cache)
                first_half_slicer[col_dim] = slice(0, half_n)
                z_cache[tuple(first_half_slicer)] = quadrant_3

                # Get the fourth quadrant of z
                quadrant_4 = slice_tensor(z, half_n, None, row_dim)
                quadrant_4 = slice_tensor(quadrant_4, half_n, None, col_dim)

                # Insert said quadrant into the rotated z-cache
                quadrant_3_slicer = empty_slicer(z_cache)
                quadrant_3_slicer[col_dim] = slice(half_n, None)

                z_cache[tuple(quadrant_3_slicer)] = quadrant_4

                return z_cache

            # Initialize the z cache to the left half of z.
            z_cache_shape = list(z.shape)
            z_cache_shape[col_dim] = half_n
            z_cache = z.new_zeros(z_cache_shape)
            z_cache_slicer = empty_slicer(z_cache)
            z_cache_slicer[col_dim] = slice(0, half_n)
            z_cache.copy_(z[tuple(z_cache_slicer)])
            z_cache_rotated = False

            # We need to reorient the z-cache at the halfway point, and we
            # don't want a single chunk to straddle that point. We contract one
            # of the chunks in the middle to address that problem.
            i_range = list(range(0, half_n, inplace_chunk_size))
            initial_offsets = [
                i_2 - i_1
                for i_1, i_2 in zip(i_range, i_range[1:] + [half_n], strict=True)
            ]
            after_half = list(range(half_n, n, inplace_chunk_size))
            after_half_offsets = [inplace_chunk_size for _ in after_half]
            combined_range_with_offsets = zip(
                i_range + after_half, initial_offsets + after_half_offsets, strict=False
            )
            for i, offset in combined_range_with_offsets:
                if not z_cache_rotated and i >= half_n:
                    z_cache = flip_z_cache_(z_cache, z)
                    z_cache_rotated = True

                z_chunk_b = slice_tensor(
                    z,
                    i,
                    i + offset,
                    b_chunk_dim,
                )
                mask_chunk = slice_tensor(
                    mask,
                    i,
                    i + offset,
                    b_chunk_dim,
                )

                z_chunk_b = z_chunk_b.clone()
                if b_chunk_dim == col_dim:
                    z_chunk_b = slice_tensor(z, i, i + offset, col_dim)
                else:  # b_chunk_dim == row_dim
                    # In this case, the b-dimension (b_chunk_dim) is partially
                    # overwritten at the end of each iteration. We need to
                    # restore the missing component from the z-cache.
                    if not z_cache_rotated:
                        z_chunk_slicer = empty_slicer(z_chunk_b)
                        z_chunk_slicer[col_dim] = slice(0, half_n)
                        z_chunk_b[tuple(z_chunk_slicer)] = slice_tensor(
                            z_cache,
                            i,
                            i + offset,
                            row_dim,
                        )
                    else:
                        z_cache_offset = i - half_n
                        z_chunk_b = slice_tensor(
                            z_cache, z_cache_offset, z_cache_offset + offset, row_dim
                        )
                b_chunk = compute_projection(
                    z_chunk_b,
                    mask_chunk,
                    a=False,
                    chunked=False,
                    use_triton=use_triton_triangle_kernels,
                )
                del z_chunk_b

                if use_triton_triangle_kernels:
                    x_chunk = torch.einsum("...ij,...jk->...ik", a, b_chunk)
                    x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
                    x_chunk = triton_layernorm(
                        x_chunk,
                        self.layer_norm_out.weight,
                        self.layer_norm_out.bias,
                        self.layer_norm_out.eps,
                    )
                    x_chunk = triton_linear(
                        x_chunk, self.linear_z.weight, self.linear_z.bias
                    )

                    # The g dimension (col_dim) is parallel to and ahead of the
                    # overwrites in z. We can extract the g chunk normally.
                    z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
                    g_input = triton_layernorm(
                        z_chunk_g,
                        self.layer_norm_in.weight,
                        self.layer_norm_in.bias,
                        self.layer_norm_in.eps,
                    )
                    del z_chunk_g

                    # Fused: sigmoid(linear(g_input)) * x_chunk [+ z_slice]
                    z_slicer = empty_slicer(z)
                    z_slicer[col_dim] = slice(i, i + offset)
                    if with_add:
                        z[tuple(z_slicer)] = triton_linear_fused(
                            g_input,
                            self.linear_g.weight,
                            self.linear_g.bias,
                            other=x_chunk,
                            add_tensor=z[tuple(z_slicer)],
                            apply_sigmoid=True,
                        )
                    else:
                        # Fused: sigmoid(linear(g_input)) * x_chunk -> z[slice]
                        z[tuple(z_slicer)] = triton_linear_fused(
                            g_input,
                            self.linear_g.weight,
                            self.linear_g.bias,
                            other=x_chunk,
                            apply_sigmoid=True,
                        )
                else:
                    x_chunk = torch.einsum("...ij,...jk->...ik", a, b_chunk)
                    x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
                    x_chunk = self.layer_norm_out(x_chunk)
                    x_chunk = self.linear_z(x_chunk)

                    # The g dimension (col_dim) is parallel to and ahead of the
                    # overwrites in z. We can extract the g chunk normally.
                    z_chunk_g = slice_tensor(z, i, i + offset, col_dim)
                    g_chunk = self.linear_g(self.layer_norm_in(z_chunk_g))
                    g_chunk.sigmoid_()
                    del z_chunk_g

                    x_chunk *= g_chunk

                    # Write the columns into z in-place
                    z_slicer = empty_slicer(z)
                    z_slicer[col_dim] = slice(i, i + offset)
                    if with_add:
                        z[tuple(z_slicer)] += x_chunk
                    else:
                        z[tuple(z_slicer)] = x_chunk

        else:
            b = compute_projection(
                z, mask, False, False, use_triton=use_triton_triangle_kernels
            )
            if use_triton_triangle_kernels:
                x = torch.einsum("...ij,...jk->...ik", a, b)
                x = triton_layernorm(
                    x,
                    self.layer_norm_out.weight,
                    self.layer_norm_out.bias,
                    self.layer_norm_out.eps,
                )
                x = triton_linear(x, self.linear_z.weight, self.linear_z.bias)
                # Fused: sigmoid(linear(z)) * x [+ z]
                if with_add:
                    triton_linear_fused(
                        z,
                        self.linear_g.weight,
                        self.linear_g.bias,
                        other=x,
                        add_tensor=z,
                        apply_sigmoid=True,
                    )
                else:
                    z[:] = triton_linear_fused(
                        z,
                        self.linear_g.weight,
                        self.linear_g.bias,
                        other=x,
                        apply_sigmoid=True,
                    )
            else:
                x = torch.einsum("...ij,...jk->...ik", a, b)
                x = self.layer_norm_out(x)
                x = self.linear_z(x)
                g = self.linear_g(z)
                g.sigmoid_()
                x *= g
                if with_add:
                    z += x
                else:
                    z = x

        return z

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: int | None = 256,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        ## NOTE: valid for inplace safe and use_cueq_triangle_kernels to be enabled
        ## inplace safe is used across the codebase and so should not
        ## be disabled. So if use_cueq_triangle_kernels is True, it will always
        ## supersede inplace_safe
        if use_cueq_triangle_kernels:
            ## VS: The cuequivariance kernel is based on the boltz implementation
            ## of triangle multiplicative update, which fuses the linear_*_p
            ## projections into a single layer (similarly for linear_*_g).
            ## this why we need to concat the projection layers here
            x = _cueq_triangle_mult(
                z=z,
                g_in_weight=torch.cat(
                    [
                        self.linear_a_g.weight,
                        self.linear_b_g.weight,
                    ]
                ),
                p_in_weight=torch.cat(
                    [
                        self.linear_a_p.weight,
                        self.linear_b_p.weight,
                    ]
                ),
                _outgoing=self._outgoing,
                mask=mask,
                norm_in_weight=self.layer_norm_in.weight,
                norm_in_bias=self.layer_norm_in.bias,
                norm_out_weight=self.layer_norm_out.weight,
                norm_out_bias=self.layer_norm_out.bias,
                p_out_weight=self.linear_z.weight,
                g_out_weight=self.linear_g.weight,
            )
            return x

        if inplace_safe:
            x = self._inference_forward(
                z,
                mask,
                inplace_chunk_size=_inplace_chunk_size,
                with_add=_add_with_inplace,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
            )
            return x

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        a = mask  # (1,s, s, 1)
        a = a * self.sigmoid(self.linear_a_g(z))
        a = a * self.linear_a_p(z)
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))
        b = b * self.linear_b_p(z)

        x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements AF2 Algorithm 11 / AF3 Algorithm 12.
    """

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements AF2 Algorithm 12 / AF3 Algorithm 13.
    """

    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


class FusedTriangleMultiplicativeUpdate(BaseTriangleMultiplicativeUpdate):
    """
    Implements AF2-Multimer version of AF2 Algorithm 11 and 12.
    """

    def __init__(
        self,
        c_z,
        c_hidden,
        _outgoing=True,
        linear_init_params=lin_init.fused_tri_mul_init,
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super().__init__(
            c_z=c_z,
            c_hidden=c_hidden,
            _outgoing=_outgoing,
            linear_init_params=linear_init_params,
        )

        self.linear_ab_p = Linear(
            self.c_z, self.c_hidden * 2, **linear_init_params.linear_ab_p
        )
        self.linear_ab_g = Linear(
            self.c_z, self.c_hidden * 2, **linear_init_params.linear_ab_g
        )

    def _triton_inference_forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        _inplace_chunk_size: int | None = None,
        with_add: bool = True,
    ):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
        Returns:
            A reference to the overwritten z
        """

        z_norm_in = triton_layernorm(
            z,
            self.layer_norm_in.weight,
            self.layer_norm_in.bias,
            self.layer_norm_in.eps,
        )

        p_g = triton_linear_fused(
            z_norm_in,
            self.linear_ab_g.weight,
            self.linear_ab_g.bias,
            apply_sigmoid=True,
        )
        p = triton_linear_fused(
            z_norm_in,
            self.linear_ab_p.weight,
            self.linear_ab_p.bias,
            other=p_g,
            mask=mask,
        )
        a = p[..., : self.c_hidden]
        b = p[..., self.c_hidden :]

        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        if _inplace_chunk_size is not None:
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                a[..., i : i + _inplace_chunk_size, :, :] = torch.einsum(
                    "...ij,...jk->...ik", a_chunk, b_chunk
                )

            x = a
        else:
            x = torch.einsum("...ij,...jk->...ik", a, b)

        x = permute_final_dims(x, (1, 2, 0))
        x = triton_layernorm(
            x,
            self.layer_norm_out.weight,
            self.layer_norm_out.bias,
            self.layer_norm_out.eps,
        )
        x = triton_linear(x, self.linear_z.weight, self.linear_z.bias)
        # Fused: sigmoid(linear(z_norm_in)) * x [+ z]
        if with_add:
            triton_linear_fused(
                z_norm_in,
                self.linear_g.weight,
                self.linear_g.bias,
                other=x,
                add_tensor=z,
                apply_sigmoid=True,
            )
        else:
            z[:] = triton_linear_fused(
                z_norm_in,
                self.linear_g.weight,
                self.linear_g.bias,
                other=x,
                apply_sigmoid=True,
            )

        return z

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        _inplace_chunk_size: int | None = None,
        with_add: bool = True,
        use_triton_triangle_kernels: bool = False,
    ):
        """
        Args:
            z:
                A [*, N, N, C_z] pair representation
            mask:
                A [*, N, N] pair mask
            with_add:
                If True, z is overwritten with (z + update). Otherwise, it is
                overwritten with (update).
            use_triton_triangle_kernels:
                If True, uses Triton kernels.
        Returns:
            A reference to the overwritten z
        """
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        if use_triton_triangle_kernels:
            return self._triton_inference_forward(
                z, mask, _inplace_chunk_size, with_add
            )

        def compute_projection_helper(pair, mask):
            p = self.linear_ab_g(pair)
            p.sigmoid_()
            p *= self.linear_ab_p(pair)
            p *= mask

            return p

        def compute_projection(pair, mask):
            p = compute_projection_helper(pair, mask)
            left = p[..., : self.c_hidden]
            right = p[..., self.c_hidden :]

            return left, right

        z_norm_in = self.layer_norm_in(z)
        a, b = compute_projection(z_norm_in, mask)
        x = self._combine_projections(a, b, _inplace_chunk_size=_inplace_chunk_size)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.linear_g(z_norm_in)
        g.sigmoid_()
        x *= g
        if with_add:
            z += x
        else:
            z = x

        return z

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        inplace_safe: bool = False,
        use_cueq_triangle_kernels: bool = False,
        use_triton_triangle_kernels: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: int | None = 256,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        # Supersede inplace_safe conditional if cueq kernel is used
        if use_cueq_triangle_kernels:
            x = _cueq_triangle_mult(
                z=z,
                g_in_weight=self.linear_ab_g.weight,
                p_in_weight=self.linear_ab_p.weight,
                _outgoing=self._outgoing,
                mask=mask,
                norm_in_weight=self.layer_norm_in.weight,
                norm_in_bias=self.layer_norm_in.bias,
                norm_out_weight=self.layer_norm_out.weight,
                norm_out_bias=self.layer_norm_out.bias,
                p_out_weight=self.linear_z.weight,
                g_out_weight=self.linear_g.weight,
            )
            return x

        if inplace_safe:
            x = self._inference_forward(
                z,
                mask,
                _inplace_chunk_size=_inplace_chunk_size,
                with_add=_add_with_inplace,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
            )
            return x

        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)
        ab = mask
        ab = ab * self.sigmoid(self.linear_ab_g(z))
        ab = ab * self.linear_ab_p(z)

        a = ab[..., : self.c_hidden]
        b = ab[..., self.c_hidden :]

        x = self._combine_projections(a, b)

        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class FusedTriangleMultiplicationOutgoing(FusedTriangleMultiplicativeUpdate):
    """
    Implements AF2-Multimer version of AF2 Algorithm 11.
    Not compatible with AF3
    """

    __init__ = partialmethod(FusedTriangleMultiplicativeUpdate.__init__, _outgoing=True)


class FusedTriangleMultiplicationIncoming(FusedTriangleMultiplicativeUpdate):
    """
    Implements AF2-Multimer version of AF2 Algorithm 12.
    Not compatible with AF3
    """

    __init__ = partialmethod(
        FusedTriangleMultiplicativeUpdate.__init__, _outgoing=False
    )


def _cueq_triangle_mult(
    z: torch.Tensor,
    g_in_weight: torch.Tensor,
    p_in_weight: torch.Tensor,
    _outgoing: bool,
    mask: torch.Tensor | None,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    norm_out_weight: torch.Tensor,
    norm_out_bias: torch.Tensor,
    p_out_weight: torch.Tensor,
    g_out_weight: torch.Tensor,
) -> torch.Tensor:
    ##VS: similar issue here as to the cueq triangle attention
    ## kernel, we need to reshape the input so that batch and
    ## n_tmpl are combined into a single dimension.

    ## only hidden dimension multiple of 32 is supported for now
    if z.shape[-1] % 32 != 0:
        raise ValueError(
            "CUEQ triangle multiplicative update only supports "
            "channel dimension multiple of 32, got: "
            f"{z.shape[-1]}"
        )

    is_batched_input = False
    if len(z.shape) > 4:
        assert len(z.shape) == 5, (
            "CUEQ triangle multiplicative update only supports "
            f"max 5 input dimensions, got: {len(z.shape)}"
        )
        is_batched_input = True
        batch, n_tmpl, n_res, _, c_in = z.shape
        z = z.view(batch * n_tmpl, *z.shape[2:])
        mask = mask.view(batch * n_tmpl, *mask.shape[2:]) if mask is not None else None

    x = triangle_multiplicative_update(
        z,
        direction="outgoing" if _outgoing else "incoming",
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        g_in_weight=g_in_weight,
        p_in_weight=p_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=1e-5,
    )
    if is_batched_input:
        x = x.view(batch, n_tmpl, *x.shape[1:])
    return x
