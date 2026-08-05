# Copyright 2026 AlQuraishi Laboratory
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

# by Liang Hong <lhong22@cse.cuhk.edu.hk>: Triton LayerNorm-to-linear
# kernels and inference launchers for contiguous and strided pair tensors.

"""Triton kernel that fuses LayerNorm followed by a Linear layer.

Targets pair-scale LN -> Linear chains in DiffusionConditioning,
NoisyPositionEmbedder, and the OpenFold3 outer trunk. The fused kernel never
materializes the [..., c_in] normalized intermediate, eliminating
~sizeof(input.dtype)*input.numel() of allocator activity per call.

Forward only in this iteration; backward is autograd-decomposed against
the saved input + LN stats.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE


if _TRITON_AVAILABLE:

    # Large tiles win on high-SMEM GPUs; (16, 32) / (16, 16) keep the
    # ieee / large-c_in specializations under T4's 64 KiB opt-in limit.
    # num_stages stays 1: the LN reduction is one-pass in registers, so
    # pipelining only doubles SMEM.
    _LN_LINEAR_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}, num_warps=2, num_stages=1),
    ]

    @triton.autotune(
        configs=_LN_LINEAR_AUTOTUNE_CONFIGS,
        # Retune per (c_out, c_in, tf32): ieee / large-K specializations
        # prune big tiles via OutOfResources; tf32 keeps larger winners.
        key=["N", "K", "ALLOW_TF32"],
        restore_value=["Y_ptr", "Mean_ptr", "Rstd_ptr"],
    )
    @triton.jit(
        do_not_specialize=[
            "stride_x_m",
            "stride_x_k",
            "stride_w_n",
            "stride_w_k",
            "stride_y_m",
            "stride_y_n",
            "M",
            "eps",
        ],
        do_not_specialize_on_alignment=[
            "X_ptr",
            "W_ptr",
            "Y_ptr",
            "Gamma_ptr",
            "Beta_ptr",
            "Bias_ptr",
            "Mean_ptr",
            "Rstd_ptr",
        ],
    )
    def _fused_ln_linear_fwd_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        Gamma_ptr,
        Beta_ptr,
        Bias_ptr,
        Mean_ptr,
        Rstd_ptr,
        stride_x_m,
        stride_x_k,
        stride_w_n,
        stride_w_k,
        stride_y_m,
        stride_y_n,
        M,
        N,
        K,
        eps,
        HAS_LN_BIAS: tl.constexpr,
        HAS_LIN_BIAS: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused LayerNorm(x) @ W^T + bias.

        X       : [M, K]   input (row-major, stride_x_m, stride_x_k)
        W       : [N, K]   Linear.weight (row-major; nn.Linear stores [out, in])
        Y       : [M, N]   output (row-major)
        Gamma   : [K]      LN scale (always present in our use; cast to fp32)
        Beta    : [K]      LN offset (optional)
        Bias    : [N]      Linear bias (optional)
        Mean    : [M]      saved fp32 row mean (for backward)
        Rstd    : [M]      saved fp32 row rstd (for backward)

        Each program computes a [BLOCK_M, BLOCK_N] output tile. K is
        loaded once into registers (BLOCK_K = next_pow2(K) covers the
        whole reduction in a single pass).
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_64 = offs_m.to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        m_mask = offs_m < M
        n_mask = offs_n < N
        k_mask = offs_k < K

        # Load X tile: [BLOCK_M, BLOCK_K], cast to fp32 in registers.
        x_ptrs = X_ptr + offs_m_64[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
        x = tl.load(
            x_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Per-row mean and rstd (LayerNorm).
        # Mean uses K (true reduction extent), not BLOCK_K (padded).
        x_sum = tl.sum(x, axis=1)  # [BLOCK_M]
        mean = x_sum / K
        x_centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        x_sq_sum = tl.sum(x_centered * x_centered, axis=1)
        var = x_sq_sum / K
        rstd = 1.0 / tl.sqrt(var + eps)

        # Save mean / rstd for backward (only program along N=0 writes them).
        if pid_n == 0:
            tl.store(Mean_ptr + offs_m, mean, mask=m_mask)
            tl.store(Rstd_ptr + offs_m, rstd, mask=m_mask)

        # Apply scale + offset.
        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_hat = x_centered * rstd[:, None] * gamma[None, :]
        if HAS_LN_BIAS:
            beta = tl.load(Beta_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x_hat + tl.where(k_mask[None, :], beta[None, :], 0.0)

        # Load W tile: [BLOCK_N, BLOCK_K] then transpose for tl.dot
        # tl.dot expects (M, K) @ (K, N) so we load W as [K, N] via swap.
        w_ptrs = W_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k
        w = tl.load(
            w_ptrs,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # x_hat: [BLOCK_M, BLOCK_K]; w: [BLOCK_N, BLOCK_K]; we want x_hat @ w.T
        # tl.dot(a, b) requires b: [K, N], so transpose w in registers.
        w_t = tl.trans(w)  # [BLOCK_K, BLOCK_N]
        if ALLOW_TF32:
            acc = tl.dot(x_hat, w_t, out_dtype=tl.float32, allow_tf32=True)
        else:
            acc = tl.dot(x_hat, w_t, out_dtype=tl.float32, input_precision="ieee")

        if HAS_LIN_BIAS:
            bias = tl.load(Bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        # Cast to output dtype (matches X dtype).
        y_ptrs = Y_ptr + offs_m_64[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
        tl.store(
            y_ptrs,
            acc.to(Y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    @triton.autotune(
        configs=_LN_LINEAR_AUTOTUNE_CONFIGS,
        key=["N", "K", "ALLOW_TF32"],
        restore_value=["Y_ptr"],
    )
    @triton.jit(
        do_not_specialize=[
            "I_dim",
            "J_dim",
            "stride_x_i",
            "stride_x_j",
            "stride_x_k",
            "stride_w_n",
            "stride_w_k",
            "stride_y_m",
            "stride_y_n",
            "eps",
        ],
        do_not_specialize_on_alignment=[
            "X_ptr",
            "W_ptr",
            "Y_ptr",
            "Gamma_ptr",
            "Beta_ptr",
            "Bias_ptr",
        ],
    )
    def _pair_ln_linear_fwd_kernel(
        X_ptr,
        W_ptr,
        Y_ptr,
        Gamma_ptr,
        Beta_ptr,
        Bias_ptr,
        I_dim,
        J_dim,
        stride_x_i,
        stride_x_j,
        stride_x_k,
        stride_w_n,
        stride_w_k,
        stride_y_m,
        stride_y_n,
        N,
        K,
        eps,
        HAS_LN_BIAS: tl.constexpr,
        HAS_LIN_BIAS: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """LayerNorm + Linear over logical [1, I, J, K] rows.

        Unlike ``_fused_ln_linear_fwd_kernel``, the row address is decoded as
        ``i = row // J`` and ``j = row % J``.  This lets triangle attention
        project a transposed pair view without first materializing a contiguous
        1U copy.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_64 = offs_m.to(tl.int64)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        M = I_dim * J_dim
        m_mask = offs_m < M
        n_mask = offs_n < N
        k_mask = offs_k < K

        offs_i = offs_m // J_dim
        offs_j = offs_m - offs_i * J_dim
        x_base = offs_i.to(tl.int64) * stride_x_i + offs_j.to(tl.int64) * stride_x_j
        x = tl.load(
            X_ptr + x_base[:, None] + offs_k[None, :] * stride_x_k,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        mean = tl.sum(x, axis=1) / K
        x_centered = tl.where(k_mask[None, :], x - mean[:, None], 0.0)
        var = tl.sum(x_centered * x_centered, axis=1) / K
        rstd = 1.0 / tl.sqrt(var + eps)

        gamma = tl.load(Gamma_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        x_hat = x_centered * rstd[:, None] * gamma[None, :]
        if HAS_LN_BIAS:
            beta = tl.load(Beta_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
            x_hat = x_hat + tl.where(k_mask[None, :], beta[None, :], 0.0)

        w = tl.load(
            W_ptr + offs_n[:, None] * stride_w_n + offs_k[None, :] * stride_w_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if ALLOW_TF32:
            acc = tl.dot(x_hat, tl.trans(w), out_dtype=tl.float32, allow_tf32=True)
        else:
            acc = tl.dot(
                x_hat, tl.trans(w), out_dtype=tl.float32, input_precision="ieee"
            )

        if HAS_LIN_BIAS:
            bias = tl.load(Bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        tl.store(
            Y_ptr + offs_m_64[:, None] * stride_y_m + offs_n[None, :] * stride_y_n,
            acc.to(Y_ptr.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    def _next_power_of_two(n: int) -> int:
        if n <= 1:
            return 1
        return 1 << (n - 1).bit_length()

    def _fused_ln_linear_triton_fwd(
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor | None,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        output_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Launch the Triton forward kernel.

        Returns (y, mean, rstd) where mean/rstd are fp32 [M] tensors saved
        for backward.
        """
        # Flatten leading dims to (M, K).
        orig_shape = x.shape
        c_in = gamma.shape[0]
        x_2d = x.contiguous().view(-1, c_in)
        M = x_2d.shape[0]
        K = c_in
        c_out = weight.shape[0]
        N = c_out

        # Output and saved-stats tensors.
        y_dtype = x.dtype if output_dtype is None else output_dtype
        y = torch.empty((M, N), dtype=y_dtype, device=x.device)
        mean = torch.empty((M,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M,), dtype=torch.float32, device=x.device)

        # BLOCK_K must cover the full reduction (one-pass LN). Tile sizes
        # (BLOCK_M / BLOCK_N / warps) come from @triton.autotune.
        BLOCK_K = _next_power_of_two(K)
        # Cap BLOCK_K to avoid register pressure on large c_in (833 -> 1024 ok).
        # If c_in ever exceeds 4096 we'd need a K-loop; assert instead so we notice.
        assert BLOCK_K <= 4096, f"BLOCK_K={BLOCK_K} too large for one-pass LN"

        # tl.dot requires its first operand's K (BLOCK_K) and second's K
        # to match. Both are BLOCK_K here. Triton requires BLOCK_K >= 16
        # for tl.dot.
        BLOCK_K_DOT = max(BLOCK_K, 16)
        allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        # Bias / beta nullity: pass dummy 1-elem tensors when absent so
        # the kernel can take a stable signature; HAS_* compile-time
        # consts gate the actual loads.
        beta_ptr = beta if beta is not None else x.new_zeros(1)
        bias_ptr = bias if bias is not None else x.new_zeros(1)

        # gamma / beta cast to fp32 happens inside the kernel via .to(tl.float32).
        # Caller is responsible for passing matching-rank / contiguous inputs.
        _fused_ln_linear_fwd_kernel[grid](
            x_2d,
            weight,
            y,
            gamma,
            beta_ptr,
            bias_ptr,
            mean,
            rstd,
            x_2d.stride(0),
            x_2d.stride(1),
            weight.stride(0),
            weight.stride(1),
            y.stride(0),
            y.stride(1),
            M,
            N,
            K,
            eps,
            HAS_LN_BIAS=beta is not None,
            HAS_LIN_BIAS=bias is not None,
            ALLOW_TF32=allow_tf32,
            BLOCK_K=BLOCK_K_DOT,
        )

        y = y.view(*orig_shape[:-1], N)
        return y, mean, rstd


class _FusedLNLinearFn(torch.autograd.Function):
    """Autograd-decomposed backward over the fused forward kernel.

    Forward: Triton fused kernel, no [..., c_in] intermediate.
    Backward: rebuild x_hat from saved (x, mean, rstd, gamma, beta),
    apply standard LayerNorm + Linear gradient formulas. This temporarily
    materializes x_hat during backward only — forward peak (the inference
    bottleneck) is unchanged.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor | None,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
    ) -> torch.Tensor:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("Triton is not available")
        if x.device.type != "cuda":
            raise RuntimeError("FusedLNLinear Triton path requires CUDA inputs")

        # Cast affine params to input dtype to avoid Triton mixed-dtype
        # issues; downstream upcasts to fp32 inside the kernel.
        gamma_d = gamma.to(dtype=x.dtype) if gamma.dtype != x.dtype else gamma
        beta_d = (
            beta.to(dtype=x.dtype)
            if (beta is not None and beta.dtype != x.dtype)
            else beta
        )
        weight_d = weight.to(dtype=x.dtype) if weight.dtype != x.dtype else weight
        bias_d = (
            bias.to(dtype=x.dtype)
            if (bias is not None and bias.dtype != x.dtype)
            else bias
        )

        y, mean, rstd = _fused_ln_linear_triton_fwd(
            x.contiguous(), gamma_d, beta_d, weight_d, bias_d, eps
        )

        ctx.save_for_backward(x, gamma, beta, weight, bias, mean, rstd)
        ctx.eps = eps
        ctx.has_ln_bias = beta is not None
        ctx.has_lin_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, gamma, beta, weight, bias, mean, rstd = ctx.saved_tensors
        c_in = gamma.shape[0]
        x_2d = x.contiguous().view(-1, c_in)
        dy_2d = dy.contiguous().view(-1, weight.shape[0])

        # Cast to fp32 for the math; cast back at the end.
        x_f = x_2d.to(torch.float32)
        gamma_f = gamma.to(torch.float32)
        dy_f = dy_2d.to(torch.float32)
        weight_f = weight.to(torch.float32)

        x_hat = (x_f - mean[:, None]) * rstd[:, None]  # [M, K]

        # Linear backward
        dx_hat_after_gamma = dy_f @ weight_f  # [M, K]
        d_weight = dy_f.t() @ (x_hat * gamma_f)  # [N, K]
        d_bias = dy_f.sum(dim=0) if ctx.has_lin_bias else None

        # LayerNorm backward
        # y_hat = x_hat (LN output before scale/offset is x_hat;
        #          but the produced "x_hat * gamma + beta" is what feeds Linear)
        # We have dx_hat_post_affine = dx_hat_after_gamma. Decompose:
        #   d_gamma = sum_m (dx_hat_post_affine * x_hat)
        #   d_beta  = sum_m (dx_hat_post_affine)
        d_gamma = (dx_hat_after_gamma * x_hat).sum(dim=0)
        d_beta = dx_hat_after_gamma.sum(dim=0) if ctx.has_ln_bias else None

        # dx via standard LN formula
        dx_hat = dx_hat_after_gamma * gamma_f  # [M, K]
        mean_dxhat = dx_hat.mean(dim=1, keepdim=True)
        mean_dxhat_xhat = (dx_hat * x_hat).mean(dim=1, keepdim=True)
        dx = rstd[:, None] * (dx_hat - mean_dxhat - x_hat * mean_dxhat_xhat)
        dx = dx.to(x.dtype).view_as(x)

        d_gamma = d_gamma.to(gamma.dtype)
        if d_beta is not None:
            d_beta = d_beta.to(beta.dtype)
        d_weight = d_weight.to(weight.dtype)
        if d_bias is not None:
            d_bias = d_bias.to(bias.dtype)

        return dx, d_gamma, d_beta, d_weight, d_bias, None


# Above this c_in the one-pass kernel cannot keep a useful TILE in SMEM
# even with the smallest @triton.autotune configs. Larger c_in falls back
# to F.layer_norm + F.linear. All openfold3 pair-tensor LN sites have
# c_in <= 384.
_MAX_FUSED_C_IN = 512

# Below this row count, the autograd-Function dispatch + Triton launch
# overhead exceeds the cuBLAS+F.layer_norm cost. Fall back to the
# reference path. Empirically tuned on RTX 4090 (SM89): cuBLAS handles
# tiny matmuls better than a one-warp Triton kernel.
_MIN_FUSED_M = 4096


def fused_ln_linear(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Public entry point: F.linear(F.layer_norm(x)) but fused.

    Falls back to the reference path on CPU, when Triton is missing,
    when c_in exceeds the kernel's SMEM budget, or when the row count
    is too small to amortize launch overhead.
    """
    if (
        _TRITON_AVAILABLE
        and x.is_cuda
        and gamma.shape[0] <= _MAX_FUSED_C_IN
        and x.numel() // gamma.shape[0] >= _MIN_FUSED_M
    ):
        return _FusedLNLinearFn.apply(x, gamma, beta, weight, bias, eps)
    x_norm = F.layer_norm(x, (gamma.shape[0],), gamma, beta, eps)
    return F.linear(x_norm, weight, bias)


def fused_ln_linear_inference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Inference-only fused LN→Linear with optional output dtype override."""
    if torch.is_grad_enabled():
        raise RuntimeError("fused_ln_linear_inference is inference-only")

    if (
        _TRITON_AVAILABLE
        and x.is_cuda
        and gamma.shape[0] <= _MAX_FUSED_C_IN
        and x.numel() // gamma.shape[0] >= _MIN_FUSED_M
    ):
        gamma_d = gamma.to(dtype=x.dtype) if gamma.dtype != x.dtype else gamma
        beta_d = (
            beta.to(dtype=x.dtype)
            if (beta is not None and beta.dtype != x.dtype)
            else beta
        )
        weight_d = weight.to(dtype=x.dtype) if weight.dtype != x.dtype else weight
        bias_d = (
            bias.to(dtype=x.dtype)
            if (bias is not None and bias.dtype != x.dtype)
            else bias
        )
        y, _, _ = _fused_ln_linear_triton_fwd(
            x.contiguous(),
            gamma_d,
            beta_d,
            weight_d,
            bias_d,
            eps,
            output_dtype=output_dtype,
        )
        return y

    x_norm = F.layer_norm(x, (gamma.shape[0],), gamma, beta, eps)
    y = F.linear(x_norm, weight, bias)
    return y if output_dtype is None else y.to(output_dtype)


def pair_ln_linear_inference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Inference-only LN→Linear for logical ``[1, I, J, C]`` pair tensors.

    The input may be a transposed pair view.  The Triton kernel indexes the
    logical ``(i, j)`` row directly, avoiding the hidden contiguous 1U clone
    that ``reshape``/``contiguous`` would create for ending triangle attention.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("pair_ln_linear_inference is inference-only")
    if x.dim() != 4 or x.shape[0] != 1:
        raise ValueError(f"expected [1, I, J, C] pair tensor, got {tuple(x.shape)}")
    if (
        not _TRITON_AVAILABLE
        or not x.is_cuda
        or gamma.shape[0] > _MAX_FUSED_C_IN
        or x.numel() // gamma.shape[0] < _MIN_FUSED_M
    ):
        x_norm = F.layer_norm(x, (gamma.shape[0],), gamma, beta, eps)
        return F.linear(x_norm, weight, bias).reshape(-1, weight.shape[0])

    I_dim, J_dim, K = x.shape[1], x.shape[2], gamma.shape[0]
    N = weight.shape[0]
    gamma_d = gamma.to(dtype=x.dtype) if gamma.dtype != x.dtype else gamma
    beta_d = (
        beta.to(dtype=x.dtype)
        if (beta is not None and beta.dtype != x.dtype)
        else beta
    )
    weight_d = weight.to(dtype=x.dtype) if weight.dtype != x.dtype else weight
    bias_d = (
        bias.to(dtype=x.dtype)
        if (bias is not None and bias.dtype != x.dtype)
        else bias
    )
    y = torch.empty((I_dim * J_dim, N), dtype=x.dtype, device=x.device)

    BLOCK_K = max(_next_power_of_two(K), 16)
    assert BLOCK_K <= 4096, f"BLOCK_K={BLOCK_K} too large for one-pass LN"
    allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)

    beta_ptr = beta_d if beta_d is not None else x.new_zeros(1)
    bias_ptr = bias_d if bias_d is not None else x.new_zeros(1)
    # BLOCK_M / BLOCK_N / warps come from @triton.autotune.
    grid = lambda meta: (
        triton.cdiv(I_dim * J_dim, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _pair_ln_linear_fwd_kernel[grid](
        x,
        weight_d,
        y,
        gamma_d,
        beta_ptr,
        bias_ptr,
        I_dim,
        J_dim,
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight_d.stride(0),
        weight_d.stride(1),
        y.stride(0),
        y.stride(1),
        N,
        K,
        eps,
        HAS_LN_BIAS=beta is not None,
        HAS_LIN_BIAS=bias is not None,
        ALLOW_TF32=allow_tf32,
        BLOCK_K=BLOCK_K,
    )
    return y
