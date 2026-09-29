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

import importlib

import pytest
import torch

from openfold3.core.kernels.cueq_utils import (
    is_cuequivariance_available,
    is_cuequivariance_installed,
)


def skip_if_rocm():
    is_rocm = torch.cuda.is_available() and torch.version.hip is not None
    return pytest.mark.skipif(is_rocm, reason="Not supported on ROCm/HIP")


def skip_unless_ds4s_installed():
    deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
    ds4s_is_installed = (
        deepspeed_is_installed
        and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
    )
    is_rocm = torch.cuda.is_available() and torch.version.hip is not None
    return pytest.mark.skipif(
        not (ds4s_is_installed and not is_rocm),
        reason="Requires DeepSpeed with version ≥ 0.10.4 (not supported on ROCm/HIP)",
    )


def skip_unless_cueq_installed():
    if not is_cuequivariance_installed():
        reason = "Requires cuequivariance to be installed"
    elif not torch.cuda.is_available():
        reason = "Requires CUDA (cuequivariance is installed but no GPU available)"
    else:
        reason = "cuequivariance not available"
    return pytest.mark.skipif(not is_cuequivariance_available(), reason=reason)


def skip_unless_triton_installed():
    triton_is_installed = importlib.util.find_spec("triton") is not None
    return pytest.mark.skipif(not triton_is_installed, reason="Requires Triton")


def skip_unless_cuda_available():
    return pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")


def _assert_abs_diff_small_base(compare_func, expected, actual, eps):
    # Helper function for comparing absolute differences of two torch tensors.
    abs_diff = torch.abs(expected - actual)
    err = compare_func(abs_diff)
    zero_tensor = torch.tensor(0, device=err.device, dtype=err.dtype)
    rtol = 1.6e-2 if err.dtype == torch.bfloat16 else 1.3e-6
    torch.testing.assert_close(err, zero_tensor, atol=eps, rtol=rtol)


def assert_max_abs_diff_small(expected, actual, eps):
    _assert_abs_diff_small_base(torch.max, expected, actual, eps)


def assert_mean_abs_diff_small(expected, actual, eps):
    _assert_abs_diff_small_base(torch.mean, expected, actual, eps)


#: fp32 unit roundoff, 2**-24.
FP32_EPS = 2.0**-24


def summation_order_atol(reference: torch.Tensor, n_terms: int, scale=None) -> float:
    """Bound on the difference between two fp32 summation orders.

    Summing ``n_terms`` values in any order differs from the exact sum by at
    most ``gamma(n-1) * sum|x|`` (Higham, *Accuracy and Stability of Numerical
    Algorithms*, eq. 3.4), where ``gamma(n) = n*u / (1 - n*u)``. Two different
    orders are independent, so they differ from each other by at most twice
    that. For a mean of ``n_terms`` values the count divides out, leaving
    ``2 * gamma(n-1) * mean|x|``, and the final division contributes at most
    ``2*u * |mean x|``.

    Both terms need ``mean|x|`` -- the magnitude of the *summed terms*. That is
    not recoverable from the reduced output, so by default this uses
    ``max|reference|`` as a proxy. That proxy is tight when the terms are of
    comparable magnitude to their mean, which holds for the atom-to-token
    reduction here (measured: the observed difference sits 10-30x below the
    resulting bound). It is *not* valid under heavy cancellation, where terms of
    large magnitude average to something small -- there ``max|reference|``
    under-states ``mean|x|`` and the bound is too tight. Pass ``scale``
    explicitly in that case.

    Args:
        reference: Aggregated output, used for its magnitude scale.
        n_terms: Number of summed terms per output element.
        scale: Optional ``max`` over output elements of ``mean|x|`` for the
            summed terms. Supply it whenever the terms can cancel.
    """
    if scale is None:
        scale = reference.abs().max().item()
    n = max(n_terms - 1, 0)
    gamma = n * FP32_EPS / (1.0 - n * FP32_EPS)
    return 2.0 * float(scale) * (gamma + FP32_EPS)


def assert_summation_order_close(actual, reference, n_terms, msg="", scale=None):
    """Assert two fp32 reductions of the same values differ only by rounding order.

    ``torch.testing.assert_close`` cannot state this bound. It compares element
    pairs against ``atol + rtol * |expected|``, which scales with the *output*
    magnitude, whereas rounding-order error scales with the magnitude of the
    *summed terms* and with the number of terms. A flat ``atol`` has to be
    guessed and is wrong somewhere; this is derived from the reduction shape.

    Args:
        actual: Reduction under test.
        reference: Reduction to compare against.
        n_terms: Number of summed terms per output element.
        msg: Prefix for the failure message.
        scale: Optional term-magnitude scale; see ``summation_order_atol``.
    """
    bound = summation_order_atol(reference, n_terms, scale=scale)
    diff = (actual.float() - reference.float()).abs().max().item()
    if diff > bound:
        raise AssertionError(
            f"{msg + ': ' if msg else ''}summation-order difference {diff:.3e} "
            f"exceeds the fp32 bound {bound:.3e} for {n_terms} terms per token"
        )
