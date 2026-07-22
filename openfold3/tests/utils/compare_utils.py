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

import functools
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


def _no_skip(reason: str):
    """Return a mark that never skips; ``reason`` records why the guard passed.

    pytest ignores ``reason`` when the condition is false, so this is purely
    documentation for the reader.
    """
    return pytest.mark.skipif(False, reason=reason)


@functools.lru_cache(maxsize=1)
def _ds4s_build_blocker() -> str | None:
    """Return why the evoformer_attn op cannot be JIT built, or None if it can.

    Importing DeepSpeed is not enough: the op is compiled on first use, which
    needs CUTLASS headers (via ``$CUTLASS_PATH``), a CUDA toolchain matching
    torch, and ninja.  Without them ``jit_load`` raises at test runtime rather
    than the test being skipped.  Probing shells out to nvcc, so the result is
    cached — this runs at collection time for every decorated test.
    """
    try:
        from deepspeed.ops.op_builder import EvoformerAttnBuilder

        if not EvoformerAttnBuilder().is_compatible(verbose=False):
            return (
                "DeepSpeed cannot build the evoformer_attn op: set $CUTLASS_PATH "
                "and ensure a CUDA toolchain matching torch is installed"
            )

        from torch.utils.cpp_extension import verify_ninja_availability

        verify_ninja_availability()
    except Exception as e:  # noqa: BLE001 - probing runs nvcc; never fail collection
        return f"DeepSpeed evoformer_attn build probe failed: {e}"

    return None


def skip_unless_ds4s_installed():
    deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
    ds4s_is_installed = (
        deepspeed_is_installed
        and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
    )
    is_rocm = torch.cuda.is_available() and torch.version.hip is not None

    if not ds4s_is_installed:
        blocker = "Requires DeepSpeed with version ≥ 0.10.4"
    elif not torch.cuda.is_available():
        # Checked before probing: the probe shells out to nvcc, and decorator
        # order cannot spare us that — every guard is called at import time.
        blocker = "Requires GPU (DeepSpeed evoformer_attn is CUDA-only)"
    elif is_rocm:
        blocker = "DeepSpeed evoformer_attn is not supported on ROCm/HIP"
    else:
        blocker = _ds4s_build_blocker()

    return pytest.mark.skipif(blocker is not None, reason=blocker or "")


def skip_unless_evo_attn_available():
    """Skip unless this platform's evoformer attention backend is usable.

    For tests that select the backend by platform — DeepSpeed's evoformer_attn
    on CUDA, Triton triangle kernels on ROCm — so DeepSpeed is only a
    requirement off ROCm.  Use :func:`skip_unless_ds4s_installed` instead for
    tests that exercise DeepSpeed on every platform.
    """
    is_rocm = torch.cuda.is_available() and torch.version.hip is not None
    if is_rocm:
        return _no_skip("ROCm uses Triton triangle kernels; DeepSpeed not required")
    return skip_unless_ds4s_installed()


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
