import importlib
import os
import unittest

import torch

# Give JAX some GPU memory discipline
# (by default it hogs 90% of GPU memory. This disables that behavior and also
# forces it to proactively free memory that it allocates)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
# TODO: Replace this with commandline flag
RUN_OF_TESTS = False


def skip_unless_ds4s_installed():
    deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
    ds4s_is_installed = (
        deepspeed_is_installed
        and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
    )
    return unittest.skipUnless(
        ds4s_is_installed, "Requires DeepSpeed with version ≥ 0.10.4"
    )


def cuda_kernels_is_installed():
    attn_core_is_installed = (
        importlib.util.find_spec("attn_core_inplace_cuda") is not None
    )
    kernels_installed = attn_core_is_installed and (
        importlib.util.find_spec("attention_core") is not None
    )
    return kernels_installed


def skip_unless_cuda_kernels_installed():
    return unittest.skipUnless(
        cuda_kernels_is_installed(), "Requires kernel installation"
    )


def skip_of2_test():
    return unittest.skipUnless(RUN_OF_TESTS, "OpenFold legacy model test")


def skip_unless_flash_attn_installed():
    fa_is_installed = importlib.util.find_spec("flash_attn") is not None
    return unittest.skipUnless(fa_is_installed, "Requires Flash Attention")


def skip_unless_triton_installed():
    triton_is_installed = importlib.util.find_spec("triton") is not None
    return unittest.skipUnless(triton_is_installed, "Requires Triton")


def skip_unless_cuda_available():
    return unittest.skipUnless(torch.cuda.is_available(), "Requires GPU")


def _assert_abs_diff_small_base(compare_func, expected, actual, eps):
    # Helper function for comparing absolute differences of two torch tensors.
    abs_diff = torch.abs(expected - actual)
    err = compare_func(abs_diff)
    zero_tensor = torch.tensor(0, dtype=err.dtype)
    rtol = 1.6e-2 if err.dtype == torch.bfloat16 else 1.3e-6
    torch.testing.assert_close(err, zero_tensor, atol=eps, rtol=rtol)


def assert_max_abs_diff_small(expected, actual, eps):
    _assert_abs_diff_small_base(torch.max, expected, actual, eps)


def assert_mean_abs_diff_small(expected, actual, eps):
    _assert_abs_diff_small_base(torch.mean, expected, actual, eps)
