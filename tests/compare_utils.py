import importlib
import unittest

import torch

# Used to disable nanobind leak warnings from gemmi project. 
import gemmi
gemmi.set_leak_warnings(False)


def skip_unless_ds4s_installed():
    deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
    ds4s_is_installed = (
        deepspeed_is_installed
        and importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
    )
    return unittest.skipUnless(
        ds4s_is_installed, "Requires DeepSpeed with version ≥ 0.10.4"
    )


def skip_unless_cueq_installed():
    cueq_is_installed = cueq_is_installed = (
        importlib.util.find_spec("cuequivariance_torch") is not None
    )
    return unittest.skipUnless(
        cueq_is_installed, "Requires CU-Equivaraince to be installed"
    )


def skip_unless_triton_installed():
    triton_is_installed = importlib.util.find_spec("triton") is not None
    return unittest.skipUnless(triton_is_installed, "Requires Triton")


def skip_unless_cuda_available():
    return unittest.skipUnless(torch.cuda.is_available(), "Requires GPU")


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
