# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
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

"""Tests for choosing the Triton ``tl.dot`` precision from torch's setting.

The settings are changed through the legacy setter because the rest of the
suite reads ``matmul.allow_tf32``, which raises once a process has written
through the newer ``fp32_precision`` API.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

triton = pytest.importorskip("triton")

# Imports after the Triton skip
# ruff: noqa: E402

import torch
import triton.language as tl

import openfold3.tests.utils.compare_utils as compare_utils
from openfold3.core.kernels.triton import dot_precision
from openfold3.core.kernels.triton.dot_precision import tl_dot
from openfold3.core.utils.precision_utils import matmul_precision

# Allowed-precision sets as the Triton backends report them, so resolution can
# be tested without the hardware.
NVIDIA_ALLOWED = ("tf32", "tf32x3", "ieee", "bf16x3", "bf16x6")
CDNA3_ALLOWED = ("bf16x3", "bf16x6", "ieee", "tf32")
CDNA4_ALLOWED = ("ieee", "bf16x3", "bf16x6")
IEEE_ONLY_ALLOWED = ("ieee",)
ALL_ALLOWED = (NVIDIA_ALLOWED, CDNA3_ALLOWED, CDNA4_ALLOWED, IEEE_ONLY_ALLOWED)


# The resolver, as a pure function. No device needed.


@pytest.mark.parametrize(
    ("allowed", "expected"),
    [
        (NVIDIA_ALLOWED, "tf32"),
        (CDNA3_ALLOWED, "tf32"),
        (CDNA4_ALLOWED, "bf16x3"),
        (IEEE_ONLY_ALLOWED, "ieee"),
    ],
)
def test_tf32_resolves_to_the_first_supported_mode(allowed, expected):
    """Picks the fastest mode the target has, falling back towards ieee."""
    assert dot_precision.resolve("tf32", allowed) == expected


@pytest.mark.parametrize("allowed", ALL_ALLOWED)
def test_ieee_never_resolves_to_a_reduced_mode(allowed):
    """ieee promises full fp32, whatever else is on offer."""
    assert dot_precision.resolve("ieee", allowed) == "ieee"


def test_resolve_rejects_an_unknown_precision():
    with pytest.raises(ValueError, match="unknown precision"):
        dot_precision.resolve("bf16", NVIDIA_ALLOWED)


def test_resolve_fails_when_no_mode_is_available():
    """A backend offering nothing usable is an error, not a silent pick."""
    with pytest.raises(RuntimeError, match="supports none of"):
        dot_precision.resolve("tf32", ("bf16x6",))


# Reading the precision from torch.


@pytest.mark.parametrize(("level", "expected"), [("highest", "ieee"), ("high", "tf32")])
def test_follows_the_torch_setting(level, expected):
    """torch's setting is the only input."""
    with matmul_precision(level):
        assert dot_precision.torch_fp32_precision() == expected


def test_reads_ieee_when_torch_has_no_setting():
    """torch reports "none" until something is set, which is full precision."""
    unset = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(fp32_precision="none"))
        )
    )
    with mock.patch.object(dot_precision, "torch", unset):
        assert dot_precision.torch_fp32_precision() == "ieee"


@pytest.mark.parametrize(("allow_tf32", "expected"), [(True, "tf32"), (False, "ieee")])
def test_reads_allow_tf32_on_torch_without_fp32_precision(allow_tf32, expected):
    """torch before 2.9 has only ``matmul.allow_tf32``. Stubbing the module is
    the only way to reach that branch on a newer torch."""
    legacy = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=allow_tf32))
        )
    )
    with mock.patch.object(dot_precision, "torch", legacy):
        assert dot_precision.torch_fp32_precision() == expected


def test_reading_the_torch_setting_leaves_the_legacy_getters_working():
    """Reading must not break a caller still using the legacy getters. Only
    writing trips torch's mixed-API check."""
    with matmul_precision("highest"):
        assert dot_precision.torch_fp32_precision() == "ieee"
        assert torch.get_float32_matmul_precision() == "highest"
        assert torch.backends.cuda.matmul.allow_tf32 is False


@pytest.mark.parametrize("level", ["highest", "high"])
def test_reduced_dtypes_pin_to_ieee(level):
    with matmul_precision(level):
        assert dot_precision.current_dot_precision(fp32_operands=False) == "ieee"


# tl_dot in a minimal kernel.

SIZE = 64
# Loose enough for any dot mode at SIZE, including tf32 that truncates its
# inputs (about 3e-2 here), but far below what a wiring mistake such as a
# dropped ``acc`` causes. The tests that use it check the kernel computes the
# right product, not its precision.
PRODUCT_ATOL = 1e-1


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_ACC: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if HAS_ACC:
        c = tl_dot(a, b, acc=tl.load(c_ptrs), input_precision=DOT_PRECISION)
    else:
        c = tl_dot(a, b, input_precision=DOT_PRECISION)
    tl.store(c_ptrs, c)


def matmul(
    a: torch.Tensor, b: torch.Tensor, acc: torch.Tensor | None = None
) -> torch.Tensor:
    """``a @ b`` (plus ``acc``, if given) in one block, at torch's precision."""
    M, K = a.shape
    N = b.shape[1]
    c = a.new_empty((M, N)) if acc is None else acc.clone()
    _matmul_kernel[(1,)](
        a,
        b,
        c,
        M,
        N,
        K,
        HAS_ACC=acc is not None,
        DOT_PRECISION=dot_precision.current_dot_precision(
            fp32_operands=a.dtype == torch.float32
        ),
    )
    return c


def matmul_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(0)
    a, b, acc = (
        torch.randn((SIZE, SIZE), device="cuda", generator=gen) for _ in range(3)
    )
    return a, b, acc


@compare_utils.skip_unless_gpu_available()
@pytest.mark.parametrize("with_acc", [False, True])
@pytest.mark.parametrize("level", ["highest", "high"])
def test_tl_dot_computes_the_product(level, with_acc):
    a, b, acc = matmul_inputs()
    if not with_acc:
        acc = None
    with matmul_precision(level):
        out = matmul(a, b, acc)
    reference = a.double() @ b.double()
    if acc is not None:
        reference += acc.double()
    torch.testing.assert_close(out.double(), reference, rtol=0, atol=PRODUCT_ATOL)


def skip_unless_reduced_precision_dot() -> None:
    """Skip unless the target has a reduced-precision fp32 matmul.

    Where it does not, tf32 correctly resolves to ieee and there is nothing to
    tell apart.
    """
    mode = dot_precision.resolve(
        dot_precision.TF32, dot_precision._allowed_dot_precisions()
    )
    if mode == dot_precision.IEEE:
        pytest.skip("no reduced-precision fp32 dot on this target")


@compare_utils.skip_unless_gpu_available()
def test_reduced_precision_is_less_accurate():
    skip_unless_reduced_precision_dot()
    a, b, _ = matmul_inputs()
    reference = a.double() @ b.double()
    errors = {}
    for level in ("high", "highest"):
        with matmul_precision(level):
            out = matmul(a, b)
        errors[level] = (out.double() - reference).abs().max().item()

    assert errors["highest"] < errors["high"], errors
