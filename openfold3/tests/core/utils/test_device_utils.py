# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Outpace Bio, Inc.
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

"""Tests for :mod:`openfold3.core.utils.device_utils`.

Beyond the helpers themselves, this module pins down the torch behaviour they
run into: ``torch.amp.autocast(device_type=..., dtype=torch.float32)`` — the
shape every call site converted by ``autocast_device_type`` uses — does *not*
mean the same thing on every backend. See
:func:`test_fp32_autocast_is_only_enabled_on_cuda`.
"""

import warnings

import pytest
import torch
from torch import nn

from openfold3.core.utils.device_utils import autocast_device_type, empty_device_cache
from openfold3.tests.utils.compare_utils import current_accelerator

#: Whether ``autocast(device_type, dtype=torch.float32)`` leaves autocast *enabled*.
#:
#: torch special-cases the CUDA backend before it validates the requested dtype
#: (``torch/amp/autocast_mode.py``), so fp32 survives there. Every other backend
#: falls through to a ``fast_dtype in device_supported_dtypes`` check, and since
#: that set is ``{float16, bfloat16}`` an fp32 request warns and disables autocast
#: outright. ROCm reports its devices as ``cuda``, so it takes the CUDA branch too.
FP32_AUTOCAST_STAYS_ENABLED = {"cuda": True, "cpu": False, "mps": False}


def _accelerator_device_type() -> str | None:
    """torch device type for this box's accelerator, or None on a CPU-only box.

    ``current_accelerator`` distinguishes ROCm from CUDA, which torch itself does
    not: a ROCm build serves HIP devices through the ``torch.cuda`` API and names
    them ``cuda``.
    """
    accelerator = current_accelerator()
    if accelerator is None:
        return None
    return "cuda" if accelerator in ("cuda", "rocm") else accelerator


#: cpu, plus the accelerator when there is one. Every test runs against both —
#: the behaviour under test is precisely where they differ.
pytestmark = pytest.mark.parametrize(
    "device_type",
    [
        pytest.param(device_type, id=device_type)
        for device_type in ["cpu", *filter(None, [_accelerator_device_type()])]
    ],
)


@pytest.mark.parametrize(
    "build_subject",
    [
        pytest.param(lambda device: torch.zeros(2, device=device), id="tensor"),
        # A module resolves through its first parameter, i.e. to where the
        # *weights* live — not to wherever a caller's input happens to be.
        pytest.param(lambda device: nn.Linear(2, 2).to(device), id="module"),
    ],
)
def test_autocast_device_type(device_type, build_subject):
    """Both overloads resolve to the device of the thing handed in."""
    assert autocast_device_type(build_subject(device_type)) == device_type


def test_empty_device_cache_dispatches_on_every_device(device_type):
    """Each backend reaches its own branch — a real ``empty_cache``, or none."""
    tensor = torch.zeros(1024, device=device_type)
    del tensor
    empty_device_cache(torch.device(device_type))


def test_fp32_autocast_is_only_enabled_on_cuda(device_type):
    """``autocast(dtype=torch.float32)`` is a no-op everywhere except CUDA.

    This is the asymmetry behind the ``UserWarning: In MPS autocast, but the
    target dtype is not supported. Disabling autocast.`` lines that inference on
    Apple Silicon emits — from `OpenFold3.forward`, `PairformerEmbedding.forward`
    and `InputEmbedderAllAtom.forward`, all of which ask for fp32 by name.
    """
    expected = FP32_AUTOCAST_STAYS_ENABLED[device_type]

    # Recorded rather than left to propagate: several openfold3 modules install a
    # global ``warnings.filterwarnings("once")`` at import, which would otherwise
    # swallow the second and later observations of the same message.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.amp.autocast(device_type=device_type, dtype=torch.float32):
            assert torch.is_autocast_enabled(device_type) is expected

    disabling = [w for w in caught if "Disabling autocast" in str(w.message)]
    assert bool(disabling) is not expected, (
        f"{device_type}: expected {'no' if expected else 'a'} 'Disabling autocast' "
        f"warning, got {[str(w.message) for w in caught]}"
    )


@pytest.mark.parametrize(
    "operand_dtype",
    [
        # Why the divergence is harmless today: inference runs at
        # ``precision: "32-true"`` (`InferenceExperimentConfig`), so the operands
        # reaching these regions are fp32 and intent and effect coincide.
        pytest.param(torch.float32, id="fp32-operands-agree-everywhere"),
        # And what it would cost: enabling bf16 on MPS silently drops the fp32
        # guarantee the call sites were written to obtain. This is the case that
        # should fail first if that is attempted.
        pytest.param(torch.bfloat16, id="bf16-operands-upcast-on-cuda-only"),
    ],
)
def test_fp32_autocast_computes_in_operand_dtype_off_cuda(device_type, operand_dtype):
    """On CUDA the region is a genuine fp32 island; elsewhere it just steps aside.

    Enabled, autocast upcasts the operands, so the matmul returns fp32 whatever
    went in. Disabled, it runs at the operands' own dtype — exactly as if the
    caller had written ``enabled=False``.
    """
    if (
        operand_dtype == torch.bfloat16
        and device_type == "mps"
        and not torch.backends.mps.is_macos_or_newer(14, 0)
    ):
        pytest.skip("MPS bfloat16 requires macOS 14+")

    a = torch.randn(4, 4, device=device_type, dtype=operand_dtype)
    b = torch.randn(4, 4, device=device_type, dtype=operand_dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.amp.autocast(device_type=device_type, dtype=torch.float32):
            product_dtype = (a @ b).dtype

    expected = (
        torch.float32 if FP32_AUTOCAST_STAYS_ENABLED[device_type] else operand_dtype
    )
    assert product_dtype == expected
