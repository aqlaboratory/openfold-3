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

"""Regression tests that Triton fused kernels reuse compiles across lengths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_fused_kernels_reuse_compile_across_lengths(tmp_path):
    """Changing sequence length should not create new same-signature kernels."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_swiglu_transition import (
    _fused_swiglu_transition_fwd_kernel,
)
from openfold3.core.kernels.triton.fused_trimul import (
    _gated_dual_gemm_kernel,
    _gated_out_gemm_residual_kernel,
    _ln_transpose_kernel,
)
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
)
from openfold3.core.model.primitives.fused_swiglu_transition import (
    fused_swiglu_transition,
)
from openfold3.core.model.primitives.fused_trimul import fused_trimul_update


def clear(*jit_fns):
    for fn in jit_fns:
        fn.device_caches.clear()


def count(cache_dir, kernel_name):
    return len(list(Path(cache_dir).rglob(f"{kernel_name}.json")))


torch.manual_seed(11)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])
lengths = (64, 96, 127)

trimul = TriangleMultiplicationOutgoing(128, 128).cuda().eval()
for linear in (
    trimul.linear_a_p,
    trimul.linear_a_g,
    trimul.linear_b_p,
    trimul.linear_b_g,
    trimul.linear_z,
    trimul.linear_g,
):
    linear.bias = None

sw_gamma = torch.ones(128, device="cuda")
sw_beta = torch.zeros(128, device="cuda")
w_a = torch.randn(512, 128, device="cuda") * 0.02
w_b = torch.randn(512, 128, device="cuda") * 0.02
w_out = torch.randn(128, 512, device="cuda") * 0.02

for n in lengths:
    z = torch.randn(1, n, n, 128, device="cuda") * 0.1
    pair_mask = torch.ones(1, n, n, device="cuda")
    x2d = torch.randn(n * n, 128, device="cuda")

    y = fused_swiglu_transition(x2d, sw_gamma, sw_beta, w_a, w_b, w_out)
    assert y.shape == x2d.shape
    clear(_fused_swiglu_transition_fwd_kernel)

    y = fused_trimul_update(trimul, z, pair_mask, with_add=False)
    assert y is not None and y.shape == z.shape
    clear(_gated_dual_gemm_kernel, _gated_out_gemm_residual_kernel, _ln_transpose_kernel)

counts = {
    "_fused_swiglu_transition_fwd_kernel": count(
        cache_dir, "_fused_swiglu_transition_fwd_kernel"
    ),
    "_gated_dual_gemm_kernel": count(cache_dir, "_gated_dual_gemm_kernel"),
    "_gated_out_gemm_residual_kernel": count(
        cache_dir, "_gated_out_gemm_residual_kernel"
    ),
    "_ln_transpose_kernel": count(cache_dir, "_ln_transpose_kernel"),
}
print(json.dumps({"counts": counts}))
assert counts == {k: 1 for k in counts}, counts
"""
    env = os.environ.copy()
    env.update(
        {
            "OPENFOLD3_FUSED_TRIMUL": "1",
            "OPENFOLD3_FUSED_SWIGLU_TRANSITION": "1",
            "OPENFOLD3_FUSED_LN_LINEAR": "1",
            "TRITON_CACHE_DIR": str(cache_dir),
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["counts"]["_fused_swiglu_transition_fwd_kernel"] == 1


def test_fused_ln_linear_reuses_compile_across_lengths(tmp_path):
    """Same-signature fused LN-linear should compile once across row counts."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_ln_linear import (
    _fused_ln_linear_fwd_kernel,
    fused_ln_linear_inference,
)

torch.manual_seed(13)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

gamma = torch.ones(128, device="cuda")
beta = torch.zeros(128, device="cuda")
weight = torch.randn(128, 128, device="cuda") * 0.02

for n in (64, 96, 127):
    x = torch.randn(n * n, 128, device="cuda")
    y = fused_ln_linear_inference(x, gamma, beta, weight, None)
    assert y.shape == x.shape
    _fused_ln_linear_fwd_kernel.device_caches.clear()

count = len(list(cache_dir.rglob("_fused_ln_linear_fwd_kernel.json")))
print(json.dumps({"_fused_ln_linear_fwd_kernel": count}))
assert count == 1, count
"""
    env = os.environ.copy()
    env.update(
        {
            "OPENFOLD3_FUSED_LN_LINEAR": "1",
            "TRITON_CACHE_DIR": str(cache_dir),
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["_fused_ln_linear_fwd_kernel"] == 1
