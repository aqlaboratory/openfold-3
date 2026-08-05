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

from openfold3.core.kernels.triton.flash_tri_attn import (
    _flash_tri_attn_kernel,
)
from openfold3.core.kernels.triton.fused_relpos_embed import (
    _fused_relpos_embed_kernel,
    fused_relpos_embed_add_,
)
from openfold3.core.kernels.triton.fused_swiglu_transition import (
    _fused_swiglu_transition_fwd_kernel,
)
from openfold3.core.kernels.triton.fused_template_coordinate import (
    _template_coordinate_projection_kernel,
    template_coordinate_projection_add_,
)
from openfold3.core.kernels.triton.fused_trimul import (
    _gated_column_gemm_kernel,
    _gated_dual_gemm_kernel,
    _gated_out_from_dm_kernel,
    _gated_out_gemm_residual_kernel,
    _ln_stats_kernel,
    _ln_transpose_kernel,
)
from openfold3.core.model.layers.triangular_attention import TriangleAttention
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
)
from openfold3.core.model.primitives.fused_swiglu_transition import (
    fused_swiglu_transition,
)
from openfold3.core.model.primitives.fused_trimul import fused_trimul_update


def jit_device_caches(fn):
    # Autotuner wraps JITFunction; device_caches lives on .fn.
    if hasattr(fn, "device_caches"):
        return fn.device_caches
    return fn.fn.device_caches


def clear(*jit_fns):
    for fn in jit_fns:
        jit_device_caches(fn).clear()


def count(cache_dir, kernel_name):
    return len(list(Path(cache_dir).rglob(f"{kernel_name}.json")))


torch.manual_seed(11)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])
lengths = (64, 96, 127)

tri_attn = TriangleAttention(128, 32, 4, starting=True).cuda().eval()
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
relpos_w = torch.randn(130, 128, device="cuda")
template_w_dgram = torch.randn(64, 39, device="cuda")
template_w_scalar = torch.randn(64, 5, device="cuda")

for n in lengths:
    z = torch.randn(1, n, n, 128, device="cuda") * 0.1
    pair_mask = torch.ones(1, n, n, device="cuda")
    x2d = torch.randn(n * n, 128, device="cuda")

    assert tri_attn(z, mask=pair_mask, chunk_size=32).shape == z.shape
    clear(_flash_tri_attn_kernel)

    relpos_idx = torch.zeros(1, n, n, dtype=torch.int64, device="cuda")
    fused_relpos_embed_add_(
        z, relpos_w, relpos_idx, relpos_idx, relpos_idx, pair_mask, 65
    )
    clear(_fused_relpos_embed_kernel)

    y = fused_swiglu_transition(x2d, sw_gamma, sw_beta, w_a, w_b, w_out)
    assert y.shape == x2d.shape
    clear(_fused_swiglu_transition_fwd_kernel)

    template_out = torch.zeros(1, n, n, 64, device="cuda")
    template_coordinate_projection_add_(
        template_out,
        torch.randn(1, n, 3, device="cuda"),
        torch.randn(1, n, 3, 3, device="cuda"),
        pair_mask[:, 0],
        pair_mask[:, 0],
        torch.ones(1, n, dtype=torch.int32, device="cuda"),
        template_w_dgram,
        template_w_scalar,
    )
    clear(_template_coordinate_projection_kernel)

    y = fused_trimul_update(trimul, z, pair_mask, with_add=False)
    assert y is not None and y.shape == z.shape
    clear(
        _gated_dual_gemm_kernel,
        _gated_out_from_dm_kernel,
        _ln_stats_kernel,
    )

counts = {
    "_flash_tri_attn_kernel": count(cache_dir, "_flash_tri_attn_kernel"),
    "_fused_relpos_embed_kernel": count(cache_dir, "_fused_relpos_embed_kernel"),
    "_fused_swiglu_transition_fwd_kernel": count(
        cache_dir, "_fused_swiglu_transition_fwd_kernel"
    ),
    "_template_coordinate_projection_kernel": count(
        cache_dir, "_template_coordinate_projection_kernel"
    ),
    "_gated_dual_gemm_kernel": count(cache_dir, "_gated_dual_gemm_kernel"),
    "_gated_out_from_dm_kernel": count(cache_dir, "_gated_out_from_dm_kernel"),
    "_ln_stats_kernel": count(cache_dir, "_ln_stats_kernel"),
}
print(json.dumps({"counts": counts}))
# Autotuned kernels compile one artifact per surviving config; non-autotuned
# kernels stay at 1. Length changes must not grow either class.
assert counts["_fused_relpos_embed_kernel"] == 1, counts
assert counts["_template_coordinate_projection_kernel"] == 1, counts
assert counts["_gated_dual_gemm_kernel"] == 1, counts
assert counts["_gated_out_from_dm_kernel"] == 1, counts
assert counts["_ln_stats_kernel"] == 1, counts
assert counts["_flash_tri_attn_kernel"] >= 1, counts
assert counts["_fused_swiglu_transition_fwd_kernel"] >= 1, counts
"""
    env = os.environ.copy()
    env.update(
        {
            "OPENFOLD3_FUSED_TRI_ATTN_V1": "1",
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
    assert all("opm" not in name for name in payload["counts"])

def test_fused_trimul_reuses_compiles_across_lengths_and_chunks(tmp_path):
    """Trimul signatures are stable across N for both directions and dimensions."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_trimul import (
    _gated_column_gemm_kernel,
    _gated_dual_gemm_kernel,
    _gated_out_from_dm_kernel,
    _gated_out_gemm_residual_kernel,
    _ln_stats_kernel,
    _ln_transpose_kernel,
)
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from openfold3.core.model.primitives.fused_trimul import fused_trimul_update


KERNELS = (
    _gated_column_gemm_kernel,
    _gated_dual_gemm_kernel,
    _gated_out_from_dm_kernel,
    _gated_out_gemm_residual_kernel,
    _ln_stats_kernel,
    _ln_transpose_kernel,
)
NAMES = tuple(kernel.fn.__name__ for kernel in KERNELS)


def clear():
    for kernel in KERNELS:
        caches = getattr(kernel, "device_caches", None)
        if caches is None:
            caches = kernel.fn.device_caches
        caches.clear()


def counts(cache_dir):
    return {
        name: len(list(Path(cache_dir).rglob(f"{name}.json")))
        for name in NAMES
    }


def run(module, n, c_z, cap):
    if cap is None:
        os.environ.pop("OPENFOLD3_TRIMUL_CHUNK_CAP", None)
    else:
        os.environ["OPENFOLD3_TRIMUL_CHUNK_CAP"] = str(cap)
    z = torch.randn(1, n, n, c_z, device="cuda") * 0.1
    mask = torch.ones(1, n, n, device="cuda")
    if cap is not None and not module._outgoing:
        out = fused_trimul_update(module, z, mask, with_add=False)
        assert out is not None and out.shape == z.shape
        z_inplace = z.clone()
        out_inplace = fused_trimul_update(
            module, z_inplace, mask, with_add=True, out=z_inplace
        )
        assert out_inplace is not None
        assert out_inplace.data_ptr() == z_inplace.data_ptr()
    else:
        out = fused_trimul_update(module, z, mask, with_add=True, out=z)
        assert out is not None and out.data_ptr() == z.data_ptr()
    torch.cuda.synchronize()
    clear()


torch.manual_seed(17)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])
records = []
for c_z in (64, 128):
    for cls in (TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming):
        module = cls(c_z, c_z).cuda().eval()
        for linear in (
            module.linear_a_p,
            module.linear_a_g,
            module.linear_b_p,
            module.linear_b_g,
            module.linear_z,
            module.linear_g,
        ):
            linear.bias = None
        for cap in (None, 64, 128, 256):
            first_n, second_n = ((383, 511) if cap == 256 else (211, 383))
            run(module, first_n, c_z, cap)
            before = counts(cache_dir)
            run(module, second_n, c_z, cap)
            after = counts(cache_dir)
            assert after == before, {
                "c_z": c_z,
                "direction": cls.__name__,
                "cap": cap,
                "before": before,
                "after": after,
            }
            records.append(
                {
                    "c_z": c_z,
                    "direction": cls.__name__,
                    "cap": cap,
                    "counts": after,
                }
            )

print(json.dumps({"records": records}))
"""
    env = os.environ.copy()
    env.update(
        {
            "OPENFOLD3_FUSED_TRIMUL": "1",
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
    assert len(payload["records"]) == 16


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

def clear(fn):
    caches = getattr(fn, "device_caches", None)
    if caches is None:
        caches = fn.fn.device_caches
    caches.clear()

def n_json(cache_dir, name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

torch.manual_seed(13)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

gamma = torch.ones(128, device="cuda")
beta = torch.zeros(128, device="cuda")
weight = torch.randn(128, 128, device="cuda") * 0.02

stable = None
for n in (192, 224, 255):
    x = torch.randn(n * n, 128, device="cuda")
    y = fused_ln_linear_inference(x, gamma, beta, weight, None)
    assert y.shape == x.shape
    clear(_fused_ln_linear_fwd_kernel)
    cur = n_json(cache_dir, "_fused_ln_linear_fwd_kernel")
    if stable is None:
        stable = cur
    assert cur == stable, (cur, stable)

print(json.dumps({"_fused_ln_linear_fwd_kernel": stable}))
assert stable >= 1, stable
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
    assert payload["_fused_ln_linear_fwd_kernel"] >= 1


def test_msa_pair_bias_ln_linear_reuses_compile_across_lengths(tmp_path):
    """MSA pair-bias LN-linear should compile once across token lengths."""
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

def clear(fn):
    caches = getattr(fn, "device_caches", None)
    if caches is None:
        caches = fn.fn.device_caches
    caches.clear()

def n_json(cache_dir, name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

torch.manual_seed(23)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

gamma = torch.ones(128, device="cuda")
beta = torch.zeros(128, device="cuda")
weight = torch.randn(4, 128, device="cuda") * 0.02
bias = torch.randn(4, device="cuda") * 0.02

stable = None
for n in (64, 96, 127):
    z = torch.randn(1, n, n, 128, device="cuda")
    y = fused_ln_linear_inference(z, gamma, beta, weight, bias)
    assert y.shape == (1, n, n, 4)
    clear(_fused_ln_linear_fwd_kernel)
    cur = n_json(cache_dir, "_fused_ln_linear_fwd_kernel")
    if stable is None:
        stable = cur
    assert cur == stable, (cur, stable)

print(json.dumps({"_fused_ln_linear_fwd_kernel": stable}))
assert stable >= 1, stable
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
    assert payload["_fused_ln_linear_fwd_kernel"] >= 1


def test_pair_ln_linear_reuses_compile_across_lengths(tmp_path):
    """Strided pair LN-linear should compile once across sequence lengths."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_ln_linear import (
    _pair_ln_linear_fwd_kernel,
    pair_ln_linear_inference,
)

def clear(fn):
    caches = getattr(fn, "device_caches", None)
    if caches is None:
        caches = fn.fn.device_caches
    caches.clear()

def n_json(cache_dir, name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

torch.manual_seed(37)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

gamma = torch.ones(128, device="cuda")
beta = torch.zeros(128, device="cuda")
weight = torch.randn(384, 128, device="cuda") * 0.02

stable = None
for n in (192, 224, 255):
    x = torch.randn(1, n, n, 128, device="cuda").transpose(-2, -3)
    y = pair_ln_linear_inference(x, gamma, beta, weight, None)
    assert y.shape == (n * n, 384)
    clear(_pair_ln_linear_fwd_kernel)
    cur = n_json(cache_dir, "_pair_ln_linear_fwd_kernel")
    if stable is None:
        stable = cur
    assert cur == stable, (cur, stable)

print(json.dumps({"_pair_ln_linear_fwd_kernel": stable}))
assert stable >= 1, stable
"""
    env = os.environ.copy()
    env.update({"TRITON_CACHE_DIR": str(cache_dir)})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["_pair_ln_linear_fwd_kernel"] >= 1


def test_v1_tri_attn_reuses_compile_across_lengths(tmp_path):
    """The custom V1 cap-block attention kernel should be length-generic."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.flash_tri_attn import _flash_tri_attn_kernel
from openfold3.core.model.layers.triangular_attention import TriangleAttention

def clear(fn):
    caches = getattr(fn, "device_caches", None)
    if caches is None:
        caches = fn.fn.device_caches
    caches.clear()

def n_json(cache_dir, name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

torch.manual_seed(37)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])
tri_attn = TriangleAttention(128, 32, 4, starting=True).cuda().eval()
with torch.no_grad():
    tri_attn.mha.linear_o.weight.normal_(0, 0.02)
    tri_attn.mha.linear_g.weight.normal_(0, 0.02)

stable = None
for n in (192, 224, 255):
    z = torch.randn(1, n, n, 128, device="cuda")
    pair_mask = torch.ones(1, n, n, device="cuda")
    assert tri_attn(z, mask=pair_mask, chunk_size=128).shape == z.shape
    clear(_flash_tri_attn_kernel)
    cur = n_json(cache_dir, "_flash_tri_attn_kernel")
    if stable is None:
        stable = cur
    assert cur == stable, (cur, stable)

print(json.dumps({"_flash_tri_attn_kernel": stable}))
assert stable >= 1, stable
"""
    env = os.environ.copy()
    env.update(
        {
            "OPENFOLD3_FUSED_TRI_ATTN_V1": "1",
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
    assert payload["_flash_tri_attn_kernel"] >= 1


def test_v1_template_tri_attn_reuses_compile_across_lengths(tmp_path):
    """The c_t=64 V1 template kernel should be length-generic."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.flash_tri_attn import _flash_tri_attn_kernel
from openfold3.core.model.layers.triangular_attention import TriangleAttention

def clear(fn):
    caches = getattr(fn, "device_caches", None)
    if caches is None:
        caches = fn.fn.device_caches
    caches.clear()

def n_json(cache_dir, name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

torch.manual_seed(41)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])
tri_attn = TriangleAttention(64, 16, 4, starting=True).cuda().eval()
with torch.no_grad():
    tri_attn.mha.linear_o.weight.normal_(0, 0.02)
    tri_attn.mha.linear_g.weight.normal_(0, 0.02)

stable = None
for n in (192, 224, 255):
    z = torch.randn(1, n, n, 64, device="cuda")
    pair_mask = torch.ones(1, n, n, device="cuda")
    assert tri_attn(z, mask=pair_mask, chunk_size=128).shape == z.shape
    clear(_flash_tri_attn_kernel)
    cur = n_json(cache_dir, "_flash_tri_attn_kernel")
    if stable is None:
        stable = cur
    assert cur == stable, (cur, stable)

print(json.dumps({"_flash_tri_attn_kernel": stable}))
assert stable >= 1, stable
"""
    env = os.environ.copy()
    env.update(
        {
            "OPENFOLD3_FUSED_TRI_ATTN_V1": "1",
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
    assert payload["_flash_tri_attn_kernel"] >= 1


def test_attention_pair_bias_reuses_fused_pair_projection_across_lengths(tmp_path):
    """Diffusion pair-bias projection should reuse fused LN-linear compiles."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_ln_linear import _fused_ln_linear_fwd_kernel
from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias

def clear(fn):
    caches = getattr(fn, "device_caches", None)
    if caches is None:
        caches = fn.fn.device_caches
    caches.clear()

def n_json(cache_dir, name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

torch.manual_seed(19)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

module = AttentionPairBias(
    c_q=768,
    c_k=768,
    c_v=768,
    c_s=384,
    c_z=128,
    c_hidden=48,
    no_heads=16,
    use_ada_layer_norm=True,
).cuda().eval()

stable = None
for n in (64, 96, 127):
    z = torch.randn(1, 1, n, n, 128, device="cuda")
    pair_bias = module.prep_static_pair_bias(z)
    assert pair_bias.shape == (1, 1, 16, n, n)
    clear(_fused_ln_linear_fwd_kernel)
    cur = n_json(cache_dir, "_fused_ln_linear_fwd_kernel")
    if stable is None:
        stable = cur
    assert cur == stable, (cur, stable)

print(json.dumps({"_fused_ln_linear_fwd_kernel": stable}))
assert stable >= 1, stable
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
    assert payload["_fused_ln_linear_fwd_kernel"] >= 1


def test_fused_softmax_reuses_compile_across_lengths(tmp_path):
    """MSA fused softmax should use one fixed tile for practical lengths."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_softmax import fused_softmax
from openfold3.core.kernels.triton.triton_softmax import softmax_mask_bias_kernel

torch.manual_seed(17)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

for n in (96, 192, 384, 768, 1264, 2049, 2800):
    x = torch.randn(1, 1, 4, n, n, device="cuda", dtype=torch.bfloat16)
    y = fused_softmax(x)
    assert y.shape == x.shape
    softmax_mask_bias_kernel.device_caches.clear()

counts = {
    "softmax_mask_bias_kernel": len(
        list(cache_dir.rglob("softmax_mask_bias_kernel.json"))
    ),
}
print(json.dumps(counts))
assert counts == {"softmax_mask_bias_kernel": 1}, counts
"""
    env = os.environ.copy()
    env.update({"TRITON_CACHE_DIR": str(cache_dir)})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["softmax_mask_bias_kernel"] == 1


def test_fused_softmax_grad_reuses_compile_across_lengths(tmp_path):
    """Backward fused softmax should use one fixed tile for practical lengths."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_softmax import fused_softmax
from openfold3.core.kernels.triton.triton_softmax import softmax_grad_kernel

torch.manual_seed(19)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

for n in (96, 192, 384, 768, 1536):
    x = torch.randn(
        1,
        1,
        4,
        n,
        n,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    y = fused_softmax(x)
    y.square().sum().backward()
    softmax_grad_kernel.device_caches.clear()

counts = {
    "softmax_grad_kernel": len(
        list(cache_dir.rglob("softmax_grad_kernel.json"))
    ),
}
print(json.dumps(counts))
assert counts == {"softmax_grad_kernel": 1}, counts
"""
    env = os.environ.copy()
    env.update({"TRITON_CACHE_DIR": str(cache_dir)})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["softmax_grad_kernel"] == 1


def test_fused_softmax_wide_reuses_compile_across_lengths(tmp_path):
    """Streaming wide-row softmax should avoid bucket recompiles above 4096."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_softmax import fused_softmax
from openfold3.core.kernels.triton.triton_softmax import (
    softmax_mask_bias_wide_kernel,
)

torch.manual_seed(31)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

for n in (4097, 5000, 8193):
    x = torch.randn(1, 1, 1, 2, n, device="cuda", dtype=torch.bfloat16)
    y = fused_softmax(x)
    ref = torch.softmax(x.float(), dim=-1).to(y.dtype)
    assert y.shape == x.shape
    torch.testing.assert_close(y, ref, atol=4e-3, rtol=4e-3)
    softmax_mask_bias_wide_kernel.device_caches.clear()

count = len(list(cache_dir.rglob("softmax_mask_bias_wide_kernel.json")))
print(json.dumps({"softmax_mask_bias_wide_kernel": count}))
assert count == 1, count
"""
    env = os.environ.copy()
    env.update({"TRITON_CACHE_DIR": str(cache_dir)})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["softmax_mask_bias_wide_kernel"] == 1


def test_fused_softmax_wide_grad_reuses_compile_across_lengths(tmp_path):
    """Streaming wide-row softmax backward should avoid bucket recompiles."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_softmax import fused_softmax
from openfold3.core.kernels.triton.triton_softmax import softmax_grad_wide_kernel

torch.manual_seed(37)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

for n in (4097, 5000, 8193):
    x = torch.randn(
        1,
        1,
        1,
        2,
        n,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    x_ref = x.detach().clone().requires_grad_(True)
    y = fused_softmax(x)
    ref = torch.softmax(x_ref, dim=-1)
    loss = y.square().sum()
    ref_loss = ref.square().sum()
    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(y, ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(x.grad, x_ref.grad, atol=2e-5, rtol=2e-5)
    softmax_grad_wide_kernel.device_caches.clear()

count = len(list(cache_dir.rglob("softmax_grad_wide_kernel.json")))
print(json.dumps({"softmax_grad_wide_kernel": count}))
assert count == 1, count
"""
    env = os.environ.copy()
    env.update({"TRITON_CACHE_DIR": str(cache_dir)})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["softmax_grad_wide_kernel"] == 1


def test_flash_diffusion_attn_reuses_compile_across_lengths(tmp_path):
    """Diffusion pair-bias FlashAttention should not specialize on token length."""
    cache_dir = tmp_path / "triton_cache"
    code = r"""
import json
import math
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.flash_diffusion_attn import (
    _flash_diffusion_attn_kernel,
    flash_diffusion_attn,
)

def clear(fn):
    caches = getattr(fn, "device_caches", None)
    if caches is None:
        caches = fn.fn.device_caches
    caches.clear()

def n_json(cache_dir, name):
    return len(list(Path(cache_dir).rglob(f"{name}.json")))

torch.manual_seed(41)
torch.set_grad_enabled(False)
cache_dir = Path(os.environ["TRITON_CACHE_DIR"])

stable = None
for n in (64, 96, 127):
    q = torch.randn(1, 2, 4, n, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = torch.ones(1, 1, 1, 1, n, device="cuda", dtype=torch.bfloat16)
    mask[..., -3:] = 0
    mask_bias = 1e9 * (mask - 1)
    pair_bias = torch.randn(
        1, 1, 4, n, n, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    scale = 1 / math.sqrt(32)
    y = flash_diffusion_attn(q, k, v, mask_bias, pair_bias, scale)
    scores = torch.einsum("bshqc,bshkc->bshqk", q.float(), k.float()) * scale
    scores = scores + mask_bias.float() + pair_bias.float()
    ref = torch.einsum(
        "bshqk,bshkc->bshqc", torch.softmax(scores, dim=-1).to(v.dtype), v
    )
    assert y.shape == q.shape
    torch.testing.assert_close(y, ref, atol=5e-2, rtol=5e-2)
    clear(_flash_diffusion_attn_kernel)
    cur = n_json(cache_dir, "_flash_diffusion_attn_kernel")
    if stable is None:
        stable = cur
    assert cur == stable, (cur, stable)

print(json.dumps({"_flash_diffusion_attn_kernel": stable}))
assert stable >= 1, stable
"""
    env = os.environ.copy()
    env.update({"TRITON_CACHE_DIR": str(cache_dir)})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["_flash_diffusion_attn_kernel"] >= 1
