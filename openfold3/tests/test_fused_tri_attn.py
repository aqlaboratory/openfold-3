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

"""Tests for the fused triangle attention path.

Compares the fused dispatch (``OPENFOLD3_FUSED_TRI_ATTN=1``) against the
eager ``TriangleAttention.forward`` for both starting and ending nodes,
in fp32 and bf16, with and without mask. Includes timing and multi-N diff.
"""

import gc
import os
import time
import unittest

import torch

# Env flags must be set before importing the modules.
os.environ["OPENFOLD3_FUSED_TRI_ATTN"] = "1"

from openfold3.core.model.layers.triangular_attention import (  # noqa: E402
    TriangleAttention,
)


def _skip_no_cuda(fn):
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA required")(fn)


def _build_module(c_in=128, c_hidden=32, no_heads=4, starting=True):
    return TriangleAttention(
        c_in=c_in,
        c_hidden=c_hidden,
        no_heads=no_heads,
        starting=starting,
    )


def _eager_forward(module, x, mask, use_cueq=False):
    """Run the eager path (env flag temporarily off)."""
    old = os.environ.get("OPENFOLD3_FUSED_TRI_ATTN")
    os.environ["OPENFOLD3_FUSED_TRI_ATTN"] = "0"
    try:
        with torch.no_grad():
            return module(
                x,
                mask=mask,
                use_cueq_triangle_kernels=use_cueq,
            )
    finally:
        if old is not None:
            os.environ["OPENFOLD3_FUSED_TRI_ATTN"] = old
        else:
            os.environ.pop("OPENFOLD3_FUSED_TRI_ATTN", None)


def _fused_forward(module, x, mask, use_cueq=False):
    """Run the fused path (env flag on)."""
    os.environ["OPENFOLD3_FUSED_TRI_ATTN"] = "1"
    with torch.no_grad():
        return module(
            x,
            mask=mask,
            use_cueq_triangle_kernels=use_cueq,
        )


class TestFusedTriAttn(unittest.TestCase):

    @_skip_no_cuda
    def test_fp32_starting(self):
        """fp32, starting node, N=64."""
        torch.manual_seed(0)
        B, N = 1, 64
        mod = _build_module(starting=True).cuda().eval()
        x = torch.randn(B, N, N, 128, device="cuda", dtype=torch.float32)
        mask = torch.ones(B, N, N, device="cuda", dtype=torch.float32)

        ref = _eager_forward(mod, x, mask)
        fused = _fused_forward(mod, x, mask)

        diff = (ref - fused).abs().max().item()
        self.assertLess(diff, 2e-4, f"fp32 starting max diff {diff}")

    @_skip_no_cuda
    def test_fp32_ending(self):
        """fp32, ending node, N=64."""
        torch.manual_seed(1)
        B, N = 1, 64
        mod = _build_module(starting=False).cuda().eval()
        x = torch.randn(B, N, N, 128, device="cuda", dtype=torch.float32)
        mask = torch.ones(B, N, N, device="cuda", dtype=torch.float32)

        ref = _eager_forward(mod, x, mask)
        fused = _fused_forward(mod, x, mask)

        diff = (ref - fused).abs().max().item()
        self.assertLess(diff, 2e-4, f"fp32 ending max diff {diff}")

    @_skip_no_cuda
    def test_bf16(self):
        """bf16, starting node, N=64."""
        torch.manual_seed(2)
        B, N = 1, 64
        mod = _build_module(starting=True).cuda().bfloat16().eval()
        x = torch.randn(B, N, N, 128, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(B, N, N, device="cuda", dtype=torch.bfloat16)

        ref = _eager_forward(mod, x, mask)
        fused = _fused_forward(mod, x, mask)

        diff = (ref - fused).abs().max().item()
        self.assertLess(diff, 8e-2, f"bf16 max diff {diff}")

    @_skip_no_cuda
    def test_no_mask(self):
        """fp32, starting node, mask=None."""
        torch.manual_seed(3)
        B, N = 1, 48
        mod = _build_module(starting=True).cuda().eval()
        x = torch.randn(B, N, N, 128, device="cuda", dtype=torch.float32)

        ref = _eager_forward(mod, x, None)
        fused = _fused_forward(mod, x, None)

        diff = (ref - fused).abs().max().item()
        self.assertLess(diff, 2e-4, f"no-mask max diff {diff}")

    @_skip_no_cuda
    def test_with_mask(self):
        """fp32, starting node, partial mask."""
        torch.manual_seed(4)
        B, N = 1, 48
        mod = _build_module(starting=True).cuda().eval()
        x = torch.randn(B, N, N, 128, device="cuda", dtype=torch.float32)
        mask = torch.ones(B, N, N, device="cuda", dtype=torch.float32)
        mask[:, :, N // 2 :] = 0.0

        ref = _eager_forward(mod, x, mask)
        fused = _fused_forward(mod, x, mask)

        diff = (ref - fused).abs().max().item()
        self.assertLess(diff, 2e-4, f"masked max diff {diff}")

    @_skip_no_cuda
    def test_larger_n(self):
        """fp32, starting node, N=128."""
        torch.manual_seed(5)
        B, N = 1, 128
        mod = _build_module(starting=True).cuda().eval()
        x = torch.randn(B, N, N, 128, device="cuda", dtype=torch.float32)
        mask = torch.ones(B, N, N, device="cuda", dtype=torch.float32)

        ref = _eager_forward(mod, x, mask)
        fused = _fused_forward(mod, x, mask)

        diff = (ref - fused).abs().max().item()
        self.assertLess(diff, 5e-4, f"N=128 max diff {diff}")

    @_skip_no_cuda
    def test_with_cueq(self):
        """fp32, starting, with cuEq attention core (if available)."""
        try:
            from openfold3.core.model.primitives.attention import (
                cueq_is_installed,
            )
        except ImportError:
            self.skipTest("cuEq not available")
        if not cueq_is_installed:
            self.skipTest("cuEq not installed")

        torch.manual_seed(6)
        B, N = 1, 128
        mod = _build_module(starting=True).cuda().eval()
        x = torch.randn(B, N, N, 128, device="cuda", dtype=torch.float32)
        mask = torch.ones(B, N, N, device="cuda", dtype=torch.float32)

        ref = _eager_forward(mod, x, mask, use_cueq=True)
        fused = _fused_forward(mod, x, mask, use_cueq=True)

        diff = (ref - fused).abs().max().item()
        self.assertLess(diff, 5e-4, f"cuEq max diff {diff}")

    @_skip_no_cuda
    def test_multi_n_diff(self):
        """Numerical diff across multiple N values (starting + ending, fp32)."""
        for starting in (True, False):
            torch.manual_seed(10)
            mod = _build_module(starting=starting).cuda().eval()
            label = "starting" if starting else "ending"
            for N in (64, 128, 256):
                x = torch.randn(1, N, N, 128, device="cuda", dtype=torch.float32)
                mask = torch.ones(1, N, N, device="cuda", dtype=torch.float32)
                ref = _eager_forward(mod, x, mask)
                fused = _fused_forward(mod, x, mask)
                abs_diff = (ref - fused).abs().max().item()
                rel_diff = (
                    (ref - fused).abs() / (ref.abs() + 1e-8)
                ).max().item()
                print(
                    f"  tri_attn {label} N={N}: "
                    f"abs={abs_diff:.2e} rel={rel_diff:.2e}"
                )
                self.assertLess(abs_diff, 5e-4, f"{label} N={N}")

    @_skip_no_cuda
    def test_timing(self):
        """Wall-clock comparison: eager vs fused at N=256 (informational)."""
        torch.manual_seed(20)
        N = 256
        mod = _build_module(starting=True).cuda().eval()
        x = torch.randn(1, N, N, 128, device="cuda", dtype=torch.float32)
        mask = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

        _ = _eager_forward(mod, x, mask)
        _ = _fused_forward(mod, x, mask)

        def _time(fn, n_runs=5):
            torch.cuda.synchronize()
            times = []
            for _ in range(n_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1e3)
            times.sort()
            return times[len(times) // 2]

        eager_ms = _time(lambda: _eager_forward(mod, x, mask))
        fused_ms = _time(lambda: _fused_forward(mod, x, mask))
        print(
            f"  tri_attn N={N}: eager={eager_ms:.2f} ms, "
            f"fused={fused_ms:.2f} ms, speedup={eager_ms / fused_ms:.2f}x"
        )

    @_skip_no_cuda
    def test_peak_memory(self):
        """Verify fused tri_attn peak with cuEq is ≤ 6.5U."""
        try:
            from openfold3.core.model.primitives.attention import (
                cueq_is_installed,
            )
        except ImportError:
            self.skipTest("cuEq not available")
        if not cueq_is_installed:
            self.skipTest("cuEq not installed")

        torch.manual_seed(30)
        N = 256
        c_z = 128
        mod = _build_module(starting=True).cuda().eval()
        x = torch.randn(1, N, N, c_z, device="cuda", dtype=torch.float32)
        mask = torch.ones(1, N, N, device="cuda", dtype=torch.float32)
        U = N * N * c_z * 4

        _ = _fused_forward(mod, x, mask, use_cueq=True)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = _fused_forward(mod, x, mask, use_cueq=True)
        peak = torch.cuda.max_memory_allocated()
        peak_u = (peak - base) / U
        del out
        print(f"  tri_attn fused+cuEq peak: {peak_u:.2f} U")
        self.assertLess(peak_u, 6.5, f"peak {peak_u:.2f} U exceeds 6.5U")


if __name__ == "__main__":
    unittest.main()
