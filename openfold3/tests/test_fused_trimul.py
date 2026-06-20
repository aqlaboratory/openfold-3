# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Numerical correctness, timing, and memory tests for the fused triangle
multiplicative update kernel.

Compares the fused Triton path (OPENFOLD3_FUSED_TRIMUL=1) against the eager
reference of the real AF3 TriangleMultiplication{Outgoing,Incoming} module,
for both directions, fp32 + bf16, mask on/off.
"""

from __future__ import annotations

import gc
import time
import unittest

import torch

from openfold3.core.kernels.triton.fused_trimul import (
    ln_transpose_fp32,
    trimul_batched_einsum_fp32,
)
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from openfold3.core.model.primitives.fused_trimul import fused_trimul_update
from openfold3.tests.compare_utils import skip_unless_cuda_available


def _randomize_observable_output(module):
    """Make final/gate projections nonzero so parity tests observe errors."""
    with torch.no_grad():
        module.linear_z.weight.normal_(0, 0.02)
        module.linear_g.weight.normal_(0, 0.02)


def _time_fn(fn, n_runs=5):
    """Time a CUDA function, return median wall-clock in ms."""
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


@skip_unless_cuda_available()
class TestFusedTrimul(unittest.TestCase):
    def _check(self, outgoing, N, dtype, use_mask, atol, rtol):
        device = "cuda"
        cls = (
            TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
        )
        torch.manual_seed(0)
        m = cls(128, 128).to(device=device, dtype=dtype).eval()
        _randomize_observable_output(m)
        z = torch.randn(1, N, N, 128, device=device, dtype=dtype) * 0.5
        mask = (
            torch.ones(1, N, N, device=device, dtype=dtype) if use_mask else None
        )
        with torch.inference_mode():
            ref = m.forward(z.clone(), mask=mask, inplace_safe=False)
            fused = fused_trimul_update(m, z.clone(), mask, with_add=False)
        self.assertIsNotNone(fused, "fused path returned None (ineligible)")
        diff = (fused - ref).abs().max().item()
        self.assertTrue(
            torch.allclose(fused, ref, atol=atol, rtol=rtol),
            f"outgoing={outgoing} N={N} {dtype} mask={use_mask}: max diff {diff:.2e}",
        )

    def test_fp32(self):
        for outgoing in (True, False):
            for N in (64, 256):
                for use_mask in (True, False):
                    self._check(outgoing, N, torch.float32, use_mask, atol=2e-4, rtol=2e-4)

    def test_bf16(self):
        for outgoing in (True, False):
            for N in (64, 256):
                self._check(outgoing, N, torch.bfloat16, True, atol=8e-2, rtol=8e-2)

    def test_with_add(self):
        device = "cuda"
        torch.manual_seed(0)
        m = TriangleMultiplicationOutgoing(128, 128).to(device, torch.float32).eval()
        _randomize_observable_output(m)
        z = torch.randn(1, 128, 128, 128, device=device) * 0.5
        mask = torch.ones(1, 128, 128, device=device)
        with torch.inference_mode():
            upd = m.forward(z.clone(), mask=mask, inplace_safe=False)
            ref = z + upd
            fused = fused_trimul_update(m, z.clone(), mask, with_add=True)
        diff = (fused - ref).abs().max().item()
        self.assertTrue(diff < 2e-4, f"with_add max diff {diff:.2e}")

    def test_with_add_inplace_out(self):
        device = "cuda"
        torch.manual_seed(0)
        for cls in (TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming):
            m = cls(128, 128).to(device, torch.float32).eval()
            _randomize_observable_output(m)
            z = torch.randn(1, 128, 128, 128, device=device) * 0.5
            mask = torch.ones(1, 128, 128, device=device)
            with torch.inference_mode():
                upd = m.forward(z.clone(), mask=mask, inplace_safe=False)
                ref = z + upd
                z_inplace = z.clone()
                fused = fused_trimul_update(
                    m,
                    z_inplace,
                    mask,
                    with_add=True,
                    out=z_inplace,
                )
            self.assertEqual(fused.data_ptr(), z_inplace.data_ptr())
            diff = (fused - ref).abs().max().item()
            self.assertTrue(diff < 2e-4, f"inplace with_add max diff {diff:.2e}")

    def test_multi_n_diff(self):
        """Numerical diff across multiple N values (outgoing + incoming, fp32)."""
        device = "cuda"
        for outgoing in (True, False):
            cls = (
                TriangleMultiplicationOutgoing
                if outgoing
                else TriangleMultiplicationIncoming
            )
            torch.manual_seed(0)
            m = cls(128, 128).to(device=device, dtype=torch.float32).eval()
            _randomize_observable_output(m)
            label = "outgoing" if outgoing else "incoming"
            for N in (64, 128, 256):
                z = torch.randn(1, N, N, 128, device=device) * 0.5
                mask = torch.ones(1, N, N, device=device)
                with torch.inference_mode():
                    ref = m.forward(z.clone(), mask=mask, inplace_safe=False)
                    fused = fused_trimul_update(m, z.clone(), mask, with_add=False)
                self.assertIsNotNone(fused)
                abs_diff = (fused - ref).abs().max().item()
                rel_diff = ((fused - ref).abs() / (ref.abs() + 1e-8)).max().item()
                print(
                    f"  trimul {label} N={N}: "
                    f"abs={abs_diff:.2e} rel={rel_diff:.2e}"
                )
                self.assertLess(abs_diff, 5e-4, f"{label} N={N}")

    def test_timing(self):
        """Wall-clock comparison: eager vs fused at N=256 (informational)."""
        device = "cuda"
        torch.manual_seed(0)
        m = TriangleMultiplicationOutgoing(128, 128).to(device, torch.float32).eval()
        _randomize_observable_output(m)
        N = 256
        z = torch.randn(1, N, N, 128, device=device)
        mask = torch.ones(1, N, N, device=device)

        with torch.inference_mode():
            _ = m.forward(z.clone(), mask=mask, inplace_safe=False)
            _ = fused_trimul_update(m, z.clone(), mask, with_add=False)

        eager_ms = _time_fn(
            lambda: m.forward(z.clone(), mask=mask, inplace_safe=False)
        )
        fused_ms = _time_fn(
            lambda: fused_trimul_update(m, z.clone(), mask, with_add=False)
        )
        print(
            f"  trimul N={N}: eager={eager_ms:.2f} ms, "
            f"fused={fused_ms:.2f} ms, speedup={eager_ms / fused_ms:.2f}x"
        )

    def test_peak_memory(self):
        """Verify fused trimul peak is ≤ 3.2U (down from 4U)."""
        device = "cuda"
        torch.manual_seed(0)
        N = 256
        c_z = 128
        m = TriangleMultiplicationOutgoing(c_z, c_z).to(device, torch.float32).eval()
        _randomize_observable_output(m)
        z = torch.randn(1, N, N, c_z, device=device)
        mask = torch.ones(1, N, N, device=device)
        U = N * N * c_z * 4

        with torch.inference_mode():
            _ = fused_trimul_update(m, z, mask, with_add=False)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        with torch.inference_mode():
            out = fused_trimul_update(m, z, mask, with_add=False)
        peak = torch.cuda.max_memory_allocated()
        peak_u = (peak - base) / U
        del out
        print(f"  trimul fused peak: {peak_u:.2f} U")
        self.assertLess(peak_u, 3.2, f"peak {peak_u:.2f} U exceeds 3.2U target")


@skip_unless_cuda_available()
class TestFusedTrimulSubOps(unittest.TestCase):
    """Tests for individual sub-op kernels used by the fused trimul pipeline."""

    def test_ln_transpose_fp32(self):
        """Transposing LN: (D, M) → (M, D) matches eager LN + permute."""
        device = "cuda"
        D = 128
        for M in (128 * 128, 200 * 200):
            torch.manual_seed(42)
            x_dm = torch.randn(D, M, device=device, dtype=torch.float32)
            w = torch.randn(D, device=device, dtype=torch.float32)
            b = torch.randn(D, device=device, dtype=torch.float32)

            out = ln_transpose_fp32(x_dm, w, b, eps=1e-5)

            x_md = x_dm.t().contiguous()
            ref = torch.nn.functional.layer_norm(x_md, (D,), w, b, eps=1e-5)

            diff = (out - ref).abs().max().item()
            self.assertLess(diff, 1e-5, f"M={M}: max diff {diff:.2e}")

    def test_ln_transpose_no_bias(self):
        """Transposing LN without bias."""
        device = "cuda"
        D, M = 128, 64 * 64
        torch.manual_seed(42)
        x_dm = torch.randn(D, M, device=device, dtype=torch.float32)
        w = torch.randn(D, device=device, dtype=torch.float32)

        out = ln_transpose_fp32(x_dm, w, None, eps=1e-5)

        x_md = x_dm.t().contiguous()
        ref = torch.nn.functional.layer_norm(x_md, (D,), w, None, eps=1e-5)

        diff = (out - ref).abs().max().item()
        self.assertLess(diff, 1e-5, f"no-bias: max diff {diff:.2e}")

    def test_batched_einsum_outgoing(self):
        """Triton batched einsum (outgoing) matches torch.einsum."""
        device = "cuda"
        D, B = 128, 1
        for L in (64, 128):
            torch.manual_seed(0)
            a = torch.randn(D, B, L, L, device=device, dtype=torch.float32)
            b = torch.randn(D, B, L, L, device=device, dtype=torch.float32)

            out = trimul_batched_einsum_fp32(a, b, outgoing=True)
            ref = torch.einsum("dbik,dbjk->dbij", a, b)

            diff = (out - ref).abs().max().item()
            self.assertLess(diff, 1e-4, f"outgoing L={L}: max diff {diff:.2e}")

    def test_batched_einsum_incoming(self):
        """Triton batched einsum (incoming) matches torch.einsum."""
        device = "cuda"
        D, B = 128, 1
        for L in (64, 128):
            torch.manual_seed(0)
            a = torch.randn(D, B, L, L, device=device, dtype=torch.float32)
            b = torch.randn(D, B, L, L, device=device, dtype=torch.float32)

            out = trimul_batched_einsum_fp32(a, b, outgoing=False)
            ref = torch.einsum("dbki,dbkj->dbij", a, b)

            diff = (out - ref).abs().max().item()
            self.assertLess(diff, 1e-4, f"incoming L={L}: max diff {diff:.2e}")


if __name__ == "__main__":
    unittest.main()
