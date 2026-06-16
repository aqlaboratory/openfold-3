# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
"""Numerical correctness for the fused SwiGLU pair-transition Triton kernel.

Compares the fused forward (and the in-place residual variant) against the
eager ``LayerNorm -> SwiGLU -> Linear -> *mask`` reference for the c_in / n /
dtype combinations the pair transition uses.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from openfold3.core.model.primitives.fused_swiglu_transition import (
    fused_swiglu_transition,
)
from openfold3.tests.compare_utils import skip_unless_cuda_available

# (c_in, n): pairformer pair_transition is (128, 4); also cover (128, 2) and
# (64, 2) for generality.
CASES = [(128, 4), (128, 2), (64, 2)]
# N so that M = N*N >= _MIN_M (4096) to exercise the kernel, plus a large one.
N_VALUES = [70, 256]


def _eager(x, g, b, wa, wb, wo, mask, residual):
    xn = F.layer_norm(x, (x.shape[-1],), g, b)
    h = F.silu(F.linear(xn, wa)) * F.linear(xn, wb)
    o = F.linear(h, wo) * mask
    if residual is not None:
        o = residual + o
    return o


@skip_unless_cuda_available()
class TestFusedSwiGLUTransition(unittest.TestCase):
    def _weights(self, c, n, dtype, device):
        torch.manual_seed(0)
        H = n * c
        g = torch.randn(c, dtype=dtype, device=device)
        b = torch.randn(c, dtype=dtype, device=device)
        wa = torch.randn(H, c, dtype=dtype, device=device) / c**0.5
        wb = torch.randn(H, c, dtype=dtype, device=device) / c**0.5
        wo = torch.randn(c, H, dtype=dtype, device=device) / H**0.5
        return g, b, wa, wb, wo

    def _check(self, c, n, N, dtype, atol, rtol):
        device = "cuda"
        g, b, wa, wb, wo = self._weights(c, n, dtype, device)
        x = torch.randn(1, N, N, c, dtype=dtype, device=device) * 0.5
        mask = torch.ones(1, N, N, 1, dtype=dtype, device=device)
        with torch.inference_mode():
            y_ref = _eager(x, g, b, wa, wb, wo, mask, None)
            y_fused = fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
        self._assert_close(y_fused, y_ref, atol, rtol, f"fwd c={c} n={n} N={N} {dtype}")

        # In-place residual: writes z + transition(z) into z.
        with torch.inference_mode():
            z = x.clone()
            y_ip = fused_swiglu_transition(z, g, b, wa, wb, wo, mask=mask, residual=z)
            y_ref_ip = _eager(x, g, b, wa, wb, wo, mask, x)
        self.assertEqual(y_ip.data_ptr(), z.data_ptr(), "in-place must alias z")
        self._assert_close(y_ip, y_ref_ip, atol, rtol, f"inplace c={c} n={n} N={N}")

    def _assert_close(self, got, ref, atol, rtol, msg):
        diff = (got.float() - ref.float()).abs().max().item()
        bound = atol + rtol * ref.float().abs().max().item()
        self.assertLess(diff, bound, f"{msg}: max_abs={diff:.3e} > {bound:.3e}")

    def test_fp32(self):
        for c, n in CASES:
            for N in N_VALUES:
                with self.subTest(c=c, n=n, N=N):
                    self._check(c, n, N, torch.float32, atol=1e-2, rtol=5e-3)

    def test_bf16(self):
        for c, n in CASES:
            for N in N_VALUES:
                with self.subTest(c=c, n=n, N=N):
                    self._check(c, n, N, torch.bfloat16, atol=2e-2, rtol=5e-2)

    def test_small_M_falls_back_to_eager(self):
        # M = N*N below _MIN_M(4096) -> eager path; must equal the reference
        # exactly (same code path, no TF32 kernel).
        c, n, N, dtype = 128, 4, 32, torch.float32
        device = "cuda"
        g, b, wa, wb, wo = self._weights(c, n, dtype, device)
        x = torch.randn(1, N, N, c, dtype=dtype, device=device) * 0.5
        mask = torch.ones(1, N, N, 1, dtype=dtype, device=device)
        with torch.inference_mode():
            y_ref = _eager(x, g, b, wa, wb, wo, mask, None)
            y = fused_swiglu_transition(x, g, b, wa, wb, wo, mask=mask)
        # Eager fallback uses cuBLAS like the reference -> very tight.
        self.assertLess((y - y_ref).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
