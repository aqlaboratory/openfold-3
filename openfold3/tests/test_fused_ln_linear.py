# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
"""Numerical correctness for the Triton-fused LayerNorm -> Linear op.

Compares the fused forward (and the autograd-decomposed backward)
against ``F.linear(F.layer_norm(x))`` for the c_in / c_out / dtype
combinations the model actually uses.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch
import torch.nn as nn

from openfold3.core.kernels.triton.fused_ln_linear import pair_ln_linear_inference
from openfold3.core.model.primitives.fused_ln_linear import (
    FusedLNLinear,
    register_legacy_remap_hook,
)
from openfold3.tests.compare_utils import skip_unless_cuda_available

# (c_in, c_out, ln_create_offset, linear_bias) covering the wired sites.
CASES = [
    # OpenFold3 outer trunk z, NoisyPositionEmbedder z
    (128, 128, True, False),
    # NoisyPositionEmbedder pair (c_atom_pair=16 in default config; pick a
    # small N to ensure the c_out << 16-floor BLOCK_N path works).
    (128, 16, False, False),
    # DiffusionConditioning z (267 = c_z + relpos_dims)
    (267, 128, False, False),
    # DiffusionConditioning n (c_fourier_emb=256)
    (256, 384, False, False),
    # OpenFold3 outer trunk s
    (384, 384, True, False),
]

# Per-call sample sizes: pair-tensor scale (256*256=65536 rows) + small.
M_VALUES = [1024, 65536]


class TestFusedLNLinearCheckpointCompatibility(unittest.TestCase):
    def test_public_checkpoint_keys_are_remapped(self):
        parent = nn.Module()
        parent.fused = FusedLNLinear(
            8,
            4,
            ln_create_offset=True,
            linear_bias=True,
        )
        with mock.patch(
            "openfold3.core.model.primitives.fused_ln_linear."
            "is_fused_ln_linear_enabled",
            return_value=True,
        ):
            register_legacy_remap_hook(
                parent,
                [("layer_norm", "linear", "fused")],
            )

        legacy_state = {
            "layer_norm.weight": torch.randn(8),
            "layer_norm.bias": torch.randn(8),
            "linear.weight": torch.randn(4, 8),
            "linear.bias": torch.randn(4),
        }
        parent.load_state_dict(legacy_state)

        torch.testing.assert_close(parent.fused.ln_weight, legacy_state["layer_norm.weight"])
        torch.testing.assert_close(parent.fused.ln_bias, legacy_state["layer_norm.bias"])
        torch.testing.assert_close(parent.fused.weight, legacy_state["linear.weight"])
        torch.testing.assert_close(parent.fused.bias, legacy_state["linear.bias"])


@skip_unless_cuda_available()
class TestFusedLNLinearForward(unittest.TestCase):
    """Forward correctness across c_in / c_out / dtype / sample-count."""

    def _check(
        self, c_in, c_out, ln_offset, lin_bias, M, dtype, atol, rtol=0.0,
        allow_tf32=None,
    ):
        torch.manual_seed(0)
        device = "cuda"
        x = torch.randn(M, c_in, dtype=dtype, device=device)
        m = FusedLNLinear(
            c_in,
            c_out,
            ln_create_offset=ln_offset,
            linear_bias=lin_bias,
            linear_init="default",
        ).to(device).to(dtype)
        # 'final' init zeros the weight; randomize for a non-trivial test.
        with torch.no_grad():
            m.weight.normal_(0, 0.5)
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        if allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        try:
            y_fused = m(x)
            y_ref = m.reference_forward(x)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
        diff = (y_fused - y_ref).abs().max().item()
        ref_scale = y_ref.abs().max().item()
        bound = atol + rtol * ref_scale
        self.assertLess(
            diff,
            bound,
            f"c_in={c_in} c_out={c_out} ln_offset={ln_offset} lin_bias={lin_bias} "
            f"M={M} dtype={dtype} max_abs={diff:.3e} > {bound:.3e} "
            f"(scale={ref_scale:.3f})",
        )

    def test_fp32(self):
        for c_in, c_out, ln_off, lin_b in CASES:
            for M in M_VALUES:
                with self.subTest(
                    c_in=c_in, c_out=c_out, ln_offset=ln_off, lin_bias=lin_b, M=M
                ):
                    # fp32 inputs but tl.dot uses TF32 (10-bit mantissa)
                    # for tensor-core throughput. Relative drift ~1e-3
                    # is the matched-input bound.
                    self._check(
                        c_in, c_out, ln_off, lin_b, M, torch.float32,
                        atol=5e-3, rtol=2e-3,
                        allow_tf32=True,
                    )

    def test_fp32_deterministic_precision(self):
        for c_in, c_out, ln_off, lin_b in CASES:
            for M in M_VALUES:
                with self.subTest(
                    c_in=c_in, c_out=c_out, ln_offset=ln_off, lin_bias=lin_b, M=M
                ):
                    self._check(
                        c_in, c_out, ln_off, lin_b, M, torch.float32,
                        atol=1e-4, rtol=1e-5,
                        allow_tf32=False,
                    )

    def test_bf16(self):
        for c_in, c_out, ln_off, lin_b in CASES:
            for M in M_VALUES:
                with self.subTest(
                    c_in=c_in, c_out=c_out, ln_offset=ln_off, lin_bias=lin_b, M=M
                ):
                    # bf16 has ~7-bit mantissa. cuBLAS bf16 GEMM and Triton
                    # tl.dot differ by reduction order, so the reasonable
                    # bound is relative — ~3% of the output scale covers
                    # both the rounding and the order-of-summation drift.
                    self._check(
                        c_in, c_out, ln_off, lin_b, M, torch.bfloat16,
                        atol=1e-2, rtol=3e-2,
                    )

    def test_higher_rank_input(self):
        """Pair-tensor shape (B, S, N, N, c_in) preserves leading dims."""
        torch.manual_seed(1)
        device = "cuda"
        x = torch.randn(1, 1, 256, 256, 128, dtype=torch.float32, device=device)
        m = FusedLNLinear(128, 128, ln_create_offset=True, linear_bias=False).to(device)
        with torch.no_grad():
            m.weight.normal_(0, 0.5)
        y_fused = m(x)
        y_ref = m.reference_forward(x)
        self.assertEqual(y_fused.shape, y_ref.shape)
        # TF32 in tl.dot ~ 1e-3 relative drift.
        scale = y_ref.abs().max().item()
        self.assertLess((y_fused - y_ref).abs().max().item(), 5e-3 + 2e-3 * scale)

    def test_large_c_in_falls_back(self):
        """c_in > 512 routes to F.layer_norm + F.linear fallback."""
        torch.manual_seed(2)
        device = "cuda"
        c_in = 833
        x = torch.randn(64, c_in, dtype=torch.float32, device=device)
        m = FusedLNLinear(c_in, 384, ln_create_offset=False, linear_bias=False).to(device)
        with torch.no_grad():
            m.weight.normal_(0, 0.5)
        # Forward must succeed and match the reference exactly (since both go
        # through the same F.layer_norm + F.linear path).
        y_fused = m(x)
        y_ref = m.reference_forward(x)
        self.assertEqual((y_fused - y_ref).abs().max().item(), 0.0)

    def test_pair_ln_linear_inference_matches_transposed_pair_view(self):
        """Pair LN-linear must avoid copies while matching transposed views."""
        torch.manual_seed(4)
        device = "cuda"
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            x_base = torch.randn(1, 96, 96, 128, device=device)
            gamma = torch.randn(128, device=device)
            beta = torch.randn(128, device=device)
            weight = torch.randn(128, 128, device=device) * 0.02
            bias = torch.randn(128, device=device) * 0.02

            with torch.inference_mode():
                for x in (x_base, x_base.transpose(-2, -3)):
                    y = pair_ln_linear_inference(x, gamma, beta, weight, bias)
                    ref = torch.nn.functional.linear(
                        torch.nn.functional.layer_norm(x, (128,), gamma, beta, 1e-5),
                        weight,
                        bias,
                    ).reshape(-1, 128)
                    self.assertEqual(y.shape, ref.shape)
                    self.assertLess((y - ref).abs().max().item(), 2e-5)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32


@skip_unless_cuda_available()
class TestFusedLNLinearBackward(unittest.TestCase):
    """Autograd-decomposed backward must match an autograd reference."""

    def test_grads_match_reference(self):
        """Backprop through the fused module with an externally-supplied
        upstream gradient, then compare to autograd through the
        reference path with the *same* upstream gradient. This isolates
        the backward correctness from the TF32 forward drift (running
        ``(y**2).sum().backward()`` would diverge through ``y``).
        """
        torch.manual_seed(3)
        device = "cuda"
        c_in, c_out = 128, 64
        m = FusedLNLinear(c_in, c_out, ln_create_offset=True, linear_bias=False).to(
            device
        )
        with torch.no_grad():
            m.weight.normal_(0, 0.5)

        x_a = torch.randn(64, c_in, dtype=torch.float32, device=device, requires_grad=True)
        x_b = x_a.detach().clone().requires_grad_()
        # Upstream gradient — same for both paths.
        gy = torch.randn(64, c_out, dtype=torch.float32, device=device)

        y_fused = m(x_a)
        y_fused.backward(gy)
        gx_fused = x_a.grad.clone()
        gln_fused = m.ln_weight.grad.clone()
        glnb_fused = m.ln_bias.grad.clone()
        gw_fused = m.weight.grad.clone()
        for p in m.parameters():
            p.grad = None

        y_ref = m.reference_forward(x_b)
        y_ref.backward(gy)
        # Backward path is fp32 autograd-decomposed for both fused
        # (manual) and ref (autograd). Diffs should be at fp32 ULP scale.
        self.assertLess((gx_fused - x_b.grad).abs().max().item(), 5e-4)
        self.assertLess((gln_fused - m.ln_weight.grad).abs().max().item(), 5e-3)
        self.assertLess((glnb_fused - m.ln_bias.grad).abs().max().item(), 5e-3)
        self.assertLess((gw_fused - m.weight.grad).abs().max().item(), 5e-3)


if __name__ == "__main__":
    unittest.main()
