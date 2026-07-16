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

"""Tests for V1 custom cap-blocked triangle attention."""

import os
import unittest
from unittest import mock

import torch

from openfold3.core.model.layers.triangular_attention import TriangleAttention
from openfold3.core.model.primitives.fused_tri_attn import fused_tri_attn_v1_forward


def _skip_no_cuda(fn):
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA required")(fn)


def _build_module(
    starting: bool,
    c_in: int = 128,
    c_hidden: int = 32,
    no_heads: int = 4,
):
    module = TriangleAttention(
        c_in=c_in,
        c_hidden=c_hidden,
        no_heads=no_heads,
        starting=starting,
    )
    with torch.no_grad():
        torch.manual_seed(1234 + int(starting))
        module.mha.linear_o.weight.normal_(mean=0.0, std=0.02)
        if module.mha.linear_g is not None:
            module.mha.linear_g.weight.normal_(mean=0.0, std=0.02)
    return module.cuda().eval()


def _forward(module, x, mask, *, v1: bool, use_cueq: bool = True):
    old_env = {
        key: os.environ.get(key)
        for key in (
            "OPENFOLD3_FUSED_TRI_ATTN_V1",
            "OPENFOLD3_TRI_ATTN_CHUNK_CAP",
        )
    }
    os.environ["OPENFOLD3_FUSED_TRI_ATTN_V1"] = "1" if v1 else "0"
    os.environ["OPENFOLD3_TRI_ATTN_CHUNK_CAP"] = "128"
    try:
        with torch.no_grad():
            return module(
                x,
                mask=mask,
                chunk_size=128,
                use_cueq_triangle_kernels=use_cueq,
            )
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestFusedTriAttnV1(unittest.TestCase):
    @_skip_no_cuda
    def test_v1_template_shape_matches_eager_starting_and_ending(self):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for starting in (True, False):
                torch.manual_seed(210 + int(starting))
                module = _build_module(
                    starting=starting,
                    c_in=64,
                    c_hidden=16,
                )
                x = torch.randn(
                    1,
                    160,
                    160,
                    64,
                    device="cuda",
                    dtype=torch.float32,
                )
                mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)
                mask[:, :, -11:] = 0

                ref = _forward(module, x, mask, v1=False, use_cueq=False)
                out = _forward(module, x, mask, v1=True, use_cueq=False)

                diff = (ref - out).abs().max().item()
                self.assertLessEqual(diff, 2e-5, f"template V1 max diff {diff}")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @_skip_no_cuda
    def test_v1_template_shape_grouped_tf32_matches_eager(self):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.manual_seed(213)
            module = _build_module(
                starting=True,
                c_in=64,
                c_hidden=16,
            )
            x = torch.randn(1, 160, 160, 64, device="cuda", dtype=torch.float32)
            mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)

            ref = _forward(module, x, mask, v1=False, use_cueq=False)
            out = _forward(module, x, mask, v1=True, use_cueq=False)

            diff = (ref - out).abs().max().item()
            self.assertLessEqual(diff, 5e-4, f"template grouped TF32 diff {diff}")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @_skip_no_cuda
    def test_v1_template_shape_residual_handles_pairblock_ending_view(self):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.manual_seed(212)
            module = _build_module(
                starting=True,
                c_in=64,
                c_hidden=16,
            )
            z = torch.randn(1, 160, 160, 64, device="cuda", dtype=torch.float32)
            mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)

            with torch.no_grad():
                z_view = z.transpose(-2, -3)
                mask_view = mask.transpose(-1, -2)
                update = fused_tri_attn_v1_forward(
                    module,
                    z_view,
                    mask_view,
                    inplace_safe=True,
                    use_cueq_triangle_kernels=False,
                    chunk_size=128,
                )
                self.assertIsNotNone(update)
                expected = z_view + update

                residual_base = z.clone()
                residual_view = residual_base.transpose(-2, -3)
                out = fused_tri_attn_v1_forward(
                    module,
                    residual_view,
                    mask_view,
                    inplace_safe=True,
                    use_cueq_triangle_kernels=False,
                    chunk_size=128,
                    residual=residual_view,
                )

            self.assertIs(out, residual_view)
            self.assertEqual(out.data_ptr(), residual_view.data_ptr())
            diff = (expected - out).abs().max().item()
            self.assertLessEqual(diff, 2e-5, f"template residual max diff {diff}")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @_skip_no_cuda
    def test_v1_direct_path_returns_output(self):
        torch.manual_seed(201)
        module = _build_module(starting=True)
        x = torch.randn(1, 160, 160, 128, device="cuda", dtype=torch.float32)
        mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            out = fused_tri_attn_v1_forward(
                module,
                x,
                mask,
                inplace_safe=False,
                use_cueq_triangle_kernels=True,
                chunk_size=128,
            )

        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.shape, x.shape)

    @_skip_no_cuda
    def test_v1_matches_eager_starting_and_ending(self):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for starting in (True, False):
                torch.manual_seed(200 + int(starting))
                module = _build_module(starting=starting)
                x = torch.randn(
                    1,
                    160,
                    160,
                    128,
                    device="cuda",
                    dtype=torch.float32,
                )
                mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)

                ref = _forward(module, x, mask, v1=False, use_cueq=False)
                out = _forward(module, x, mask, v1=True)

                diff = (ref - out).abs().max().item()
                self.assertLessEqual(diff, 2e-5, f"V1 max diff {diff}")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @_skip_no_cuda
    def test_v1_grouped_tf32_matches_eager_tolerance(self):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.manual_seed(204)
            module = _build_module(starting=True)
            x = torch.randn(1, 160, 160, 128, device="cuda", dtype=torch.float32)
            mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)

            ref = _forward(module, x, mask, v1=False, use_cueq=False)
            out = _forward(module, x, mask, v1=True, use_cueq=False)

            diff = (ref - out).abs().max().item()
            self.assertLessEqual(diff, 5e-4, f"V1 grouped TF32 max diff {diff}")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @_skip_no_cuda
    def test_v1_explicit_dense_mask_matches_no_mask(self):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.manual_seed(205)
            module = _build_module(starting=True)
            x = torch.randn(1, 160, 160, 128, device="cuda", dtype=torch.float32)
            mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)

            ref = _forward(module, x, None, v1=True, use_cueq=False)
            out = _forward(module, x, mask, v1=True, use_cueq=False)

            diff = (ref - out).abs().max().item()
            self.assertLessEqual(diff, 1e-6, f"V1 dense-mask max diff {diff}")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @_skip_no_cuda
    def test_v1_explicit_padding_mask_matches_eager(self):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            for starting in (True, False):
                torch.manual_seed(206 + int(starting))
                module = _build_module(starting=starting)
                x = torch.randn(
                    1, 160, 160, 128, device="cuda", dtype=torch.float32
                )
                mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)
                mask[:, :, -13:] = 0
                mask[:, -7:, :] = 0

                ref = _forward(module, x, mask, v1=False, use_cueq=False)
                out = _forward(module, x, mask, v1=True, use_cueq=False)

                diff = (ref - out).abs().max().item()
                self.assertLessEqual(diff, 2e-5, f"V1 masked max diff {diff}")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @_skip_no_cuda
    def test_v1_residual_fusion_matches_update_and_mutates_in_place(self):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.manual_seed(207)
            module = _build_module(starting=True)
            x = torch.randn(1, 160, 160, 128, device="cuda", dtype=torch.float32)
            mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)

            with torch.no_grad():
                update = fused_tri_attn_v1_forward(
                    module,
                    x,
                    mask,
                    inplace_safe=True,
                    use_cueq_triangle_kernels=False,
                    chunk_size=128,
                )
                self.assertIsNotNone(update)
                expected = x + update

                residual = x.clone()
                out = fused_tri_attn_v1_forward(
                    module,
                    residual,
                    mask,
                    inplace_safe=True,
                    use_cueq_triangle_kernels=False,
                    chunk_size=128,
                    residual=residual,
                )

            self.assertIs(out, residual)
            self.assertEqual(out.data_ptr(), residual.data_ptr())
            diff = (expected - out).abs().max().item()
            self.assertLessEqual(diff, 2e-5, f"V1 residual max diff {diff}")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @_skip_no_cuda
    def test_v1_residual_fusion_handles_pairblock_ending_view(self):
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            torch.manual_seed(208)
            module = _build_module(starting=True)
            z = torch.randn(1, 160, 160, 128, device="cuda", dtype=torch.float32)
            mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)

            with torch.no_grad():
                z_view = z.transpose(-2, -3)
                mask_view = mask.transpose(-1, -2)
                update = fused_tri_attn_v1_forward(
                    module,
                    z_view,
                    mask_view,
                    inplace_safe=True,
                    use_cueq_triangle_kernels=False,
                    chunk_size=128,
                )
                self.assertIsNotNone(update)
                expected = z_view + update

                residual_base = z.clone()
                residual_view = residual_base.transpose(-2, -3)
                out = fused_tri_attn_v1_forward(
                    module,
                    residual_view,
                    mask_view,
                    inplace_safe=True,
                    use_cueq_triangle_kernels=False,
                    chunk_size=128,
                    residual=residual_view,
                )

            self.assertIs(out, residual_view)
            self.assertEqual(out.data_ptr(), residual_view.data_ptr())
            diff = (expected - out).abs().max().item()
            self.assertLessEqual(diff, 2e-5, f"V1 ending-view max diff {diff}")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @_skip_no_cuda
    def test_v1_kernel_is_not_used_with_grad_enabled(self):
        torch.manual_seed(209)
        module = _build_module(starting=True).train()
        x = torch.randn(
            1,
            160,
            160,
            128,
            device="cuda",
            dtype=torch.float32,
            requires_grad=True,
        )
        mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)
        residual = x.detach().clone()
        residual_before = residual.clone()

        with (
            mock.patch(
                "openfold3.core.model.primitives.fused_tri_attn."
                "flash_tri_attn_v1_block",
                side_effect=AssertionError("inference kernel called with grad enabled"),
            ),
            torch.enable_grad(),
        ):
            out = fused_tri_attn_v1_forward(
                module,
                x,
                mask,
                inplace_safe=False,
                use_cueq_triangle_kernels=False,
                chunk_size=128,
                residual=residual,
            )

        self.assertIsNone(out)
        torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)

    @_skip_no_cuda
    def test_v1_runs_without_cueq_flag(self):
        torch.manual_seed(202)
        module = _build_module(starting=True)
        x = torch.randn(1, 160, 160, 128, device="cuda", dtype=torch.float32)
        mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)

        ref = _forward(module, x, mask, v1=False, use_cueq=False)
        out = _forward(module, x, mask, v1=True, use_cueq=False)

        diff = (ref - out).abs().max().item()
        self.assertLessEqual(diff, 2e-5, f"V1 no-cueq max diff {diff}")

    @_skip_no_cuda
    def test_v1_does_not_call_cueq_triangle_attention(self):
        torch.manual_seed(203)
        module = _build_module(starting=True)
        x = torch.randn(1, 160, 160, 128, device="cuda", dtype=torch.float32)
        mask = torch.ones(1, 160, 160, device="cuda", dtype=torch.float32)

        if not hasattr(torch.ops, "cuequivariance"):
            out = _forward(module, x, mask, v1=True, use_cueq=False)
            self.assertEqual(out.shape, x.shape)
            return

        original = getattr(torch.ops.cuequivariance, "triangle_attention", None)
        if original is None:
            out = _forward(module, x, mask, v1=True, use_cueq=False)
            self.assertEqual(out.shape, x.shape)
            return

        def fail(*args, **kwargs):
            raise AssertionError("V1 must not call cuEq triangle_attention")

        torch.ops.cuequivariance.triangle_attention = fail
        try:
            out = _forward(module, x, mask, v1=True, use_cueq=True)
            self.assertEqual(out.shape, x.shape)
        finally:
            torch.ops.cuequivariance.triangle_attention = original


if __name__ == "__main__":
    unittest.main()
