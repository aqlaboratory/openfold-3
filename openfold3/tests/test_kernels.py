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

"""
Unit tests to compare components of OpenFold run with the DeepSpeed memory-efficient
attention kernel, DS4Sci_EvoformerAttention vs. a stock PyTorch attention
implementation.
"""

import os
from unittest import mock

import pytest
import torch
from torch.nn import functional as F

import openfold3.tests.compare_utils as compare_utils
from openfold3.core.model.latent.pairformer import PairFormerStack
from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicativeUpdate,
)
from openfold3.core.model.primitives.attention import Attention
from openfold3.core.model.primitives.initialization import lecun_normal_init_
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts
from openfold3.tests.data_utils import (
    random_attention_inputs,
)

# Needed to do backward for cuEq kernels with FP32
torch.backends.cuda.matmul.allow_tf32 = True
pytestmark = [pytest.mark.slow]

torch.backends.cuda.preferred_blas_library("cublas")


@compare_utils.skip_unless_cuda_available()
class TestKernels:
    def _compare_attn_kernel_forward(
        self,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=False,
        use_triton_triangle_kernels=False,
        dtype=torch.float32,
    ):
        """Compare attention with and without using DeepSpeed Evoformer kernel."""
        batch_size = consts.batch_size
        n_seq = 18
        n_res = 200  # Avoid cuEq seq len constraints
        c_hidden = 32
        no_heads = 4
        eps = 2e-2

        q, kv, mask, biases = random_attention_inputs(
            batch_size=batch_size,
            n_seq=n_seq,
            n=n_res,
            no_heads=no_heads,
            c_hidden=c_hidden,
            dtype=dtype,
        )

        a = Attention(
            c_hidden,
            c_hidden,
            c_hidden,
            c_hidden,
            no_heads,
        ).cuda()

        # Change output params init for testing since they are initialized with 'final'
        # init (zeros) Otherwise both will just return zero.
        with torch.no_grad():
            lecun_normal_init_(a.linear_g.weight)
            lecun_normal_init_(a.linear_o.weight)

            real_out = a(q, kv, biases=biases).cpu()

            kernel_out = a(
                q,
                kv,
                biases=biases,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
            ).cpu()

        err = torch.max(torch.abs(kernel_out - real_out))
        assert err < eps, f"Error: {err}"

    @compare_utils.skip_unless_ds4s_installed()
    def test_dsk_forward_bf16(self):
        self._compare_attn_kernel_forward(
            use_deepspeed_evo_attention=True,
            use_cueq_triangle_kernels=False,
            dtype=torch.bfloat16,
        )

    @compare_utils.skip_unless_ds4s_installed()
    def test_dsk_forward_fp32(self):
        self._compare_attn_kernel_forward(
            use_deepspeed_evo_attention=True,
            use_cueq_triangle_kernels=False,
            dtype=torch.float32,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_cueq_forward_fp32(self):
        self._compare_attn_kernel_forward(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.float32,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_cueq_forward_bf16(self):
        self._compare_attn_kernel_forward(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.bfloat16,
        )

    @compare_utils.skip_unless_triton_installed()
    def test_triton_forward_bf16(self):
        self._compare_attn_kernel_forward(
            use_triton_triangle_kernels=True,
            dtype=torch.bfloat16,
        )

    @compare_utils.skip_unless_triton_installed()
    def test_triton_forward_fp32(self):
        self._compare_attn_kernel_forward(
            use_triton_triangle_kernels=True,
            dtype=torch.float32,
        )

    def _compare_attn_kernel_backward(
        self,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=False,
        use_triton_triangle_kernels=False,
        dtype=torch.float32,
    ):
        """
        Compare backward pass for regular attention vs. DeepSpeed Evoformer kernel.
        """
        batch_size = consts.batch_size
        n_seq = 18
        n_res = 200  # Avoid cuEq seq len constraints
        c_hidden = 32
        no_heads = 4
        eps = consts.eps

        q, kv, _, biases = random_attention_inputs(
            batch_size=batch_size,
            n_seq=n_seq,
            n=n_res,
            no_heads=no_heads,
            c_hidden=c_hidden,
            requires_grad=True,
            dtype=dtype,
        )

        attn = Attention(
            c_hidden,
            c_hidden,
            c_hidden,
            c_hidden,
            no_heads,
        ).cuda()

        with torch.no_grad():
            lecun_normal_init_(attn.linear_g.weight)
            lecun_normal_init_(attn.linear_o.weight)

        def clone(t):
            # Create new params, clone values
            t = t.clone()
            if t.requires_grad:
                t.retain_grad()
            return t

        def init_attn():
            # Create new attention object with same initial weights
            a_clone = Attention(
                c_hidden,
                c_hidden,
                c_hidden,
                c_hidden,
                no_heads,
            ).cuda()

            a_clone.load_state_dict(attn.state_dict())
            return a_clone

        # Clone param values and run attention with DS kernel
        q_repro = clone(q)
        kv_repro = clone(kv)
        biases_repro = [clone(b) for b in biases]

        a_repro = init_attn()
        out_repro = a_repro(
            q_repro,
            kv_repro,
            biases=biases_repro,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_cueq_triangle_kernels=use_cueq_triangle_kernels,
            use_triton_triangle_kernels=use_triton_triangle_kernels,
        )
        loss_repro = torch.mean(out_repro)
        loss_repro.backward()

        q_gt = clone(q)
        kv_gt = clone(kv)
        biases_gt = [clone(b) for b in biases]

        # Clone param values and run attention without DS kernel
        a_gt = init_attn()
        out_gt = a_gt(q_gt, kv_gt, biases=biases_gt)

        loss_gt = torch.mean(out_gt)
        loss_gt.backward()

        # Compare the grads of attention inputs
        pairs = zip(
            [q_repro, kv_repro, biases_repro[1]],
            [q_gt, kv_gt, biases_gt[1]],
            strict=False,
        )
        for i, item in enumerate(pairs):
            t_repro, t_gt = item
            err = torch.max(torch.abs(t_repro.grad.cpu() - t_gt.grad.cpu()))
            assert err < eps, f"Error item #{i}: {err}"

        # Compare the grads of model weights
        a_repro_params = dict(a_repro.named_parameters())
        a_gt_params = dict(a_gt.named_parameters())
        for name in a_gt_params:
            t_repro = a_repro_params[name]
            t_gt = a_gt_params[name]
            err = torch.max(torch.abs(t_repro.grad.cpu() - t_gt.grad.cpu()))
            assert err < eps, f"Error item {name}: {err}"

    @compare_utils.skip_unless_ds4s_installed()
    def test_dsk_backward_bf16(self):
        self._compare_attn_kernel_backward(
            use_deepspeed_evo_attention=True,
            use_cueq_triangle_kernels=False,
            dtype=torch.bfloat16,
        )

    @compare_utils.skip_unless_ds4s_installed()
    def test_dsk_backward_fp32(self):
        self._compare_attn_kernel_backward(
            use_deepspeed_evo_attention=True,
            use_cueq_triangle_kernels=False,
            dtype=torch.float32,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_cueq_backward_fp32(self):
        self._compare_attn_kernel_backward(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.float32,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_cueq_backward_bf16(self):
        self._compare_attn_kernel_backward(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.bfloat16,
        )

    @compare_utils.skip_unless_triton_installed()
    def test_triton_backward_bf16(self):
        self._compare_attn_kernel_backward(
            use_triton_triangle_kernels=True,
            dtype=torch.bfloat16,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_cueq_tri_mult_fwd(self):
        batch = consts.batch_size
        n_tmpl = 20
        seq_len = 84
        c_z = 128
        c_hidden = 128
        outgoing = True
        tm = TriangleMultiplicativeUpdate(
            c_z=c_z,
            c_hidden=c_hidden,
            _outgoing=outgoing,
        ).to("cuda")
        z = torch.randn(batch, n_tmpl, seq_len, seq_len, c_z).to("cuda")
        mask = torch.ones(batch, n_tmpl, seq_len, seq_len).to("cuda")
        with torch.no_grad():
            lecun_normal_init_(tm.linear_g.weight)
            lecun_normal_init_(tm.linear_z.weight)
            lecun_normal_init_(tm.linear_a_p.weight)
            lecun_normal_init_(tm.linear_a_g.weight)
            lecun_normal_init_(tm.linear_b_p.weight)
            lecun_normal_init_(tm.linear_b_g.weight)

            fwd_reg = tm(
                z=z,
                mask=mask,
                use_cueq_triangle_kernels=False,
            )
            fwd_cueq = tm(
                z=z,
                mask=mask,
                use_cueq_triangle_kernels=True,
            )
        err = torch.max(torch.abs(fwd_reg - fwd_cueq))
        eps = 2e-2
        assert err < eps, f"Error: {err}"

    @compare_utils.skip_unless_cueq_installed()
    def test_cueq_tri_mult_bwd(self):
        batch = consts.batch_size
        n_tmpl = 20
        seq_len = 84
        c_z = 128
        c_hidden = 128
        outgoing = True
        eps = consts.eps

        tm = TriangleMultiplicativeUpdate(
            c_z=c_z,
            c_hidden=c_hidden,
            _outgoing=outgoing,
        ).to("cuda")
        z = torch.randn(batch, n_tmpl, seq_len, seq_len, c_z, requires_grad=True).to(
            "cuda"
        )
        mask = torch.ones(batch, n_tmpl, seq_len, seq_len, requires_grad=False).to(
            "cuda"
        )
        with torch.no_grad():
            lecun_normal_init_(tm.linear_g.weight)
            lecun_normal_init_(tm.linear_z.weight)
            lecun_normal_init_(tm.linear_a_p.weight)
            lecun_normal_init_(tm.linear_a_g.weight)
            lecun_normal_init_(tm.linear_b_p.weight)
            lecun_normal_init_(tm.linear_b_g.weight)

        def clone(t):
            # Create new params, clone values
            t = t.clone()
            if t.requires_grad:
                t.retain_grad()
            return t

        def init_tm():
            # Create new attention object with same initial weights
            tm_clone = TriangleMultiplicativeUpdate(
                c_z=c_z,
                c_hidden=c_hidden,
                _outgoing=outgoing,
            ).to("cuda")

            tm_clone.load_state_dict(tm.state_dict())
            return tm_clone

        z_repro = clone(z)
        mask_repro = clone(mask)
        tm_repro = init_tm()
        out_repro = tm_repro(
            z=z_repro,
            mask=mask_repro,
            use_cueq_triangle_kernels=True,
        )
        loss_repro = torch.mean(out_repro)
        loss_repro.backward()

        z_gt = clone(z)
        mask_gt = clone(mask)
        tm_gt = init_tm()
        out_gt = tm_gt(
            z=z_gt,
            mask=mask_gt,
            use_cueq_triangle_kernels=False,
        )
        loss_gt = torch.mean(out_gt)
        loss_gt.backward()
        # Compare the grads of attention inputs
        tm_repro_params = dict(tm_repro.named_parameters())
        tm_gt_params = dict(tm_gt.named_parameters())
        for name in tm_gt_params:
            t_repro = tm_repro_params[name]
            t_gt = tm_gt_params[name]
            err = torch.max(torch.abs(t_repro.grad.cpu() - t_gt.grad.cpu()))
            assert err < eps, f"Error item {name}: {err}"

    def _initialize_model_weights(self, model):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                with torch.no_grad():
                    lecun_normal_init_(module.weight)

    def _compare_pairformer(
        self,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=False,
        use_triton_triangle_kernels=False,
        dtype=torch.float32,
        chunk_size=None,
        eps=2e-2,
    ):
        """
        Compare Pairformer output with and without using optimized kernels
        Set dtype to confirm the kernel can be used during both training (BF16)
        and inference (FP32), since the kernels can run with either BF16 or FP16
        precision. Notably, for cueq kernels when use_cueq_triangle_kernels is
        true, both the triangle_attention and triangle_multiplicative_update
        kernels will be active

        TODO: Change the test to use a loaded Pairformer block from the trained model
          instead of a newly initialized block.
        """
        batch_size = consts.batch_size
        if chunk_size is not None and (
            use_deepspeed_evo_attention or use_triton_triangle_kernels
        ):
            # Chunk tuning is not supported with batch size > 1 for these kernels
            batch_size = 1

        n_res = 200  # Avoid cuEq seq len constraints
        c_s = consts.c_s
        c_z = consts.c_z
        c_hidden_pair_bias = 24
        no_heads_pair_bias = 16
        c_hidden_mul = 128
        c_hidden_pair_att = 32
        no_heads_pair = 4
        no_blocks = 2
        transition_type = "swiglu"
        transition_n = 2
        pair_dropout = 0.25
        inf = 1e9

        block = (
            PairFormerStack(
                c_s=c_s,
                c_z=c_z,
                c_hidden_pair_bias=c_hidden_pair_bias,
                no_heads_pair_bias=no_heads_pair_bias,
                c_hidden_mul=c_hidden_mul,
                c_hidden_pair_att=c_hidden_pair_att,
                no_heads_pair=no_heads_pair,
                no_blocks=no_blocks,
                transition_type=transition_type,
                transition_n=transition_n,
                pair_dropout=pair_dropout,
                fuse_projection_weights=False,
                blocks_per_ckpt=None,
                inf=inf,
                tune_chunk_size=chunk_size is not None,
            )
            .eval()
            .to(device="cuda", dtype=dtype)
        )

        self._initialize_model_weights(block)

        s = torch.rand(batch_size, n_res, consts.c_s, device="cuda", dtype=dtype)
        z = torch.rand(batch_size, n_res, n_res, consts.c_z, device="cuda", dtype=dtype)

        s_mask = torch.randint(0, 2, (batch_size, n_res), device="cuda", dtype=dtype)
        z_mask = torch.randint(
            0, 2, (batch_size, n_res, n_res), device="cuda", dtype=dtype
        )

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out_repro_single, out_repro_pair = block(
                s=s,
                z=z,
                single_mask=s_mask,
                pair_mask=z_mask,
                use_deepspeed_evo_attention=False,
                chunk_size=None,  # Test against non-chunked version
            )

            # In practice, layer norms applied later in the network make any
            # kernel rounding errors negligible
            out_repro_single = F.layer_norm(out_repro_single, (consts.c_s,)).cpu()
            out_repro_pair = F.layer_norm(out_repro_pair, (consts.c_z,)).cpu()

            out_repro_single_ds, out_repro_pair_ds = block(
                s=s,
                z=z,
                single_mask=s_mask,
                pair_mask=z_mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
                chunk_size=chunk_size,
            )
            out_repro_single_ds = F.layer_norm(out_repro_single_ds, (consts.c_s,)).cpu()
            out_repro_pair_ds = F.layer_norm(out_repro_pair_ds, (consts.c_z,)).cpu()

            compare_utils.assert_mean_abs_diff_small(
                out_repro_single, out_repro_single_ds, eps
            )

            compare_utils.assert_mean_abs_diff_small(
                out_repro_pair, out_repro_pair_ds, eps
            )

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_pairformer_dsk_bf16(self):
        """Run Pairformer comparison test with BF16 precision."""
        self._compare_pairformer(
            use_deepspeed_evo_attention=True,
            use_cueq_triangle_kernels=False,
            dtype=torch.bfloat16,
            eps=4e-2,
        )

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_pairformer_dsk_fp32(self):
        """Run Pairformer comparison test with FP32 precision."""
        self._compare_pairformer(
            use_deepspeed_evo_attention=True,
            use_cueq_triangle_kernels=False,
            dtype=torch.float32,
            eps=2e-2,
        )

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_pairformer_dsk_fp32_chunk(self):
        """Run Pairformer comparison test with chunk tuning enabled."""
        self._compare_pairformer(
            use_deepspeed_evo_attention=True,
            use_cueq_triangle_kernels=False,
            dtype=torch.float32,
            chunk_size=4,
            eps=4e-2,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_compare_pairformer_cueq_bf16(self):
        """Run Pairformer comparison test with BF16 precision."""
        self._compare_pairformer(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.bfloat16,
            eps=2e-2,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_compare_pairformer_cueq_fp32(self):
        """Run Pairformer comparison test with FP32 precision."""
        self._compare_pairformer(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.float32,
            eps=2e-2,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_compare_pairformer_cueq_fp32_chunk(self):
        """Run Pairformer comparison test with chunk tuning enabled."""
        self._compare_pairformer(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.float32,
            chunk_size=4,
            eps=4e-2,
        )

    @compare_utils.skip_unless_triton_installed()
    def test_compare_pairformer_triton_bf16(self):
        """Run Pairformer comparison test with Triton kernel and BF16 precision."""
        self._compare_pairformer(
            use_triton_triangle_kernels=True,
            dtype=torch.bfloat16,
            eps=2e-2,
        )

    @compare_utils.skip_unless_triton_installed()
    def test_compare_pairformer_triton_fp32(self):
        """Run Pairformer comparison test with Triton kernel and FP32 precision."""
        self._compare_pairformer(
            use_triton_triangle_kernels=True,
            dtype=torch.float32,
            eps=2e-2,
        )

    @compare_utils.skip_unless_triton_installed()
    def test_compare_pairformer_triton_fp32_chunk(self):
        """Run Pairformer comparison test with Triton kernel and chunk tuning enabled."""
        self._compare_pairformer(
            use_triton_triangle_kernels=True,
            dtype=torch.float32,
            chunk_size=4,
            eps=4e-2,
        )

    def _compare_diffusion_transformer(
        self,
        use_deepspeed_evo_attention=False,
        use_fused_diffusion_attention=False,
        dtype=torch.float32,
        eps=2e-2,
    ):
        """
        Compare DiffusionTransformer output with and without using optimized kernels

        TODO: Change the test to use a loaded DiffusionTransformer block from the
          trained model instead of a newly initialized block.
        """
        batch_size = consts.batch_size
        n_sample = 5
        n_res = consts.n_res
        c_a = 768
        c_s = consts.c_s
        c_z = consts.c_z
        c_hidden = 48
        no_heads = 16
        no_blocks = 2
        n_transition = 2
        inf = 1e9

        block = (
            DiffusionTransformer(
                c_a=c_a,
                c_s=c_s,
                c_z=c_z,
                c_hidden=c_hidden,
                no_heads=no_heads,
                no_blocks=no_blocks,
                n_transition=n_transition,
                use_ada_layer_norm=True,
                n_query=None,
                n_key=None,
                inf=inf,
            )
            .eval()
            .to(device="cuda", dtype=dtype)
        )

        self._initialize_model_weights(block)

        a = torch.rand(batch_size, n_sample, n_res, c_a, device="cuda", dtype=dtype)
        s = torch.rand(
            batch_size, n_sample, n_res, consts.c_s, device="cuda", dtype=dtype
        )
        z = torch.rand(
            batch_size, 1, n_res, n_res, consts.c_z, device="cuda", dtype=dtype
        )

        mask = torch.randint(0, 2, (batch_size, 1, n_res), device="cuda", dtype=dtype)

        old_env = os.environ.get("OPENFOLD3_FUSED_DIFFUSION_ATTN")
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = "0"
                out_repro_a = block(
                    a=a,
                    s=s,
                    z=z,
                    mask=mask,
                    use_deepspeed_evo_attention=False,
                )

                # In practice, layer norms applied later in the network make any
                # kernel rounding errors negligible
                out_repro_a = F.layer_norm(out_repro_a, (c_a,)).cpu()

                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = (
                    "1" if use_fused_diffusion_attention else "0"
                )
                out_repro_a_ds = block(
                    a=a,
                    s=s,
                    z=z,
                    mask=mask,
                    use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                )
                out_repro_a_ds = F.layer_norm(out_repro_a_ds, (c_a,)).cpu()

                compare_utils.assert_mean_abs_diff_small(
                    out_repro_a, out_repro_a_ds, eps
                )
        finally:
            if old_env is None:
                os.environ.pop("OPENFOLD3_FUSED_DIFFUSION_ATTN", None)
            else:
                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = old_env

    @compare_utils.skip_unless_triton_installed()
    @compare_utils.skip_unless_cuda_available()
    def test_fused_diffusion_attention_dispatch_policy(self):
        """Fused diffusion attention should avoid slow single-sample medium N."""
        from openfold3.core.model.primitives.attention import (
            _can_use_fused_diffusion_attention,
        )

        old_min_tokens = os.environ.get("OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS")
        try:
            os.environ.pop("OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS", None)
            with torch.no_grad():
                for samples, n_token, expected in (
                    (1, 590, False),
                    (1, 1024, True),
                    (5, 590, True),
                ):
                    q = torch.empty(
                        1, samples, n_token, 768, device="cuda", dtype=torch.bfloat16
                    )
                    mask_bias = torch.empty(
                        1, 1, 1, 1, n_token, device="cuda", dtype=torch.bfloat16
                    )
                    pair_bias = torch.empty(
                        1,
                        1,
                        16,
                        n_token,
                        n_token,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    assert (
                        _can_use_fused_diffusion_attention(
                            q, q, [mask_bias, pair_bias], 16
                        )
                        is expected
                    )

                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS"] = "0"
                q = torch.empty(1, 1, 590, 768, device="cuda", dtype=torch.bfloat16)
                mask_bias = torch.empty(
                    1, 1, 1, 1, 590, device="cuda", dtype=torch.bfloat16
                )
                pair_bias = torch.empty(
                    1, 1, 16, 590, 590, device="cuda", dtype=torch.bfloat16
                )
                assert _can_use_fused_diffusion_attention(
                    q, q, [mask_bias, pair_bias], 16
                )
        finally:
            if old_min_tokens is None:
                os.environ.pop("OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS", None)
            else:
                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS"] = old_min_tokens

    def test_fused_diffusion_attention_default_cutoff_is_1024(self):
        """Lock the S=1 token cutoff at 1024.

        The microbench in scripts/dev/bench_diffusion_attn_module.py showed
        that the kernel does not beat eager at S=1 N<1024 in either bf16 or
        fp32, but wins from N>=1024 onward in fp32 (1.13x at N=1024,
        1.15x at N=1264). Keep the default cutoff at 1024 so the kernel
        only engages where it actually helps at single-sample.
        """
        from openfold3.core.model.primitives.attention import (
            _fused_diffusion_attention_min_tokens,
        )

        old = os.environ.pop("OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS", None)
        try:
            assert _fused_diffusion_attention_min_tokens() == 1024
        finally:
            if old is not None:
                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS"] = old

    @compare_utils.skip_unless_triton_installed()
    @compare_utils.skip_unless_cuda_available()
    def test_diffusion_transformer_pair_bias_cache_equivalence(self):
        """Cached pair_bias_h produces identical output to uncached.

        ``DiffusionTransformer.prepare_pair_bias_cache(z)`` returns one
        precomputed ``LN_z(z) @ Wz`` projection per block. Passing those
        through as ``pair_bias_cache`` must match the per-call eager path
        bitwise (same kernel, same math, same input). Runs once with each
        of the eager and fused diffusion-attn paths.
        """
        torch.manual_seed(0)
        batch_size = 1
        n_sample = 5
        n_res = 384
        c_a = 768
        c_s = consts.c_s
        c_z = consts.c_z

        block = (
            DiffusionTransformer(
                c_a=c_a, c_s=c_s, c_z=c_z, c_hidden=48, no_heads=16,
                no_blocks=2, n_transition=2, use_ada_layer_norm=True,
                n_query=None, n_key=None, inf=1e9,
            )
            .eval().to(device="cuda", dtype=torch.float32)
        )
        self._initialize_model_weights(block)

        a = torch.rand(batch_size, n_sample, n_res, c_a, device="cuda")
        s = torch.rand(batch_size, n_sample, n_res, c_s, device="cuda")
        z = torch.rand(batch_size, n_res, n_res, c_z, device="cuda")
        mask = torch.randint(0, 2, (batch_size, n_res), device="cuda").float()

        old_attn = os.environ.get("OPENFOLD3_FUSED_DIFFUSION_ATTN")
        old_min = os.environ.get("OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS")
        try:
            for fused_attn in ("0", "1"):
                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = fused_attn
                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS"] = "0"
                with torch.no_grad():
                    out_uncached = block(a=a, s=s, z=z, mask=mask).clone()
                    cache = block.prepare_pair_bias_cache(z)
                    out_cached = block(
                        a=a, s=s, z=z, mask=mask, pair_bias_cache=cache,
                    ).clone()
                # Both paths consume the same input and call the same kernel
                # math; the cache only changes WHEN prep_static_pair_bias is
                # called, not its output. Outputs must be bitwise equal.
                torch.testing.assert_close(
                    out_uncached, out_cached, rtol=0, atol=0,
                    msg=f"pair-bias cache != uncached (fused_attn={fused_attn})",
                )
        finally:
            if old_attn is None:
                os.environ.pop("OPENFOLD3_FUSED_DIFFUSION_ATTN", None)
            else:
                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = old_attn
            if old_min is None:
                os.environ.pop("OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS", None)
            else:
                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN_MIN_TOKENS"] = old_min

    def test_sample_diffusion_drops_triangle_flags_for_fused_attention(self):
        """Diffusion flash attention should not be blocked by triangle flags."""
        from openfold3.core.model.structure.diffusion_module import SampleDiffusion

        class RecordingDiffusionModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def prepare_diffusion_conditioning_cache(
                self, batch, si_input, si_trunk, zij_trunk, **kwargs
            ):
                return {"zij_conditioned": zij_trunk}

            def prepare_atom_rep_cache(self, batch, si_trunk, zij_conditioned):
                return {}

            def forward(self, xl_noisy, **kwargs):
                self.calls.append(kwargs)
                return xl_noisy

        old_env = os.environ.get("OPENFOLD3_FUSED_DIFFUSION_ATTN")
        try:
            os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = "1"
            diffusion_module = RecordingDiffusionModule()
            sampler = SampleDiffusion(
                gamma_0=0.0,
                gamma_min=1.0,
                noise_scale=1.0,
                step_scale=1.0,
                diffusion_module=diffusion_module,
            )
            batch = {
                "atom_mask": torch.ones(1, 3),
                "token_mask": torch.ones(1, 2),
            }
            si_input = torch.zeros(1, 2, 4)
            si_trunk = torch.zeros(1, 2, 4)
            zij_trunk = torch.zeros(1, 2, 2, 4)
            sampler(
                batch=batch,
                si_input=si_input,
                si_trunk=si_trunk,
                zij_trunk=zij_trunk,
                noise_schedule=torch.tensor([1.0, 0.5]),
                no_rollout_samples=1,
                use_cueq_triangle_kernels=True,
                use_triton_triangle_kernels=True,
            )

            assert len(diffusion_module.calls) == 1
            call = diffusion_module.calls[0]
            assert call["use_cueq_triangle_kernels"] is False
            assert call["use_triton_triangle_kernels"] is False
        finally:
            if old_env is None:
                os.environ.pop("OPENFOLD3_FUSED_DIFFUSION_ATTN", None)
            else:
                os.environ["OPENFOLD3_FUSED_DIFFUSION_ATTN"] = old_env

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_diffusion_transformer_dsk_bf16(self):
        """Run Diffusion Transformer comparison test with BF16 precision."""
        self._compare_diffusion_transformer(
            use_deepspeed_evo_attention=True,
            dtype=torch.bfloat16,
            eps=4e-2,
        )

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_diffusion_transformer_dsk_fp32(self):
        """Run Diffusion Transformer comparison test with FP32 precision."""
        self._compare_diffusion_transformer(
            use_deepspeed_evo_attention=True,
            dtype=torch.float32,
            eps=2e-2,
        )

    @compare_utils.skip_unless_triton_installed()
    def test_compare_diffusion_transformer_fused_attention_bf16(self):
        """Run Diffusion Transformer comparison test with fused pair-bias attention."""
        self._compare_diffusion_transformer(
            use_deepspeed_evo_attention=False,
            use_fused_diffusion_attention=True,
            dtype=torch.bfloat16,
            eps=4e-2,
        )

    def _compare_template_stack(
        self,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=False,
        use_triton_triangle_kernels=False,
        dtype=torch.float32,
        chunk_size=None,
        eps=2e-2,
    ):
        """
        Compare Template Stack output with and without using different optimized
        attention kernels. Kernel can be used for Triangle Attention in the
        Template Pair Stack.
        """
        batch_size = consts.batch_size
        if chunk_size is not None and (
            use_deepspeed_evo_attention or use_triton_triangle_kernels
        ):
            # Chunk tuning is not supported with batch size > 1 for these kernels
            batch_size = 1

        n_templ = 3
        n_token = 200  # Avoid cuEq seq len constraints

        of3_proj_entry = OF3ProjectEntry()
        of3_config = of3_proj_entry.get_model_config_with_presets()
        c_in = of3_config.architecture.template.template_pair_embedder.c_in

        embedder = (
            TemplateEmbedderAllAtom(of3_config.architecture.template)
            .eval()
            .to(device="cuda")
        )
        self._initialize_model_weights(embedder)

        batch = {
            "token_mask": torch.ones((batch_size, n_token)),
            "asym_id": torch.ones((batch_size, n_token)),
            "template_restype": torch.ones((batch_size, n_templ, n_token, 32)),
            "template_pseudo_beta_mask": torch.ones((batch_size, n_templ, n_token)),
            "template_backbone_frame_mask": torch.ones((batch_size, n_templ, n_token)),
            "template_distogram": torch.ones(
                (batch_size, n_templ, n_token, n_token, 39)
            ),
            "template_unit_vector": torch.ones(
                (batch_size, n_templ, n_token, n_token, 3)
            ),
        }

        def to_device(t):
            return t.to(device=torch.device("cuda"))

        batch = tensor_tree_map(to_device, batch)

        z = torch.ones((batch_size, n_token, n_token, c_in))
        pair_mask = torch.randint(0, 2, size=(batch_size, n_token, n_token))

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            args = (
                batch,
                torch.as_tensor(z).cuda(),
                torch.as_tensor(pair_mask).cuda(),
            )

            out_repro = embedder(
                *args,
                inplace_safe=False,
                use_deepspeed_evo_attention=False,
                chunk_size=None,  # Test against non-chunked version
            )

            out_repro_ds = embedder(
                *args,
                inplace_safe=False,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_cueq_triangle_kernels=use_cueq_triangle_kernels,
                use_triton_triangle_kernels=use_triton_triangle_kernels,
            )

            compare_utils.assert_max_abs_diff_small(out_repro, out_repro_ds, eps)

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_template_stack_dsk_fp32(self):
        self._compare_template_stack(
            use_deepspeed_evo_attention=True,
            dtype=torch.float32,
        )

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_template_stack_dsk_bf16(self):
        self._compare_template_stack(
            use_deepspeed_evo_attention=True,
            dtype=torch.bfloat16,
            eps=4e-2,
        )

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_template_stack_dsk_fp32_chunk(self):
        self._compare_template_stack(
            use_deepspeed_evo_attention=True,
            dtype=torch.float32,
            chunk_size=4,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_compare_template_stack_cueq_fp32(self):
        self._compare_template_stack(
            use_cueq_triangle_kernels=True,
            dtype=torch.float32,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_compare_template_stack_cueq_bf16(self):
        self._compare_template_stack(
            use_cueq_triangle_kernels=True,
            dtype=torch.bfloat16,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_compare_template_stack_cueq_fp32_chunk(self):
        self._compare_template_stack(
            use_cueq_triangle_kernels=True,
            dtype=torch.float32,
            chunk_size=4,
        )

    @compare_utils.skip_unless_triton_installed()
    def test_compare_template_stack_triton_fp32_chunk(self):
        self._compare_template_stack(
            use_triton_triangle_kernels=True,
            dtype=torch.float32,
            chunk_size=4,
        )

    @compare_utils.skip_unless_triton_installed()
    def test_compare_template_stack_triton_fp32(self):
        self._compare_template_stack(
            use_triton_triangle_kernels=True,
            dtype=torch.float32,
        )

    @compare_utils.skip_unless_triton_installed()
    def test_compare_template_stack_triton_bf16(self):
        self._compare_template_stack(
            use_triton_triangle_kernels=True,
            dtype=torch.bfloat16,
        )

    @compare_utils.skip_unless_cuda_available()
    def test_template_stack_empty_masks_match_full_path(self):
        """Empty template masks still run the learned template path.

        MSA-free/no-template inference supplies padded template slots with zero
        masks. Those slots are not mathematically equivalent to returning a
        zero template embedding because the template pair stack has learned
        parameters and normalization layers.
        """
        batch_size = 1
        n_templ = 2
        n_token = 32

        of3_proj_entry = OF3ProjectEntry()
        of3_config = of3_proj_entry.get_model_config_with_presets()
        c_in = of3_config.architecture.template.template_pair_embedder.c_in

        embedder = (
            TemplateEmbedderAllAtom(of3_config.architecture.template)
            .eval()
            .to(device="cuda")
        )
        self._initialize_model_weights(embedder)

        batch = {
            "token_mask": torch.ones((batch_size, n_token)),
            "asym_id": torch.ones((batch_size, n_token)),
            "template_restype": torch.ones((batch_size, n_templ, n_token, 32)),
            "template_pseudo_beta_mask": torch.zeros(
                (batch_size, n_templ, n_token)
            ),
            "template_backbone_frame_mask": torch.zeros(
                (batch_size, n_templ, n_token)
            ),
            "template_distogram": torch.ones(
                (batch_size, n_templ, n_token, n_token, 39)
            ),
            "template_unit_vector": torch.ones(
                (batch_size, n_templ, n_token, n_token, 3)
            ),
        }
        batch = tensor_tree_map(lambda t: t.cuda(), batch)
        z = torch.randn((batch_size, n_token, n_token, c_in), device="cuda")
        pair_mask = torch.ones((batch_size, n_token, n_token), device="cuda")

        with torch.no_grad():
            out = embedder(
                batch,
                z,
                pair_mask,
                inplace_safe=True,
                use_deepspeed_evo_attention=False,
                use_cueq_triangle_kernels=False,
                use_triton_triangle_kernels=False,
                chunk_size=4,
            )

        assert out.shape == (batch_size, n_token, n_token, of3_config.architecture.template.c_z)
        assert out.abs().max() > 0

    @compare_utils.skip_unless_cuda_available()
    @compare_utils.skip_unless_triton_installed()
    def test_template_stack_streaming_fused_trimul_matches_cueq(self):
        """Fused trimul must handle template c_z=64 in the streaming path."""
        batch_size = 1
        n_templ = 4
        n_token = 76

        of3_proj_entry = OF3ProjectEntry()
        of3_config = of3_proj_entry.get_model_config_with_presets()
        c_in = of3_config.architecture.template.template_pair_embedder.c_in

        embedder = (
            TemplateEmbedderAllAtom(of3_config.architecture.template)
            .eval()
            .cuda()
        )
        self._initialize_model_weights(embedder)

        batch = {
            "token_mask": torch.ones((batch_size, n_token), device="cuda"),
            "asym_id": torch.ones((batch_size, n_token), device="cuda"),
            "template_restype": torch.ones(
                (batch_size, n_templ, n_token, 32), device="cuda"
            ),
            "template_pseudo_beta_mask": torch.zeros(
                (batch_size, n_templ, n_token), device="cuda"
            ),
            "template_backbone_frame_mask": torch.zeros(
                (batch_size, n_templ, n_token), device="cuda"
            ),
            "template_distogram": torch.ones(
                (batch_size, n_templ, n_token, n_token, 39), device="cuda"
            ),
            "template_unit_vector": torch.ones(
                (batch_size, n_templ, n_token, n_token, 3), device="cuda"
            ),
        }
        z = torch.randn((batch_size, n_token, n_token, c_in), device="cuda")
        pair_mask = torch.ones((batch_size, n_token, n_token), device="cuda")

        with torch.inference_mode():
            with mock.patch.dict(
                os.environ,
                {"OPENFOLD3_FUSED_TRIMUL": "0"},
            ):
                ref = embedder(
                    batch,
                    z.clone(),
                    pair_mask,
                    chunk_size=4,
                    use_cueq_triangle_kernels=True,
                    inplace_safe=True,
                )
            with mock.patch.dict(
                os.environ,
                {"OPENFOLD3_FUSED_TRIMUL": "1"},
            ):
                fused = embedder(
                    batch,
                    z.clone(),
                    pair_mask,
                    chunk_size=4,
                    use_cueq_triangle_kernels=True,
                    inplace_safe=True,
                )

        compare_utils.assert_max_abs_diff_small(ref, fused, 2e-3)

    @pytest.mark.parametrize("N", [76, 256, 512])
    def test_fused_relpos_embed_matches_eager(self, N):
        """Fused relpos gather-add kernel matches the 4 sequential gather-adds."""
        from openfold3.core.kernels.triton.fused_relpos_embed import (
            fused_relpos_embed_add_,
        )

        C = 128
        vocab = 130
        same_entity_offset = 65

        z = torch.randn(1, N, N, C, device="cuda", dtype=torch.float32)
        w = torch.randn(vocab, C, device="cuda", dtype=torch.float32)
        idx1 = torch.randint(0, vocab, (1, N, N), device="cuda", dtype=torch.int64)
        idx2 = torch.randint(0, vocab, (1, N, N), device="cuda", dtype=torch.int64)
        idx3 = torch.randint(0, vocab, (1, N, N), device="cuda", dtype=torch.int64)
        same_entity = torch.randint(0, 2, (1, N, N), device="cuda", dtype=torch.bool)

        z_ref = z.clone()
        z_ref.add_(w[idx1])
        z_ref.add_(w[idx2])
        z_ref.add_(w[idx3])
        z_ref.add_(same_entity[..., None].to(dtype=z.dtype) * w[same_entity_offset])

        z_fused = z.clone()
        fused_relpos_embed_add_(
            z_fused, w, idx1, idx2, idx3, same_entity, same_entity_offset
        )

        compare_utils.assert_max_abs_diff_small(z_ref, z_fused, 1e-5)
