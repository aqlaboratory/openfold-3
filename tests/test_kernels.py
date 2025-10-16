# Copyright 2021 AlQuraishi Laboratory
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

import unittest

import torch
from torch.nn import functional as F

import tests.compare_utils as compare_utils
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
from tests.config import consts
from tests.data_utils import (
    random_attention_inputs,
)


@compare_utils.skip_unless_cuda_available()
class TestKernels(unittest.TestCase):
    def _compare_attn_kernel_forward(
        self,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=False,
        dtype=torch.float32,
    ):
        """Compare attention with and without using DeepSpeed Evoformer kernel."""
        batch_size = consts.batch_size
        n_seq = 18
        n_res = 20
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
            ).cpu()

        err = torch.max(torch.abs(kernel_out - real_out))
        self.assertTrue(err < eps, f"Error: {err}")

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

    def _compare_attn_kernel_backward(
        self,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=False,
        dtype=torch.float32,
    ):
        """
        Compare backward pass for regular attention vs. DeepSpeed Evoformer kernel.
        """
        batch_size = consts.batch_size
        n_seq = 18
        n_res = 20
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
            self.assertTrue(err < eps, f"Error item #{i}: {err}")

        # Compare the grads of model weights
        a_repro_params = dict(a_repro.named_parameters())
        a_gt_params = dict(a_gt.named_parameters())
        for name in a_gt_params:
            t_repro = a_repro_params[name]
            t_gt = a_gt_params[name]
            err = torch.max(torch.abs(t_repro.grad.cpu() - t_gt.grad.cpu()))
            self.assertTrue(err < eps, f"Error item {name}: {err}")

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
        self.assertTrue(err < eps, f"Error: {err}")

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
            self.assertTrue(err < eps, f"Error item {name}: {err}")

    def _initialize_model_weights(self, model):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                with torch.no_grad():
                    lecun_normal_init_(module.weight)

    def _compare_pairformer(
        self,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=False,
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
        if chunk_size is not None and use_deepspeed_evo_attention:
            # Chunk tuning is not supported with batch size > 1 for DeepSpeed kernel
            batch_size = 1

        n_res = consts.n_res
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

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_pairformer_cueq_fp32_chunk(self):
        """Run Pairformer comparison test with chunk tuning enabled."""
        self._compare_pairformer(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.float32,
            chunk_size=4,
            eps=4e-2,
        )

    def _compare_diffusion_transformer(
        self,
        use_deepspeed_evo_attention=False,
        dtype=torch.float32,
        chunk_size=None,
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
                tune_chunk_size=chunk_size is not None,
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

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            out_repro_a = block(
                a=a,
                s=s,
                z=z,
                mask=mask,
                use_deepspeed_evo_attention=False,
                chunk_size=None,  # Test against non-chunked version
            )

            # In practice, layer norms applied later in the network make any
            # kernel rounding errors negligible
            out_repro_a = F.layer_norm(out_repro_a, (c_a,)).cpu()

            out_repro_a_ds = block(
                a=a,
                s=s,
                z=z,
                mask=mask,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                chunk_size=chunk_size,
            )
            out_repro_a_ds = F.layer_norm(out_repro_a_ds, (c_a,)).cpu()

            compare_utils.assert_mean_abs_diff_small(out_repro_a, out_repro_a_ds, eps)

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

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_diffusion_transformer_dsk_fp32_chunk(self):
        """Run Diffusion Transformer comparison test with chunk tuning enabled."""
        self._compare_diffusion_transformer(
            use_deepspeed_evo_attention=True,
            dtype=torch.float32,
            chunk_size=4,
            eps=4e-2,
        )

    def _compare_template_stack(
        self,
        use_deepspeed_evo_attention=False,
        use_cueq_triangle_kernels=False,
        dtype=torch.float32,
        chunk_size=None,
    ):
        """
        Compare Template Stack output with and without using DeepSpeed Evoformer
        attention kernel. Kernel can be used for Triangle Attention in the Template Pair
        Stack.
        """
        batch_size = consts.batch_size
        if chunk_size is not None and use_deepspeed_evo_attention:
            # Chunk tuning is not supported with batch size > 1 for DeepSpeed kernel
            batch_size = 1

        n_templ = 3
        n_token = 10

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
            )

            compare_utils.assert_max_abs_diff_small(out_repro, out_repro_ds, 2e-2)

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_template_stack_dsk_fp32(self):
        self._compare_template_stack(
            use_deepspeed_evo_attention=True,
            use_cueq_triangle_kernels=False,
            dtype=torch.float32,
        )

    @compare_utils.skip_unless_ds4s_installed()
    def test_compare_template_stack_dsk_bf16(self):
        self._compare_template_stack(
            use_deepspeed_evo_attention=True,
            use_cueq_triangle_kernels=False,
            dtype=torch.bfloat16,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_compare_template_stack_dsk_fp32_chunk(self):
        self._compare_template_stack(
            use_deepspeed_evo_attention=True,
            use_cueq_triangle_kernels=False,
            dtype=torch.float32,
            chunk_size=4,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_compare_template_stack_cueq_fp32(self):
        self._compare_template_stack(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.float32,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_compare_template_stack_cueq_bf16(self):
        self._compare_template_stack(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.bfloat16,
        )

    @compare_utils.skip_unless_cueq_installed()
    def test_compare_template_stack_cueq_fp32_chunk(self):
        self._compare_template_stack(
            use_deepspeed_evo_attention=False,
            use_cueq_triangle_kernels=True,
            dtype=torch.float32,
            chunk_size=4,
        )


if __name__ == "__main__":
    unittest.main()
