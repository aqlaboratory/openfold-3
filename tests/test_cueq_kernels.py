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
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicativeUpdate,
)
from openfold3.core.model.primitives.attention import Attention
from openfold3.core.model.primitives.initialization import lecun_normal_init_
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from tests.config import consts


@compare_utils.skip_unless_cueq_installed()
@compare_utils.skip_unless_cuda_available()
class TestCuEqKernels(unittest.TestCase):
    def test_cueq_tri_attn_fwd(self):
        """test cueq triangle attn forward pass."""
        ## NOTE: this tests the forwards pass as seen in
        ## the template module
        batch_size = consts.batch_size
        n_tmpl = 20
        n_res = 64
        c_in = 128
        c_hidden = 32
        no_heads = 4
        eps = 2e-2
        x = torch.randn(batch_size, n_tmpl, n_res, n_res, c_in).to("cuda")
        mask_bias = torch.zeros(batch_size, n_tmpl, n_res, 1, 1, n_res).to("cuda")
        triangle_bias = torch.randn(batch_size, n_tmpl, 1, no_heads, n_res, n_res).to(
            "cuda"
        )
        biases = [mask_bias, triangle_bias]

        a = Attention(
            c_q=c_in,
            c_k=c_in,
            c_v=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
        ).to("cuda")

        # Change output params init for testing since they are initialized with 'final'
        # init (zeros) Otherwise both will just return zero.
        with torch.no_grad():
            lecun_normal_init_(a.linear_g.weight)
            lecun_normal_init_(a.linear_o.weight)

            real_out = a(x, x, biases=biases, use_cueq_triangle_kernel=False).cpu()

            cueq_out = a(x, x, biases=biases, use_cueq_triangle_kernel=True).cpu()

        err = torch.max(torch.abs(cueq_out - real_out))
        print(f"Max error in cueq triangle attention forward: {err.item()}")
        self.assertTrue(err < eps, f"Error: {err}")

    def test_cueq_tri_attn_bwd(self):
        """
        test cu eq triangle attention backward pass. Right now
        only bf16 is supported
        """
        batch_size = consts.batch_size
        n_tmpl = 20
        n_res = 64
        c_in = 128
        c_hidden = 32
        no_heads = 4
        eps = consts.eps
        # NOTE: fp32 is not supported for the cueq kernel
        # this is intentionally placed to trigger a warning
        # that types get auto converted to bf16
        dtype = torch.float32
        x = torch.randn(
            batch_size, n_tmpl, n_res, n_res, c_in, dtype=dtype, requires_grad=True
        ).to("cuda")
        q = x.clone()
        kv = x.clone()
        mask_bias = torch.zeros(batch_size, n_tmpl, n_res, 1, 1, n_res, dtype=dtype).to(
            "cuda"
        )
        triangle_bias = torch.randn(
            batch_size,
            n_tmpl,
            1,
            no_heads,
            n_res,
            n_res,
            dtype=dtype,
            requires_grad=True,
        ).to("cuda")
        biases = [mask_bias, triangle_bias]

        attn = Attention(
            c_q=c_in,
            c_k=c_in,
            c_v=c_in,
            c_hidden=c_hidden,
            no_heads=no_heads,
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
                c_q=c_in,
                c_k=c_in,
                c_v=c_in,
                c_hidden=c_hidden,
                no_heads=no_heads,
            ).cuda()

            a_clone.load_state_dict(attn.state_dict())
            return a_clone

        # Clone param values and run attention with DS kernel
        q_repro = clone(q)
        kv_repro = clone(kv)
        biases_repro = [clone(b) for b in biases]

        a_repro = init_attn()

        attn.train()
        a_repro.train()

        out_repro = a_repro(
            q_repro, kv_repro, biases=biases_repro, use_cueq_triangle_kernel=True
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
        pairs = zip([q_repro, kv_repro, biases_repro[1]], [q_gt, kv_gt, biases_gt[1]])
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
                use_cueq_triangle_kernel=False,
            )
            fwd_cueq = tm(
                z=z,
                mask=mask,
                use_cueq_triangle_kernel=True,
            )
        err = torch.max(torch.abs(fwd_reg - fwd_cueq))
        print(f"Max error in cueq triangle multiplicative update forward: {err.item()}")
        eps = 2e-2
        self.assertTrue(err < eps, f"Error: {err}")

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
            use_cueq_triangle_kernel=True,
        )
        loss_repro = torch.mean(out_repro)
        loss_repro.backward()

        z_gt = clone(z)
        mask_gt = clone(mask)
        tm_gt = init_tm()
        out_gt = tm_gt(
            z=z_gt,
            mask=mask_gt,
            use_cueq_triangle_kernel=False,
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

    def compare_pairformer(self, dtype, eps):
        """
        Compare Pairformer output with and without using cueq kernels
        kernel. Set dtype to confirm the kernel can be used during both training (BF16)
        and inference (FP32), since the kernel itself can run with either BF16 or FP16
        precision.

        TODO: Change the test to use a loaded Pairformer block from the trained model
          instead of a newly initialized block.
        """
        batch_size = consts.batch_size
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
            )
            .eval()
            .to(device="cuda", dtype=dtype)
        )

        s = torch.rand(batch_size, n_res, consts.c_s, device="cuda", dtype=dtype)
        z = torch.rand(batch_size, n_res, n_res, consts.c_z, device="cuda", dtype=dtype)

        s_mask = torch.randint(0, 2, (batch_size, n_res), device="cuda", dtype=dtype)
        z_mask = torch.randint(
            0, 2, (batch_size, n_res, n_res), device="cuda", dtype=dtype
        )
        block.eval()
        with torch.amp.autocast("cuda", dtype=dtype):
            out_repro_single, out_repro_pair = block(
                s=s.clone(),
                z=z.clone(),
                single_mask=s_mask.clone(),
                pair_mask=z_mask.clone(),
                use_deepspeed_evo_attention=False,
                use_cueq_triangle_kernel=False,
            )

            # In practice, layer norms applied later in the network make any
            # kernel rounding errors negligible
            out_repro_single = F.layer_norm(out_repro_single, (consts.c_s,)).cpu()
            out_repro_pair = F.layer_norm(out_repro_pair, (consts.c_z,)).cpu()

            out_repro_single_ds, out_repro_pair_ds = block(
                s=s.clone(),
                z=z.clone(),
                single_mask=s_mask.clone(),
                pair_mask=z_mask.clone(),
                use_deepspeed_evo_attention=False,
                use_cueq_triangle_kernel=True,
            )
            out_repro_single_ds = F.layer_norm(out_repro_single_ds, (consts.c_s,)).cpu()
            out_repro_pair_ds = F.layer_norm(out_repro_pair_ds, (consts.c_z,)).cpu()

            compare_utils.assert_mean_abs_diff_small(
                out_repro_single, out_repro_single_ds, eps
            )

            compare_utils.assert_mean_abs_diff_small(
                out_repro_pair, out_repro_pair_ds, eps
            )

    def test_compare_pairformer_bf16(self):
        """Run evoformer comparison test with BF16 precision."""
        self.compare_pairformer(dtype=torch.bfloat16, eps=4e-2)

    def test_compare_pairformer_fp32(self):
        """Run evoformer comparison test with FP32 precision."""
        self.compare_pairformer(dtype=torch.float32, eps=2e-2)

    def test_compare_template_stack(self):
        """
        Compare Template Stack output with and without using DeepSpeed Evoformer
        attention kernel. Kernel can be used for Triangle Attention in the Template Pair
        Stack.
        """
        batch_size = 1
        n_templ = 3
        n_token = 10

        of3_proj_entry = OF3ProjectEntry()
        of3_config = of3_proj_entry.get_model_config_with_presets()
        c_in = of3_config.architecture.template.template_pair_embedder.c_in

        embedder = TemplateEmbedderAllAtom(of3_config.architecture.template).to(
            device="cuda"
        )

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

        with torch.no_grad():
            args = (
                batch,
                torch.as_tensor(z).cuda(),
                torch.as_tensor(pair_mask).cuda(),
            )

            out_repro = embedder(
                *args,
                inplace_safe=False,
                chunk_size=None,
                use_deepspeed_evo_attention=False,
                use_cueq_triangle_kernel=False,
            )

            out_repro_ds = embedder(
                *args,
                inplace_safe=False,
                chunk_size=None,
                use_deepspeed_evo_attention=False,
                use_cueq_triangle_kernel=True,
            )

            compare_utils.assert_max_abs_diff_small(
                out_repro.cpu(), out_repro_ds.cpu(), 2e-2
            )


if __name__ == "__main__":
    unittest.main()
