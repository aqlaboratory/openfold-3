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

import tests.compare_utils as compare_utils
from openfold3.core.model.layers.triangular_multiplicative_update import (
    TriangleMultiplicativeUpdate,
)
from openfold3.core.model.primitives.attention import Attention
from openfold3.core.model.primitives.initialization import lecun_normal_init_
from tests.config import consts


@compare_utils.skip_unless_cueq_installed()
@compare_utils.skip_unless_cuda_available()
class TestCuEqKernels(unittest.TestCase):
    def test_cueq_tri_attn_fwd(self):
        """test cueq triangle attn forward pass."""
        batch_size = consts.batch_size
        n_res = 64
        c_in = 128
        c_hidden = 32
        no_heads = 4
        eps = 2e-2
        x = torch.randn(batch_size, n_res, n_res, c_in).to("cuda")
        mask_bias = torch.zeros(batch_size, n_res, 1, 1, n_res).to("cuda")
        triangle_bias = torch.randn(batch_size, 1, no_heads, n_res, n_res).to("cuda")
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
        test cu eq triangle attention backward pass
        """
        batch_size = consts.batch_size

        n_res = 64
        c_in = 128
        c_hidden = 32
        no_heads = 4
        eps = consts.eps

        x = torch.randn(
            batch_size, n_res, n_res, c_in, dtype=torch.bfloat16, requires_grad=True
        ).to("cuda")
        q = x.clone()
        kv = x.clone()
        mask_bias = torch.zeros(
            batch_size, n_res, 1, 1, n_res, dtype=torch.bfloat16
        ).to("cuda")
        triangle_bias = torch.randn(
            batch_size,
            1,
            no_heads,
            n_res,
            n_res,
            dtype=torch.bfloat16,
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
        batch = 1
        seq_len = 84
        c_z = 128
        c_hidden = 128
        outgoing = True
        tm = TriangleMultiplicativeUpdate(
            c_z=c_z,
            c_hidden=c_hidden,
            _outgoing=outgoing,
        ).to("cuda")
        z = torch.randn(1, batch, seq_len, seq_len, c_z).to("cuda")
        mask = torch.ones(1, batch, seq_len, seq_len).to("cuda")
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
        batch = 1
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
        z = torch.randn(1, batch, seq_len, seq_len, c_z, requires_grad=True).to("cuda")
        mask = torch.ones(1, batch, seq_len, seq_len, requires_grad=False).to("cuda")
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


if __name__ == "__main__":
    unittest.main()
