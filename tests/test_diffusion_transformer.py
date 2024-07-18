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

import unittest

import torch

from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
from openfold3.core.model.layers.transition import ConditionedTransitionBlock
from tests.config import consts


class TestDiffusionTransformer(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_res = consts.n_res
        c_a = 768
        c_s = consts.c_s
        c_z = consts.c_z
        c_hidden = 16
        no_heads = 3
        no_blocks = 2
        n_transition = 2
        inf = 1e9

        dt = DiffusionTransformer(
            c_a,
            c_s,
            c_z,
            c_hidden,
            no_heads,
            no_blocks,
            n_transition,
            use_ada_layer_norm=True,
            use_block_sparse_attn=False,
            block_size=None,
            inf=inf,
        ).eval()

        a = torch.rand((batch_size, n_res, c_a))
        s = torch.rand((batch_size, n_res, c_s))
        z = torch.rand((batch_size, n_res, n_res, c_z))
        beta = torch.rand((batch_size, n_res, n_res))
        single_mask = torch.randint(0, 2, size=(batch_size, n_res))

        shape_a_before = a.shape

        a = dt(a, s, z, mask=single_mask, beta=beta)

        self.assertTrue(a.shape == shape_a_before)


class TestConditionedTransitionBlock(unittest.TestCase):
    def test_shape(self):
        batch_size = 2
        n_r = 5
        c_a = 14
        c_s = 7
        n = 11

        ct = ConditionedTransitionBlock(c_a=c_a, c_s=c_s, n=n)

        a = torch.rand((batch_size, n_r, c_a))
        s = torch.rand((batch_size, n_r, c_s))

        shape_before = a.shape
        a = ct(a=a, s=s)
        shape_after = a.shape

        self.assertTrue(shape_before == shape_after)


if __name__ == "__main__":
    unittest.main()
