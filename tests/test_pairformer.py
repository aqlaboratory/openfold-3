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

import torch
import unittest
from openfold3.core.model.latent import PairFormerStack

from tests.config import consts


class TestPairFormer(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_res = consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z
        c_hidden_pair_bias = 12
        no_heads_pair_bias = 3
        c_hidden_mul = 19
        c_hidden_pair_att = 14
        no_heads_pair = 7
        no_blocks = 2
        transition_n = 2
        pair_dropout = 0.25
        inf = 1e9
        eps = 1e-10

        pfs = PairFormerStack(
            c_s,
            c_z,
            c_hidden_pair_bias,
            no_heads_pair_bias,
            c_hidden_mul,
            c_hidden_pair_att,
            no_heads_pair,
            no_blocks,
            transition_n,
            pair_dropout,
            fuse_projection_weights=False,
            blocks_per_ckpt=None,
            inf=inf,
            eps=eps,
        ).eval()

        s = torch.rand((batch_size, n_res, c_s))
        z = torch.rand((batch_size, n_res, n_res, c_z))
        single_mask = torch.randint(0, 2, size=(batch_size, n_res,))
        pair_mask = torch.randint(0, 2, size=(batch_size, n_res, n_res))

        shape_s_before = s.shape
        shape_z_before = z.shape

        s, z = pfs(
            s, z, single_mask=single_mask, pair_mask=pair_mask, chunk_size=4
        )

        self.assertTrue(s.shape == shape_s_before)
        self.assertTrue(z.shape == shape_z_before)