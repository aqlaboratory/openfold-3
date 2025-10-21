# Copyright 2025 AlQuraishi Laboratory
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

from openfold3.core.model.latent.msa_module import MSAModuleStack
from openfold3.core.model.layers.transition import SwiGLUTransition
from tests.config import consts


class TestMSAModule(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_seq = consts.n_seq
        n_res = consts.n_res
        c_m = consts.c_m
        c_z = consts.c_z
        c_hidden_msa_att = 12
        c_hidden_opm = 17
        c_hidden_mul = 19
        c_hidden_pair_att = 14
        no_heads_msa = 3
        no_heads_pair = 7
        no_blocks = 2
        transition_type = "swiglu"
        transition_n = 2
        msa_dropout = 0.15
        pair_dropout = 0.25
        inf = 1e9
        eps = 1e-10

        ms = MSAModuleStack(
            c_m=c_m,
            c_z=c_z,
            c_hidden_msa_att=c_hidden_msa_att,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            no_blocks=no_blocks,
            transition_type=transition_type,
            transition_n=transition_n,
            msa_dropout=msa_dropout,
            pair_dropout=pair_dropout,
            opm_first=True,
            fuse_projection_weights=False,
            blocks_per_ckpt=None,
            inf=inf,
            eps=eps,
        ).eval()

        m = torch.rand((batch_size, n_seq, n_res, c_m))
        z = torch.rand((batch_size, n_res, n_res, c_z))
        msa_mask = torch.randint(0, 2, size=(batch_size, n_seq, n_res))
        pair_mask = torch.randint(0, 2, size=(batch_size, n_res, n_res))

        shape_z_before = z.shape

        z = ms(m, z, chunk_size=4, msa_mask=msa_mask, pair_mask=pair_mask)

        self.assertTrue(z.shape == shape_z_before)


class TestMSAModuleTransition(unittest.TestCase):
    def test_shape(self):
        batch_size = 2
        s_t = 3
        n_r = 5
        c_m = 7
        n = 11

        mt = SwiGLUTransition(c_in=c_m, n=n)

        m = torch.rand((batch_size, s_t, n_r, c_m))

        shape_before = m.shape
        m = mt(m, chunk_size=4)
        shape_after = m.shape

        self.assertTrue(shape_before == shape_after)


if __name__ == "__main__":
    unittest.main()
