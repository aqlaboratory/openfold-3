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

import re
import unittest

import torch

from openfold3.core.model.latent.evoformer import EvoformerStack
from openfold3.core.model.latent.extra_msa import ExtraMSAStack
from openfold3.core.model.layers.transition import ReLUTransition
from tests.config import consts


class TestEvoformerStack(unittest.TestCase):
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
        c_s = consts.c_s
        no_heads_msa = 3
        no_heads_pair = 7
        no_blocks = 2
        transition_type = "relu"
        transition_n = 2
        msa_dropout = 0.15
        pair_dropout = 0.25
        opm_first = consts.is_multimer
        fuse_projection_weights = bool(
            re.fullmatch("^model_[1-5]_multimer_v3$", consts.model_preset)
        )
        inf = 1e9
        eps = 1e-10

        es = EvoformerStack(
            c_m=c_m,
            c_z=c_z,
            c_hidden_msa_att=c_hidden_msa_att,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            c_s=c_s,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            no_blocks=no_blocks,
            transition_type=transition_type,
            transition_n=transition_n,
            msa_dropout=msa_dropout,
            pair_dropout=pair_dropout,
            no_column_attention=False,
            opm_first=opm_first,
            fuse_projection_weights=fuse_projection_weights,
            blocks_per_ckpt=None,
            inf=inf,
            eps=eps,
        ).eval()

        m = torch.rand((batch_size, n_seq, n_res, c_m))
        z = torch.rand((batch_size, n_res, n_res, c_z))
        msa_mask = torch.randint(0, 2, size=(batch_size, n_seq, n_res))
        pair_mask = torch.randint(0, 2, size=(batch_size, n_res, n_res))

        shape_m_before = m.shape
        shape_z_before = z.shape

        m, z, s = es(m, z, chunk_size=4, msa_mask=msa_mask, pair_mask=pair_mask)

        self.assertTrue(m.shape == shape_m_before)
        self.assertTrue(z.shape == shape_z_before)
        self.assertTrue(s.shape == (batch_size, n_res, c_s))

    def test_shape_without_column_attention(self):
        batch_size = consts.batch_size
        n_seq = consts.n_seq
        n_res = consts.n_res
        c_m = consts.c_m
        c_z = consts.c_z
        c_hidden_msa_att = 12
        c_hidden_opm = 17
        c_hidden_mul = 19
        c_hidden_pair_att = 14
        c_s = consts.c_s
        no_heads_msa = 3
        no_heads_pair = 7
        no_blocks = 2
        transition_type = "relu"
        transition_n = 2
        msa_dropout = 0.15
        pair_dropout = 0.25
        inf = 1e9
        eps = 1e-10

        es = EvoformerStack(
            c_m=c_m,
            c_z=c_z,
            c_hidden_msa_att=c_hidden_msa_att,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            c_s=c_s,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            no_blocks=no_blocks,
            transition_type=transition_type,
            transition_n=transition_n,
            msa_dropout=msa_dropout,
            pair_dropout=pair_dropout,
            no_column_attention=True,
            opm_first=False,
            fuse_projection_weights=False,
            blocks_per_ckpt=None,
            inf=inf,
            eps=eps,
        ).eval()

        m_init = torch.rand((batch_size, n_seq, n_res, c_m))
        z_init = torch.rand((batch_size, n_res, n_res, c_z))
        msa_mask = torch.randint(0, 2, size=(batch_size, n_seq, n_res))
        pair_mask = torch.randint(0, 2, size=(batch_size, n_res, n_res))

        shape_m_before = m_init.shape
        shape_z_before = z_init.shape

        m, z, s = es(
            m_init, z_init, chunk_size=4, msa_mask=msa_mask, pair_mask=pair_mask
        )

        self.assertTrue(m.shape == shape_m_before)
        self.assertTrue(z.shape == shape_z_before)
        self.assertTrue(s.shape == (batch_size, n_res, c_s))


class TestExtraMSAStack(unittest.TestCase):
    def test_shape(self):
        batch_size = 2
        s_t = 23
        n_res = 5
        c_m = 7
        c_z = 11
        c_hidden_msa_att = 12
        c_hidden_opm = 17
        c_hidden_mul = 19
        c_hidden_tri_att = 16
        no_heads_msa = 3
        no_heads_pair = 8
        no_blocks = 2
        transition_type = "relu"
        transition_n = 5
        msa_dropout = 0.15
        pair_stack_dropout = 0.25
        opm_first = consts.is_multimer
        fuse_projection_weights = bool(
            re.fullmatch("^model_[1-5]_multimer_v3$", consts.model_preset)
        )
        inf = 1e9
        eps = 1e-10

        es = (
            ExtraMSAStack(
                c_m,
                c_z,
                c_hidden_msa_att,
                c_hidden_opm,
                c_hidden_mul,
                c_hidden_tri_att,
                no_heads_msa,
                no_heads_pair,
                no_blocks,
                transition_type,
                transition_n,
                msa_dropout,
                pair_stack_dropout,
                opm_first,
                fuse_projection_weights,
                ckpt=False,
                inf=inf,
                eps=eps,
            )
            .eval()
            .cuda()
        )

        m = torch.rand((batch_size, s_t, n_res, c_m), device="cuda")
        z = torch.rand((batch_size, n_res, n_res, c_z), device="cuda")
        msa_mask = torch.randint(
            0,
            2,
            size=(
                batch_size,
                s_t,
                n_res,
            ),
            device="cuda",
        ).float()
        pair_mask = torch.randint(
            0,
            2,
            size=(
                batch_size,
                n_res,
                n_res,
            ),
            device="cuda",
        ).float()

        shape_z_before = z.shape

        z = es(m, z, chunk_size=4, msa_mask=msa_mask, pair_mask=pair_mask)

        self.assertTrue(z.shape == shape_z_before)


class TestMSATransition(unittest.TestCase):
    def test_shape(self):
        batch_size = 2
        s_t = 3
        n_r = 5
        c_m = 7
        n = 11

        mt = ReLUTransition(c_in=c_m, n=n)

        m = torch.rand((batch_size, s_t, n_r, c_m))

        shape_before = m.shape
        m = mt(m, chunk_size=4)
        shape_after = m.shape

        self.assertTrue(shape_before == shape_after)


if __name__ == "__main__":
    unittest.main()
