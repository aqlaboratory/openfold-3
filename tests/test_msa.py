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

from openfold3.core.model.layers.msa import (
    MSAColumnAttention,
    MSAColumnGlobalAttention,
    MSAPairWeightedAveraging,
    MSARowAttentionWithPairBias,
)
from tests.config import consts


class TestMSARowAttentionWithPairBias(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_seq = consts.n_seq
        n_res = consts.n_res
        c_m = consts.c_m
        c_z = consts.c_z
        c = 52
        no_heads = 4
        chunk_size = None

        mrapb = MSARowAttentionWithPairBias(
            c_m,
            c_z,
            c,
            no_heads,
        )

        m = torch.rand((batch_size, n_seq, n_res, c_m))
        z = torch.rand((batch_size, n_res, n_res, c_z))

        shape_before = m.shape
        m = mrapb(m, z=z, chunk_size=chunk_size)
        shape_after = m.shape

        self.assertTrue(shape_before == shape_after)


class TestMSAColumnAttention(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_seq = consts.n_seq
        n_res = consts.n_res
        c_m = consts.c_m
        c = 44
        no_heads = 4

        msaca = MSAColumnAttention(c_m, c, no_heads)

        x = torch.rand((batch_size, n_seq, n_res, c_m))

        shape_before = x.shape
        x = msaca(x, chunk_size=None)
        shape_after = x.shape

        self.assertTrue(shape_before == shape_after)


class TestMSAColumnGlobalAttention(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_seq = consts.n_seq
        n_res = consts.n_res
        c_m = consts.c_m
        c = 44
        no_heads = 4

        msagca = MSAColumnGlobalAttention(
            c_m,
            c,
            no_heads,
        )

        x = torch.rand((batch_size, n_seq, n_res, c_m))

        shape_before = x.shape
        x = msagca(x, chunk_size=None)
        shape_after = x.shape

        self.assertTrue(shape_before == shape_after)


class TestMSAPairWeightedAveraging(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_seq = consts.n_seq
        n_res = consts.n_res
        c_m = consts.c_m
        c_z = consts.c_z
        c = 52
        no_heads = 4

        mrapb = MSAPairWeightedAveraging(
            c_in=c_m,
            c_hidden=c,
            c_z=c_z,
            no_heads=no_heads,
        )

        m = torch.rand((batch_size, n_seq, n_res, c_m))
        z = torch.rand((batch_size, n_res, n_res, c_z))

        shape_before = m.shape
        m = mrapb(m, z=z, mask=None)
        shape_after = m.shape

        self.assertTrue(shape_before == shape_after)


if __name__ == "__main__":
    unittest.main()
