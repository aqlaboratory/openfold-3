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

import re
import unittest

import torch

from openfold3.core.model.layers.triangular_multiplicative_update import (
    FusedTriangleMultiplicationOutgoing,
    TriangleMultiplicationOutgoing,
)
from openfold3.tests.config import consts


# Updates pair representation z[i,j] by projecting to two gated vectors (a, b),
# contracting along a shared dimension (outgoing vs incoming), then projecting
# back. "Outgoing" contracts over the starting node, "Incoming" over the ending
# node. Shape-preserving: [*, N, N, C_z] -> [*, N, N, C_z].
class TestTriangularMultiplicativeUpdate(unittest.TestCase):
    def test_shape(self):
        # c_z: pair representation channel dim (128 in production)
        c_z = consts.c_z
        # c: hidden projection dim (production uses ~128; smaller here for speed)
        c = 11

        # Multimer v3 uses a fused variant (single projection split into a, b)
        # vs separate projections for each
        if re.fullmatch("^model_[1-5]_multimer_v3$", consts.model_preset):
            tm = FusedTriangleMultiplicationOutgoing(
                c_z,
                c,
            )
        else:
            tm = TriangleMultiplicationOutgoing(
                c_z,
                c,
            )

        # NOTE: n_res is set to c_z (128) here, not consts.n_res (22)
        n_res = consts.c_z
        batch_size = consts.batch_size

        # Pair representation: [batch, N_residues, N_residues, C_z]
        x = torch.rand((batch_size, n_res, n_res, c_z))
        # Binary mask: which residue pairs are valid
        mask = torch.randint(0, 2, size=(batch_size, n_res, n_res))
        shape_before = x.shape
        x = tm(x, mask)
        shape_after = x.shape

        # Shape must be preserved for the residual addition z = z + tri_mul(z)
        self.assertTrue(shape_before == shape_after)


if __name__ == "__main__":
    unittest.main()
