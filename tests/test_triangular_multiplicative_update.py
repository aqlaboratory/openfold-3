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

from openfold3.core.model.layers.triangular_multiplicative_update import (
    FusedTriangleMultiplicationOutgoing,
    TriangleMultiplicationOutgoing,
)
from tests.config import consts


class TestTriangularMultiplicativeUpdate(unittest.TestCase):
    def test_shape(self):
        c_z = consts.c_z
        c = 11

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

        n_res = consts.c_z
        batch_size = consts.batch_size

        x = torch.rand((batch_size, n_res, n_res, c_z))
        mask = torch.randint(0, 2, size=(batch_size, n_res, n_res))
        shape_before = x.shape
        x = tm(x, mask)
        shape_after = x.shape

        self.assertTrue(shape_before == shape_after)


if __name__ == "__main__":
    unittest.main()
