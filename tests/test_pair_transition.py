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

import numpy as np
import torch

from openfold3.core.model.layers.transition import ReLUTransition
from openfold3.core.utils.tensor_utils import tree_map
from tests.config import consts


class TestPairTransition(unittest.TestCase):
    def test_shape(self):
        c_z = consts.c_z
        n = 4

        pt = ReLUTransition(c_in=c_z, n=n)

        batch_size = consts.batch_size
        n_res = consts.n_res

        z = torch.rand((batch_size, n_res, n_res, c_z))
        mask = torch.randint(0, 2, size=(batch_size, n_res, n_res))
        shape_before = z.shape
        z = pt(z, mask=mask, chunk_size=None)
        shape_after = z.shape

        self.assertTrue(shape_before == shape_after)


if __name__ == "__main__":
    unittest.main()
