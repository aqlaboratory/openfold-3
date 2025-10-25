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

from openfold3.core.model.layers.outer_product_mean import OuterProductMean
from openfold3.tests.config import consts


class TestOuterProductMean(unittest.TestCase):
    def test_shape(self):
        c = 31

        opm = OuterProductMean(consts.c_m, consts.c_z, c)

        m = torch.rand((consts.batch_size, consts.n_seq, consts.n_res, consts.c_m))
        mask = torch.randint(0, 2, size=(consts.batch_size, consts.n_seq, consts.n_res))
        m = opm(m, mask=mask, chunk_size=None)

        self.assertTrue(
            m.shape == (consts.batch_size, consts.n_res, consts.n_res, consts.c_z)
        )


if __name__ == "__main__":
    unittest.main()
