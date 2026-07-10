# Copyright 2026 AlQuraishi Laboratory
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

import pytest
import torch

from openfold3.core.loss.loss_utils import loss_masked_batch_mean


@pytest.mark.usefixtures("seeded_rng")
class TestLossUtils(unittest.TestCase):
    def test_loss_masked_batch_mean_is_precise(self):
        # Confirm we don't lose precision. The input is constructed such that if
        # we used bfloat16 to sum the mask divisor, we would lose precision
        # (because 50,000 is not precisely representable in bf16).
        n = 50_000
        value = 3.0
        loss = torch.full((n, 1), value)
        weight = torch.ones((n, 1))

        result = loss_masked_batch_mean(
            loss=loss, weight=weight, apply_weight=False, eps=1e-8
        )

        self.assertAlmostEqual(result.item(), value, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
