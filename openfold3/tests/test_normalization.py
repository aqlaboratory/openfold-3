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

import openfold3.tests.compare_utils as compare_utils
from openfold3.core.model.primitives.normalization import LayerNorm


@compare_utils.skip_unless_cuda_available()
# Test both the vectorized (C%4==0) and non-vectorized (C%4!=0) code paths for both batch>=2^23 and numel>=2^32
@pytest.mark.parametrize(
    ("batch", "C"),
    [
        pytest.param(2**23, 4, id="2^23_4"),
        pytest.param(2**23 - 1, 3, id="2^23-1_3"),
        # In testing with HIP, there's actually an "invalid configuration argument" error at exactly batch=2^23 as opposed to bad output
        pytest.param(2**23, 3, id="2^23_3"),
        pytest.param(2**23 + 1, 3, id="2^23+1_3"),
        pytest.param(2**23 + 1, 4, id="2^23+1_4"),
        pytest.param(2**22, 2**10, marks=pytest.mark.slow, id="2^22_2^10"),
        pytest.param(2**22 + 1, 2**10, marks=pytest.mark.slow, id="2^22+1_2^10"),
    ],
)
def test_layer_norm_overflow_bug_workaround(batch, C, seeded_rng):
    """Test we don't hit torch bugs in very large layernorms.

    See comments in layer norm implementation for details.
    """
    ln = LayerNorm(C).cuda()

    input_row = torch.randn(C, device="cuda")
    x = input_row.to(torch.bfloat16).unsqueeze(0).expand(batch, C).contiguous()

    expected_row = (input_row - torch.mean(input_row)) / torch.sqrt(
        torch.var(input_row, correction=0) + ln.eps
    )
    expected = expected_row.to(torch.bfloat16).unsqueeze(0).expand(batch, C)

    out = ln(x)

    torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    unittest.main()
