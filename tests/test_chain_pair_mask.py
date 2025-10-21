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

import torch

from openfold3.core.data.framework.single_datasets.validation import (
    make_chain_pair_mask_padded,
)


def test_chain_pair_mask_padded():
    # Chain IDs for each token [n_token]
    chain_id = torch.tensor([1, 1, 2, 1, 3, 3, 2, 1])
    interfaces_to_include = [(1, 3), (2, 3)]

    expected_chain_pair_mask = torch.tensor(
        [
            [0, 0, 0, 0],  # 0th row padding
            [0, 0, 0, 1],  # chain 1
            [0, 0, 0, 1],  # chain 2
            [0, 1, 1, 0],  # chain 3
        ]
    )
    actual_chain_pair_mask = make_chain_pair_mask_padded(
        chain_id, interfaces_to_include
    )
    assert torch.equal(expected_chain_pair_mask, actual_chain_pair_mask)
