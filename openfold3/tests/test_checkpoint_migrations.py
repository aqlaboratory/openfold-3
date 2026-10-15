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

import pytest
import torch

from openfold3.projects.of3_all_atom.checkpoint_migrations import migrate_state_dict


@pytest.mark.parametrize(
    "state_dict",
    [
        {"version_tensor": torch.tensor([1.0, 0.0, 0.0]), "a": torch.zeros(1)},
        {"a": torch.zeros(1)},
    ],
    ids=["no_migration_path", "no_version"],
)
def test_state_dicts_without_a_path_are_unchanged(
    state_dict: dict[str, torch.Tensor],
) -> None:
    before = dict(state_dict)
    migrate_state_dict(state_dict, (2, 1, 0))
    assert state_dict.keys() == before.keys()
    for k, v in before.items():
        assert state_dict[k] is v
