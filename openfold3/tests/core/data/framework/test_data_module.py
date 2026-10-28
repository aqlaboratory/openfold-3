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

import torch
from biotite.structure import AtomArray

from openfold3.core.data.framework.data_module import DataModule


def test_transfer_batch_to_device_passes_atom_arrays_through():
    data_module = object.__new__(DataModule)

    atom_array = AtomArray(5)
    batch = {"atom_array": [atom_array], "seed": torch.tensor([123])}

    batch = data_module.transfer_batch_to_device(batch, torch.device("cpu"), 0)

    # biotite>=2 AtomArrays are Sequences of Atoms, which will be misinterpreted
    # as a Sequence of Tensors by PyTorch's default collate function.
    # This test ensures that the atom array is not changed by PLT.
    assert batch["atom_array"][0] is atom_array
    assert torch.equal(batch["seed"], torch.tensor([123]))
