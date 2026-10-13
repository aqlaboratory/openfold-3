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
from torch.multiprocessing.reductions import StorageWeakRef


def storage_weak_ref(t: torch.Tensor) -> StorageWeakRef:
    """Return a weak reference to the storage (memory) behind `t`.

    Used in offload-inference paths to check that dropping a tensor actually
    releases its memory. Take the reference before dropping what should be the
    last reference to the memory, and assert that it expired afterwards::

        storage = storage_weak_ref(t)
        del t
        assert storage.expired()

    The reference tracks the storage, not the tensor object. Python reference
    counting on the tensor object isn't enough: other tensors can share its
    storage (views, `detach()`) without holding a reference to it, so the
    object can be dropped while the memory stays alive.

    Expired means PyTorch released the storage. The allocator may keep the
    memory for reuse (the CUDA caching allocator does) rather than return it to
    the device.
    """
    return StorageWeakRef(t.untyped_storage())


def move_and_free(
    tensors: list[torch.Tensor],
    index: int,
    device: torch.device | str,
) -> None:
    """Move ``tensors[index]`` to `device` and assert its old memory is freed.

    The container slot must hold the only reference to the tensor's storage
    (see `storage_weak_ref`). Nothing happens if the tensor is already on
    `device`.
    """
    storage = storage_weak_ref(tensors[index])
    moved = tensors[index].to(device)
    if moved is tensors[index]:
        return
    tensors[index] = moved
    assert storage.expired(), (
        f"Memory of tensor {index} is still referenced after moving it to {device}"
    )
