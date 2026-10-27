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

"""Tests for ``storage_weak_ref`` and ``move_and_free``."""

import contextlib

import pytest
import torch

from openfold3.core.model.memory import move_and_free, storage_weak_ref

# Torch has subtly different behavior with views under these different
# settings, so test all of them.
GRAD_MODES = {
    "grad": contextlib.nullcontext,
    "no_grad": torch.no_grad,
    "inference_mode": torch.inference_mode,
}

# Where to move a tensor from each test device. Moving to "meta" allocates no
# memory but is a real device change, so the CPU tests need no accelerator.
TARGET_DEVICE = {"cpu": "meta", "cuda": "cpu"}

N_ELEM = 1024


@pytest.fixture(params=list(GRAD_MODES.values()), ids=list(GRAD_MODES))
def grad_mode(request):
    with request.param():
        yield


@pytest.mark.usefixtures("grad_mode")
class TestStorageWeakRef:
    def test_sole_tensor(self, device):
        t = torch.zeros(N_ELEM, device=device)
        storage = storage_weak_ref(t)
        del t
        assert storage.expired()

    def test_extra_holder(self, device):
        t = torch.zeros(N_ELEM, device=device)
        alias = t
        storage = storage_weak_ref(t)
        del t
        assert not storage.expired()
        del alias
        assert storage.expired()

    def test_sole_view(self, device):
        t = torch.zeros(N_ELEM, device=device)
        view = t[2:]
        del t
        storage = storage_weak_ref(view)
        del view
        assert storage.expired()

    def test_view_held(self, device):
        t = torch.zeros(N_ELEM, device=device)
        view = t[2:]
        storage = storage_weak_ref(t)
        del t
        assert not storage.expired()
        del view
        assert storage.expired()

    def test_other_view_held(self, device):
        t = torch.zeros(N_ELEM, device=device)
        view = t[2:]
        other_view = t[3:]
        del t
        storage = storage_weak_ref(view)
        assert not storage.expired()
        del view
        assert not storage.expired()
        del other_view
        assert storage.expired()

    def test_drop_from_container(self, device):
        tensors = {"a": torch.zeros(N_ELEM, device=device)}
        storage = storage_weak_ref(tensors["a"])
        del tensors["a"]
        assert storage.expired()


@pytest.mark.usefixtures("grad_mode")
class TestMoveAndFree:
    def test_sole_holder(self, device):
        tensors = [
            torch.zeros(N_ELEM, device=device),
            torch.ones(N_ELEM, device=device),
        ]
        if device == "cuda":
            allocated = torch.cuda.memory_allocated()
        move_and_free(tensors, 1, TARGET_DEVICE[device])
        assert tensors[1].device == torch.device(TARGET_DEVICE[device])
        if device == "cuda":
            # The CPU doesn't have the same sort of allocated memory tracking.
            assert torch.cuda.memory_allocated() < allocated

    def test_extra_holder_raises(self, device):
        tensors = [
            torch.zeros(N_ELEM, device=device),
            torch.ones(N_ELEM, device=device),
        ]
        alias = tensors[1]
        with pytest.raises(AssertionError, match="still referenced"):
            move_and_free(tensors, 1, TARGET_DEVICE[device])
        del alias

    def test_view_held_raises(self, device):
        tensors = [
            torch.zeros(N_ELEM, device=device),
            torch.ones(N_ELEM, device=device),
        ]
        view = tensors[1][2:]
        with pytest.raises(AssertionError, match="still referenced"):
            move_and_free(tensors, 1, TARGET_DEVICE[device])
        del view

    def test_same_device_is_noop(self, device):
        tensors = [
            torch.zeros(N_ELEM, device=device),
            torch.ones(N_ELEM, device=device),
        ]
        alias = tensors[1]
        move_and_free(tensors, 1, device)
        assert tensors[1] is alias
