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

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from openfold3.entry_points import import_utils

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA GPU",
)


def _env_without_alloc_conf() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k not in import_utils._ALLOC_CONF_ENV_VARS
    }


@pytest.fixture
def clean_env():
    with patch.dict(os.environ, _env_without_alloc_conf(), clear=True):
        yield


@pytest.fixture
def nvidia_gpu():
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.version, "hip", None),
    ):
        yield


@pytest.fixture
def set_allocator_settings(nvidia_gpu):
    with patch.object(
        torch._C, "_accelerator_setAllocatorSettings", create=True
    ) as setter:
        yield setter


def test_enables_expandable_segments_by_default(clean_env, set_allocator_settings):
    import_utils._configure_cuda_allocator()
    set_allocator_settings.assert_called_once_with("expandable_segments:True")


def test_falls_back_to_deprecated_setter_on_older_torch(clean_env, nvidia_gpu):
    with (
        patch.object(torch, "_C", SimpleNamespace()),
        patch.object(torch.cuda.memory, "_set_allocator_settings") as setter,
    ):
        import_utils._configure_cuda_allocator()
    setter.assert_called_once_with("expandable_segments:True")


def test_does_not_mutate_environment(clean_env, set_allocator_settings):
    before = dict(os.environ)
    import_utils._configure_cuda_allocator()
    assert dict(os.environ) == before


def test_opt_out(clean_env, set_allocator_settings):
    import_utils._configure_cuda_allocator(expandable_segments=False)
    set_allocator_settings.assert_not_called()


@pytest.mark.parametrize("key", import_utils._ALLOC_CONF_ENV_VARS)
def test_respects_user_env(clean_env, set_allocator_settings, key):
    with patch.dict(os.environ, {key: "max_split_size_mb:128"}):
        import_utils._configure_cuda_allocator()
    set_allocator_settings.assert_not_called()


def test_skipped_on_rocm(clean_env, set_allocator_settings):
    with patch.object(torch.version, "hip", "6.4"):
        import_utils._configure_cuda_allocator()
    set_allocator_settings.assert_not_called()


def test_skipped_without_cuda(clean_env, set_allocator_settings):
    with patch.object(torch.cuda, "is_available", return_value=False):
        import_utils._configure_cuda_allocator()
    set_allocator_settings.assert_not_called()


@requires_cuda
def test_expandable_segments_take_effect_via_cli_import_path():
    # Allocator settings are process-global, so check in a fresh interpreter
    # that imports openfold3 the same way the CLI does.
    code = textwrap.dedent(
        """
        import torch
        from openfold3.run_openfold import _configure_torch_backend

        _configure_torch_backend()
        x = torch.empty(64 * 2**20, device="cuda", dtype=torch.uint8)
        segments = torch.cuda.memory._snapshot()["segments"]
        assert all(s["is_expandable"] for s in segments), segments
        """
    )
    subprocess.run(
        [sys.executable, "-c", code], env=_env_without_alloc_conf(), check=True
    )
