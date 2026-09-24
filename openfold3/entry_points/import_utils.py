# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
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

"""
Manage imports run_openfold.py
"""
# ruff: noqa: F821
# ruff: noqa: F401

import logging
import os

logger = logging.getLogger(__name__)


def _enable_tf32():
    import torch

    torch_versions = torch.__version__.split(".")
    torch_major_version = int(torch_versions[0])
    torch_minor_version = int(torch_versions[1])
    if torch_major_version > 1 or (
        torch_major_version == 1 and torch_minor_version >= 12
    ):
        # Gives a large speedup on Ampere-class GPUs
        torch.set_float32_matmul_precision("high")


# Any of these means the user configured the allocator themselves. Note that
# PYTORCH_CUDA_ALLOC_CONF *replaces* PYTORCH_ALLOC_CONF (it does not merge), so
# we must never set one when the user has set the other.
_ALLOC_CONF_ENV_VARS = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
)


def _configure_cuda_allocator(expandable_segments: bool = True) -> None:
    """Enable expandable segments in the CUDA caching allocator.

    Uses the runtime allocator API rather than an environment variable: CUDA may
    already be initialised by the time this runs (e.g. ``import deepspeed``), at
    which point the environment variable is no longer read.
    """
    import torch

    if not expandable_segments or not torch.cuda.is_available():
        return
    if torch.version.hip is not None:
        return
    user_set = [k for k in _ALLOC_CONF_ENV_VARS if k in os.environ]
    if user_set:
        logger.info(f"Allocator configured via {user_set}; leaving it untouched")
        return

    set_allocator_settings = getattr(
        torch._C,
        "_accelerator_setAllocatorSettings",
        torch.cuda.memory._set_allocator_settings,
    )
    set_allocator_settings("expandable_segments:True")
    logger.info("Enabled CUDA allocator expandable_segments")


def _configure_torch_backend(expandable_segments: bool = True):
    """Apply backend settings"""
    _configure_cuda_allocator(expandable_segments)

    import torch

    # Force the cuBLAS backend on AMD/ROCm to match the numerics of
    # NVIDIA-trained models.
    if torch.cuda.is_available() and torch.version.hip is not None:
        torch.backends.cuda.preferred_blas_library("cublas")
