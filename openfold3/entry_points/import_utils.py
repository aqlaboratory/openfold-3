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


def _configure_torch_backend():
    """Apply backend settings"""

    # NOTE: only set if we are NOT on AMD
    if "PYTORCH_HIP_ALLOC_CONF" not in os.environ:
        # NOTE: PYTORCH_ALLOC_CONF is used more often in newer torch versions, but would require
        # a version pin
        key = "PYTORCH_CUDA_ALLOC_CONF"
        value = "expandable_segments:True"
        os.environ.setdefault(key, value)
        logger.info(f"Setting env var ${key}={value}")

    import torch

    # Force the cuBLAS backend on AMD/ROCm to match the numerics of
    # NVIDIA-trained models.
    if torch.cuda.is_available() and torch.version.hip is not None:
        torch.backends.cuda.preferred_blas_library("cublas")
