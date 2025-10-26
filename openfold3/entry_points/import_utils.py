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

"""
Manage imports run_openfold.py
"""
# ruff: noqa: F821


def _torch_gpu_setup():
    torch_versions = torch.__version__.split(".")
    torch_major_version = int(torch_versions[0])
    torch_minor_version = int(torch_versions[1])
    if torch_major_version > 1 or (
        torch_major_version == 1 and torch_minor_version >= 12
    ):
        # Gives a large speedup on Ampere-class GPUs
        torch.set_float32_matmul_precision("high")


def import_modules_for_inference():
    _torch_gpu_setup()


def import_modules_for_training():
    _torch_gpu_setup()
