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

__all__ = ["core", "projects", "entry_points", "run_openfold"]

import importlib.util

import gemmi
from packaging import version

from . import hacks  # noqa: F401

if version.parse(gemmi.__version__) >= version.parse("0.7.3"):
    gemmi.set_leak_warnings(False)

if importlib.util.find_spec("deepspeed") is not None:
    import deepspeed

    # TODO: Resolve this later
    # This is a hack to prevent deepspeed from doing the triton matmul autotuning
    # This has weird effects with hanging if libaio is not installed and can
    # cause restart errors if run is preempted in the middle of autotuning
    deepspeed.HAS_TRITON = False
