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

__all__ = ["core", "projects", "entry_points", "run_openfold"]

import gemmi
from packaging import version

if version.parse(gemmi.__version__) >= version.parse("0.7.3"):
    gemmi.set_leak_warnings(False)

# NOTE: deepspeed is imported lazily where it is actually used (the DS Evoformer
# kernel, activation checkpointing, ZeRO checkpoint loading, and training), so that
# importing openfold3 never triggers deepspeed's import-time side effects. The
# deepspeed.HAS_TRITON autotuning hack now lives in _deepspeed_evo_attn.
