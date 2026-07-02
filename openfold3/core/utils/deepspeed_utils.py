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

"""Helpers for querying DeepSpeed state without importing DeepSpeed.

openfold3 imports deepspeed lazily so that the default (non-DeepSpeed) code path
never triggers deepspeed's import-time side effects (including a Python 3.14+
``torch.jit.script`` deprecation warning). These predicates answer "is DeepSpeed
active?" by looking at ``sys.modules``: if deepspeed was never imported it cannot
have initialized distributed comm or configured activation checkpointing, so they
return False without importing it.
"""

import sys


def deepspeed_is_initialized() -> bool:
    """Whether DeepSpeed distributed comm is initialized. Never imports deepspeed."""
    ds = sys.modules.get("deepspeed")
    return ds is not None and ds.comm.comm.is_initialized()


def deepspeed_checkpointing_is_configured() -> bool:
    """Whether DeepSpeed activation checkpointing is configured.

    Never imports deepspeed.
    """
    ds = sys.modules.get("deepspeed")
    return ds is not None and ds.checkpointing.is_configured()
