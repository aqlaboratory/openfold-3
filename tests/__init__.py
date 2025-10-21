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

import importlib

if importlib.util.find_spec("deepspeed") is not None:
    import deepspeed

    # TODO: Resolve this
    # This is a hack to prevent deepspeed from doing the triton matmul autotuning
    # I'm not sure why it's doing this by default, but it's causing the tests to hang
    deepspeed.HAS_TRITON = False
