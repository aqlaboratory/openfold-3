# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Outpace Bio, Inc.
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

"""Helpers for writing device-agnostic torch.amp.autocast calls."""

import torch


def autocast_device_type(x: torch.Tensor | torch.nn.Module) -> str:
    """Resolve the `device_type` to pass to `torch.amp.autocast`.

    Accepts either a tensor (uses its device) or a module (uses its first
    parameter's device), so call sites can pass whatever tensor or `self`
    they already have in scope instead of hardcoding `"cuda"`.
    """
    if isinstance(x, torch.nn.Module):
        return next(x.parameters()).device.type
    return x.device.type


def empty_device_cache(device: torch.device) -> None:
    """Release an accelerator's cached-but-unused allocator memory.

    `offload_inference` frees a tensor by moving it off `device` with
    `.cpu()`; the accelerator's allocator still holds the vacated memory in
    its own cache until this is called. No-op for `cpu` (no allocator cache
    to release) and for any accelerator without a matching `empty_cache`.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
