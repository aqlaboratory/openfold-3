# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Intel Corporation
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

"""PyTorch Lightning accelerator plugin for Intel GPUs (``torch.device("xpu")``).

PyTorch Lightning ships built-in `Accelerator` implementations for CUDA, MPS, and
TPU, but not XPU (its own docs use a "hypothetical XPU" as the worked example for
writing a custom accelerator plugin). This module supplies that missing piece so
`pl.Trainer(accelerator="xpu")` (and `"auto"`/`"gpu"` when an Intel GPU is the only
accelerator present) resolves correctly.

Modeled on the XPU accelerator plugin in
https://github.com/open-edge-platform/anomalib (`engine/accelerator/xpu.py`).

Importing this module registers `XPUAccelerator` into Lightning's
`AcceleratorRegistry` as a side effect.
"""

from typing import Any

import torch
from pytorch_lightning.accelerators import Accelerator, AcceleratorRegistry


class XPUAccelerator(Accelerator):
    """Support for Intel GPUs (`torch.device("xpu")`) in PyTorch Lightning."""

    @staticmethod
    def name() -> str:
        """Name required for accelerators by pytorch-lightning >= 2.5.6."""
        return "xpu"

    @staticmethod
    def setup_device(device: torch.device) -> None:
        """Set up the specified device."""
        if device.type != "xpu":
            raise RuntimeError(f"Device should be xpu, got {device} instead")

        torch.xpu.set_device(device)

    @staticmethod
    def parse_devices(devices: str | list | torch.device) -> list:
        """Parse the `devices` Trainer argument into a list."""
        if isinstance(devices, list):
            return devices
        return [devices]

    @staticmethod
    def get_parallel_devices(devices: list) -> list[torch.device]:
        """Generate a list of parallel devices from device indices."""
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Return the number of XPU devices available."""
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        """Check whether an XPU is available."""
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    def get_device_stats(device: torch.device | str | int) -> dict[str, Any]:
        """Return XPU device stats.

        No standardized stats dict is exposed for XPU today (unlike
        `torch.cuda.memory_stats`); returns an empty dict like other minimal
        accelerator plugins.
        """
        del device  # Unused
        return {}

    def teardown(self) -> None:
        """No teardown needed; required override of the abstract base method."""


AcceleratorRegistry.register(
    XPUAccelerator.name(),
    XPUAccelerator,
    description="Accelerator supports Intel GPU (XPU) devices",
)
