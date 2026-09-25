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

"""Helpers for measuring CUDA caching allocator behaviour in tests."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

MiB = 1024**2


@dataclass(frozen=True)
class CudaMemoryMetrics:
    """Snapshot of CUDA caching allocator state after a workload."""

    peak_allocated_bytes: int
    """What tensors actually needed (the useful work)."""
    peak_reserved_bytes: int
    """What the caching allocator reserved from CUDA (the real GPU cost)."""
    peak_inactive_split_bytes: int
    """Free blocks created by splitting larger segments — the most direct
    measure of fragmentation."""
    num_alloc_retries: int
    """Times the allocator failed to find a block, freed cached memory, and
    retried. The practical cost of fragmentation."""
    num_ooms: int
    """Out-of-memory errors (catastrophic fragmentation)."""
    peak_segments: int
    """Number of cudaMalloc segments at peak."""

    def __str__(self) -> str:
        return (
            f"allocated={self.peak_allocated_bytes / MiB:.0f} MiB "
            f"reserved={self.peak_reserved_bytes / MiB:.0f} MiB "
            f"inactive_split={self.peak_inactive_split_bytes / MiB:.0f} MiB "
            f"segments={self.peak_segments} "
            f"alloc_retries={self.num_alloc_retries} ooms={self.num_ooms}"
        )


def get_cuda_memory_metrics(device: torch.device | str = "cuda") -> CudaMemoryMetrics:
    """Collect CUDA caching allocator metrics after a workload."""
    stats = torch.cuda.memory_stats(device)
    return CudaMemoryMetrics(
        peak_allocated_bytes=stats["allocated_bytes.all.peak"],
        peak_reserved_bytes=stats["reserved_bytes.all.peak"],
        peak_inactive_split_bytes=stats["inactive_split_bytes.all.peak"],
        num_alloc_retries=stats["num_alloc_retries"],
        num_ooms=stats["num_ooms"],
        peak_segments=stats["segment.all.peak"],
    )


def measure_cuda_memory(
    workload: Callable[[], object], device: torch.device | str = "cuda"
) -> CudaMemoryMetrics:
    """Run ``workload`` from an empty allocator cache and return its metrics."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    workload()
    torch.cuda.synchronize(device)
    return get_cuda_memory_metrics(device)
