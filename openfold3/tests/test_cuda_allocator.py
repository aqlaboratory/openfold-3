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

"""CUDA caching allocator behaviour on OF3 training steps, expandable segments off/on.

Each configuration runs ``openfold3.tests.utils.allocator_workload`` in a fresh
interpreter, since allocator settings are process-global. Per-crop metrics are
compared pairwise and pinned in a platform-dependent snapshot; regenerate with
``pytest openfold3/tests/test_cuda_allocator.py -m slow --force-regen``.
"""

import dataclasses
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from openfold3.entry_points import import_utils
from openfold3.tests.utils.allocator_workload import DEFAULT_N_TOKENS
from openfold3.tests.utils.cuda_memory import CudaMemoryMetrics, MiB

pytestmark = [
    pytest.mark.slow,
    pytest.mark.platform_dependent_snapshot,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.version.hip is not None,
        reason="requires an NVIDIA GPU",
    ),
]


@dataclass(frozen=True)
class WorkloadResult:
    is_expandable: bool
    n_tokens: list[int]
    steps: list[CudaMemoryMetrics]


def _run_workload(expandable_segments: bool, out: Path) -> WorkloadResult:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in import_utils._ALLOC_CONF_ENV_VARS
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "openfold3.tests.utils.allocator_workload",
            "--expandable-segments",
            "on" if expandable_segments else "off",
            "--out",
            str(out),
        ],
        env=env,
        check=True,
    )
    result = json.loads(out.read_text())
    return WorkloadResult(
        is_expandable=result["is_expandable"],
        n_tokens=result["n_tokens"],
        steps=[CudaMemoryMetrics(**step) for step in result["steps"]],
    )


def _format_table(off: WorkloadResult, on: WorkloadResult) -> str:
    rows = [
        f"{'step':>4} {'tokens':>6} | {'allocated MiB':>15} | {'reserved MiB':>15}"
        f" | {'inactive split MiB':>18} | {'reserved/alloc':>14}",
        f"{'':>11} | {'off':>7} {'on':>7} | {'off':>7} {'on':>7}"
        f" | {'off':>8} {'on':>9} | {'off':>6} {'on':>7}",
    ]
    for i, (n_token, a, b) in enumerate(zip(off.n_tokens, off.steps, on.steps)):
        rows.append(
            f"{i:>4} {n_token:>6}"
            f" | {a.peak_allocated_bytes / MiB:>7.0f} {b.peak_allocated_bytes / MiB:>7.0f}"
            f" | {a.peak_reserved_bytes / MiB:>7.0f} {b.peak_reserved_bytes / MiB:>7.0f}"
            f" | {a.peak_inactive_split_bytes / MiB:>8.0f}"
            f" {b.peak_inactive_split_bytes / MiB:>9.0f}"
            f" | {a.peak_reserved_bytes / a.peak_allocated_bytes:>5.2f}x"
            f" {b.peak_reserved_bytes / b.peak_allocated_bytes:>6.2f}x"
        )
    return "\n".join(rows)


@pytest.fixture(scope="module")
def workloads(tmp_path_factory) -> tuple[WorkloadResult, WorkloadResult]:
    tmp_path = tmp_path_factory.mktemp("allocator_workload")
    off = _run_workload(False, tmp_path / "off.json")
    on = _run_workload(True, tmp_path / "on.json")
    print(f"\nexpandable_segments off vs on, per crop:\n{_format_table(off, on)}")
    return off, on


def test_setting_applied(workloads):
    off, on = workloads
    assert not off.is_expandable
    assert on.is_expandable
    assert off.n_tokens == on.n_tokens


@pytest.mark.parametrize(
    "step",
    range(len(DEFAULT_N_TOKENS)),
    ids=[f"crop{i}-{n}tok" for i, n in enumerate(DEFAULT_N_TOKENS)],
)
def test_per_crop_off_vs_on(workloads, step):
    off, on = (w.steps[step] for w in workloads)
    n_token = workloads[0].n_tokens[step]
    context = f"crop {step} ({n_token} tokens)\n  off: {off}\n  on:  {on}"

    assert off.num_ooms == on.num_ooms == 0, context
    # Same: both runs did the same work...
    assert on.peak_allocated_bytes == pytest.approx(
        off.peak_allocated_bytes, rel=0.02
    ), context
    # ...different: the allocator held on to less memory to do it,
    assert on.peak_reserved_bytes < off.peak_reserved_bytes, context
    # because it no longer leaves unusable slivers in split blocks.
    assert on.peak_inactive_split_bytes <= off.peak_inactive_split_bytes, context
    assert on.peak_inactive_split_bytes == 0, context


def test_fragmentation_stays_sensible(workloads):
    """With expandable segments, reserved tracks the largest crop seen so far.

    Memory is cached across steps, so a small crop after a large one is
    compared with the large one's allocation, not its own.
    """
    _, on = workloads
    largest_allocated = 0
    for step, metrics in enumerate(on.steps):
        largest_allocated = max(largest_allocated, metrics.peak_allocated_bytes)
        assert metrics.peak_reserved_bytes <= 1.15 * largest_allocated, (
            f"crop {step}: {metrics}"
        )


def test_matches_snapshot(workloads, ndarrays_regression):
    """Pin every metric for every crop, both settings, to catch drift in either."""
    off, on = workloads
    arrays = {"n_tokens": np.array(off.n_tokens)}
    for mode, result in (("off", off), ("on", on)):
        for field in dataclasses.fields(CudaMemoryMetrics):
            arrays[f"{mode}_{field.name}"] = np.array(
                [getattr(m, field.name) for m in result.steps]
            )
    ndarrays_regression.check(arrays, default_tolerance=dict(rtol=0.02, atol=0))
