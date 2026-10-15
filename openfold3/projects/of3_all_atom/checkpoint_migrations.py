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

"""
Converts OpenFold3 state dicts between model versions (the `version_tensor`).

Each Migration converts a state dict between two adjacent versions, upwards
(`up`) and downwards (`down`). migrate_state_dict() chains them to reach a
target version. A state dict with no path to the target is left unchanged, so
the version check at load time still rejects it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

Version = tuple[int, int, int]
StateDict = dict[str, torch.Tensor]

VERSION_KEY = "version_tensor"


@dataclass(frozen=True)
class Migration:
    lower: Version
    upper: Version
    up: Callable[[StateDict, str], None]
    down: Callable[[StateDict, str], None]


_DIFFUSION_MODULE = "diffusion_module."
_SAMPLE_DIFFUSION_ALIAS = "sample_diffusion.diffusion_module."


def _drop_sample_diffusion_alias(state_dict: StateDict, prefix: str) -> None:
    """2.0.0 -> 2.1.0: SampleDiffusion no longer registers the DiffusionModule."""
    alias = prefix + _SAMPLE_DIFFUSION_ALIAS
    for k in [k for k in state_dict if k.startswith(alias)]:
        del state_dict[k]


def _add_sample_diffusion_alias(state_dict: StateDict, prefix: str) -> None:
    """2.1.0 -> 2.0.0: restores the alias keys, sharing the DiffusionModule tensors."""
    source = prefix + _DIFFUSION_MODULE
    alias = prefix + _SAMPLE_DIFFUSION_ALIAS
    for k in [k for k in state_dict if k.startswith(source)]:
        state_dict[alias + k[len(source) :]] = state_dict[k]


MIGRATIONS: list[Migration] = [
    Migration(
        lower=(2, 0, 0),
        upper=(2, 1, 0),
        up=_drop_sample_diffusion_alias,
        down=_add_sample_diffusion_alias,
    ),
]


def as_version(version: torch.Tensor | Sequence[int | str]) -> Version:
    """Parses a version_tensor or a sequence such as (2, 1, 0) or ["2", "1", "0"]."""
    values = version.long().tolist() if isinstance(version, torch.Tensor) else version
    major, minor, patch = (int(v) for v in values)
    return major, minor, patch


def migrate_state_dict(
    state_dict: StateDict, target: torch.Tensor | Version, prefix: str = ""
) -> None:
    """
    Migrates, in place, a state dict to the target model version.

    Args:
        state_dict:
            Flat state dict holding `<prefix>version_tensor`. State dicts without
            a version are left unchanged.
        target:
            Version to migrate to, e.g. MODEL_VERSION
        prefix:
            Prefix of the OpenFold3 keys, e.g. "model." for a LightningModule
    """
    key = prefix + VERSION_KEY
    if key not in state_dict:
        return

    stored = state_dict[key]
    version = as_version(stored)
    target_version = as_version(target)
    while version != target_version:
        if version < target_version:
            step = next((m for m in MIGRATIONS if m.lower == version), None)
            if step is None or step.upper > target_version:
                return
            step.up(state_dict, prefix)
            version = step.upper
        else:
            step = next((m for m in MIGRATIONS if m.upper == version), None)
            if step is None or step.lower < target_version:
                return
            step.down(state_dict, prefix)
            version = step.lower
        state_dict[key] = torch.tensor(
            version, dtype=stored.dtype, device=stored.device
        )


def migrate_checkpoint(ckpt: dict[str, Any], target: torch.Tensor | Version) -> None:
    """
    Migrates, in place, every OpenFold3 state dict in a loaded checkpoint.

    Handles parameter files (a flat state dict), Lightning checkpoints
    ("state_dict" and "ema") and DeepSpeed model states ("module").
    """
    state_dicts = [
        ckpt,
        ckpt.get("state_dict"),
        ckpt.get("module"),
        (ckpt.get("ema") or {}).get("params"),
    ]
    for state_dict in state_dicts:
        if not isinstance(state_dict, dict):
            continue
        version_keys = [
            k for k in state_dict if isinstance(k, str) and k.endswith(VERSION_KEY)
        ]
        for k in version_keys:
            migrate_state_dict(state_dict, target, prefix=k[: -len(VERSION_KEY)])
