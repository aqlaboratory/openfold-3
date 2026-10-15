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

import io
from typing import Any

import pytest
import torch

from openfold3.projects.of3_all_atom.checkpoint_migrations import (
    as_version,
    migrate_checkpoint,
    migrate_state_dict,
)
from openfold3.projects.of3_all_atom.model import MODEL_VERSION, OpenFold3
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry


def v2_0_0_state_dict(prefix: str = "") -> dict[str, torch.Tensor]:
    """Minimal 2.0.0 layout: the diffusion module is listed twice, as two copies."""
    weight = torch.randn(4, 4)
    return {
        f"{prefix}version_tensor": torch.tensor([2.0, 0.0, 0.0]),
        f"{prefix}diffusion_module.linear.weight": weight,
        f"{prefix}sample_diffusion.diffusion_module.linear.weight": weight.clone(),
        f"{prefix}pairformer_stack.linear.weight": torch.randn(4, 4),
    }


@pytest.mark.parametrize("prefix", ["", "model.", "_forward_module.model."])
def test_up_drops_alias_and_down_restores_it(prefix: str) -> None:
    original = v2_0_0_state_dict(prefix)
    state_dict = dict(original)

    migrate_state_dict(state_dict, (2, 1, 0), prefix)
    assert as_version(state_dict[f"{prefix}version_tensor"]) == (2, 1, 0)
    assert not any("sample_diffusion" in k for k in state_dict)

    migrate_state_dict(state_dict, (2, 0, 0), prefix)
    assert as_version(state_dict[f"{prefix}version_tensor"]) == (2, 0, 0)
    assert set(state_dict) == set(original)
    for k, v in original.items():
        assert torch.equal(state_dict[k], v), k
    # The restored alias shares the diffusion module tensor rather than copying it
    assert (
        state_dict[f"{prefix}sample_diffusion.diffusion_module.linear.weight"]
        is state_dict[f"{prefix}diffusion_module.linear.weight"]
    )


def test_downgraded_alias_is_stored_once_on_disk() -> None:
    state_dict = {
        "version_tensor": torch.tensor([2.0, 1.0, 0.0]),
        "diffusion_module.linear.weight": torch.randn(256, 256),
    }
    sizes: dict[tuple[int, int, int], int] = {}
    for target in [(2, 1, 0), (2, 0, 0)]:
        migrate_state_dict(state_dict, target)
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        sizes[target] = buffer.tell()
    # Only the extra key name is stored, not a second copy of the 256 KiB tensor
    assert sizes[(2, 0, 0)] - sizes[(2, 1, 0)] < 1024


@pytest.mark.parametrize(
    "state_dict",
    [
        {"version_tensor": torch.tensor([1.0, 0.0, 0.0]), "a": torch.zeros(1)},
        {"a": torch.zeros(1)},
    ],
    ids=["no_migration_path", "no_version"],
)
def test_state_dicts_without_a_path_are_unchanged(
    state_dict: dict[str, torch.Tensor],
) -> None:
    before = dict(state_dict)
    migrate_state_dict(state_dict, (2, 1, 0))
    assert state_dict.keys() == before.keys()
    for k, v in before.items():
        assert state_dict[k] is v


def test_migrate_checkpoint_handles_lightning_layout() -> None:
    ckpt: dict[str, Any] = {
        "state_dict": v2_0_0_state_dict(prefix="model."),
        "ema": {"params": v2_0_0_state_dict(), "decay": 0.999},
        "global_step": 10,
    }
    migrate_checkpoint(ckpt, MODEL_VERSION)
    assert as_version(ckpt["state_dict"]["model.version_tensor"]) == (2, 1, 0)
    assert as_version(ckpt["ema"]["params"]["version_tensor"]) == (2, 1, 0)
    assert not any("sample_diffusion" in k for k in ckpt["state_dict"])
    assert not any("sample_diffusion" in k for k in ckpt["ema"]["params"])


def test_model_loads_2_0_0_layout_strictly() -> None:
    model = OpenFold3(OF3ProjectEntry().get_model_config_with_presets())
    old = {k: v.clone() for k, v in model.state_dict().items()}
    migrate_state_dict(old, (2, 0, 0))
    assert any(k.startswith("sample_diffusion.diffusion_module.") for k in old)

    model.load_state_dict(old, strict=True)

    assert as_version(model.state_dict()["version_tensor"]) == as_version(MODEL_VERSION)
