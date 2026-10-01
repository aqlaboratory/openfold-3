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

"""End-to-end reproducibility tests for inference.

These cover the two independent factors behind the determinism fix in PR #320:

- per-datapoint feature seeding in ``InferenceDataset``, and
- deterministic segmented atom-to-token aggregation in eval mode.

Both are exercised the way a user meets them: build a real runner, retrieve a
datapoint, and run the model, with the global RNG deliberately polluted in
between. Assertions are bitwise equality rather than a tolerance.

The two tests differ in what they can detect, and the distinction matters when
reading a green run:

- ``test_features_are_bitwise_repeatable_under_rng_pollution`` is the
  discriminating test for the seeding fix. Verified as a negative control:
  with per-datapoint seeding stubbed out it fails, and with the fix it passes.
- ``test_model_outputs_are_bitwise_repeatable`` is a guard on the forward path,
  not a detector for the seeding bug. It passes either way, because production
  ``predict_step`` reseeds after feature creation and before the forward, which
  overwrites any leaked generator state. It is kept so that a future change
  which breaks forward reproducibility fails here rather than in a user's run.

**Do not add the ``device`` pytest fixture to these tests.** It turns on
``torch.use_deterministic_algorithms``, which makes ``scatter_add``
deterministic and would mask the very nondeterminism under test.

Run with:
    pytest openfold3/tests/inference/test_inference_determinism.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from openfold3.core.config import config_utils
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
from openfold3.entry_points.validator import InferenceExperimentConfig
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from openfold3.tests.utils.compare_utils import skip_unless_accelerator_available

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QUERY_JSON = (
    REPO_ROOT / "examples" / "example_inference_inputs" / "query_ubiquitin.json"
)
DEFAULT_RUNNER_YAML = (
    REPO_ROOT / "examples" / "example_runner_yamls" / "smoke_inference.yml"
)

#: Features whose bitwise stability depends on the per-datapoint seeding. The
#: conformer-dependent entries are the sensitive ones: ligand conformers are
#: generated with RDKit drawing from the global RNG.
FEATURE_KEYS = ("ref_pos", "ref_mask", "ref_space_uid")

#: Model outputs compared bitwise between repeats.
OUTPUT_KEYS = ("atom_positions_predicted", "plddt_logits", "pae_logits")

SEED = 42


def _pollute_rng() -> None:
    """Advance every global RNG stream, so a repeat cannot pass by inertia."""
    for _ in range(512):
        random.random()
    np.random.randn(16)
    torch.randn(16)


def _flatten_tensors(value, prefix: str = "") -> dict[str, torch.Tensor]:
    """Collect leaf tensors of a nested output structure under dotted paths."""
    tensors: dict[str, torch.Tensor] = {}
    if torch.is_tensor(value):
        tensors[prefix or "output"] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            tensors.update(_flatten_tensors(child, name))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            name = f"{prefix}.{index}" if prefix else str(index)
            tensors.update(_flatten_tensors(child, name))
    return tensors


def _capture_outputs(output) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    for key, tensor in _flatten_tensors(output).items():
        for wanted in OUTPUT_KEYS:
            if key == wanted or key.endswith(wanted):
                captured[wanted] = tensor.detach().clone()
                break
    return captured


def _assert_bitwise_equal(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor], label: str
) -> None:
    shared = sorted(set(first) & set(second))
    assert shared, f"{label}: no comparable keys in both captures"
    for key in shared:
        a, b = first[key], second[key]
        assert a.shape == b.shape, (
            f"{label}: {key} shape changed between repeats "
            f"({tuple(a.shape)} vs {tuple(b.shape)})"
        )
        assert a.dtype == b.dtype, f"{label}: {key} dtype changed between repeats"
        assert torch.equal(a, b), (
            f"{label}: {key} is not bitwise identical between repeats "
            f"(max abs diff {(a.float() - b.float()).abs().max().item():.6g})"
        )


@pytest.fixture(scope="module")
def runner() -> InferenceExperimentRunner:
    """A real inference runner with one seed, no network, no diffusion sampling."""
    if not DEFAULT_QUERY_JSON.exists() or not DEFAULT_RUNNER_YAML.exists():
        pytest.skip("Example inference inputs are not available")

    runner_args = config_utils.load_yaml(DEFAULT_RUNNER_YAML)
    runner_args.setdefault("experiment_settings", {})["seeds"] = [SEED]
    runner_args.setdefault("data_module_args", {})["num_workers"] = 0

    inference_runner = InferenceExperimentRunner(
        InferenceExperimentConfig(**runner_args),
        num_diffusion_samples=1,
        use_msa_server=False,
        use_templates=False,
    )
    # Keep everything resident: offloading would change which device the
    # aggregation runs on and confound what this test is measuring.
    memory = inference_runner.model_config.settings.memory.eval
    memory.offload_inference.token_cutoff = 10_000_000
    memory.use_deepspeed_evo_attention = False
    memory.use_triton_triangle_kernels = False

    try:
        inference_runner.setup()
    except ValueError as e:
        if "is not a valid file or directory" in str(e):
            pytest.skip("No checkpoint files available")
        raise

    inference_runner.inference_query_set = InferenceQuerySet.from_json(
        DEFAULT_QUERY_JSON
    )
    return inference_runner


def _get_batch(runner: InferenceExperimentRunner) -> dict:
    """Retrieve one batch from the runner's predict dataloader."""
    data_module = runner.lightning_data_module
    data_module.prepare_data()
    data_module.setup()
    for batch in data_module.predict_dataloader():
        return tensor_tree_map(lambda t: t.to("cuda"), batch)
    raise RuntimeError("Predict dataloader yielded no batches")


def _capture_features(batch: dict) -> dict[str, torch.Tensor]:
    return {key: batch[key].detach().clone() for key in FEATURE_KEYS if key in batch}


@skip_unless_accelerator_available()
def test_features_are_bitwise_repeatable_under_rng_pollution(runner):
    """Two retrievals of the same datapoint give bitwise-identical features.

    The global RNG is advanced between the retrievals. Without per-datapoint
    seeding the second retrieval inherits that advanced state and produces
    different features, so this fails on the pre-fix code.
    """
    _pollute_rng()
    first = _capture_features(_get_batch(runner))

    _pollute_rng()
    second = _capture_features(_get_batch(runner))

    _assert_bitwise_equal(first, second, label="features")


@skip_unless_accelerator_available()
def test_model_outputs_are_bitwise_repeatable(runner):
    """Two forwards over the same batch give bitwise-identical outputs.

    The batch is held fixed and the RNG is polluted before each forward,
    following the same reseed-before-forward contract the production
    ``predict_step`` uses. This is the end-to-end determinism claim: the same
    seed replays to the same coordinates.
    """
    batch = _get_batch(runner)
    module = runner.lightning_module.to("cuda").eval()

    def run_once() -> dict[str, torch.Tensor]:
        _pollute_rng()
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        with torch.inference_mode():
            return _capture_outputs(module(batch))

    _assert_bitwise_equal(run_once(), run_once(), label="outputs")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv"]))
