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

"""Integration test for training on the small local PDB subset.

Requires the subset produced by ``scripts/datasets/generate_subset_cache.py`` +
``scripts/datasets/download_subset.py`` to already exist locally under
``<openfold3-directory>/datasets`` (skips otherwise -- these files are gitignored,
not fetched by CI, and must be generated/downloaded by hand).

The runner yaml (``datasets/train_pdb_subset.yaml``) already bakes in a small
test case -- 8 train / 4 val structures, gradient checkpointing, small MSA chunk size,
diffusion loss chunking, etc, picked by ``generate_subset_cache.py``. ``full_subset``
runs it as checked in (only ``output_dir`` is redirected). ``smoke`` additionally trims
epoch length/count and disables dataloader workers, for a faster opt-in sanity check.

To mimic a real user's workflow (not just the internal Python API), both cases invoke
the actual ``run_openfold train --runner-yaml ...`` console script as a subprocess --
the same command documented in docs/source/training.md -- against a materialized copy
of the runner yaml with the per-case overrides applied. Its output (including
Lightning's per-step progress bar) streams live to the real terminal via
``capsys.disabled()``, regardless of pytest's capture mode -- no need for ``-s``.

Run with:
    pytest openfold3/tests/test_training_full.py
    pytest openfold3/tests/test_training_full.py -k smoke  # fast case only
"""

import logging
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

import openfold3
from openfold3.core.config import config_utils
from openfold3.entry_points.validator import TrainingExperimentConfig
from openfold3.tests.utils.compare_utils import skip_unless_cuda_available

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# OPENFOLD_PDB_SUBSET_DIR lets CI point this at wherever the cached subset
# lives (e.g. outside the checked-out workspace); defaults to the local dev
# convention of <openfold3-directory>/datasets otherwise.
DATASETS_DIR = Path(
    os.environ.get("OPENFOLD_PDB_SUBSET_DIR")
    or (Path(openfold3.__file__).resolve().parent.parent / "datasets")
)
RUNNER_YAML = DATASETS_DIR / "train_pdb_subset.yaml"
PDB_TRAINING_SET_DIR = DATASETS_DIR / "pdb_training_set"

RUN_OPENFOLD = shutil.which("run_openfold")


@dataclass(frozen=True)
class TrainCase:
    name: str
    # Deep-merged onto the checked-in yaml; empty means "run it as-is".
    overrides: dict = field(default_factory=dict)
    timeout_s: int = 900


# The default runner.yaml settings can be found in
# scripts/datasets/pdb_subset_helpers.py `build_runner_yaml_config`
CASES = [
    pytest.param(
        TrainCase(
            "smoke",
            overrides={
                # No dataloader worker subprocesses (each forks a copy of the dataset
                # cache/process state -- a real contributor to memory blowups on
                # constrained machines) and a single-batch epoch.
                "data_module_args": {
                    "num_workers": 0,
                    "num_workers_validation": 0,
                    "epoch_len": 1,
                },
                "pl_trainer_args": {
                    "max_epochs": 1,
                    "log_every_n_steps": 1,
                },
            },
            timeout_s=600,
        ),
        marks=pytest.mark.slow,
        id="smoke",
    ),
    pytest.param(
        TrainCase("full_subset", timeout_s=1800),
        marks=pytest.mark.slow,
        id="full_subset",
    ),
]


def _require_local_subset() -> None:
    """Skip if the local PDB subset hasn't been generated/downloaded yet."""
    missing = [p for p in (RUNNER_YAML, PDB_TRAINING_SET_DIR) if not p.exists()]
    if missing:
        pytest.skip(
            "Local PDB training subset not found: "
            f"{', '.join(str(p) for p in missing)}. Run "
            "`python scripts/datasets/generate_subset_cache.py` and "
            "`python scripts/datasets/download_subset.py` first "
            "(see scripts/datasets/), or `pixi run setup-pdb-subset`."
        )


def _run_streaming(cmd: list[str], timeout_s: int, capsys) -> tuple[int, str]:
    """Run `cmd`, printing its combined stdout/stderr live as it arrives.

    Unlike `subprocess.run(capture_output=True)`, this gives real-time progress
    visibility (e.g. Lightning's per-step progress bar) instead of a silent
    block until the process exits. `capsys.disabled()` forces the output to
    the real terminal regardless of pytest's capture mode, so it's visible
    without needing to remember `-s`. A watchdog timer kills the process after
    `timeout_s` even if it produces no output at all (a true hang), which
    `for line in proc.stdout` alone wouldn't catch.

    Returns (returncode, combined_output).
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    timed_out = threading.Event()

    def _kill_on_timeout():
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, _kill_on_timeout)
    timer.start()

    lines = []
    try:
        with capsys.disabled():
            for line in proc.stdout:
                print(line, end="", flush=True)
                lines.append(line)
    finally:
        timer.cancel()
        proc.wait()

    output = "".join(lines)
    if timed_out.is_set():
        pytest.fail(
            f"`{' '.join(cmd)}` timed out after {timeout_s}s and was killed.\n"
            f"--- output (last 4000 chars) ---\n{output[-4000:]}"
        )
    return proc.returncode, output


@skip_unless_cuda_available()
@pytest.mark.training_verification
@pytest.mark.parametrize("case", CASES)
def test_train(case: TrainCase, tmp_path, capsys):
    """`run_openfold train --runner-yaml ...` on the local subset writes a checkpoint."""
    _require_local_subset()
    if RUN_OPENFOLD is None:
        pytest.skip("`run_openfold` console script not found on PATH")

    config_dict = config_utils.load_yaml(RUNNER_YAML)
    config_dict["experiment_settings"]["output_dir"] = str(tmp_path)
    if case.overrides:
        config_dict = config_utils.deep_update(config_dict, case.overrides)

    # Fail fast with a clear pydantic error here rather than parsing it out of
    # subprocess stderr, before materializing/launching the actual CLI call.
    TrainingExperimentConfig(**config_dict)

    runner_yaml = tmp_path / "runner_config.yaml"
    runner_yaml.write_text(yaml.safe_dump(config_dict, sort_keys=False))

    returncode, output = _run_streaming(
        [RUN_OPENFOLD, "train", "--runner-yaml", str(runner_yaml)],
        case.timeout_s,
        capsys,
    )
    assert returncode == 0, (
        f"`run_openfold train` exited {returncode}\n"
        f"--- output (last 4000 chars) ---\n{output[-4000:]}"
    )

    checkpoints = list((tmp_path / "checkpoints").glob("*.ckpt"))
    assert checkpoints, f"No checkpoint written to {tmp_path / 'checkpoints'}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv"]))
