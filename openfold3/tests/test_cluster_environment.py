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

"""Cluster-environment resolution (issue #47).

Lightning's autodetection adopts SLURMEnvironment for any process inside a
SLURM allocation. When the batch step runs `run_openfold` directly (sbatch
without srun), no sibling ranks are ever launched and distributed init blocks
forever at 0% GPU. These tests pin the resolution rules that prevent that:

- `auto` (default) defers to Lightning, EXCEPT inside a SLURM allocation that
  was not launched via srun (no SLURM_STEP_ID), where it falls back to the
  local launcher.
- `local` / `slurm` / `mpi` force the corresponding environment.
- the legacy `mpi_plugin` flag still forces MPIEnvironment.
"""

import importlib.util
import os
from unittest import mock

import pytest
from lightning_fabric.plugins.environments.lightning import LightningEnvironment
from lightning_fabric.plugins.environments.mpi import MPIEnvironment
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment

from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
from openfold3.entry_points.validator import InferenceExperimentConfig, PlTrainerArgs

requires_mpi4py = pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None,
    reason="MPIEnvironment requires mpi4py",
)

# A batch step running the process directly: allocation vars, no SLURM_STEP_ID.
SBATCH_WITHOUT_SRUN = {"SLURM_NTASKS": "8", "SLURM_JOB_NAME": "of3-job"}
# An srun-launched task within the allocation.
SRUN_LAUNCHED = {**SBATCH_WITHOUT_SRUN, "SLURM_STEP_ID": "0"}


def scrubbed_slurm_env(extra_env: dict[str, str]) -> mock._patch_dict:
    """Patch os.environ to contain no SLURM variables except `extra_env`."""
    base = {k: v for k, v in os.environ.items() if not k.startswith("SLURM")}
    return mock.patch.dict(os.environ, {**base, **extra_env}, clear=True)


@pytest.fixture()
def make_runner(tmp_path):
    def _make(**pl_trainer_args) -> InferenceExperimentRunner:
        ckpt = tmp_path / "dummy.ckpt"
        ckpt.touch()
        config = InferenceExperimentConfig(
            inference_ckpt_path=ckpt,
            pl_trainer_args=PlTrainerArgs(**pl_trainer_args),
        )
        return InferenceExperimentRunner(config)

    return _make


@pytest.mark.parametrize(
    ("env", "pl_trainer_args", "expected"),
    [
        pytest.param({}, {}, None, id="auto-outside-slurm-defers-to-lightning"),
        pytest.param(
            SBATCH_WITHOUT_SRUN,
            {},
            LightningEnvironment,
            id="auto-sbatch-without-srun-falls-back-to-local",
        ),
        pytest.param(
            SRUN_LAUNCHED,
            {},
            None,
            id="auto-srun-launched-defers-to-lightning",
        ),
        pytest.param(
            # Lightning treats job name bash/interactive as "not SLURM-managed"
            # already; auto must not override that.
            {**SBATCH_WITHOUT_SRUN, "SLURM_JOB_NAME": "bash"},
            {},
            None,
            id="auto-interactive-job-name-defers-to-lightning",
        ),
        pytest.param(
            SRUN_LAUNCHED,
            {"cluster_environment": "local"},
            LightningEnvironment,
            id="local-forced-even-under-srun",
        ),
        pytest.param(
            {},
            {"cluster_environment": "slurm"},
            SLURMEnvironment,
            id="slurm-forced-even-outside-slurm",
        ),
        pytest.param(
            {},
            {"cluster_environment": "mpi"},
            MPIEnvironment,
            id="mpi-forced",
            marks=requires_mpi4py,
        ),
        pytest.param(
            {},
            {"mpi_plugin": True},
            MPIEnvironment,
            id="legacy-mpi-plugin-flag",
            marks=requires_mpi4py,
        ),
    ],
)
def test_cluster_environment_resolution(make_runner, env, pl_trainer_args, expected):
    with scrubbed_slurm_env(env):
        resolved = make_runner(**pl_trainer_args).cluster_environment
    if expected is None:
        assert resolved is None
    else:
        assert isinstance(resolved, expected)


def test_invalid_value_rejected(make_runner):
    with pytest.raises(ValueError):
        make_runner(cluster_environment="k8s")


def test_knob_not_forwarded_to_pl_trainer(make_runner):
    """pl.Trainer must never see the cluster_environment kwarg."""
    with scrubbed_slurm_env({}):
        runner = make_runner(accelerator="cpu", devices=1, cluster_environment="local")
        trainer = runner.trainer  # raises TypeError if the kwarg leaks through
        assert isinstance(
            trainer._accelerator_connector.cluster_environment,
            LightningEnvironment,
        )


def test_distributed_strategy_carries_env(make_runner):
    with scrubbed_slurm_env(SBATCH_WITHOUT_SRUN):
        runner = make_runner(accelerator="cpu", devices=8)
        assert isinstance(runner.strategy.cluster_environment, LightningEnvironment)
