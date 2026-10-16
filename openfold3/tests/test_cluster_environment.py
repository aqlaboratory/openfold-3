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

import pytest
from lightning_fabric.plugins.environments.lightning import LightningEnvironment
from lightning_fabric.plugins.environments.mpi import MPIEnvironment
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment

from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
from openfold3.entry_points.validator import InferenceExperimentConfig, PlTrainerArgs


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


@pytest.fixture()
def no_slurm_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("SLURM"):
            monkeypatch.delenv(key)


@pytest.fixture()
def sbatch_env_without_srun(no_slurm_env, monkeypatch):
    """A batch step running the process directly: no SLURM_STEP_ID."""
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_JOB_NAME", "of3-job")


@pytest.fixture()
def srun_env(sbatch_env_without_srun, monkeypatch):
    """An srun-launched task within the allocation."""
    monkeypatch.setenv("SLURM_STEP_ID", "0")


class TestAutoResolution:
    def test_no_slurm_defers_to_lightning(self, make_runner, no_slurm_env):
        assert make_runner().cluster_environment is None

    def test_sbatch_without_srun_falls_back_to_local(
        self, make_runner, sbatch_env_without_srun
    ):
        env = make_runner().cluster_environment
        assert isinstance(env, LightningEnvironment)

    def test_srun_launched_defers_to_lightning(self, make_runner, srun_env):
        # Lightning's own detection will (correctly) pick SLURMEnvironment.
        assert make_runner().cluster_environment is None

    def test_interactive_job_name_defers_to_lightning(
        self, make_runner, sbatch_env_without_srun, monkeypatch
    ):
        # Lightning treats job name bash/interactive as "not SLURM-managed"
        # already; auto must not override that.
        monkeypatch.setenv("SLURM_JOB_NAME", "bash")
        assert make_runner().cluster_environment is None


class TestExplicitResolution:
    def test_local_forced(self, make_runner, srun_env):
        env = make_runner(cluster_environment="local").cluster_environment
        assert isinstance(env, LightningEnvironment)

    def test_slurm_forced(self, make_runner, no_slurm_env):
        env = make_runner(cluster_environment="slurm").cluster_environment
        assert isinstance(env, SLURMEnvironment)

    @pytest.mark.skipif(
        importlib.util.find_spec("mpi4py") is None,
        reason="MPIEnvironment requires mpi4py",
    )
    def test_mpi_forced(self, make_runner, no_slurm_env):
        env = make_runner(cluster_environment="mpi").cluster_environment
        assert isinstance(env, MPIEnvironment)

    @pytest.mark.skipif(
        importlib.util.find_spec("mpi4py") is None,
        reason="MPIEnvironment requires mpi4py",
    )
    def test_legacy_mpi_plugin_flag(self, make_runner, no_slurm_env):
        env = make_runner(mpi_plugin=True).cluster_environment
        assert isinstance(env, MPIEnvironment)

    def test_invalid_value_rejected(self, make_runner):
        with pytest.raises(ValueError):
            make_runner(cluster_environment="k8s")


class TestTrainerIntegration:
    def test_knob_not_forwarded_to_pl_trainer(self, make_runner, no_slurm_env):
        """pl.Trainer must never see the cluster_environment kwarg."""
        runner = make_runner(
            accelerator="cpu", devices=1, cluster_environment="local"
        )
        trainer = runner.trainer  # raises TypeError if the kwarg leaks through
        assert isinstance(
            trainer._accelerator_connector.cluster_environment,
            LightningEnvironment,
        )

    def test_distributed_strategy_carries_env(
        self, make_runner, sbatch_env_without_srun
    ):
        runner = make_runner(accelerator="cpu", devices=8)
        assert isinstance(
            runner.strategy.cluster_environment, LightningEnvironment
        )
