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

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.rank_zero import (
    rank_zero_only,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SecondsPerIterationProgressBar(pl.callbacks.TQDMProgressBar):
    """Progress bar that displays s/it instead of it/s.

    Lightning hardcodes {rate_noinv_fmt} in BAR_FORMAT which forces it/s.
    Overriding with {rate_inv_fmt} to always show s/it.
    """

    BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
        "{rate_inv_fmt}{postfix}]"
    )


class MemorySnapshot(pl.Callback):
    """
    Records memory history and dumps a snapshot pickle.
    The pickle can be visualized at https://pytorch.org/memory_viz

    Args:
        output_path: Path to save the .pickle snapshot file.
        step: Step (batch_idx) at which to record and dump a single-step
            snapshot. Set to `None` to disable step-based snapshotting (useful
            when only dump_on_oom is needed). Defaults to 0.
        dump_on_oom: If True, attaches an OOM observer that dumps the snapshot
            on out-of-memory. History is reset each step to keep the snapshot
            small.
        stacks: Stack trace mode for record_memory_history. Defaults to "all".
    """

    def __init__(
        self,
        output_path: str = "memory_snapshot.pickle",
        step: int | None = 0,
        dump_on_oom: bool = False,
        stacks: str = "all",
    ):
        super().__init__()
        self.output_path = output_path
        self.step = step
        self.dump_on_oom = dump_on_oom
        self.stacks = stacks
        self._oom_dumped = False
        self._setup_done = False
        self._global_rank = 0

    def _tagged_path(self, suffix: str) -> str:
        """Build output path with rank and suffix before the extension."""
        p = Path(self.output_path)
        return str(p.with_stem(f"{p.stem}{suffix}_rank{self._global_rank}"))

    def _oom_observer(self, device, alloc, device_allocated, device_free):
        if self._oom_dumped:
            return

        self._oom_dumped = True
        oom_path = self._tagged_path("_oom")
        logger.error(
            f"MemorySnapshot: OOM on rank {self._global_rank} (tried to allocate "
            f"{alloc / 1024**3:.2f} GiB, {device_free / 1024**3:.2f} GiB free). "
            f"Dumping snapshot to {oom_path}"
        )
        torch.cuda.memory._dump_snapshot(oom_path)

    @property
    def step_recording_enabled(self):
        return self.step is not None

    def setup(self, trainer, pl_module, stage=None):
        # Only attach oom observer once
        if self._setup_done:
            return

        self._setup_done = True
        self._global_rank = trainer.global_rank

        if self.dump_on_oom:
            torch.cuda.memory._record_memory_history(stacks=self.stacks)
            if trainer.is_global_zero:
                logger.info("MemorySnapshot: Enabling OOM observer")
            torch._C._cuda_attach_out_of_memory_observer(self._oom_observer)

    def _on_batch_start(self, batch_idx: int):
        record_step = self.step_recording_enabled and batch_idx == self.step
        if self.dump_on_oom or record_step:
            # Reset history so the snapshot only contains the current step
            torch.cuda.memory._record_memory_history(enabled=None)
            torch.cuda.memory._record_memory_history(stacks=self.stacks)

    def _on_batch_end(self, batch_idx: int):
        record_step = self.step_recording_enabled and batch_idx == self.step
        if not record_step:
            return

        output_path = self._tagged_path("")
        logger.info(f"MemorySnapshot: Dumping snapshot to {output_path}")
        torch.cuda.memory._dump_snapshot(output_path)

        if not self.dump_on_oom:
            torch.cuda.memory._record_memory_history(enabled=None)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
        self._on_batch_start(batch_idx=batch_idx)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, **kwargs
    ):
        self._on_batch_end(batch_idx=batch_idx)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
        self._on_batch_start(batch_idx=batch_idx)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, **kwargs
    ):
        self._on_batch_end(batch_idx=batch_idx)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
        self._on_batch_start(batch_idx=batch_idx)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, **kwargs
    ):
        self._on_batch_end(batch_idx=batch_idx)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
        self._on_batch_start(batch_idx=batch_idx)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, **kwargs
    ):
        self._on_batch_end(batch_idx=batch_idx)


class PredictTimer(pl.Callback):
    def __init__(self, output_dir: Path | None):
        super().__init__()
        self.output_dir = output_dir

        # For recording runtime per batch
        self.batch_start_time = None

    def _get_start_time(self, sync=True):
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        self.batch_start_time = time.perf_counter()

    def _get_runtime(self, sync=True):
        """Record the runtime for the current batch."""
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_end_time = time.perf_counter()

        return batch_end_time - self.batch_start_time

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0
    ):
        self._get_start_time()

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        # Get batch runtime
        runtime = self._get_runtime()

        if pl_module.logger is not None:
            pl_module.logger.log_metrics(
                {"seconds_per_iteration": runtime}, step=pl_module.global_step
            )

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0
    ):
        self._get_start_time()

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        # Get batch runtime
        runtime = self._get_runtime()

        # Skip repeated samples
        if batch.get("repeated_sample") or outputs is None:
            return

        batch_size = len(batch["atom_array"])

        # Calculate an average runtime for each sample in the batch
        # This is always one sample for now
        runtime_per_sample = runtime / batch_size

        # Iterate over all predictions in the batch
        for b in range(batch_size):
            seed = batch["seed"][b]
            query_id = batch["query_id"][b]

            output_subdir = Path(self.output_dir) / query_id / f"seed_{seed}"

            # Save runtime for the batch
            runtime_file = output_subdir / "timing.json"
            runtime_json = {"runtime_s": runtime_per_sample}
            runtime_file.write_text(json.dumps(runtime_json, indent=4))


def set_seed_for_rank(seed: int, rank: int) -> None:
    """
    Sets the seed for all relevant random number generators on a specific rank.

    Args:
        seed (int): The base seed to use.
        rank (int): The process rank, used to create a unique seed for the process.
    """
    # Calculate a unique seed for each rank
    rank_specific_seed = seed + rank

    # Set seed for Python's random module
    random.seed(rank_specific_seed)

    # Set seed for NumPy
    np.random.seed(rank_specific_seed)

    # Set seed for PyTorch on CPU and CUDA
    torch.manual_seed(rank_specific_seed)
    torch.cuda.manual_seed_all(rank_specific_seed)  # Seeds all GPUs


class RankSpecificSeedCallback(pl.Callback):
    """
    Callback to set a unique seed for each distributed process from a starting
    base seed. This de-synchronizes randomness in the model across ranks.

    The DataModule will use the data_seed, which wil not change across ranks.

    Args:
        base_seed (int): The starting seed. The seed for each rank `r` will
            be `base_seed + r`.
        log_seed (bool): If True, logs the seed used for rank 0.
    """

    def __init__(self, base_seed: int, log_seed: bool = True):
        super().__init__()
        self.base_seed = base_seed
        self.log_seed = log_seed
        self._has_been_set = False

    def setup(
        self,
        trainer: "pl.Trainer",  # noqa: F821
        pl_module: "pl.LightningModule",  # noqa: F821
        stage: str,
    ) -> None:
        """
        Called by Lightning when preparing for training, validation, testing,
        or predicting. This is the ideal hook to set the seed because the trainer
        object is available and the distributed environment is fully configured.
        """
        if self._has_been_set:
            return

        rank = trainer.global_rank

        set_seed_for_rank(self.base_seed, rank)
        self._has_been_set = True

        logging.info(
            f"[rank: {trainer.global_rank}] Model base seed set to {self.base_seed}, "
            f"rank initialized with seed {self.base_seed + rank}"
        )


class LogInferenceQuerySet(pl.Callback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    @rank_zero_only
    def on_predict_start(self, trainer, pl_module):
        log_path = self.output_dir / "inference_query_set.json"
        with open(log_path, "w") as fp:
            fp.write(
                pl_module.trainer.datamodule.inference_config.query_set.model_dump_json(
                    indent=4
                )
            )
