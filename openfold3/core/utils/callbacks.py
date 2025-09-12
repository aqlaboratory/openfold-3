import operator
import os
import random
import time

import dllogger as logger
import numpy as np
import torch
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities import rank_zero_info
from torch.cuda import profiler as profiler


class EarlyStoppingVerbose(EarlyStopping):
    """
    The default EarlyStopping callback's verbose mode is too verbose.
    This class outputs a message only when it's getting ready to stop.
    """

    def _evalute_stopping_criteria(self, *args, **kwargs):
        should_stop, reason = super()._evalute_stopping_criteria(*args, **kwargs)
        if should_stop:
            rank_zero_info(f"{reason}\n")

        return should_stop, reason


class PerformanceLoggingCallback(Callback):
    def __init__(
        self, log_file, global_batch_size, warmup_steps: int = 0, profile: bool = False
    ):
        logger.init(
            backends=[
                JSONStreamBackend(Verbosity.VERBOSE, log_file),
                StdOutBackend(Verbosity.VERBOSE),
            ]
        )
        self.warmup_steps = warmup_steps
        self.global_batch_size = global_batch_size
        self.step = 0
        self.profile = profile
        self.timestamps = []

    def do_step(self):
        self.step += 1
        if self.profile and self.step == self.warmup_steps:
            profiler.start()
        if self.step > self.warmup_steps:
            self.timestamps.append(time.time())

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        self.do_step()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        self.do_step()

    def process_performance_stats(self, deltas):
        def _round3(val):
            return round(val, 3)

        throughput_imgps = _round3(self.global_batch_size / np.mean(deltas))
        timestamps_ms = 1000 * deltas
        stats = {
            "throughput": throughput_imgps,
            "latency_mean": _round3(timestamps_ms.mean()),
        }
        for level in [90, 95, 99]:
            stats.update(
                {f"latency_{level}": _round3(np.percentile(timestamps_ms, level))}
            )

        return stats

    def _log(self):
        def is_main_process():
            return int(os.getenv("LOCAL_RANK", "0")) == 0

        if is_main_process():
            diffs = list(map(operator.sub, self.timestamps[1:], self.timestamps[:-1]))
            deltas = np.array(diffs)
            stats = self.process_performance_stats(deltas)
            logger.log(step=(), data=stats)
            logger.flush()

    def on_train_end(self, trainer, pl_module):
        if self.profile:
            profiler.stop()
        self._log()

    def on_epoch_end(self, trainer, pl_module):
        self._log()


def set_seed_for_rank(seed: int, rank: int, deterministic: bool = False) -> None:
    """
    Sets the seed for all relevant random number generators on a specific rank.

    Args:
        seed (int): The base seed to use.
        rank (int): The process rank, used to create a unique seed for the process.
        deterministic (bool): Whether to set torch deterministic flags.
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

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RankSpecificSeedCallback(Callback):
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

        print(
            f"SEEDING: Base seed set to {self.base_seed}. Rank {trainer.global_rank} "
            f"initialized with seed {self.base_seed + rank}."
        )
