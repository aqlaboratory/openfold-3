import operator
import os
import time

import dllogger as logger
import numpy as np
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
            f"throughput": throughput_imgps,
            f"latency_mean": _round3(timestamps_ms.mean()),
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
