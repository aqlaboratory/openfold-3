import json
import random
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.rank_zero import (
    rank_zero_only,
)


class PredictTimer(pl.Callback):
    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir

        # For recording runtime per batch
        self.batch_start_time = None

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0
    ):
        self.batch_start_time = time.perf_counter()

    def _get_runtime(self):
        """Record the runtime for the current batch."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_end_time = time.perf_counter()

        return batch_end_time - self.batch_start_time

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

        print(
            f"SEEDING: Base seed set to {self.base_seed}. Rank {trainer.global_rank} "
            f"initialized with seed {self.base_seed + rank}."
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
