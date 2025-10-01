import random
import time

import numpy as np
import pytorch_lightning as pl
import torch


class PredictTimer(pl.Callback):
    def on_predict_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_predict_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        print(f"Inference runtime: {elapsed} seconds")


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
