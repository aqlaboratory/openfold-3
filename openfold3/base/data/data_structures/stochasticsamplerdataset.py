# TODO add license

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


class OpenFoldStochasticSamplerDataset(Dataset):
    """A dataset class for combining multiple OpenFoldSingleDataset instances and iteratively sampling from them with the provided probabilities."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        probabilities: Sequence[float],
        virtual_epoch_len: int,
        num_virtual_epochs: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()

        self.datasets = datasets
        self.probabilities = probabilities
        self.virtual_epoch_len = virtual_epoch_len
        self.num_virtual_epochs = num_virtual_epochs
        self.generator = generator

    def __len__(self):
        return self.virtual_epoch_len

    def __getitem__(self, index: tuple[int, int]) -> Any:
        dataset_idx, datapoint_idx = self.indices[index]
        return self.datasets[dataset_idx][datapoint_idx]

    def resample_epoch(self):
        """Resample virtual_epoch_len number of samples according to the provided probabilities."""
        raise NotImplementedError(
            "OpenFoldStochasticSamplerDataset.resample() is not yet implemented."
        )
        # <functions to generate index tuples self.indices for (dataset_idx, datapoint_idx)>
        pass

    def calculate_coverage(self):
        """Calculate dataset coverage - low priority functionality."""
