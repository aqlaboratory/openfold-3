# TODO add license

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


class OpenFoldStochasticSamplerDataset(Dataset):
    """A dataset class for combining multiple OpenFoldSingleDataset instances and
    sampling from them with the provided probabilities."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        dataset_probabilities: Sequence[float],
        epoch_len: int,
        num_epochs: int,
        generator: torch.Generator,
    ) -> None:
        """Initializes the OpenFoldStochasticSamplerDataset class.

        Args:
            datasets (Sequence[Dataset]):
                List of datasets to sample from.
            dataset_probabilities (Sequence[float]):
                Probabilities of sampling each dataset.
            epoch_len (int):
                Number of datapoints to sample in total for each virtual epoch.
            num_epochs (int):
                Total number of virtual epochs.
            generator (torch.Generator):
                torch.Generator instance for reproducibility.
        """
        super().__init__()

        self.datasets = datasets
        self.dataset_probabilities = torch.tensor(dataset_probabilities)
        self.epoch_len = epoch_len
        self.num_epochs = num_epochs
        self.generator = generator

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index: tuple[int, int]) -> Any:
        dataset_idx, datapoint_idx = self.indices[index]
        return self.datasets[dataset_idx][datapoint_idx]

    def resample_epoch(self):
        """Resample epoch_len number of samples according to the provided
        probabilities."""
        # Sample dataset indices
        dataset_indices = torch.multinomial(
            input=self.dataset_probabilities,
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )

        # For each dataset, sample datapoint indices
        datapoint_indices = torch.zeros(self.epoch_len, dtype=torch.long)
        for dataset_idx, num_datapoints in zip(
            torch.unique(dataset_indices), torch.bincount(dataset_indices)
        ):
            # QUESTION should this use a different generator like below?
            datapoint_idx_generator = torch.Generator(
                device=self.generator.device
            ).manual_seed(
                torch.randint(
                    low=0, high=100000, size=(1,), generator=self.generator
                ).item()
            )

            # Retrieve datapoint probabilities for give dataset
            datapoint_probabilities = torch.tensor(
                self.datasets[dataset_idx].datapoint_probabilities
            )

            # Sample datapoint indices
            datapoint_indices_i = torch.multinomial(
                input=datapoint_probabilities,
                num_samples=num_datapoints,
                replacement=True,
                generator=datapoint_idx_generator,
            )

            # Add to datapoint index container to pair with dataset indices
            datapoint_indices[torch.where(dataset_indices == dataset_idx)] = (
                datapoint_indices_i
            )

        self.indices = torch.stack((dataset_indices, datapoint_indices), dim=1).tolist()

    def calculate_coverage(self):
        """Calculate dataset coverage - low priority functionality."""
        pass
