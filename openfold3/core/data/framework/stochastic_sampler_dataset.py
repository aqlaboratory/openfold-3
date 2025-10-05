"""This module contains the SamplerDataset class.

The amplerDataset class is a pytorch Dataset class that wraps one or
more SingleDataset instances and samples a desired number of datapoints based on
the provided dataset and datapoint probabilities. The sampling is done by
generating a list of index tuples for a given dataset-datapoint pair per
sample and can be regenerated at the start of each virtual epoch.

The steps below outline how datapoints get from raw datapoints to the model
and highlight where you currently are in the process:

0. Dataset filtering and cache generation
    raw data -> filtered data
1. PreprocessingPipeline
    filtered data -> preprocessed data
2. SampleProcessingPipeline and FeaturePipeline
    preprocessed data -> parsed/processed data -> FeatureDict
3. SingleDataset
    datapoints -> __getitem__ -> FeatureDict
4. SamplerDataset (optional) [YOU ARE HERE]
    Sequence[SingleDataset] -> __getitem__ -> FeatureDict
5. DataLoader
    FeatureDict -> batched data
6. DataModule
    SingleDataset/SamplerDataset -> DataLoader
7. ModelRunner
    batched data -> model
"""

# TODO: rename module from stochastic_sampler_dataset.py to sampler_dataset.py
import logging
from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from openfold3.core.data.framework.single_datasets.base_of3 import BaseOF3Dataset

logger = logging.getLogger(__name__)


class SamplerDataset(Dataset):
    """A dataset class for combining multiple SingleDataset instances and
    sampling from them with the provided probabilities."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        dataset_probabilities: Sequence[float],
        epoch_len: int,
        generator: torch.Generator,
        next_dataset_indices: dict[str, Any],
    ) -> None:
        """Initializes the SamplerDataset class.

        Args:
            datasets (Sequence[Dataset]):
                List of datasets to sample from.
            dataset_probabilities (Sequence[float]):
                Probabilities of sampling each dataset.
            epoch_len (int):
                Number of datapoints to sample in total for each virtual epoch.
            generator (torch.Generator):
                torch.Generator instance for reproducibility.
            next_dataset_indices:
                Record of last used indices for datasets that use in-order sampling
        """
        super().__init__()

        self.datasets = datasets
        self.dataset_probabilities = torch.tensor(dataset_probabilities)
        self.epoch_len = epoch_len
        self.generator = generator
        self.next_dataset_indices = next_dataset_indices
        self.indices = None

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, index: int) -> Any:
        """Wrapper getitem for indexing into the unrolled examples.

        Args:
            index (int):
                Index of the example to retrieve, passed from the DataLoader.
        """
        if not self.indices:
            # This resampling should only be used for plightning training sanity check
            logger.debug(f"Resampling epoch in getitem(), {self.next_dataset_indices=}")
            self.resample_epoch()

        # Get the dataset-datapoint pair for the given index
        dataset_idx, datapoint_idx = self.indices[index]

        # Set datapoint index in sampled dataset - used for logging in the treadmill
        self.datasets[dataset_idx].set_current_datapoint_index(index)

        # Index into the list of datasets then datapoints for the given dataset
        # This calls the __getitem__ method of the SingleDataset class
        return self.datasets[dataset_idx][datapoint_idx]

    def get_random_subset(
        self, dataset: BaseOF3Dataset, num_examples: int, generator: torch.Generator
    ) -> torch.Tensor:
        """Selects random indices from dataset based on dataset probabilities."""
        # Retrieve datapoint probabilities for given dataset
        datapoint_probabilities = torch.tensor(
            dataset.datapoint_cache["datapoint_probabilities"].to_numpy()
        )

        # Sample datapoint indices
        datapoint_indices_i = torch.multinomial(
            input=datapoint_probabilities,
            num_samples=num_examples,
            replacement=True,
            generator=generator,
        )
        return datapoint_indices_i

    def get_ordered_subset(
        self, dataset: BaseOF3Dataset, num_examples: int
    ) -> torch.Tensor:
        """Selects indices based on sliced examples from the dataset."""

        datapoint_probabilities = torch.tensor(
            dataset.datapoint_cache["datapoint_probabilities"].to_numpy()
        )
        if not torch.all(torch.eq(datapoint_probabilities, 1.0)):
            raise ValueError(
                "Ordered slicing of datasets not supported for "
                "datasets with nonuniform probabilities"
            )

        start_idx = self.next_dataset_indices[dataset.name]
        end_idx = start_idx + num_examples

        slice_indices = torch.arange(start_idx, min(end_idx, len(dataset)))

        if end_idx > len(dataset):
            end_idx = end_idx - len(dataset)
            logger.warning(
                f"Reached the end of the dataset {dataset.name},"
                f"missing {end_idx} examples,"
                "Sampling will loop back from beginning of dataset."
            )
            slice_indices = torch.concat((slice_indices, torch.arange(0, end_idx)))

        self.next_dataset_indices[dataset.name] = end_idx

        return slice_indices

    def resample_epoch(self):
        """Resample epoch_len number of samples according to the provided
        probabilities."""
        # Sample dataset indices
        n_datasets = len(self.datasets)
        dataset_indices = torch.multinomial(
            input=self.dataset_probabilities,
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )

        # For each dataset, sample datapoint indices
        datapoint_indices = torch.zeros(self.epoch_len, dtype=torch.long)
        for dataset_idx, num_datapoints_per_dataset in zip(
            torch.arange(n_datasets),
            torch.bincount(dataset_indices, minlength=n_datasets),
        ):
            if num_datapoints_per_dataset == 0:
                continue
            dataset = self.datasets[dataset_idx]
            generator_seed = torch.randint(
                low=0, high=100000, size=(1,), generator=self.generator
            ).item()
            datapoint_idx_generator = torch.Generator(
                device=self.generator.device
            ).manual_seed(generator_seed)

            if dataset.name in self.next_dataset_indices:
                datapoint_indices_i = self.get_ordered_subset(
                    dataset, num_datapoints_per_dataset
                )
            else:
                datapoint_indices_i = self.get_random_subset(
                    dataset, num_datapoints_per_dataset, datapoint_idx_generator
                )

            # Add to datapoint index container to pair with dataset indices
            datapoint_indices[torch.where(dataset_indices == dataset_idx)] = (
                datapoint_indices_i
            )

        self.indices = torch.stack((dataset_indices, datapoint_indices), dim=1).tolist()

    def get_worker_path(self, subdirs: list[str] | None, fname: str) -> str:
        """Returns the path to the worker output directory or file.

        Note: Treadmill logging utility. This function only works if individual datasets
        passed to the StochasticSamplerDataset were wrapped with the LoggingMixin.

        Args:
            subdirs (list[str] | None):
                List of subdirectories to append to the path.
            fname (str):
                Filename to append to the path.
        """
        # Get the dataset-specific path
        dataset_path = self.datasets[0].get_worker_path(subdirs=subdirs, fname=fname)

        return dataset_path
