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

"""This module contains the SamplerDataset and OF3DistributedSampler classes.

The SamplerDataset class is a pytorch Dataset class that wraps one or more
SingleDataset instances. The OF3DistributedSampler class samples a desired number
of datapoints from the SamplerDataset based on the provided dataset and datapoint
probabilities. The sampling is done by generating a list of index tuples for a
given dataset-datapoint pair per sample that is regenerated at the start of
each virtual epoch.

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
4. OF3DistributedSampler + SamplerDataset (optional) [YOU ARE HERE]
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
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from openfold3.core.data.framework.single_datasets.abstract_single import SingleDataset

logger = logging.getLogger(__name__)


class SamplerDataset(Dataset):
    """
    A dataset class for combining multiple SingleDataset instances.
    Accepts (dataset_idx, datapoint_idx) tuples from the OF3DistributedSampler
    to fetch the actual data.
    """

    def __init__(
        self,
        datasets: Sequence[SingleDataset],
        epoch_len: int = None,
    ) -> None:
        """
        Args:
            datasets (Sequence[SingleDataset]):
                List of datasets to sample from.
            epoch_len (int):
                Number of datapoints to sample in total for each virtual epoch.
        """
        super().__init__()
        self.datasets = datasets
        self.epoch_len = epoch_len

    def __len__(self):
        # This length is nominal; the Sampler controls the actual __iter__ length
        return self.epoch_len

    def __getitem__(self, index_tuple: tuple[int, int]) -> Any:
        """
        Wrapper getitem for indexing into the unrolled examples.

        Args:
            index_tuple: A tuple (dataset_idx, datapoint_idx) yielded by the Sampler.
        """
        # Dataset-datapoint pair for the given index
        dataset_idx, datapoint_idx = index_tuple

        # Set datapoint index in sampled dataset - used for logging in the treadmill
        self.datasets[dataset_idx].set_current_datapoint_index(datapoint_idx)

        # Index into the list of datasets then datapoints for the given dataset
        # This calls the __getitem__ method of the SingleDataset class
        return self.datasets[dataset_idx][datapoint_idx]

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


class OF3DistributedSampler(DistributedSampler):
    """
    A dataset class for combining multiple SingleDataset instances and
    sampling from them with the provided probabilities.
    """

    def __init__(
        self,
        dataset: SamplerDataset,
        dataset_probabilities: Sequence[float],
        next_dataset_indices: dict[str, Any],
        epoch_len: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Initializes the SamplerDataset class.

        Args:
            dataset (Dataset):
                Composite SamplerDataset to sample from.
            dataset_probabilities (Sequence[float]):
                Probabilities of sampling each dataset.
            epoch_len (int):
                Number of datapoints to sample in total for each virtual epoch.
            next_dataset_indices: dict[str, Any]
                Record of last used indices for datasets that use in-order
                sampling
            num_replicas:
                Number of processes participating in distributed training
            rank:
                Rank of the current process
            seed:
                Random seed used to shuffle the sampler if shuffle=True
        """
        logger.debug(
            f"Rank {rank} - Initializing OF3DistributedSampler with "
            f"{dataset_probabilities=}, {next_dataset_indices=}, {epoch_len=}, {seed=}"
        )
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=drop_last,
        )
        # Access the underlying SingleDatasets
        self.datasets = dataset.datasets
        self.dataset_probabilities = torch.tensor(dataset_probabilities)
        self.epoch_len = epoch_len
        self.next_dataset_indices = next_dataset_indices

    @staticmethod
    def _get_random_subset(
        dataset: SingleDataset, num_examples: int, generator: torch.Generator
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

    def _get_ordered_subset(
        self, dataset: SingleDataset, num_examples: int
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

        logger.debug(
            f"Fetching ordered subset for epoch {self.epoch} {dataset.name}: "
            f"start={start_idx}, n={num_examples}, end={end_idx}"
        )

        return slice_indices

    def get_datapoint_indices(
        self, dataset_indices: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        n_datasets = len(self.datasets)
        datapoint_indices = torch.zeros(self.epoch_len, dtype=torch.long)

        for dataset_idx, num_datapoints_per_dataset in zip(
            torch.arange(n_datasets),
            torch.bincount(dataset_indices, minlength=n_datasets),
            strict=True,
        ):
            if num_datapoints_per_dataset == 0:
                continue

            dataset = self.datasets[dataset_idx]

            # Create a dataset-specific generator derived from the main one
            generator_seed = torch.randint(
                low=0, high=100000, size=(1,), generator=generator
            ).item()
            datapoint_idx_generator = torch.Generator().manual_seed(generator_seed)

            if dataset.name in self.next_dataset_indices:
                datapoint_indices_i = self._get_ordered_subset(
                    dataset=dataset, num_examples=num_datapoints_per_dataset
                )
            else:
                datapoint_indices_i = self._get_random_subset(
                    dataset=dataset,
                    num_examples=num_datapoints_per_dataset,
                    generator=datapoint_idx_generator,
                )

            # Add to datapoint index container to pair with dataset indices
            datapoint_indices[torch.where(dataset_indices == dataset_idx)] = (
                datapoint_indices_i
            )

        return datapoint_indices

    def get_dataset_indices(self, generator: torch.Generator) -> torch.Tensor:
        return torch.multinomial(
            input=self.dataset_probabilities,
            num_samples=self.epoch_len,
            replacement=True,
            generator=generator,
        )

    def __iter__(self) -> Iterator[tuple[int, int]]:
        """
        Sample epoch_len number of samples according to the provided probabilities.
        """

        # Seed with self.seed + self.epoch to ensure all ranks generate the
        # exact same global list before slicing.
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # Sample dataset indices
        dataset_indices = self.get_dataset_indices(generator=generator)

        # For each dataset, sample datapoint indices
        datapoint_indices = self.get_datapoint_indices(
            dataset_indices=dataset_indices, generator=generator
        )

        # Create the global list of dataset-datapoint index tuples
        global_pairs = torch.stack((dataset_indices, datapoint_indices), dim=1).tolist()

        logger.debug(f"Sampled batch indices: {global_pairs=}")

        # super().__iter__() yields the indices for this rank.
        # Use those indices to pick the correct tuples from the global list
        indices_for_this_rank = list(super().__iter__())

        logger.debug(
            f"Called OF3DistributedSampler.__iter__ in rank {self.rank}: "
            f"epoch {self.epoch}, seed {self.seed + self.epoch}, sampled dataset "
            f"indices {dataset_indices.tolist()}, sampled datapoint indices "
            f"{datapoint_indices.tolist()}, indices_for_this_rank "
            f"{indices_for_this_rank}."
        )

        for i in indices_for_this_rank:
            dataset_idx, datapoint_idx = global_pairs[i]
            yield int(dataset_idx), int(datapoint_idx)
