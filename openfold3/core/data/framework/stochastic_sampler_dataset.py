"""This module contains the StochasticSamplerDataset class.

The StochasticSamplerDataset class is a pytorch Dataset class that wraps one or
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
4. StochasticSamplerDataset (optional) [YOU ARE HERE]
    Sequence[SingleDataset] -> __getitem__ -> FeatureDict
5. DataLoader
    FeatureDict -> batched data
6. DataModule
    SingleDataset/StochasticSamplerDataset -> DataLoader
7. ModelRunner
    batched data -> model
"""

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


class StochasticSamplerDataset(Dataset):
    """A Dataset class for combining and sampling from multiple SingleDatasets."""

    def __init__(
        self,
        datasets: Sequence[Dataset],
        probabilities: Sequence[float],
        virtual_epoch_len: int,
        num_virtual_epochs: int,
        generator: torch.Generator,
    ) -> None:
        super().__init__()
        if len(datasets) != len(probabilities):
            raise RuntimeError("Number of datasets and probabilities must be equal.")
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
        """Resample virtual_epoch_len number of samples."""
        raise NotImplementedError(
            "OpenFoldStochasticSamplerDataset.resample() is not yet implemented."
        )
        # <functions to generate index tuples self.indices for (dataset_idx,
        # datapoint_idx)>
        pass

    def calculate_coverage(self):
        """Calculate dataset coverage - low priority functionality."""
