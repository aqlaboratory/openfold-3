"""This module contains the SingleDataset class and its subclasses.

A SingleDataset class is a pytorch Dataset class which specified the way datapoints
need to be parsed/process and embedded into feature tensors using a pair of
PreprocessingPipeline and FeaturePipeline. SingleDataset also has an optional
calculate_datapoint_probabilities which implements a strategy for calculating
the probability of sampling all of the datapoints from a precomputed data cache.

The steps below outline how datapoints get from raw datapoints to the model
and highlight where you currently are in the process:

0. Dataset filtering and cache generation
    raw data -> filtered data
1. PreprocessingPipeline
    filtered data -> preprocessed data
2. SampleProcessingPipeline and FeaturePipeline
    preprocessed data -> parsed/processed data -> FeatureDict
3. SingleDataset [YOU ARE HERE]
    datapoints -> __getitem__ -> FeatureDict
4. StochasticSamplerDataset (optional)
    Sequence[SingleDataset] -> __getitem__ -> FeatureDict
5. DataLoader
    FeatureDict -> batched data
6. DataModule
    SingleDataset/StochasticSamplerDataset -> DataLoader
7. ModelRunner
    batched data -> model
"""

from abc import ABC, abstractmethod, property
from typing import Any

from torch.utils.data import Dataset

DATASET_REGISTRY = {}


def register_dataset(cls):
    """Register a specific OpenFoldSingleDataset class in the DATASET_REGISTRY.

    Args:
        cls (Type[OpenFoldSingleDataset]): The class to register.

    Returns:
        Type[OpenFoldSingleDataset]: The registered class.
    """
    DATASET_REGISTRY[cls.__name__] = cls
    cls._registered = True
    return cls


class DatasetNotRegisteredError(Exception):
    """A custom error for for unregistered SingleDatasets."""

    def __init__(self, dataset_name: str) -> None:
        super().__init__()
        self.dataset_name = dataset_name

    def __str__(self):
        return (f"SingleDataset {self.dataset_name} missing from dataset registry."
                "Wrap your class with the register_dataset decorator.")


class SingleDataset(ABC, Dataset):
    """Abstract class wrapping a pair of preprocessing and feature pipelines."""

    def __init__(self) -> None:
        if not self.__class__._registered:
            raise DatasetNotRegisteredError(self.__class__.__name__)

    @property
    @abstractmethod
    def preprocessing_pipeline(self):
        """Calls the forward pass of a PreprocessingPipeline instance."""
        pass

    @property
    @abstractmethod
    def feature_pipeline(self):
        """Calls the forward pass of a FeaturePipeline instace."""
        pass

    def calculate_datapoint_probabilities(self) -> float:
        """Calculates datapoint probabilities for stochastic sampling.

        Datapoint probabilities are calculated from the self.data_cache attribute and
        are used in the StochasticSamplerDataset class. By default datapoints are
        sampled uniformly."""

        self.datapoint_probabilities = 1 / len(self.data_cache)

    def __getitem__(self, index) -> Any:
        data = self.preprocessing_pipeline(index)
        features = self.feature_pipeline(data)
        return features
