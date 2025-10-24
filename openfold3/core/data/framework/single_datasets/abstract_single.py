# Copyright 2025 AlQuraishi Laboratory
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
4. SamplerDataset (optional)
    Sequence[SingleDataset] -> __getitem__ -> FeatureDict
5. DataLoader
    FeatureDict -> batched data
6. DataModule
    SingleDataset/SamplerDataset -> DataLoader
7. ModelRunner
    batched data -> model
"""

from abc import ABC, abstractmethod

import torch
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
        return (
            f"SingleDataset {self.dataset_name} missing from dataset registry."
            "Wrap your class with the register_dataset decorator."
        )


class SingleDataset(ABC, Dataset):
    """Abstract Dataset class implementing necessery attributes and methods.

    A child class of SingleDataset
        - must be decorated with the register_dataset decorator
        - must implement the __getitem__ method
        - must implement the dataset_cache property
        - must implement the datapoint_cache property
    """

    def __init__(self) -> None:
        if not self.__class__._registered:
            raise DatasetNotRegisteredError(self.__class__.__name__)

        self.datapoint_index = None

    def __post_init__(self) -> None:
        if self.dataset_cache is None:
            raise ValueError(
                f"No dataset_cache was created for {self.get_class_name()}. "
                "Assign this attribute in the __init__ of this class."
            )

        if self.datapoint_cache is None:
            raise ValueError(
                f"No datapoint_cache was created for {self.get_class_name()}. "
                "Assign this attribute in the __init__ of this class."
            )

    def get_class_name(self) -> str:
        """Returns the name of the class."""
        return self.__class__.__name__

    def set_current_datapoint_index(self, index: int) -> None:
        """Set the current datapoint index for logging purposes."""
        self.datapoint_index = index

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Getitem of a specific SingleDataset class.

        Called by the DataLoader directly or indirectly via the SamplerDataset
        getitem method and indexes into the data cache. Implements a series of steps to
        process the raw data into intermediate arrays via pipelines from
        pipelines.sample_processing and tensorize these arrays to create tensors for the
        model from pipelines.featurization.

        Args:
            index (int):
                Index of the datapoint to retrieve.

        Returns:
            dict[str, Union[torch.Tensor]]:
                Featuredict.
        """
        pass
