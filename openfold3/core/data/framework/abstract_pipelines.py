"""Contains the SampleProcessingPipeline and FeaturePipeline abstract classes.

SampleProcessingPipeline classes are used to implement the data parsing and filtering
that needs to happen for individual samples returned in the __getitem__ method of a
specific SingleDataset class. The logic and order of preprocessing steps are defined in
the forward method using primitives from data/primitives/sequence and 
data/primitive/structure.

FeaturePipelines accept parse and processed data from a SampleProcessingPipeline, embed
features into tensors necessary for a specific model (and are hence specific to a
model), and return a feature dictionary. The logic and order of tensorization steps are
defined in the forward method using primitives from data/primitives/featurization.

0. Dataset filtering and cache generation
    raw data -> filtered data
1. PreprocessingPipeline
    filtered data -> preprocessed data
2. SampleProcessingPipeline and FeaturePipeline [YOU ARE HERE]
    preprocessed data -> parsed/processed data -> FeatureDict
3. SingleDataset
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

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch


# TODO import parsing primitives from parsing_pipeline_primitives.py
# TODO implement checks that a PreprocessingPipeline is used with the
# correct SingleDataset
class PreprocessingPipeline(ABC):
    """An abstract class for implementing a PreprocessingPipeline class."""

    @abstractmethod
    def forward(index):
        """Implements data parsing and filtering logic.

        Args:
            index (int): datapoint index
        """
        pass

    def __call__(self, index) -> Any:
        return self.forward(index=index)


# TODO import parsing primitives from feature_pipeline_primitives.py
# TODO implement checks that a FeaturePipeline is used with the correct SingleDataset
class FeaturePipeline(ABC):
    """An abstract class for implementing a FeaturePipeline class."""

    @abstractmethod
    def forward(self, data_dict: dict) -> dict[Sequence[torch.Tensor]]:
        """Implements featurization and embedding logic.

        Args:
            data_dict (dict):
                dictionary of parsed features to be tensorized
        Returns:
            dict:
                feature dictionary of embedded feature tensors
        """
        pass

    def __call__(self, parsed_features: dict) -> dict[Sequence[torch.Tensor]]:
        return self.forward(parsed_features=parsed_features)
