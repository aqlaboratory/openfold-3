""" This module contains the FeaturePipeline abstract class and its subclasses.

Feature pipelines accept parse and processed data from a PreprocessingPipeline,
embed features into tensors necessary for a specific model (and are hence specific
to a model), and return a feature dictionary. The logic and order of tensorization
steps are defined in the forward method using primitives from 
feature_pipeline_primitives.

The steps below outline how datapoints get from raw datapoints to the model
and highlight where you currently are in the process:

0. Dataset filtering and cache generation
    raw data -> filtered data
1. PreprocessingPipeline
    filtered data -> processed data
2. *FeaturePipeline* [YOU ARE HERE]
    processed data -> FeatureDict
3. SingleDataset
    datapoints -> __getitem__ -> FeatureDict
4. StochasticSamplerDataset (optional)
    Sequence[SingeDataset] -> __getitem__ -> FeatureDict
5. DataLoader
    FeatureDict -> batched data
6. DataModule
    SingleDataset/StochasticSamplerDataset -> DataLoader
7. ModelRunner
    batched data -> model
"""

from abc import ABC, abstractmethod
from typing import Sequence
import torch


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


class AF3FeaturePipeline(FeaturePipeline):
    """An FeaturePipeline for embedding AF3 features."""

    def forward(self, parsed_features) -> dict:
        """_summary_

        Args:
            parsed_features (_type_): set of coordinates from parsed mmCIF and parsed MSAs, chain/interface metadata, center atom around which to crop

        Returns:
            dict: feature dictionary (Table 5)
        """
        # Featurization
        # Token features
        # Reference conformer features
        # MSA features
        # Template features
        # Bond features
