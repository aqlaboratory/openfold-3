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

import json
from abc import ABC, abstractmethod, property
from typing import Any

from torch.utils.data import Dataset

# from openfold3.core.data.featurization.feature_pipelines import (
#     AF3BioAssemblyFeaturePipeline,
# )
# from openfold3.core.data.preprocessing.preprocessing_pipelines import (
#     BioAssemblyPreprocessingPipeline,
#     TFDNAPreprocessingPipeline,
# )

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
        return f"""SingleDataset {self.dataset_name} missing from dataset registry. \
                Wrap your class with the register_dataset decorator."""


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


@register_dataset
class WeightedPDBDataset(SingleDataset):
    """Implements a Dataset class for the Weighted PDB training dataset for AF3."""

    def __init__(self, dataset_config) -> None:
        super().__init__()

        # argument error checks here

        self._preprocessing_pipeline = "BioAssemblyPreprocessingPipeline(dataset_config)"
        self._feature_pipeline = "AF3BioAssemblyFeaturePipeline(dataset_config)"

        # Parse data cache
        # with open(dataset_config['data_cache'], 'r') as f:
        #     self.data_cache = json.load(f)

        # Calculate datapoint probabilities
        self.calculate_datapoint_probabilities()

    @property
    def preprocessing_pipeline(self):
        return self._preprocessing_pipeline

    @property
    def feature_pipeline(self):
        return self._feature_pipeline

    def calculate_datapoint_probabilities(self):
        """Implements equation 1 from section 2.5.1 of the AF3 SI."""
        self.datapoint_probabilities = "<...>"  # TODO
        return

    def __len__(self):
        return len(self.data_cache)


@register_dataset
class ProteinMonomerDataset(SingleDataset):
    """For monomeric protein distillation datasets for AF3."""

    def __init__(self, data_config) -> None:
        super().__init__()
        self._preprocessing_pipeline = (
            "<ProteinMonomerPreprocessingPipeline(data_config)>"
        )
        self._feature_pipeline = "<AF3ProteinFeaturePipeline(data_config)>"

        with open(data_config["data_cache"]) as f:
            self.data_cache = json.load(f)

        "<assign other attributes here>"

    @property
    def preprocessing_pipeline(self):
        return self._preprocessing_pipeline

    @property
    def feature_pipeline(self):
        return self._feature_pipeline

    def calculate_datapoint_probabilities(self, cache_entry) -> float:
        # uniform sampling
        return

    def __len__(self):
        return len(self.data_cache)


@register_dataset
class RNAStructureDataset(SingleDataset):
    """Implements a Dataset class for the RNA distillation dataset for AF3."""

    def __init__(self, data_config) -> None:
        super().__init__()
        self._preprocessing_pipeline = "<RNAPreprocessingPipeline(data_config)>"
        self._feature_pipeline = "<AF3RNAFeaturePipeline(data_config)>"

        with open(data_config["data_cache"]) as f:
            self.data_cache = json.load(f)

        "<assign other attributes here>"

    @property
    def preprocessing_pipeline(self):
        return self._preprocessing_pipeline

    @property
    def feature_pipeline(self):
        return self._feature_pipeline

    def calculate_datapoint_probabilities(self, cache_entry) -> float:
        # uniform sampling
        return

    def __len__(self):
        return len(self.data_cache)


@register_dataset
class TFPositiveDataset(SingleDataset):
    """For transcription factor-DNA positive distillation dataset for AF3."""

    def __init__(self, data_config) -> None:
        super().__init__()
        self._preprocessing_pipeline = "TFDNAPreprocessingPipeline(data_config)"
        self._feature_pipeline = "AF3BioAssemblyFeaturePipeline(data_config)"

        with open(data_config["data_cache"]) as f:
            self.data_cache = json.load(f)

        "<assign other attributes here>"

    @property
    def preprocessing_pipeline(self):
        return self._preprocessing_pipeline

    @property
    def feature_pipeline(self):
        return self._feature_pipeline

    def calculate_datapoint_probabilities(self, cache_entry) -> float:
        # uniform sampling
        return

    def __len__(self):
        return len(self.data_cache)


@register_dataset
class TFNegativeDataset(SingleDataset):
    """For the Transcription factor-DNA negative distillation dataset for AF3."""

    def __init__(self, data_config) -> None:
        super().__init__()
        self._preprocessing_pipeline = "TFDNAPreprocessingPipeline(data_config)"
        self._feature_pipeline = "AF3BioAssemblyFeaturePipeline(data_config)"

        with open(data_config["data_cache"]) as f:
            self.data_cache = json.load(f)

        "<assign other attributes here>"

    @property
    def preprocessing_pipeline(self):
        return self._preprocessing_pipeline

    @property
    def feature_pipeline(self):
        return self._feature_pipeline

    def calculate_datapoint_probabilities(self, cache_entry) -> float:
        # uniform sampling
        return

    def __len__(self):
        return len(self.data_cache)


# QUESTION how to do validation/test datasets?


@register_dataset
class InferenceDataset(SingleDataset):
    """Implements a Dataset class for the inference dataset for AF3."""

    def __init__(self, data_config) -> None:
        super().__init__()
        self._preprocessing_pipeline = "<InferencePreprocessingPipeline(data_config)>"
        self._feature_pipeline = "<AF3InferenceFeaturePipeline(data_config)>"

        with open(data_config["data_cache"]) as f:
            self.data_cache = json.load(f)

        "<assign other attributes here>"

    @property
    def preprocessing_pipeline(self):
        return self._preprocessing_pipeline

    @property
    def feature_pipeline(self):
        return self._feature_pipeline

    def calculate_datapoint_probabilities(self, cache_entry) -> float:
        # no sampling !!!
        return

    def __len__(self):
        return len(self.data_cache)
