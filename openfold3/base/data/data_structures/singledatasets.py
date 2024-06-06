# TODO add license

from abc import ABC, abstractmethod, property
from typing import Any
import json

from torch.utils.data import Dataset

from openfold3.base.data.preprocessing.preprocessing_pipelines import BioAssemblyPreprocessingPipeline
from openfold3.base.data.feature_pipelines.feature_pipelines import AF3BioAssemblyFeaturePipeline

DATASET_REGISTRY = {}


def register_dataset(cls):
    """Register a specific OpenFoldSingleDataset class in the DATASET_REGISTRY.

    Args:
        cls (Type[OpenFoldSingleDataset]): The class to register.

    Returns:
        Type[OpenFoldSingleDataset]: The registered class.
    """
    DATASET_REGISTRY[cls.__name__] = cls
    # cls._registered = True  # QUESTION do we want to enforce class registration with this decorator? Part A.
    return cls


class OpenFoldSingleDataset(ABC, Dataset):
    """Abstract SingleDataset class for implementing preprocessing and feature pipeline methods. Requires register_dataset decorator."""

    # def __init__(self) -> None:  # do we want to enforce class registration with this decorator? Part B
    #     if not self.__class__._registered:
    #         raise DatasetNotRegisteredError()
        
    @property
    @abstractmethod
    def preprocessing_pipeline(self):
        """A function call to a PreprocessingPipeline instance assigned to the self.preprocessing_pipeline attribute."""
        pass

    @property
    @abstractmethod
    def feature_pipeline(self):
        """A function call to a FeaturePipeline instace assigned to the self.feature_pipeline attribute."""
        pass

    @abstractmethod
    def calculate_datapoint_probabilities(self) -> float:
        """Calculates datapoint probabilities from the data_cache using the dataset-specific sampling formula.
        """
        pass

    def __getitem__(self, index) -> Any:
        data = self.preprocessing_pipeline(index)
        features = self.feature_pipeline(data)
        return features


@register_dataset
class WeightedPDBDataset(OpenFoldSingleDataset):
    """Weighted PDB training dataset for AF3. 
    """

    def __init__(self, dataset_config) -> None:
        super().__init__()

        # argument error checks here
        
        self._preprocessing_pipeline = BioAssemblyPreprocessingPipeline(dataset_config)
        self._feature_pipeline = AF3BioAssemblyFeaturePipeline(dataset_config)

        # Parse data cache
        # with open(dataset_config['data_cache'], 'r') as f:
        #     self.data_cache = json.load(f)

        # Calculate datapoint probabilities

        "<assign other attributes here>"

    @property
    def preprocessing_pipeline(self):
        return self._preprocessing_pipeline

    @property
    def feature_pipeline(self):
        return self._feature_pipeline
    
    def calculate_datapoint_probabilities(self):
        """Implements equation 1 from section 2.5.1 of the AF3 SI.
        """
        return
    
    def __len__(self):
        return len(self.data_cache)


@register_dataset
class TFPositiveDataset(OpenFoldSingleDataset):
    """TF-DNA positive distillation dataset for AF3.
    """

    def __init__(self, data_config) -> None:
        super().__init__()
        # self._preprocessing_pipeline = ProteinDNAPreprocessingPipeline(data_config)
        # self._feature_pipeline = AF3ProteinDNAFeaturePipeline(data_config)

        with open(data_config['data_cache'], 'r') as f:
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