# TODO add license

from abc import ABC, abstractmethod, property
from typing import Any
import json

from torch.utils.data import Dataset

from openfold3.base.data.feature_pipelines.feature_pipelines import AF3BioAssemblyFeaturePipeline

DATASET_REGISTRY = {}


def register_dataset(cls):
    DATASET_REGISTRY[cls.__name__] = cls
    # cls._registered = True
    return cls


# currently requires the definition of methods parsing_pipeline, feature_pipeline in the inheritee SingleDataset
# to return the respective attributes, which I don't like
class OpenFoldSingleDataset(ABC, Dataset):
    """Abstract SingleDataset class for implementing parsing and feature pipeline methods. Requires register_dataset decorator."""

    # def __init__(self) -> None:
    #     if not self.__class__._registered:
    #         raise DatasetNotRegisteredError()
        
    @property
    @abstractmethod
    def parsing_pipeline(self):
        """A function call to a ParsingPipeline instance assigned to the self.parsing_pipeline attribute."""
        pass

    @property
    @abstractmethod
    def feature_pipeline(self):
        """A function call to a FeaturePipeline instace assigned to the self.feature_pipeline attribute."""
        pass

    @abstractmethod
    def calculate_datapoint_probabilities(self, cache_entry) -> float:
        """Calculates datapoint probabilities. - TODO do for entire cache
        """
        pass

    def __getitem__(self, index) -> Any:
        data = self.parsing_pipeline(index)
        features = self.feature_pipeline(data)
        return features


class OpenFoldSingleDataset2(ABC, Dataset):
    """Abstract SingleDataset class for implementing parser and feature pipeline methods."""

    def __getitem__(self, index) -> Any:
        cache_entry = self.data_cache[index]
        if self.feature_pipeline is not None:
            return self.feature_pipeline(cache_entry)
        else:
            raise NotImplementedError(
                "Any OpenFoldSingleDataset must implement a parser_pipeline attribute."
            )
        
    def __getitem__(self, index):
        if not hasattr(self, 'parser_pipeline'):
            raise NotImplementedError
        elif not hasattr(self, 'feature_pipeline'):
            raise NotImplementedError
        else:
            data = self._parsing_pipeline(index)
            features = self._feature_pipeline(data)
            return features
        
    @abstractmethod
    def compute_datapoint_probabilities(self):
        """A function call to compute datapoint probabilities."""
        pass


# Currently agreed to have separate SingleDatasets for train/eval/test/predict as one monolithic SingleDataset for
# a specific model goes against the basic goal of having separate SingleDatasets for each specific dataset. The
# SingleDatasets can still have a hierarchy so that they don't reimplement identical pipelines.
@register_dataset
class WeightedPDBDataset(OpenFoldSingleDataset):
    """Weighted PDB training dataset for AF3. 
    """

    def __init__(self, data_config) -> None:
        super().__init__()
        self._parser_pipeline = BioAssemblyParserPipeline(data_config)
        self._feature_pipeline = AF3BioAssemblyFeaturePipeline(data_config)

        with open(data_config['data_cache'], 'r') as f:
            self.data_cache = json.load(f)

        "<assign other attributes here>"

    @property
    def parsing_pipeline(self):
        return self._parsing_pipeline

    @property
    def feature_pipeline(self):
        return self._feature_pipeline
    
    def calculate_datapoint_probabilities(self, cache_entry) -> float:
        # AF3 sampling formula with entry-specific values
        return
    
    def __len__(self):
        return len(self.data_cache)


@register_dataset
class TFPositiveDataset(OpenFoldSingleDataset):
    """TF-DNA positive distillation dataset for AF3.
    """

    def __init__(self, data_config) -> None:
        super().__init__()
        self._parser_pipeline = ProteinDNAParserPipeline(data_config)
        self._feature_pipeline = AF3ProteinDNAFeaturePipeline(data_config)

        with open(data_config['data_cache'], 'r') as f:
            self.data_cache = json.load(f)

        "<assign other attributes here>"

    @property
    def parsing_pipeline(self):
        return self._parsing_pipeline

    @property
    def feature_pipeline(self):
        return self._feature_pipeline

    def __getitem__(self, index) -> Any:
        return self._feature_pipeline(self._parsing_pipeline(index))
    
    def calculate_datapoint_probabilities(self, cache_entry) -> float:
        # uniform sampling
        return
    
    def __len__(self):
        return len(self.data_cache)