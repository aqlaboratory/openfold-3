# TODO add license

from abc import ABC, abstractmethod, property
from typing import Any

from torch.utils.data import Dataset


# currently requires the definition of methods parsing_pipeline, feature_pipeline in the inheritee SingleDataset
# to return the respective attributes, which I don't like
class OpenFoldSingleDataset(ABC):
    """Abstract SingleDataset class for implementing parsing and feature pipeline methods."""

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

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self, "parsing_pipeline") or self.parsing_pipeline is None:
            raise NotImplementedError(
                "Any OpenFoldSingleDataset must implement a parsing_pipeline attribute."
            )
        if not hasattr(self, "feature_pipeline") or self.feature_pipeline is None:
            raise NotImplementedError(
                "Any OpenFoldSingleDataset must implement a feature_pipeline attribute."
            )


# Currently agreed to have separate SingleDatasets for train/eval/test/predict as one monolithic SingleDataset for
# a specific model goes against the basic goal of having separate SingleDatasets for each specific dataset. The
# SingleDatasets can still have a hierarchy so that they don't reimplement identical pipelines.
class OpenFold3TrainSingleDataset(OpenFoldSingleDataset, Dataset):
    """SingleDataset for the core training dataset of OpenFold3.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parsing_pipeline = "<here assign a ParsingPipeline class object>"
        self.feature_pipeline = "<here assign a FeaturePipeline class object>"
        self.data_cache = '<...>'

        "<assign other attributes here>"

    def parsing_pipeline(self):
        return super().parsing_pipeline()

    def feature_pipeline(self):
        return super().feature_pipeline()

    def __getitem__(self, index) -> Any:
        return self.feature_pipeline(self.parsing_pipeline(index))
    
    def __len__(self):
        return len(self.data_cache)


class OpenFold3EvalSingleDataset(OpenFoldSingleDataset, Dataset):
    """SingleDataset for the validation dataset of OpenFold3.
    """

    def __init__(self) -> None:  # do we want a class for handling the arguments of SingleDatasets?
        super().__init__()
        self.parsing_pipeline = "<here assign a ParsingPipeline class object>"
        self.feature_pipeline = "<here assign a FeaturePipeline class object>"
        self.data_cache = '<...>'

        "<assign other attributes here>"

    def parsing_pipeline(self):
        return super().parsing_pipeline()

    def feature_pipeline(self):
        return super().feature_pipeline()

    def __getitem__(self, index) -> Any:
        return self.feature_pipeline(self.parsing_pipeline(index))
    
    def __len__(self):
        return len(self.data_cache)
