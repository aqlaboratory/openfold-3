# TODO add license

from abc import ABC, abstractmethod
from typing import Any

# TODO import parsing primitives from feature_pipeline_primitives.py
# TODO implement checks that a FeaturePipeline is used with the correct SingleDataset
class FeaturePipeline(ABC):
    """An abstract class for implementing a FeaturePipeline.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, index):
        """Implements featurization and embedding logic.

        Args:
            index (int): datapoint index
        """
        pass

    def __call__(self, index) -> Any:
        return self.forward(index=index)