# TODO add license

from abc import ABC, abstractmethod
from typing import Any

# TODO import parsing primitives from parsing_pipeline_primitives.py
# TODO implement checks that a ParsingPipeline is used with the correct SingleDataset
class ParsingPipeline(ABC):
    """An abstract class for implementing a ParsingPipeline.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, index):
        """Implements data parsing and filtering logic.

        Args:
            index (int): datapoint index
        """
        pass

    def __call__(self, index) -> Any:
        return self.forward(index=index)