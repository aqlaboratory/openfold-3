# TODO add license

from abc import ABC, abstractmethod
from typing import Any

# TODO import parsing primitives from parsing_pipeline_primitives.py
# TODO implement checks that a ParserPipeline is used with the correct SingleDataset
class ParserPipeline(ABC):
    """An abstract class for implementing a ParserPipeline.
    """

    @abstractmethod
    def forward(index):
        pass

    def __call__(self, index) -> Any:
        """Implements data parsing and filtering logic.

        Args:
            index (int): datapoint index
        """
        return self.forward(index=index)
    
# BioAssemblyParserPipeline: MSA parsing, template mmCIF parsing and prefiltering, training target mmCIF parsing