# TODO add license

from abc import ABC, abstractmethod
from typing import Any


# TODO import parsing primitives from feature_pipeline_primitives.py
# TODO implement checks that a FeaturePipeline is used with the correct SingleDataset
class FeaturePipeline(ABC):
    """An abstract class for implementing a FeaturePipeline."""

    @abstractmethod
    def forward(self, index):
        """Implements featurization and embedding logic.

        Args:
            index (int): datapoint index
        """
        pass

    def __call__(self, parsed_features: dict) -> Any:
        return self.forward(parsed_features=parsed_features)


class AF3BioAssemblyFeaturePipeline(FeaturePipeline):
    def forward(self, parsed_features) -> dict:
        """_summary_

        Args:
            parsed_features (_type_): set of coordinates from parsed mmCIF and parsed
            MSAs, chain/interface metadata, center atom around which to crop

        Returns:
            dict: feature dictionary (Table 5)
        """
        # Featurization
        # Token features
        # Reference conformer features
        # MSA features
        # Template features
        # Bond features


# class AF3ProteinDNAFeaturePipeline(FeaturePipeline):
#     def forward(self, parsed_features) -> dict:
#         """_summary_

#         Args: parsed_features (_type_): set of coordinates from parsed mmCIF and
#             parsed MSAs, center atom around which to crop

#         Returns:
#             dict: feature dictionary (Table 5)
#         """
#         # Featurization
#         # Token features
#         # Reference conformer features
#         # MSA features
#         # Template features
#         # Bond features
