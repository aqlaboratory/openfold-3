# TODO add license

from abc import ABC, abstractmethod
from typing import Any

from preprocessing_primitives import (
    align_template,
    calculate_charges,
    crop_template,
    generate_reference_conformers,
    parse_target_structure_mmCIF,
    search_template,
    select_cropping_strategy,
)


# TODO import parsing primitives from parsing_pipeline_primitives.py
# TODO implement checks that a PreprocessingPipeline is used with the
# correct SingleDataset
class PreprocessingPipeline(ABC):
    """An abstract class for implementing a specific PreprocessingPipeline class."""

    @abstractmethod
    def forward(index):
        pass

    def __call__(self, index) -> Any:
        """Implements data parsing and filtering logic.

        Args:
            index (int): datapoint index
        """
        return self.forward(index=index)


# QUESTION can we have separate pipelines for each molecule type and composite pipelines
# like BioAssemblyPreprocessingPipeline and TFDNAPreprocessingPipeline calling the
# appropriate sub-pipelines?


class BioAssemblyPreprocessingPipeline(PreprocessingPipeline):
    def forward(self, index: int) -> dict:
        """_summary_

        Args:
            index (int): datapoint index

        Returns:
            dict: parsed raw data dictionary
        """

        # Parse target structure mmCIF
        parse_target_structure_mmCIF()

        # Select cropping strategy and crop
        crop = select_cropping_strategy()
        crop()

        # Target feature preprocessing
        # - reference conformer generation
        generate_reference_conformers()
        # - charge calculation
        calculate_charges()
        # -> target_raw_data

        # # for each recycling step process MSAs
        # process_MSAs()
        # for _ in self.recycling_steps:
        #     # - parse
        #     parse()
        #     # - subsample
        #     subsample()
        #     # - apply crop
        #     apply_crop()
        # # -> msa_raw_data

        # Template preprocessing
        # - search
        search_template()
        # - alignment
        align_template()
        # - cropping
        crop_template()
        # -> template_raw_data

        # return {target_raw_data, msa_raw_data, template_raw_data}


class TFDNAPreprocessingPipeline(PreprocessingPipeline):
    def forward(self, index: int) -> dict:
        pass
