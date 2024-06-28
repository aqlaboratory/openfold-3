# TODO add license

from abc import ABC, abstractmethod
from typing import Any

from preprocessing_primitives import (
    parse_target_structure_mmCIF,
    select_cropping_strategy,
    generate_reference_conformers,
    calculate_charges,
    process_MSAs,
    parse,
    subsample,
    apply_crop,
    search_template,
    align_template,
    crop_template,
)

PREPROCESSING_PIPELINE_MAP_DOCSTRING = """
    Preprocessing pipelines are used to implement the data parsing and filtering 
    that needs to happen for individual samples returned in the __getitem__ method of a
    specific SingleDataset class. The logic and order of preprocessing steps are
    defined in the forward method using primitives from 
    preprocessing_primitives

    The steps below outline how datapoints get from raw datapoints to the model
    and highlight where you currently are in the process:
    
    0. Dataset filtering and cache generation
        raw data -> filtered data
    1. *PreprocessingPipeline* [YOU ARE HERE]
        filtered data -> processed data
    2. FeaturePipeline
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


# TODO import parsing primitives from parsing_pipeline_primitives.py
# TODO implement checks that a PreprocessingPipeline is used with the correct SingleDataset
class PreprocessingPipeline(ABC):
    """An abstract class for implementing a PreprocessingPipeline class.
    
    {data_pipeline_map}
    """.format(data_pipeline_map=PREPROCESSING_PIPELINE_MAP_DOCSTRING)

    @abstractmethod
    def forward(index):
        """Implements data parsing and filtering logic.

        Args:
            index (int): datapoint index
        """
        pass

    def __call__(self, index) -> Any:
        return self.forward(index=index)


# QUESTION can we have separate pipelines for each molecule type and composite pipelines like BioAssemblyPreprocessingPipeline and TFDNAPreprocessingPipeline calling the appropriate sub-pipelines?


class BioAssemblyPreprocessingPipeline(PreprocessingPipeline):
    """A PreprocessingPipeline for SingeDataset(s) that need to handle arbitrary molecule types,
    including protein, DNA, RNA, and ligands.

    {data_pipeline_map}
    """.format(data_pipeline_map=PREPROCESSING_PIPELINE_MAP_DOCSTRING)

    def forward(self, index: int) -> dict:
        """Implements data parsing and filtering logic.

        Args:
            index (int): datapoint index
        """
        pass

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

        # for each recycling step process MSAs
        process_MSAs()
        for step in self.recycling_steps:
            # - parse
            parse()
            # - subsample
            subsample()
            # - apply crop
            apply_crop()
        # -> msa_raw_data

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
    """A PreprocessingPipeline for SingeDataset(s) that implement the preprocessing
    logic of the Transcription Factor Positive Distillation dataset of AF3 (SI 2.5.2.).
    
    {data_pipeline_map}
    """.format(data_pipeline_map=PREPROCESSING_PIPELINE_MAP_DOCSTRING)

    def forward(self, index: int) -> dict:
        pass
