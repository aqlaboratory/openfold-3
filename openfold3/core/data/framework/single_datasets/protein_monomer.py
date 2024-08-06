import json

from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    SingleDataset,
    register_dataset,
)


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
