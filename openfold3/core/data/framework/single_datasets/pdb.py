from openfold3.core.data.framework.single_datasets import (
    SingleDataset,
    register_dataset,
)


@register_dataset
class WeightedPDBDataset(SingleDataset):
    """Implements a Dataset class for the Weighted PDB training dataset for AF3."""

    def __init__(self, dataset_config) -> None:
        super().__init__()

        # argument error checks here

        self._preprocessing_pipeline = "<some pipeline>"
        self._feature_pipeline = "<some pipeline>"

        # Parse data cache
        # with open(dataset_config['data_cache'], 'r') as f:
        #     self.data_cache = json.load(f)

        # Calculate datapoint probabilities
        self.calculate_datapoint_probabilities()

    @property
    def preprocessing_pipeline(self):
        return self._preprocessing_pipeline

    @property
    def feature_pipeline(self):
        return self._feature_pipeline

    def calculate_datapoint_probabilities(self):
        """Implements equation 1 from section 2.5.1 of the AF3 SI."""
        self.datapoint_probabilities = "<...>"  # TODO
        return

    def __len__(self):
        return len(self.data_cache)