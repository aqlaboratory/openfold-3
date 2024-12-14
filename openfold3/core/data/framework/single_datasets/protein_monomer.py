# %%
import pandas as pd

from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.base_single_dataset import (
    BaseSingleDataset
)


@register_dataset
class ProteinMonomerDataset(BaseSingleDataset):
    def create_datapoint_cache(self):
        """
        Same premise as the ValidationDatasetClass - no need for chain/interface
        distinctions or sample weighting - just need to iterate over all entries
        - "datapoint" is used here -> preferred_chain_or_interface = datapoint["datapoint"]
        setting this to None uses just the chain
        """
        self.datapoint_cache = pd.DataFrame(
            list(self.dataset_cache.structure_data.keys()), columns=["pdb_id"]
        ).assign(weight=1.0) 
        ## weight column requried by StochasticSampler. Setting this to 1 
        ## weights all samples equally(weight is passed to torch.multinomial)
