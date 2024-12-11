#%%
import json
import pandas as pd 
from openfold3.core.data.framework.single_datasets.abstract_single_dataset import register_dataset
from openfold3.core.data.framework.single_datasets.pdb import WeightedPDBDataset

@register_dataset
class MonomerDistillationDataset(WeightedPDBDataset):
    def create_datapoint_cache(self):
        """
        Same premise as the ValidationDatasetClass - no need for chain/interface
        distinctions or sample weighting - just need to iterate over all entries
        - "datapoint" is used here -> preferred_chain_or_interface = datapoint["datapoint"]
        setting this to None uses just the chain 
        """
        pdb_ids = []
        for entry, _ in self.dataset_cache.structure_data.items():
            pdb_ids.append((entry, None, -1))

        self.datapoint_cache = pd.DataFrame(
            pdb_ids, columns=["pdb_id", "datapoint", "weight"]
        )
