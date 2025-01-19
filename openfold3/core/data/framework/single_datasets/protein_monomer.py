# %%
import logging
import random
import traceback

import pandas as pd
import torch

from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.base_af3 import (
    BaseAF3Dataset,
)
from openfold3.core.data.framework.single_datasets.pdb import is_invalid_feature_dict

logger = logging.getLogger(__name__)

DEBUG_PROTEIN_MONOMER_BLACKLIST = ["MGYP000285232442"]


@register_dataset
class ProteinMonomerDataset(BaseAF3Dataset):
    def __init__(self, dataset_config: dict) -> None:
        """Initializes a ProteinMonomerDataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__(dataset_config)

        # Datapoint cache
        self.create_datapoint_cache()

        # Dataset configuration
        self.apply_crop = True
        self.crop = dataset_config["custom"]["crop"]

        # All samples are protein
        self.single_moltype = "PROTEIN"

    def create_datapoint_cache(self):
        """Creates the datapoint_cache for uniform sampling.

        Creates a Dataframe storing a flat list of structure_data keys and sets
        corresponding datapoint probabilities all to 1. Used for mapping FROM the
        dataset_cache in the StochasticSamplerDataset and TO the dataset_cache in the
        getitem.
        """
        sample_ids = list(self.dataset_cache.structure_data.keys())
        self.datapoint_cache = pd.DataFrame(
            {
                "pdb_id": sample_ids,
                "datapoint_probabilities": [1.0] * len(sample_ids),
            }
        )

    def __getitem__(
        self, index: int
    ) -> dict[str : torch.Tensor | dict[str, torch.Tensor]]:
        """Returns a single datapoint from the dataset.

        Note: The data pipeline is modularized at the getitem level to enable
        subclassing for profiling without code duplication. See
        logging_datasets.py for an example."""

        # Get PDB ID from the datapoint cache and the preferred chain/interface
        datapoint = self.datapoint_cache.iloc[index]
        pdb_id = datapoint["pdb_id"]

        # TODO: Remove debug logic
        if not self.debug_mode:
            sample_data = self.create_all_features(
                pdb_id=datapoint["pdb_id"],
                preferred_chain_or_interface=None,
                return_atom_arrays=False,
                return_crop_strategy=False,
            )

            features = sample_data["features"]
            features["pdb_id"] = pdb_id
            return features
        else:
            try:
                if pdb_id in DEBUG_PROTEIN_MONOMER_BLACKLIST:
                    logger.warning(f"Skipping blacklisted pdb id {pdb_id}")
                    index = random.randint(0, len(self) - 1)
                    return self.__getitem__(index)

                sample_data = self.create_all_features(
                    pdb_id=datapoint["pdb_id"],
                    preferred_chain_or_interface=None,
                    return_atom_arrays=False,
                    return_crop_strategy=False,
                )

                features = sample_data["features"]

                features["pdb_id"] = pdb_id
                if is_invalid_feature_dict(features):
                    index = random.randint(0, len(self) - 1)
                    return self.__getitem__(index)

                return features

            except Exception as e:
                tb = traceback.format_exc()
                logger.warning(
                    "-" * 40
                    + "\n"
                    + f"Failed to process ProteinMonomerDataset entry {pdb_id}:"
                    + f" {str(e)}\n"
                    + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                    + "-" * 40
                )
                index = random.randint(0, len(self) - 1)
                return self.__getitem__(index)
