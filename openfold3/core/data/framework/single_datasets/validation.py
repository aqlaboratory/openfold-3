import logging
import random
import traceback

import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray

from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.base_af3 import BaseAF3Dataset
from openfold3.core.data.framework.single_datasets.pdb import is_invalid_feature_dict
from openfold3.core.data.primitives.featurization.structure import (
    extract_starts_entities,
)

logger = logging.getLogger(__name__)


def make_chain_pair_mask_padded(
    all_chains: torch.Tensor, interfaces_to_include: list[tuple[int, int]]
) -> torch.Tensor:
    """Creates a pairwise mask for chains given a list of chain tuples.
    Args:
        all_chains: tensor containing all chain ids in complex
        interfaces_to_include: tuples with pairwise interactions to include
    Returns:
        torch.Tensor [n_chains + 1, n_chains + 1] where:
            - each value [i,j] represents
            - a 0th row and 0th column of all zeros is added as padding
    """
    largest_chain_index = torch.max(all_chains)
    chain_mask = torch.zeros(
        (largest_chain_index + 1, largest_chain_index + 1), dtype=torch.int
    )

    for interface_tuple in interfaces_to_include:
        chain_mask[int(interface_tuple[0]), int(interface_tuple[1])] = 1
        chain_mask[int(interface_tuple[1]), int(interface_tuple[0])] = 1

    return chain_mask


@register_dataset
class ValidationPDBDataset(BaseAF3Dataset):
    """Validation Dataset class."""

    def __init__(self, dataset_config: dict) -> None:
        """Initializes a ValidationDataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__(dataset_config)

        # Dataset/datapoint cache
        self.create_datapoint_cache()

        # Cropping is turned off
        self.apply_crop = False

    def create_datapoint_cache(self):
        """Creates the datapoint_cache for iterating over each sample.

        Creates a Dataframe storing a flat list of structure_data keys. Used for mapping
        TO the dataset_cache in the getitem. Note that the validation set is not wrapped
        in a StoachasticSamplerDataset.
        """
        self.datapoint_cache = pd.DataFrame(
            {
                "pdb_id": list(self.dataset_cache.structure_data.keys()),
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

        if not self.debug_mode:
            sample_data = self.create_all_features(
                pdb_id=pdb_id,
                preferred_chain_or_interface=None,
                return_atom_arrays=False,
            )
            features = sample_data["features"]
            features["pdb_id"] = pdb_id
            return features

        else:
            try:
                sample_data = self.create_all_features(
                    pdb_id=pdb_id,
                    preferred_chain_or_interface=None,
                    return_atom_arrays=False,
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
                    + f"Failed to process ValidationPDBDataset entry {pdb_id}:"
                    + f" {str(e)}\n"
                    + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                    + "-" * 40
                )
                index = random.randint(0, len(self) - 1)
                return self.__getitem__(index)

    def get_validation_homology_features(
        self, pdb_id: str, atom_array: AtomArray
    ) -> dict:
        """Create masks for validation metrics analysis.

        Args:
            pdb_id: PDB id for example found in dataset_cache
            atom_array: structure data for given pdb_id
        Returns:
            dict with two features:
            - use_for_intra_validation [*, n_tokens]
                mask indicating if token should be used for intrachain metrics
            - use_for_inter_validation [*, n_tokens]
                mask indicating if token should be used for intrachain metrics
        """
        features = {}

        structure_entry = self.dataset_cache.structure_data[pdb_id]

        chains_for_intra_metrics = [
            cid
            for cid, cdata in structure_entry.chains.items()
            if cdata.use_intrachain_metrics
        ]

        interfaces_to_include = []
        for interface_id, cluster_data in structure_entry.interfaces.items():
            if cluster_data.use_interchain_metrics:
                interface_chains = tuple(interface_id.split("_"))
                interfaces_to_include.append(interface_chains)

        # Create token mask for validation intra and inter metrics
        token_starts_with_stop, _ = extract_starts_entities(atom_array)
        token_starts = token_starts_with_stop[:-1]
        token_chain_id = atom_array.chain_id[token_starts].astype(int)

        features["use_for_intra_validation"] = torch.tensor(
            np.isin(token_chain_id, chains_for_intra_metrics),
            dtype=torch.int32,
        )

        token_chain_id = torch.tensor(token_chain_id, dtype=torch.int32)

        chain_mask_padded = make_chain_pair_mask_padded(
            token_chain_id, interfaces_to_include
        )

        # [n_token, n_token] for pairwise interactions
        features["use_for_inter_validation"] = chain_mask_padded[
            token_chain_id.unsqueeze(0), token_chain_id.unsqueeze(1)
        ]

        return features

    def create_all_features(
        self,
        pdb_id: str,
        preferred_chain_or_interface: str,
        return_atom_arrays: bool,
    ) -> dict:
        """Calls the parent create_all_features, and then adds features for homology
        similarity."""
        sample_data = super().create_all_features(
            pdb_id, preferred_chain_or_interface, return_atom_arrays=True
        )

        validation_homology_filters = self.get_validation_homology_features(
            pdb_id, sample_data["atom_array"]
        )
        sample_data["features"].update(validation_homology_filters)
        sample_data["features"]["atom_array"] = sample_data["atom_array"]

        # Remove atom arrays if they are not needed
        if not return_atom_arrays:
            del sample_data["atom_array"]
            del sample_data["atom_array_gt"]
            del sample_data["atom_array_cropped"]

        return sample_data
