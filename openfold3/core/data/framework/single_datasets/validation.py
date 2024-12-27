import logging

import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray

from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.pdb import WeightedPDBDataset
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_target_gt_structure_af3,
)
from openfold3.core.data.pipelines.sample_processing.structure import (
    process_target_structure_af3,
)
from openfold3.core.data.primitives.featurization.structure import (
    extract_starts_entities,
)
from openfold3.core.data.primitives.structure.cropping import (
    NO_CROPPING_TOKEN_BUDGET_SENTINEL,
)


def make_chain_mask_padded(all_chains, interfaces_to_include):
    largest_chain_index = torch.max(all_chains)
    chain_mask = torch.zeros((largest_chain_index + 1, largest_chain_index + 1), dtype=torch.int)

    for interface_tuple in interfaces_to_include:
        chain_mask[interface_tuple[0], interface_tuple[1]] = 1
        chain_mask[interface_tuple[1], interface_tuple[0]] = 1
    
    return chain_mask

@register_dataset
class ValidationPDBDataset(WeightedPDBDataset):
    """Dataset class for the validation set of the WeightedPDBDataset."""

    def __init__(self, dataset_config: dict) -> None:
        """Initializes a ValidationDataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__(dataset_config)

        # Dataset/datapoint cache
        self.datapoint_cache = {}
        self.create_datapoint_cache()
        self.datapoint_probabilities = self.datapoint_cache["weight"].to_numpy()

    def create_datapoint_cache(self):
        """
        The WeightedPDBDataset uses the per-chain/per-interface datapoint cache as the
        base item, but all pre-processing code is applied to the complete structure,
        then subset based on the crop. We want the validation set to be per-complete
        structure - so we need to create a new cache However, the actual datapoint cache
        values won't actually be ever used:

        - The `datapoint` value is normally used as part of the crop generation process.
          However, it only ends up being used when the complete structure is has more
          tokens than the token budget, which is never the case for the validation set.
        - The `weight` value is used within the datapoint cache to determine the
          probability of sampling that chain - but appears to only ever be stored and
          unused within the actual PDB dataset class.

        Therefore creating a data cache with dummy values lets us re-use the whole PDB
        dataset class. This does break if we have more tokens than the token budget
        """
        pdb_ids = []
        for entry, _ in self.dataset_cache.structure_data.items():
            pdb_ids.append((entry, -1, -1))

        self.datapoint_cache = pd.DataFrame(
            pdb_ids, columns=["pdb_id", "datapoint", "weight"]
        )

    def create_target_structure_features(
        self,
        pdb_id: str,
        preferred_chain_or_interface: str,
        unused_return_atom_arrays: bool,
    ) -> tuple[dict, AtomArray]:
        """Creates the target structure features.

        IMPORTANT: Because there is no cropping in the validation set, this method will
        additionally overwrite self.token_budget to match the number of tokens in the
        PDB structure.
        """

        # Target structure and duplicate-expanded GT structure features
        target_structure_data = process_target_structure_af3(
            target_structures_directory=self.target_structures_directory,
            pdb_id=pdb_id,
            crop_weights=self.crop_weights,
            token_budget=NO_CROPPING_TOKEN_BUDGET_SENTINEL,
            preferred_chain_or_interface=preferred_chain_or_interface,
            structure_format="pkl",
            return_full_atom_array=True,
        )
        # NOTE that for now we avoid the need for permutation alignment by providing the
        # cropped atom array as the ground truth atom array
        # target_structure_features.update(
        #     featurize_target_gt_structure_af3(
        #         atom_array_cropped, atom_array_gt, self.token_budget
        #     )
        # )
        n_tokens = len(set(target_structure_data["atom_array"].token_id))

        # Overwrite token_budget to the number of tokens in this example
        logging.info(f"Overwriting token budget to {n_tokens} for {pdb_id}")
        self.token_budget = n_tokens

        target_structure_features = featurize_target_gt_structure_af3(
            target_structure_data["atom_array"],
            target_structure_data["atom_array"],
            self.token_budget,
        )
        target_structure_data["target_structure_features"] = target_structure_features

        # Overwrite self.token_budget to be the number of tokens in this example
        return target_structure_data

    def get_validation_homology_features(self, pdb_id: str, atom_array: AtomArray)-> dict:
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
        print(f"{pdb_id=} {chains_for_intra_metrics=}")

        interfaces_to_include = [] 
        for interface_id, cluster_data in structure_entry.interfaces.items():
            if cluster_data.use_interchain_metrics:
                 print(f"{pdb_id=} {interface_id=} to be included")
                 interface_chains = tuple(interface_id.split("_")) 
                 interfaces_to_include.append(interface_chains)
        
        # Create token mask for validation intra and inter metrics
        token_starts_with_stop, _ = extract_starts_entities(atom_array)
        token_starts = token_starts_with_stop[:-1]
        token_chain_id = atom_array.chain_id[token_starts]

        features["use_for_intra_validation"] = torch.tensor(
            np.isin(token_chain_id, chains_for_intra_metrics),
            dtype=torch.int32,
        )

        all_chain_ids = torch.unique(token_chain_id)
        chain_mask_padded = make_chain_mask_padded(all_chain_ids, interfaces_to_include)

        # [n_token, n_token] for pairwise interactions
        features["use_for_inter_validation"] = chain_mask_padded[token_chain_id.unsqueeze(0), token_chain_id.unsqueeze(1)]

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
            pdb_id, preferred_chain_or_interface, return_atom_arrays = True
        )

        validation_homology_filters = self.get_validation_homology_features(
            pdb_id, sample_data["atom_array"]
        )
        sample_data["features"].update(validation_homology_filters)

        # If we have all the datasets write their own create_all_features, we can avoid
        # recording and then removing the atom_arrays
        if not return_atom_arrays:
            del sample_data["atom_array"]
            del sample_data["atom_array_gt"]
            del sample_data["atom_array_cropped"]
        
        return sample_data
