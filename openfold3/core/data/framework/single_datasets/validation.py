import logging

import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray

from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.pdb import WeightedPDBDataset
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_af3,
)
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_target_gt_structure_af3,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    get_reference_conformer_data_af3,
)
from openfold3.core.data.pipelines.sample_processing.structure import (
    process_target_structure_af3,
)
from openfold3.core.data.primitives.featurization.structure import (
    extract_starts_entities,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.cropping import (
    NO_CROPPING_TOKEN_BUDGET_SENTINEL,
)
from openfold3.core.data.primitives.structure.tokenization import add_token_positions

logger = logging.getLogger(__name__)


def make_chain_pair_mask_padded(all_chains: torch.Tensor, interfaces_to_include: list[tuple[int, int]])-> torch.Tensor:
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

    # TODO: Factor out redundancy with WeightedPDBDataset
    @log_runtime_memory(runtime_dict_key="runtime-create-structure-features")
    def create_structure_features(
        self,
        pdb_id: str,
        preferred_chain_or_interface: str,
        return_full_atom_array: bool,
    ) -> tuple[dict, AtomArray | torch.Tensor]:
        """Creates the target structure and conformer features."""

        # Processed ground-truth structure with added annotations and masks
        atom_array_gt, crop_strategy = process_target_structure_af3(
            target_structures_directory=self.target_structures_directory,
            pdb_id=pdb_id,
            crop_weights=self.crop_weights,
            token_budget=NO_CROPPING_TOKEN_BUDGET_SENTINEL,
            preferred_chain_or_interface=preferred_chain_or_interface,
            structure_format="pkl",
            per_chain_metadata=self.dataset_cache.structure_data[pdb_id].chains,
        )

        # Processed reference conformers
        processed_reference_molecules = get_reference_conformer_data_af3(
            atom_array=atom_array_gt,
            per_chain_metadata=self.dataset_cache.structure_data[pdb_id].chains,
            reference_mol_metadata=self.dataset_cache.reference_molecule_data,
            reference_mol_dir=self.reference_molecule_directory,
        )

        if return_full_atom_array:
            atom_array_full = atom_array_gt.copy()

        # Necessary for compatibility with downstream function signatures
        atom_array_cropped = atom_array_gt.copy()

        # Necessary positional indices for MSA and template processing
        add_token_positions(atom_array_cropped)

        # Overwrite token_budget to the number of tokens in this example
        n_tokens = np.unique(atom_array_gt.token_id).size
        logger.info(f"Overwriting token budget to {n_tokens} for {pdb_id}")
        self.token_budget = n_tokens

        # Compute target and ground-truth structure features
        target_structure_features = featurize_target_gt_structure_af3(
            atom_array_cropped=atom_array_cropped,
            atom_array_gt=atom_array_gt,
            token_budget=self.token_budget,
        )

        # Compute reference conformer features
        reference_conformer_features = featurize_reference_conformers_af3(
            processed_ref_mol_list=processed_reference_molecules
        )

        # Wrap up features
        target_structure_data = {
            "atom_array_gt": atom_array_gt,
            "atom_array_cropped": atom_array_cropped,
            "target_structure_features": target_structure_features,
            "reference_conformer_features": reference_conformer_features,
        }

        if return_full_atom_array:
            target_structure_data["atom_array"] = atom_array_full

        return target_structure_data

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

        chain_mask_padded = make_chain_pair_mask_padded(token_chain_id, interfaces_to_include)

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

        # If we have all the datasets write their own create_all_features, we can avoid
        # recording and then removing the atom_arrays
        if not return_atom_arrays:
            del sample_data["atom_array"]
            del sample_data["atom_array_gt"]
            del sample_data["atom_array_cropped"]

        return sample_data
