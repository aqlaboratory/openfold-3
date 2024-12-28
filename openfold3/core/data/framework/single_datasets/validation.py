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
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.cropping import (
    NO_CROPPING_TOKEN_BUDGET_SENTINEL,
)
from openfold3.core.data.primitives.structure.tokenization import add_token_positions

logger = logging.getLogger(__name__)


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
