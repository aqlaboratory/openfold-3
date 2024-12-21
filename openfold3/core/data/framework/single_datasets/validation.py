import logging

import pandas as pd
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
from openfold3.core.data.primitives.structure.cropping import (
    NO_CROPPING_TOKEN_BUDGET_SENTINEL,
)


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
