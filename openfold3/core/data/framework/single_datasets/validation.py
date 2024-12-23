import logging

import pandas as pd
import torch
from biotite.structure import AtomArray

from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.base_af3 import BaseAF3Dataset
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
        # TODO: refactor cropping logic to enable cleaner way of turning it off
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

    def __getitem__(
        self, index: int
    ) -> dict[str : torch.Tensor | dict[str, torch.Tensor]]:
        """Returns a single datapoint from the dataset.

        Note: The data pipeline is modularized at the getitem level to enable
        subclassing for profiling without code duplication. See
        logging_datasets.py for an example."""

        # Get PDB ID from the datapoint cache and the preferred chain/interface
        datapoint = self.datapoint_cache.iloc[index]
        sample_data = self.create_all_features(
            pdb_id=datapoint["pdb_id"],
            preferred_chain_or_interface=None,
            return_atom_arrays=False,
        )
        return sample_data["features"]
