from typing import Union

import torch

from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    SingleDataset,
    register_dataset,
)
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_target_gt_structure_af3,
)
from openfold3.core.data.pipelines.sample_processing.msa import process_msas_af3
from openfold3.core.data.pipelines.sample_processing.structure import (
    process_target_structure_af3,
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

    def __getitem__(
        self, index
    ) -> dict[str : Union[torch.Tensor, dict[str, torch.Tensor]]]:
        # Parse training cache
        pdb_id = self.data_cache[index]["pdb_id"]

        # Target structure and duplicate-expanded GT structure features
        atom_array_cropped, atom_array_gt = process_target_structure_af3(
            target_path=self.target_path,
            pdb_id=pdb_id,
            crop_weights=self.crop_weights,
            token_budget=self.token_budget,
            preferred_chain_or_interface=self.data_cache[index]['preferred_chain_or_interface'],
            ciftype=".bcif",
        )
        features = featurize_target_gt_structure_af3(
            atom_array_cropped, atom_array_gt, self.token_budget
        )

        # MSA features
        query_seq, paired_msa, main_msas = process_msas_af3(
            chain_ids=self.data_cache[index]["chain_ids"],
            alignments_path=self.alignments_path,
            max_seq_counts=self.max_seq_counts,
            use_alignment_database=self.use_alignment_database,
            alignment_index=self.alignment_index,
        )
        """<MSA featurization pipeline>"""

        # Template features
        """<Dummy template featurization pipeline until review>"""

        # Reference conformer features
        """<Lukas' reference conformer pipelines>"""

        # Loss switches
        features["resolution"] = self.data_cache[index]["resolution"]
        features["is_distillation"] = torch.tensor(False)
        return features

    def __len__(self):
        return len(self.data_cache)
