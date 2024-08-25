import json
import math
from typing import Union

import torch

from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    SingleDataset,
    register_dataset,
)
from openfold3.core.data.pipelines.featurization.msa import featurize_msa_af3
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_target_gt_structure_af3,
)
from openfold3.core.data.pipelines.featurization.template import (
    featurize_templates_dummy_af3,
)
from openfold3.core.data.pipelines.sample_processing.msa import process_msas_cropped_af3
from openfold3.core.data.pipelines.sample_processing.structure import (
    process_target_structure_af3,
)


@register_dataset
class WeightedPDBDataset(SingleDataset):
    """Implements a Dataset class for the Weighted PDB training dataset for AF3."""

    def __init__(self, dataset_config) -> None:
        super().__init__()

        self.target_path = dataset_config["target_path"]
        self.alignments_path = dataset_config["alignments_path"]
        self.crop_weights = dataset_config["crop_weights"]
        self.token_budget = dataset_config["token_budget"]
        self.n_templates = dataset_config["n_templates"]
        self.use_alignment_database = dataset_config["use_alignment_database"]

        if self.use_alignment_database:
            with open(dataset_config["alignment_index_path"]) as f:
                self.alignment_index = json.load(f)
        else:
            self.alignment_index = None
        with open(dataset_config["dataset_cache_path"]) as f:
            self.data_cache = json.load(f)

        # Create datapoint cache (flat list of chains and interfaces)

        # Calculate datapoint probabilities
        self.calculate_datapoint_probabilities()

    def calculate_datapoint_probabilities(self):
        """Implements equation 1 from section 2.5.1 of the AF3 SI."""
        self.datapoint_probabilities = "<...>"  # TODO
        return

    def __getitem__(
        self, index
    ) -> dict[str : Union[torch.Tensor, dict[str, torch.Tensor]]]:
        # Parse training cache
        pdb_id = self.data_cache[index]["pdb_id"]

        features = {}

        # Target structure and duplicate-expanded GT structure features
        atom_array_cropped, atom_array_gt = process_target_structure_af3(
            target_path=self.target_path,
            pdb_id=pdb_id,
            crop_weights=self.crop_weights,
            token_budget=self.token_budget,
            preferred_chain_or_interface=self.data_cache[index][
                "preferred_chain_or_interface"
            ],
            ciftype=".bcif",
        )
        features.update(
            featurize_target_gt_structure_af3(
                atom_array_cropped, atom_array_gt, self.token_budget
            )
        )

        # MSA features
        msa_processed, _ = process_msas_cropped_af3(
            atom_array_cropped,
            self.data_cache[pdb_id]["chains"],
            self.alignments_path,
            max_seq_counts={
                "uniref90_hits": 10000,
                "uniprot_hits": 50000,
                "bfd_uniclust_hits": math.inf,
                "bfd_uniref_hits": math.inf,
                "mgnify_hits": 5000,
                "rfam_hits": 10000,
                "rnacentral_hits": 10000,
                "nucleotide_collection_hits": 10000,
            },
            use_alignment_database=self.use_alignment_database,
            token_budget=self.token_budget,
            max_rows_paired=8191,
            alignment_index=self.alignment_index,
        )
        features.update(featurize_msa_af3(msa_processed))

        # Template features
        features.update(
            featurize_templates_dummy_af3(1, self.n_templates, self.token_budget)
        )

        # Reference conformer features
        """<Lukas' reference conformer pipelines>"""

        # Loss switches
        features["resolution"] = self.data_cache[index]["resolution"]
        features["is_distillation"] = torch.tensor(False)
        return features

    def __len__(self):
        return len(self.data_cache)
