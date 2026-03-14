# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
import traceback

import numpy as np
import pandas as pd
import torch

from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.base_of3 import (
    BaseOF3Dataset,
)
from openfold3.core.data.framework.single_datasets.dataset_utils import (
    check_invalid_feature_dict,
    pad_to_world_size,
)
from openfold3.core.data.primitives.featurization.structure import (
    extract_starts_entities,
    make_chain_pair_labels_padded,
    make_chain_pair_mask_padded,
)
from openfold3.core.data.resources.lists import (
    AB_AG_CHAIN_PAIR_TYPES,
    AB_AG_CHAIN_TYPES,
)
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms

logger = logging.getLogger(__name__)


@register_dataset
class ValidationPDBDataset(BaseOF3Dataset):
    """Validation Dataset class."""

    def __init__(self, dataset_config: dict, world_size: int | None = None) -> None:
        """Initializes a ValidationDataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__(dataset_config)

        self.world_size = world_size

        # Dataset/datapoint cache
        self.create_datapoint_cache()

        # Cropping should be disabled for validation datasets
        if self.crop["token_crop"]["enabled"]:
            logger.warning(
                "Token cropping is enabled for ValidationPDBDataset. Make sure this is"
                " intended."
            )
        if self.crop["chain_crop"]["enabled"]:
            logger.warning(
                "Chain cropping is enabled for ValidationPDBDataset. Make sure this is"
                " intended."
            )

    def create_datapoint_cache(self):
        """Creates the datapoint_cache for iterating over each sample.

        Creates a Dataframe storing a flat list of structure_data keys. Used for mapping
        TO the dataset_cache in the getitem. Note that the validation set is not wrapped
        in a StochasticSamplerDataset.
        """
        # Order by token count so that the run times are more consistent across GPUs
        pdb_ids = list(self.dataset_cache.structure_data.keys())

        def null_safe_token_count(x):
            elem = self.dataset_cache.structure_data[x]
            token_count = elem.token_count if hasattr(elem, "token_count") else None
            return token_count if token_count is not None else 0

        pdb_ids = sorted(
            pdb_ids,
            key=null_safe_token_count,
            reverse=False,
        )

        _datapoint_cache = pd.DataFrame({"pdb_id": pdb_ids})
        self.datapoint_cache = pad_to_world_size(_datapoint_cache, self.world_size)

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
        is_repeated_sample = bool(datapoint["repeated_sample"])

        if not self.debug_mode:
            sample_data = self.create_all_features(
                pdb_id=pdb_id,
                preferred_chain_or_interface=None,
                return_atom_arrays=True,
                return_crop_strategy=False,
            )
            features = sample_data["features"]
            features["pdb_id"] = pdb_id
            features["preferred_chain_or_interface"] = "none"
            features["repeated_sample"] = torch.tensor(
                [is_repeated_sample], dtype=torch.bool
            )

            return features

        else:
            try:
                sample_data = self.create_all_features(
                    pdb_id=pdb_id,
                    preferred_chain_or_interface=None,
                    return_atom_arrays=True,
                    return_crop_strategy=False,
                )

                features = sample_data["features"]
                features["pdb_id"] = pdb_id
                features["preferred_chain_or_interface"] = "none"

                check_invalid_feature_dict(features)

                features["repeated_sample"] = torch.tensor(
                    [is_repeated_sample], dtype=torch.bool
                )

                return features

            except Exception as e:
                tb = traceback.format_exc()
                dataset_name = self.get_class_name()
                logger.warning(
                    "-" * 40
                    + "\n"
                    + f"Failed to process {dataset_name} entry {pdb_id}:"
                    + f" {str(e)}\n"
                    + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                    + "-" * 40
                )
                index = random.randint(0, len(self) - 1)
                return self.__getitem__(index)

    # TODO factor out primitives
    def get_validation_homology_features(self, pdb_id: str, sample_data: dict) -> dict:
        """Create masks for validation metrics analysis.

        Args:
            pdb_id: PDB id for example found in dataset_cache
            sample_data: dictionary containing features for the sample and atom array
        Returns:
            dict with two features:
            - use_for_intra_validation [*, n_atoms]
                mask indicating if token should be used for intrachain metrics
            - use_for_inter_validation [*, n_atoms, n_atoms]
                mask indicating if token should be used for intrachain metrics
        """
        features = {}

        structure_entry = self.dataset_cache.structure_data[pdb_id]

        def _use_metrics(x):
            """Check if the chain or interface should be used for metrics."""
            return x.use_metrics if hasattr(x, "use_metrics") else True

        chains_for_intra_metrics = [
            int(cid)
            for cid, cdata in structure_entry.chains.items()
            if _use_metrics(cdata)
        ]

        interfaces_to_include = []
        for interface_id, cluster_data in structure_entry.interfaces.items():
            if _use_metrics(cluster_data):
                interface_chains = tuple(int(ci) for ci in interface_id.split("_"))
                interfaces_to_include.append(interface_chains)

        # Create token mask for validation intra and inter metrics
        atom_array = sample_data["atom_array"]
        token_starts_with_stop, _ = extract_starts_entities(atom_array)
        token_starts = token_starts_with_stop[:-1]
        token_chain_id = atom_array.chain_id[token_starts].astype(int)

        token_mask = sample_data["features"]["token_mask"]
        num_atoms_per_token = sample_data["features"]["num_atoms_per_token"]

        use_for_intra = torch.tensor(
            np.isin(token_chain_id, chains_for_intra_metrics),
            dtype=torch.int32,
        )
        intra_filter_atomized = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=use_for_intra,
        ).bool()

        features["intra_filter_atomized"] = intra_filter_atomized

        token_chain_id = torch.tensor(token_chain_id, dtype=torch.int32)
        chain_mask_padded = make_chain_pair_mask_padded(
            token_chain_id, interfaces_to_include
        )

        # [n_token, n_token] for pairwise interactions
        use_for_inter = chain_mask_padded[
            token_chain_id.unsqueeze(0), token_chain_id.unsqueeze(1)
        ]

        # convert use_for_inter: [*, n_token, n_token] into [*, n_atom, n_atom]
        inter_filter_atomized = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=use_for_inter,
            token_dim=-2,
        )
        inter_filter_atomized = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=inter_filter_atomized.transpose(-1, -2),
            token_dim=-2,
        )
        inter_filter_atomized = inter_filter_atomized.transpose(-1, -2).bool()
        features["inter_filter_atomized"] = inter_filter_atomized

        return features

    # TODO factor out primitives
    def get_ab_ag_features(self, pdb_id, sample_data):
        """Create AB/AG label features for validation metrics.

        Label tensors encode the following types:
            Chain labels:
                1. antibody heavy chain
                2. antibody light chain
                3. antigen chain
            Chain pair labels:
                1. antibody heavy chain - antibody heavy chain
                2. antibody heavy chain - antibody light chain
                3. antibody heavy chain - antigen chain
                4. antibody light chain - antibody light chain
                5. antibody light chain - antigen chain
                6. antigen chain - antigen chain

        Args:
            pdb_id:
                PDB id for example found in dataset_cache
            sample_data:
                dictionary containing features for the sample and atom array
        Returns:
            dict with two features:
            - intra_ab_ag_type_atomized [*, n_atoms]
                per atom integer labels indicating which chain type an atom belongs to
            - inter_ab_ag_type_atomized [*, n_atoms, n_atoms]
                per atom pair integer labels indicating which chain pair type an atom
                pair belongs to
        """
        features = {}
        structure_entry = self.dataset_cache.structure_data[pdb_id]

        # Create AB/AG type features
        # Per-chain maps
        ab_ag_type_to_chain_id = {t: [] for t in AB_AG_CHAIN_TYPES}
        for chain_id, chain_data in structure_entry.chains.items():
            sabdab_annotation = chain_data.sabdab_annotation
            if sabdab_annotation:
                ab_ag_type_to_chain_id[sabdab_annotation].append(int(chain_id))
        # Per-interface maps
        ab_ag_type_to_chain_id_pair = {t: [] for t in AB_AG_CHAIN_PAIR_TYPES}
        for chain_id_pair in structure_entry.interfaces:
            chain_id_i, chain_id_j = chain_id_pair.split("_")
            chain_data_i = structure_entry.chains[chain_id_i]
            chain_data_j = structure_entry.chains[chain_id_j]
            sabdab_annotation_i = chain_data_i.sabdab_annotation
            sabdab_annotation_j = chain_data_j.sabdab_annotation
            if sabdab_annotation_i and sabdab_annotation_j:
                if (
                    sabdab_annotation_i,
                    sabdab_annotation_j,
                ) in ab_ag_type_to_chain_id_pair:
                    ab_ag_type_to_chain_id_pair[
                        (sabdab_annotation_i, sabdab_annotation_j)
                    ].append((int(chain_id_i), int(chain_id_j)))
                else:
                    ab_ag_type_to_chain_id_pair[
                        (sabdab_annotation_j, sabdab_annotation_i)
                    ].append((int(chain_id_j), int(chain_id_i)))

        atom_array = sample_data["atom_array"]
        token_starts_with_stop, _ = extract_starts_entities(atom_array)
        token_starts = token_starts_with_stop[:-1]
        token_chain_id = atom_array.chain_id[token_starts].astype(int)

        token_mask = sample_data["features"]["token_mask"]
        num_atoms_per_token = sample_data["features"]["num_atoms_per_token"]

        token_ab_ag_type = torch.zeros_like(
            torch.tensor(token_chain_id),
            dtype=torch.int32,
        )

        # Intra
        for ab_ag_type_int, ab_ag_type_str in enumerate(AB_AG_CHAIN_TYPES, start=1):
            mask = torch.tensor(
                np.isin(token_chain_id, ab_ag_type_to_chain_id[ab_ag_type_str]),
                dtype=torch.int32,
            )
            token_ab_ag_type += mask * ab_ag_type_int

        intra_ab_ag_type_atomized = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=token_ab_ag_type,
        ).to(torch.int32)

        features["intra_ab_ag_type_atomized"] = intra_ab_ag_type_atomized

        # Inter
        token_chain_id = torch.tensor(token_chain_id, dtype=torch.int32)
        chain_pair_labels = make_chain_pair_labels_padded(
            token_chain_id, AB_AG_CHAIN_PAIR_TYPES, ab_ag_type_to_chain_id_pair
        )
        token_pair_labels = chain_pair_labels[
            token_chain_id.unsqueeze(0), token_chain_id.unsqueeze(1)
        ]

        inter_ab_ag_type_atomized = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=token_pair_labels,
            token_dim=-2,
        )
        inter_ab_ag_type_atomized = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=inter_ab_ag_type_atomized.transpose(-1, -2),
            token_dim=-2,
        )
        inter_ab_ag_type_atomized = inter_ab_ag_type_atomized.transpose(-1, -2)
        features["inter_ab_ag_type_atomized"] = inter_ab_ag_type_atomized

        return features

    def create_all_features(
        self,
        pdb_id: str,
        preferred_chain_or_interface: str | None,
        return_atom_arrays: bool,
        return_crop_strategy: bool,
    ) -> dict:
        """Calls the parent create_all_features, and then adds features for homology
        similarity."""
        sample_data = super().create_all_features(
            pdb_id,
            preferred_chain_or_interface,
            return_atom_arrays=return_atom_arrays,
            return_crop_strategy=return_crop_strategy,
        )

        sample_data["features"]["ground_truth"].update(
            self.get_validation_homology_features(pdb_id, sample_data)
        )
        sample_data["features"]["ground_truth"].update(
            self.get_ab_ag_features(pdb_id, sample_data)
        )
        sample_data["features"]["atom_array"] = sample_data["atom_array"]

        # Remove atom arrays if they are not needed
        if not return_atom_arrays:
            del sample_data["atom_array"]
            del sample_data["atom_array_gt"]
            del sample_data["atom_array_cropped"]

        return sample_data
