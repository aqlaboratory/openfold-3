import copy
import dataclasses
import logging
import random
import traceback
from collections import Counter
from enum import IntEnum

import pandas as pd
import torch

from openfold3.core.data.framework.data_module import openfold_batch_collator
from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.base_af3 import (
    BaseAF3Dataset,
)
from openfold3.core.data.pipelines.featurization.loss_weights import (
    set_loss_weights_for_disordered_set,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
)
from openfold3.core.utils.permutation_alignment import (
    multi_chain_permutation_alignment,
    naive_alignment,
)

logger = logging.getLogger(__name__)


DEBUG_PDB_BLACKLIST = ["3e81", "6fg3", "6dra", "6dr2", "6dqj", "7lx0"]


# TODO: Remove debug logic
def is_invalid_feature_dict(features: dict) -> bool:
    """
    Validate the feature dictionary for a single datapoint.
    Do not fail early to log all potential errors.

    Args:
        features (dict):
            Feature dictionary for a single datapoint

    Returns:
        skip (bool):
            Whether the feature dictionary is invalid and should be skipped
    """
    skip = False
    pdb_id = features["pdb_id"]

    # Check that the sum of the number of atoms per token is equal to the
    # number of atoms in the reference conformer
    num_atoms_sum = torch.max(torch.sum(features["num_atoms_per_token"], dim=-1)).int()
    num_ref_atoms = features["ref_pos"].shape[-2]
    if num_atoms_sum != num_ref_atoms:
        logger.warning(
            f"Size mismatch {pdb_id} with {num_atoms_sum} vs {num_ref_atoms} atoms"
        )
        skip = True

    # Check that for overlapping token indices, the total number of
    # atoms in the ground truth is equal to the number of atoms in the
    # reference conformer
    gt_token_ids_atomized = broadcast_token_feat_to_atoms(
        token_mask=features["ground_truth"]["token_mask"].bool(),
        num_atoms_per_token=features["ground_truth"]["num_atoms_per_token"],
        token_feat=features["ground_truth"]["token_index"],
    )
    atom_selection_mask = torch.isin(gt_token_ids_atomized, features["token_index"])
    gt_atom_indices = torch.nonzero(atom_selection_mask, as_tuple=True)[0]
    if gt_atom_indices.shape[0] != num_ref_atoms:
        logger.warning(
            f"Size mismatch between ground truth {gt_atom_indices.shape[0]} atoms vs "
            f"crop {num_ref_atoms} atoms for the same token indices."
        )
        skip = True

    # Check that the crop has some resolved atoms
    if not features["ground_truth"]["atom_resolved_mask"].any():
        logger.warning(f"Skipping {pdb_id}: no resolved atoms")
        skip = True

    # Check that the ligand and atomized masks are consistent
    if (features["is_ligand"] & ~features["is_atomized"]).any():
        logger.warning(f"Skipping {pdb_id}: contains non-atomized ligands")
        skip = True

    # Check that the number of tokens per atom is less than the maximum expected
    if (features["num_atoms_per_token"] > 23).any():
        logger.warning(
            f"Skipping {pdb_id}: token contains number of atoms > max expected (23)"
        )
        skip = True

    # Check that all input features are finite
    for k, v in features.items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            logger.warning(f"Non-finite values in {pdb_id} for {k}")
            skip = True
        if isinstance(v, dict):
            for i, j in v.items():
                if isinstance(j, torch.Tensor) and not torch.isfinite(j).all():
                    logger.warning(f"Non-finite values in {pdb_id} for {i}")
                    skip = True

    # Run the permutation alignment to skip over samples that may fail in the model
    # This could throw an exception that is handled in the __getitem__
    feats_perm = openfold_batch_collator([copy.deepcopy(features)])
    multi_chain_permutation_alignment(
        batch=feats_perm,
        atom_positions_predicted=torch.randn_like(feats_perm["ref_pos"]),
    )
    naive_alignment(
        batch=feats_perm,
        atom_positions_predicted=torch.randn_like(feats_perm["ref_pos"]),
    )

    return skip


class DatapointType(IntEnum):
    CHAIN = 0
    INTERFACE = 1


@dataclasses.dataclass(frozen=False)
class DatapointCollection:
    """Dataclass to tally chain/interface properties."""

    pdb_id: list[str]
    datapoint: list[str | tuple[str, str]]
    n_prot: list[int]
    n_nuc: list[int]
    n_ligand: list[int]
    type: list[str]
    n_clust: list[int]
    metadata = pd.DataFrame()

    @classmethod
    def create_empty(cls):
        """Create an empty instance of the dataclass."""
        return cls(
            pdb_id=[],
            datapoint=[],
            n_prot=[],
            n_nuc=[],
            n_ligand=[],
            type=[],
            n_clust=[],
        )

    def append(
        self,
        pdb_id: str,
        datapoint: str | tuple[str, str],
        moltypes: str | tuple[str, str],
        type: DatapointType,
        n_clust: int,
    ) -> None:
        """Append datapoint metadata to the tally.

        Args:
            pdb_id (str):
                PDB ID.
            datapoint (int | tuple[int, int]):
                Chain or interface ID.
            moltypes (str | tuple[str, str]):
                Molecule types in the datapoint.
            type (DatapointType):
                Datapoint type. One of chain or interface.
            n_clust (int):
                Size of the cluster the datapoint belongs to.
        """
        self.pdb_id.append(pdb_id)
        self.datapoint.append(datapoint)
        n_prot, n_nuc, n_ligand = self.count_moltypes(moltypes)
        self.n_prot.append(n_prot)
        self.n_nuc.append(n_nuc)
        self.n_ligand.append(n_ligand)
        self.type.append(type)
        self.n_clust.append(n_clust)

    def count_moltypes(self, moltypes: str | tuple[str, str]) -> tuple[int, int, int]:
        """Count the number of molecule types.

        Args:
            moltypes (str | tuple[str, str]):
                Molecule type of the chain or types of the interface datapoint.

        Returns:
            tuple[int, int, int]:
                Number of protein, nucleic acid and ligand molecules
        """
        moltypes = (
            [MoleculeType[moltypes]]
            if isinstance(moltypes, str)
            else [MoleculeType[m] for m in moltypes]
        )
        moltype_count = Counter(moltypes)
        return (
            moltype_count.get(MoleculeType.PROTEIN, 0),
            moltype_count.get(MoleculeType.RNA, 0)
            + moltype_count.get(MoleculeType.DNA, 0),
            moltype_count.get(MoleculeType.LIGAND, 0),
        )

    def convert_to_dataframe(self) -> None:
        """Internally convert the tallies to a DataFrame."""
        self.metadata = pd.DataFrame(
            {
                "pdb_id": self.pdb_id,
                "preferred_chain_or_interface": self.datapoint,
                "n_prot": self.n_prot,
                "n_nuc": self.n_nuc,
                "n_ligand": self.n_ligand,
                "type": self.type,
                "n_clust": self.n_clust,
            }
        )

    def create_datapoint_cache(self, sample_weights) -> pd.DataFrame:
        """Creates the datapoint_cache with chain/interface probabilities."""
        datapoint_type_weight = {
            DatapointType.CHAIN: sample_weights["w_chain"],
            DatapointType.INTERFACE: sample_weights["w_interface"],
        }

        def calculate_datapoint_probability(row):
            """Algorithm 1. from Section 2.5.1 of the AF3 SI."""
            return (datapoint_type_weight[row["type"]] / row["n_clust"]) * (
                sample_weights["a_prot"] * row["n_prot"]
                + sample_weights["a_nuc"] * row["n_nuc"]
                + sample_weights["a_ligand"] * row["n_ligand"]
            )

        self.metadata["datapoint_probabilities"] = self.metadata.apply(
            calculate_datapoint_probability, axis=1
        )

        return self.metadata[
            [
                "pdb_id",
                "preferred_chain_or_interface",
                "datapoint_probabilities",
                "n_clust",
            ]
        ]


@register_dataset
class WeightedPDBDataset(BaseAF3Dataset):
    """Implements a Dataset class for the Weighted PDB training dataset for AF3."""

    def __init__(self, dataset_config: dict) -> None:
        """Initializes a WeightedPDBDataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__(dataset_config)

        # Dataset configuration
        self.apply_crop = True
        self.crop = dataset_config["custom"]["crop"]
        self.sample_weights = dataset_config["custom"]["sample_weights"]

        # Datapoint cache
        self.create_datapoint_cache()

    def create_datapoint_cache(self) -> None:
        """Creates the datapoint_cache with chain/interface probabilities.

        Creates a Dataframe storing a flat list of chains and interfaces and
        corresponding datapoint probabilities. Used for mapping FROM the dataset_cache
        in the SamplerDataset and TO the dataset_cache in the getitem."""
        datapoint_collection = DatapointCollection.create_empty()
        for entry, entry_data in self.dataset_cache.structure_data.items():
            # Append chains
            for chain, chain_data in entry_data.chains.items():
                datapoint_collection.append(
                    entry,
                    str(chain),
                    chain_data.molecule_type,
                    DatapointType.CHAIN,
                    int(chain_data.cluster_size),
                )

            # Append interfaces
            for interface_id, cluster_data in entry_data.interfaces.items():
                interface_chains = interface_id.split("_")
                cluster_size = int(cluster_data.cluster_size)
                chain_moltypes = [
                    entry_data.chains[chain].molecule_type for chain in interface_chains
                ]

                datapoint_collection.append(
                    entry,
                    interface_chains,
                    chain_moltypes,
                    DatapointType.INTERFACE,
                    cluster_size,
                )

        datapoint_collection.convert_to_dataframe()
        self.datapoint_cache = datapoint_collection.create_datapoint_cache(
            self.sample_weights
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
        preferred_chain_or_interface = datapoint["preferred_chain_or_interface"]

        # TODO: Remove debug logic
        if not self.debug_mode:
            sample_data = self.create_all_features(
                pdb_id=pdb_id,
                preferred_chain_or_interface=preferred_chain_or_interface,
                return_atom_arrays=False,
                return_crop_strategy=False,
            )
            features = sample_data["features"]
            features["pdb_id"] = pdb_id
            return features
        else:
            try:
                if pdb_id in DEBUG_PDB_BLACKLIST:
                    logger.warning(f"Skipping blacklisted pdb id {pdb_id}")
                    index = random.randint(0, len(self) - 1)
                    return self.__getitem__(index)

                sample_data = self.create_all_features(
                    pdb_id=pdb_id,
                    preferred_chain_or_interface=preferred_chain_or_interface,
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
                dataset_name = self.get_class_name()
                logger.warning(
                    "-" * 40
                    + "\n"
                    + f"Failed to process {dataset_name} entry {pdb_id}: {str(e)}\n"
                    + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                    + "-" * 40
                )
                index = random.randint(0, len(self) - 1)
                return self.__getitem__(index)


@register_dataset
class DisorderedPDBDataset(WeightedPDBDataset):
    """Implements a Dataset class for the Disordered PDB training dataset for AF3."""

    def __init__(self, dataset_config: dict) -> None:
        """Initializes a DisorderedPDBDataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__(dataset_config)
        self.custom_settings = dataset_config["custom"]

    def create_loss_features(self, pdb_id: str) -> dict:
        """Creates the loss features for the disordered PDB set."""

        loss_features = {}
        loss_features["loss_weights"] = set_loss_weights_for_disordered_set(
            self.loss,
            self.dataset_cache.structure_data[pdb_id].resolution,
            self.custom_settings,
        )
        return loss_features
