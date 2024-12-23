import dataclasses
from collections import Counter
from enum import IntEnum

import pandas as pd
import torch

from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.base_af3 import (
    BaseAF3Dataset,
)
from openfold3.core.data.resources.residues import MoleculeType


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
            ["pdb_id", "preferred_chain_or_interface", "datapoint_probabilities"]
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
        self.sample_weights = dataset_config["custom"]["sample_weights"]

        # Datapoint cache
        self.create_datapoint_cache()

    def create_datapoint_cache(self) -> None:
        """Creates the datapoint_cache with chain/interface probabilities.

        Creates a Dataframe storing a flat list of chains and interfaces and
        corresponding datapoint probabilities. Used for mapping FROM the dataset_cache
        in the StochasticSamplerDataset and TO the dataset_cache in the getitem."""
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
        sample_data = self.create_all_features(
            pdb_id=datapoint["pdb_id"],
            preferred_chain_or_interface=datapoint["preferred_chain_or_interface"],
            return_atom_arrays=False,
        )
        return sample_data["features"]
