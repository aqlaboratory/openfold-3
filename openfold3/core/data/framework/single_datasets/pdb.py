import dataclasses
import json
import math
from collections import Counter
from enum import IntEnum
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from biotite.structure.io import pdbx

from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    SingleDataset,
    register_dataset,
)
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_ref_conformers_af3,
)
from openfold3.core.data.pipelines.featurization.loss_weights import set_loss_weights
from openfold3.core.data.pipelines.featurization.msa import featurize_msa_af3
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_target_gt_structure_af3,
)
from openfold3.core.data.pipelines.featurization.template import (
    featurize_templates_dummy_af3,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    get_reference_conformer_data_af3,
)
from openfold3.core.data.pipelines.sample_processing.msa import process_msas_cropped_af3
from openfold3.core.data.pipelines.sample_processing.structure import (
    process_target_structure_af3,
)
from openfold3.core.data.resources.residues import MoleculeType


class DatapointType(IntEnum):
    CHAIN = 0
    INTERFACE = 1


@dataclasses.dataclass(frozen=False)
class DatapointCollection:
    """Dataclass to tally chain/interface properties."""

    pdb_id: list[str]
    datapoint: list[Union[int, tuple[int, int]]]
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
        datapoint: Union[int, tuple[int, int]],
        moltypes: Union[str, tuple[str, str]],
        type: DatapointType,
        n_clust: int,
    ) -> None:
        """Append datapoint metadata to the tally.

        Args:
            pdb_id (str):
                PDB ID.
            datapoint (Union[int, tuple[int, int]]):
                Chain or interface ID.
            moltypes (Union[str, tuple[str, str]]):
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

    def count_moltypes(
        self, moltypes: Union[str, tuple[str, str]]
    ) -> tuple[int, int, int]:
        """Count the number of molecule types.

        Args:
            moltypes (Union[str, tuple[str, str]]):
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
                "datapoint": self.datapoint,
                "n_prot": self.n_prot,
                "n_nuc": self.n_nuc,
                "n_ligand": self.n_ligand,
                "type": self.type,
                "n_clust": self.n_clust,
            }
        )

    def create_datapoint_cache(self) -> pd.DataFrame:
        """Creates the datapoint_cache with chain/interface probabilities."""
        datapoint_type_weigth = {DatapointType.CHAIN: 0.5, DatapointType.INTERFACE: 1}
        a_prot = 3
        a_nuc = 3
        a_ligand = 1

        def calculate_datapoint_probability(row):
            """Algorithm 1. from Section 2.5.1 of the AF3 SI."""
            return (datapoint_type_weigth[row["type"]] / row["n_clust"]) * (
                a_prot * row["n_prot"]
                + a_nuc * row["n_nuc"]
                + a_ligand * row["n_ligand"]
            )

        self.metadata["weight"] = self.metadata.apply(
            calculate_datapoint_probability, axis=1
        )

        return self.metadata[["pdb_id", "datapoint", "weight"]]


@register_dataset
class WeightedPDBDataset(SingleDataset):
    """Implements a Dataset class for the Weighted PDB training dataset for AF3."""

    def __init__(self, dataset_config: dict) -> None:
        """Initializes a WeightedPDBDataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__()

        # Paths/IO
        self.target_structures_directory = dataset_config["dataset_paths"][
            "target_structures_directory"
        ]
        self.alignments_directory = dataset_config["dataset_paths"][
            "alignments_directory"
        ]
        self.alignment_db_directory = dataset_config["dataset_paths"][
            "alignment_db_directory"
        ]
        if self.alignment_db_directory is not None:
            with open(self.alignment_db_directory / Path("alignment_db.index")) as f:
                self.alignment_index = json.load(f)
        else:
            self.alignment_index = None
        self.template_cache_directory = dataset_config["dataset_paths"][
            "template_cache_directory"
        ]
        self.template_structures_directory = dataset_config["dataset_paths"][
            "template_structures_directory"
        ]
        self.reference_molecule_directory = dataset_config["dataset_paths"][
            "reference_molecule_directory"
        ]

        # Dataset/datapoint cache
        self.datapoint_cache = {}
        with open(dataset_config["dataset_paths"]["dataset_cache_file"]) as f:
            self.dataset_cache = json.load(f)
        self.create_datapoint_cache()
        self.datapoint_probabilities = self.datapoint_cache["weight"].to_numpy()

        # CCD
        self.ccd = pdbx.CIFFile.read(dataset_config["dataset_paths"]["ccd_file"])

        # Dataset configuration
        self.crop_weights = dataset_config["crop_weights"]
        self.token_budget = dataset_config["token_budget"]
        self.n_templates = dataset_config["n_templates"]
        self.loss_settings = dataset_config["loss"]

    def create_datapoint_cache(self) -> None:
        """Creates the datapoint_cache with chain/interface probabilities.

        Creates a Dataframe storing a flat list of chains and interfaces and
        correspoinding datapoint probabilities. Used for mapping FROM the dataset_cache
        in the StochasticSamplerDataset and TO the dataset_cache in the getitem."""
        datapoint_collection = DatapointCollection.create_empty()
        for entry, entry_data in self.dataset_cache["structure_data"].items():
            # Append chains
            _ = [
                datapoint_collection.append(
                    entry,
                    str(chain),
                    chain_data["molecule_type"],
                    DatapointType.CHAIN,
                    int(chain_data["cluster_size"]),
                )
                for chain, chain_data in entry_data["chains"].items()
            ]
            # Append interfaces
            _ = [
                datapoint_collection.append(
                    entry,
                    [str(chain) for chain in interface.split("_")],
                    [
                        entry_data["chains"][chain]["molecule_type"]
                        for chain in interface.split("_")
                    ],
                    DatapointType.INTERFACE,
                    int(cluster_size),
                )
                for interface, cluster_size in entry_data[
                    "interface_cluster_sizes"
                ].items()
            ]

        datapoint_collection.convert_to_dataframe()
        self.datapoint_cache = datapoint_collection.create_datapoint_cache()

    def __getitem__(
        self, index: int
    ) -> dict[str : Union[torch.Tensor, dict[str, torch.Tensor]]]:
        """Returns a single datapoint from the dataset."""

        # Get PDB ID from the datapoint cache and the preferred chain/interface
        datapoint = self.datapoint_cache.iloc[index]
        pdb_id = datapoint["pdb_id"]
        preferred_chain_or_interface = datapoint["datapoint"]
        features = {}

        # Target structure and duplicate-expanded GT structure features
        atom_array_cropped, atom_array_gt = process_target_structure_af3(
            target_structures_directory=self.target_structures_directory,
            pdb_id=pdb_id,
            crop_weights=self.crop_weights,
            token_budget=self.token_budget,
            preferred_chain_or_interface=preferred_chain_or_interface,
            ciftype=".bcif",
        )
        # NOTE that for now we avoid the need for permutation alignment by providing the
        # cropped atom array as the ground truth atom array
        # features.update(
        #     featurize_target_gt_structure_af3(
        #         atom_array_cropped, atom_array_gt, self.token_budget
        #     )
        # )
        features.update(
            featurize_target_gt_structure_af3(
                atom_array_cropped, atom_array_cropped, self.token_budget
            )
        )

        # MSA features
        msa_processed = process_msas_cropped_af3(
            alignments_directory=self.alignments_directory,
            alignment_db_directory=self.alignment_db_directory,
            alignment_index=self.alignment_index,
            atom_array=atom_array_cropped,
            data_cache_entry_chains=self.dataset_cache["structure_data"][pdb_id][
                "chains"
            ],
            max_seq_counts={
                "uniref90_hits": 10000,
                "uniprot": 50000,
                "bfd_uniclust_hits": math.inf,
                "bfd_uniref_hits": math.inf,
                "mgnify_hits": 5000,
                "rfam_hits": 10000,
                "rnacentral_hits": 10000,
                "nucleotide_collection_hits": 10000,
            },
            token_budget=self.token_budget,
            max_rows_paired=8191,
        )
        features.update(featurize_msa_af3(msa_processed))

        # Dummy template features
        features.update(
            featurize_templates_dummy_af3(self.n_templates, self.token_budget)
        )

        # Reference conformer features
        processed_reference_molecules = get_reference_conformer_data_af3(
            atom_array=atom_array_cropped,
            per_chain_metadata=self.dataset_cache["structure_data"][pdb_id]["chains"],
            reference_mol_metadata=self.dataset_cache["reference_molecule_data"],
            reference_mol_dir=self.reference_molecule_directory,
        )
        features.update(featurize_ref_conformers_af3(processed_reference_molecules))

        # Loss switches
        features["loss_weight"] = set_loss_weights(
            self.loss_settings,
            self.dataset_cache["structure_data"][pdb_id]["resolution"],
        )
        return features

    def __len__(self):
        return len(self.datapoint_cache)
