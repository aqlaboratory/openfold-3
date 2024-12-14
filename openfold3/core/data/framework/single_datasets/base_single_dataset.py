import dataclasses
import json
from collections import Counter
from enum import IntEnum
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    SingleDataset,
    register_dataset,
)
from openfold3.core.data.io.dataset_cache import read_datacache
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_ref_conformers_af3,
)
from openfold3.core.data.pipelines.featurization.loss_weights import set_loss_weights
from openfold3.core.data.pipelines.featurization.msa import featurize_msa_af3
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_target_gt_structure_af3,
)
from openfold3.core.data.pipelines.featurization.template import (
    featurize_template_structures_af3,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    get_ref_conformer_data_af3,
)
from openfold3.core.data.pipelines.sample_processing.msa import (
    process_msas_af3,
)
from openfold3.core.data.pipelines.sample_processing.structure import (
    process_target_structure_af3,
)
from openfold3.core.data.pipelines.sample_processing.template import (
    process_template_structures_af3,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.resources.residues import MoleculeType



class BaseSingleDataset(SingleDataset):
    """ A re-usable dataset class for that implements the same 
    preprocessing and featurization logicv across different 
    datasets
    """

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
        self.alignment_array_directory = dataset_config["dataset_paths"][
            "alignment_array_directory"
        ]
        self.template_cache_directory = dataset_config["dataset_paths"][
            "template_cache_directory"
        ]
        self.template_structures_directory = dataset_config["dataset_paths"][
            "template_structures_directory"
        ]
        self.template_file_format = dataset_config["dataset_paths"][
            "template_file_format"
        ]
        self.reference_molecule_directory = dataset_config["dataset_paths"][
            "reference_molecule_directory"
        ]

        # Dataset/datapoint cache
        
        self.dataset_cache = read_datacache(
            dataset_config["dataset_paths"]["dataset_cache_file"]
        )
        self.create_datapoint_cache()
        self.datapoint_probabilities = self.datapoint_cache["weight"].to_numpy()

        # CCD
        self.ccd = pdbx.CIFFile.read(dataset_config["dataset_paths"]["ccd_file"])

        # Dataset configuration
        self.crop_weights = dataset_config["crop_weights"]
        self.token_budget = dataset_config["token_budget"]
        self.loss_settings = dataset_config["loss"]
        self.msa = dataset_config["msa"]
        self.template = dataset_config["template"]

    def create_datapoint_cache(self) -> None:
        """
        Creates a cache of datapoints for the dataset.
        At its base, it's simply a list of sequence IDs, corresponding to 
        entries in the dataset cache. For other datasets, adds 
        sample-specific information such as the preferred chain or interface,
        and the weight of the datapoint.
        """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def __getitem__(
        self, index: int
    ) -> dict[str : Union[torch.Tensor, dict[str, torch.Tensor]]]:
        """Returns a single datapoint from the dataset.

        Note: The data pipeline is modularized at the getitem level to enable
        subclassing for profiling without code duplication. See
        logging_datasets.py for an example."""

        # Get PDB ID from the datapoint cache and the preferred chain/interface
        datapoint = self.datapoint_cache.iloc[index]
        pdb_id = datapoint["pdb_id"]
        preferred_chain_or_interface = datapoint["datapoint"] if "datapoint" in datapoint else None
        sample_data = self.create_all_features(
            pdb_id=pdb_id,
            preferred_chain_or_interface=preferred_chain_or_interface,
            return_atom_arrays=False,
        )
        return sample_data["features"]

    @log_runtime_memory(runtime_dict_key="runtime-create-target-structure-features")
    def create_target_structure_features(
        self, pdb_id: str, preferred_chain_or_interface: str, return_atom_arrays: bool
    ) -> tuple[dict, AtomArray | torch.Tensor]:
        """Creates the target structure features."""

        # Target structure and duplicate-expanded GT structure features
        target_structure_data = process_target_structure_af3(
            target_structures_directory=self.target_structures_directory,
            pdb_id=pdb_id,
            crop_weights=self.crop_weights,
            token_budget=self.token_budget,
            preferred_chain_or_interface=preferred_chain_or_interface,
            structure_format="pkl",
            return_full_atom_array=return_atom_arrays,
        )
        # NOTE that for now we avoid the need for permutation alignment by providing the
        # cropped atom array as the ground truth atom array
        # target_structure_features.update(
        #     featurize_target_gt_structure_af3(
        #         target_structure_data["atom_array_cropped"],
        #         target_structure_data["atom_array_gt"],
        #         self.token_budget
        #     )
        # )
        target_structure_features = featurize_target_gt_structure_af3(
            target_structure_data["atom_array_cropped"],
            target_structure_data["atom_array_cropped"],
            self.token_budget,
        )
        target_structure_data["target_structure_features"] = target_structure_features
        return target_structure_data

    @log_runtime_memory(runtime_dict_key="runtime-create-msa-features")
    def create_msa_features(self, pdb_id: str, atom_array_cropped: AtomArray) -> dict:
        """Creates the MSA features."""

        msa_array_collection = process_msas_af3(
            pdb_id=pdb_id,
            atom_array=atom_array_cropped,
            dataset_cache=self.dataset_cache,
            alignments_directory=self.alignments_directory,
            alignment_db_directory=self.alignment_db_directory,
            alignment_index=self.alignment_index,
            alignment_array_directory=self.alignment_array_directory,
            max_seq_counts=self.msa.max_seq_counts,
            aln_order=self.msa.aln_order,
            max_rows_paired=self.msa.max_rows_paired,
        )
        msa_features = featurize_msa_af3(
            atom_array=atom_array_cropped,
            msa_array_collection=msa_array_collection,
            max_rows=self.msa.max_rows,
            max_rows_paired=self.msa.max_rows_paired,
            token_budget=self.token_budget,
            subsample_with_bands=self.msa.subsample_with_bands,
        )

        return msa_features

    @log_runtime_memory(runtime_dict_key="runtime-create-template-features")
    def create_template_features(
        self, pdb_id: str, atom_array_cropped: AtomArray
    ) -> dict:
        """Creates the template features."""

        template_slice_collection = process_template_structures_af3(
            atom_array=atom_array_cropped,
            n_templates=self.template.n_templates,
            take_top_k=False,
            template_cache_directory=self.template_cache_directory,
            dataset_cache=self.dataset_cache,
            pdb_id=pdb_id,
            template_structures_directory=self.template_structures_directory,
            template_file_format=self.template_file_format,
            ccd=self.ccd,
        )

        template_features = featurize_template_structures_af3(
            template_slice_collection=template_slice_collection,
            n_templates=self.template.n_templates,
            token_budget=self.token_budget,
            min_bin=self.template.distogram.min_bin,
            max_bin=self.template.distogram.max_bin,
            n_bins=self.template.distogram.n_bins,
        )

        return template_features

    @log_runtime_memory(runtime_dict_key="runtime-create-ref-conformer-features")
    def create_ref_conformer_features(
        self, pdb_id: str, atom_array_cropped: AtomArray
    ) -> dict:
        """Creates the reference conformer features."""

        processed_reference_molecules = get_ref_conformer_data_af3(
            atom_array=atom_array_cropped,
            per_chain_metadata=self.dataset_cache.structure_data[pdb_id].chains,
            reference_mol_metadata=self.dataset_cache.reference_molecule_data,
            reference_mol_dir=self.reference_molecule_directory,
        )
        reference_conformer_features = featurize_ref_conformers_af3(
            processed_reference_molecules
        )

        return reference_conformer_features

    def create_loss_features(self, pdb_id: str) -> dict:
        """Creates the loss features."""

        loss_features = {}
        loss_features["loss_weights"] = set_loss_weights(
            self.loss_settings,
            self.dataset_cache.structure_data[pdb_id].resolution,
        )
        return loss_features

    @log_runtime_memory(runtime_dict_key="runtime-create-all-features")
    def create_all_features(
        self,
        pdb_id: str,
        preferred_chain_or_interface: str,
        return_atom_arrays: bool,
    ) -> dict:
        """Creates all features for a single datapoint."""

        sample_data = {"features": {}}

        # Target structure features
        target_structure_data = self.create_target_structure_features(
            pdb_id, preferred_chain_or_interface, return_atom_arrays
        )
        sample_data["features"].update(
            target_structure_data["target_structure_features"]
        )

        # MSA features
        msa_features = self.create_msa_features(
            pdb_id,
            target_structure_data["atom_array_cropped"],
        )
        sample_data["features"].update(msa_features)

        # Template features
        template_features = self.create_template_features(
            pdb_id, target_structure_data["atom_array_cropped"]
        )
        sample_data["features"].update(template_features)

        # Reference conformer features
        reference_conformer_features = self.create_ref_conformer_features(
            pdb_id, target_structure_data["atom_array_cropped"]
        )
        sample_data["features"].update(reference_conformer_features)

        # Loss switches
        loss_features = self.create_loss_features(pdb_id)
        sample_data["features"].update(loss_features)

        if return_atom_arrays:
            sample_data["atom_array"] = target_structure_data["atom_array"]
            sample_data["atom_array_gt"] = target_structure_data["atom_array_gt"]
            sample_data["atom_array_cropped"] = target_structure_data[
                "atom_array_cropped"
            ]

        return sample_data

    def __len__(self):
        return len(self.datapoint_cache)
