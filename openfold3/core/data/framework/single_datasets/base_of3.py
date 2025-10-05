import json
import logging
from abc import ABC
from pathlib import Path
from typing import Any

import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from openfold3.core.data.framework.single_datasets.abstract_single import (
    SingleDataset,
    register_dataset,
)
from openfold3.core.data.io.dataset_cache import read_datacache
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from openfold3.core.data.pipelines.featurization.loss_weights import set_loss_weights
from openfold3.core.data.pipelines.featurization.msa import featurize_msa_of3
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_target_gt_structure_of3,
)
from openfold3.core.data.pipelines.featurization.template import (
    featurize_template_structures_of3,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    get_reference_conformer_data_of3,
)
from openfold3.core.data.pipelines.sample_processing.msa import (
    process_msas_of3,
)
from openfold3.core.data.pipelines.sample_processing.structure import (
    process_target_structure_of3,
)
from openfold3.core.data.pipelines.sample_processing.template import (
    process_template_structures_of3,
)
from openfold3.core.data.primitives.permutation.mol_labels import (
    separate_cropped_and_gt,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.tokenization import add_token_positions

logger = logging.getLogger(__name__)


# TODO: update docstring with inputs
@register_dataset
class BaseOF3Dataset(SingleDataset, ABC):
    """Implements a general SingleDataset for handling inputs for OF3.

    The BaseOF3Dataset dataset
    - implements a set of general class methods for processing and featurizing inputs
    for AF3
    - assigns general class attributes, including the dataset_cache property

    As required by the SingleDataset class, child classes of BaseAF3Dataset must
    - implement the __getitem__ method
    - implement the datapoint_cache property and
    - decorate the class with the register_dataset decorator.

    In addition, child classes of BaseAF3Dataset must set whether cropping is performed
    by setting the self.apply_crop attribute.
    """

    # TODO: add typehint - currently causes circular import issues
    # dataset_config: DefaultDatasetConfigSection
    def __init__(self, dataset_config) -> None:
        """Initializes a BaseOF3Dataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__()
        self.name = dataset_config.name

        # Paths/IO
        self.target_structures_directory = (
            dataset_config.dataset_paths.target_structures_directory
        )
        self.target_structure_file_format = (
            dataset_config.dataset_paths.target_structure_file_format
        )
        self.alignments_directory = dataset_config.dataset_paths.alignments_directory
        self.alignment_db_directory = (
            dataset_config.dataset_paths.alignment_db_directory
        )
        if self.alignment_db_directory is not None:
            with open(self.alignment_db_directory / Path("alignment_db.index")) as f:
                self.alignment_index = json.load(f)
        else:
            self.alignment_index = None
        self.alignment_array_directory = (
            dataset_config.dataset_paths.alignment_array_directory
        )
        self.template_cache_directory = (
            dataset_config.dataset_paths.template_cache_directory
        )
        self.template_structures_directory = (
            dataset_config.dataset_paths.template_structures_directory
        )
        self.template_structure_array_directory = (
            dataset_config.dataset_paths.template_structure_array_directory
        )
        self.template_file_format = dataset_config.dataset_paths.template_file_format
        self.reference_molecule_directory = (
            dataset_config.dataset_paths.reference_molecule_directory
        )

        self.use_roda_monomer_format = (
            dataset_config.dataset_paths.use_roda_monomer_format
        )

        # Dataset/datapoint cache
        # TODO: rename dataset_cache_file to dataset_cache_path to signal that it can be
        # a directory or a file
        # TODO: potentially expose the LMDB database encoding types
        self.dataset_cache = read_datacache(
            dataset_config.dataset_paths.dataset_cache_file
        )
        self.datapoint_cache = {}

        # CCD - only used if template structures are not preprocessed
        if dataset_config.dataset_paths.template_structure_array_directory is not None:
            self.ccd = None
        else:
            self.ccd = pdbx.CIFFile.read(dataset_config.dataset_paths.ccd_file)

        # Dataset configuration
        # n_tokens can be set in the getitem method separately for each sample using
        # the output of create_target_structure_features
        self.apply_crop = None
        self.crop = {}
        self.loss = dataset_config.loss.model_dump()
        self.msa = dataset_config.msa
        self.template = dataset_config.template

        # Misc
        self.single_moltype = None
        self.debug_mode = dataset_config.debug_mode

    def __post_init__(self):
        if self.apply_crop is None:
            raise ValueError(
                "Attribute self.apply_crop must be set in the __init__ of"
                f"{self.get_class_name()}."
            )

    @log_runtime_memory(runtime_dict_key="runtime-create-structure-features")
    def create_structure_features(
        self,
        pdb_id: str,
        preferred_chain_or_interface: str | list[str, str] | None,
        return_full_atom_array: bool,
        return_crop_strategy: bool,
    ) -> tuple[dict, AtomArray | torch.Tensor]:
        """Creates the target structure features."""

        # Processed ground-truth structure with added annotations and masks
        atom_array_gt, crop_strategy, self.n_tokens = process_target_structure_of3(
            target_structures_directory=self.target_structures_directory,
            pdb_id=pdb_id,
            apply_crop=self.apply_crop,
            crop_config=self.crop,
            preferred_chain_or_interface=preferred_chain_or_interface,
            structure_format=self.target_structure_file_format,
            per_chain_metadata=self.dataset_cache.structure_data[pdb_id].chains,
            use_roda_monomer_format=self.use_roda_monomer_format,
        )

        # Processed reference conformers
        processed_reference_molecules = get_reference_conformer_data_of3(
            atom_array=atom_array_gt,
            per_chain_metadata=self.dataset_cache.structure_data[pdb_id].chains,
            reference_mol_metadata=self.dataset_cache.reference_molecule_data,
            reference_mol_dir=self.reference_molecule_directory,
        )

        if return_full_atom_array:
            atom_array_full = atom_array_gt.copy()

        # Apply crop and subset GT to only contain the atoms that are symmetry-related
        # to atoms in the crop and necessary for the permutation alignment
        atom_array_cropped, atom_array_gt = separate_cropped_and_gt(
            atom_array_gt=atom_array_gt,
            crop_strategy=crop_strategy,
            processed_ref_mol_list=processed_reference_molecules,
        )

        # Necessary positional indices for MSA and template processing
        add_token_positions(atom_array_cropped)

        # Compute target and ground-truth structure features
        target_structure_features = featurize_target_gt_structure_of3(
            atom_array=atom_array_cropped,
            atom_array_gt=atom_array_gt,
            n_tokens=self.n_tokens,
        )

        # Compute reference conformer features
        reference_conformer_features = featurize_reference_conformers_of3(
            processed_ref_mol_list=processed_reference_molecules
        )

        # Wrap up features
        # TODO: is there a reason to return atom_array_gt?
        target_structure_data = {
            "atom_array_gt": atom_array_gt,
            "atom_array_cropped": atom_array_cropped,
            "target_structure_features": target_structure_features,
            "reference_conformer_features": reference_conformer_features,
        }

        if return_full_atom_array:
            target_structure_data["atom_array"] = atom_array_full

        if return_crop_strategy:
            target_structure_data["crop_strategy"] = crop_strategy

        return target_structure_data

    @log_runtime_memory(runtime_dict_key="runtime-create-msa-features")
    def create_msa_features(self, pdb_id: str, atom_array: AtomArray) -> dict:
        """Creates the MSA features."""

        msa_array_collection = process_msas_of3(
            atom_array=atom_array,
            assembly_data=self.fetch_fields_for_chains(
                pdb_id=pdb_id,
                fields=["alignment_representative_id", "molecule_type"],
                defaults=[None, self.single_moltype],
            ),
            alignments_directory=self.alignments_directory,
            alignment_db_directory=self.alignment_db_directory,
            alignment_index=self.alignment_index,
            alignment_array_directory=self.alignment_array_directory,
            max_seq_counts=self.msa.max_seq_counts,
            aln_order=self.msa.aln_order,
            max_rows_paired=self.msa.max_rows_paired,
            min_chains_paired_partial=self.msa.min_chains_paired_partial,
            pairing_mask_keys=self.msa.pairing_mask_keys,
            moltypes=self.msa.moltypes,
            msas_to_pair=self.msa.msas_to_pair,
            use_roda_monomer_format=self.use_roda_monomer_format,
        )
        msa_features = featurize_msa_of3(
            atom_array=atom_array,
            msa_array_collection=msa_array_collection,
            max_rows=self.msa.max_rows,
            max_rows_paired=self.msa.max_rows_paired,
            n_tokens=self.n_tokens,
            subsample_with_bands=self.msa.subsample_with_bands,
        )

        return msa_features

    @log_runtime_memory(runtime_dict_key="runtime-create-template-features")
    def create_template_features(self, pdb_id: str, atom_array: AtomArray) -> dict:
        """Creates the template features."""

        template_slice_collection = process_template_structures_of3(
            atom_array=atom_array,
            n_templates=self.template.n_templates,
            take_top_k=self.template.take_top_k,
            template_cache_directory=self.template_cache_directory,
            assembly_data=self.fetch_fields_for_chains(
                pdb_id=pdb_id,
                fields=["alignment_representative_id", "template_ids"],
                defaults=[None, []],
            ),
            template_structures_directory=self.template_structures_directory,
            template_structure_array_directory=self.template_structure_array_directory,
            template_file_format=self.template_file_format,
            ccd=self.ccd,
            use_roda_monomer_format=self.use_roda_monomer_format,
        )

        template_features = featurize_template_structures_of3(
            template_slice_collection=template_slice_collection,
            n_templates=self.template.n_templates,
            n_tokens=self.n_tokens,
            min_bin=self.template.distogram.min_bin,
            max_bin=self.template.distogram.max_bin,
            n_bins=self.template.distogram.n_bins,
        )

        return template_features

    def create_loss_features(self, pdb_id: str) -> dict:
        """Creates the loss features."""

        loss_features = {}
        loss_features["loss_weights"] = set_loss_weights(
            self.loss,
            getattr(self.dataset_cache.structure_data[pdb_id], "resolution", None),
        )
        return loss_features

    @log_runtime_memory(runtime_dict_key="runtime-create-all-features")
    def create_all_features(
        self,
        pdb_id: str,
        preferred_chain_or_interface: str | list[str, str] | None,
        return_atom_arrays: bool,
        return_crop_strategy: bool,
    ) -> dict:
        """Creates all features for a single datapoint."""

        sample_data = {"features": {}}

        # Target & GT structure and conformer features
        target_structure_data = self.create_structure_features(
            pdb_id,
            preferred_chain_or_interface,
            return_atom_arrays,
            return_crop_strategy,
        )
        sample_data["features"].update(
            target_structure_data["target_structure_features"]
        )
        sample_data["features"].update(
            target_structure_data["reference_conformer_features"]
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

        # Loss switches
        loss_features = self.create_loss_features(pdb_id)
        sample_data["features"].update(loss_features)

        if return_atom_arrays:
            sample_data["atom_array"] = target_structure_data["atom_array"]
            sample_data["atom_array_gt"] = target_structure_data["atom_array_gt"]
            sample_data["atom_array_cropped"] = target_structure_data[
                "atom_array_cropped"
            ]
        if return_crop_strategy:
            sample_data["crop_strategy"] = target_structure_data["crop_strategy"]

        return sample_data

    def fetch_fields_for_chains(
        self, pdb_id: str, fields: list[str], defaults: list[Any]
    ) -> dict[str, Any]:
        """Fetches values for fields for all chains of a PDB entry in the cache.

        Requires the dataset cache to contain the structure_data field storing metadata
        per sample.

        Args:
            pdb_id (str):
                The PDB ID of the target structure.
            fields (list[str]):
                List of fields to fetch for all chains.
            defaults (list[Any]):
                List of default values for the fields if they are not found in the
                cache. Values for ALL chains are set to this value if the field is not
                found.

        Returns:
            dict[str, Any]:
                Dictionary containing the fetched values for the fields for all chains.
        """
        if len(fields) != len(defaults):
            raise ValueError("Fields and defaults must have the same length.")

        assembly_data = {}

        for chain_id, chain_data in self.dataset_cache.structure_data[
            pdb_id
        ].chains.items():
            assembly_data[chain_id] = {}
            for field, default in zip(fields, defaults):
                if hasattr(chain_data, field):
                    assembly_data[chain_id][field] = getattr(chain_data, field)
                else:
                    assembly_data[chain_id][field] = default

        return assembly_data

    def __getitem__(
        self, index: int
    ) -> dict[str : torch.Tensor | dict[str, torch.Tensor]]:
        """Getitem method to be implemented by child classes."""

        raise NotImplementedError("Missing __getitem__ method.")

    def __len__(self):
        return len(self.datapoint_cache)
