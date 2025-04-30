"""
Inference class template for first inference pipeline prototype.
"""

import logging

import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from torch.utils.data import Dataset

from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_af3,
)
from openfold3.core.data.pipelines.featurization.msa import featurize_msa_af3
from openfold3.core.data.pipelines.featurization.structure import (
    featurize_target_gt_structure_af3,
)
from openfold3.core.data.pipelines.featurization.template import (
    featurize_template_structures_af3,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.tokenization import (
    get_token_count,
)
from openfold3.projects.af3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
    Query,
)

logger = logging.getLogger(__name__)


# TODO: Replace these with actual appropriate implementations
def atom_array_from_query(*args, **kwargs) -> AtomArray: ...


def preprocess_atom_array(*args, **kwargs) -> AtomArray: ...


def get_reference_conformer_data_inference_af3(*args, **kwargs) -> list: ...


def process_msas_inference_af3(*args, **kwargs): ...


def process_template_structures_inference_af3(*args, **kwargs): ...


def do_seeding(seed: int): ...


# TODO: update docstring with inputs
@register_dataset
class InferenceDataset(Dataset):
    """Dataset class for running inference on a set of queries."""

    # TODO: Can accept a dataset_config here if we want
    def __init__(self, query_set: InferenceQuerySet) -> None:
        """Initializes the InferenceDataset.

        Args:
            dataset_config (dict):
                Input config. See openfold3/examples/pdb_sample_dataset_config.yml for
                an example.
        """
        super().__init__()

        self._query_set = query_set

        # TODO: Any seeding configuration like in the old InferenceDataset could go here
        ...

        # TODO: Any other settings, e.g. from dataset_config, could go here
        ...

        # NOTE: We can rename this to dataset_cache for consistency if we want, then we
        # would be allowed to inherit from SingleDataset
        self.query_cache = query_set.queries

        # Expose for notation convenience
        self._msa_directory_path = query_set.msa_directory_path

        # Parse CCD
        if query_set.ccd_file_path is not None:
            logger.debug("Parsing CCD file.")
            self.ccd = pdbx.CIFFile.read(query_set.ccd_file_path)
            logger.debug("Done parsing CCD file.")

        # Create individual datapoint cache (allows rerunning the same query with
        # different seeds)
        self.create_datapoint_cache()

    # TODO: Will pair datapoints with seeds and handle any required sorting (e.g. by
    # token_count)
    def create_datapoint_cache(self) -> None: ...

    @log_runtime_memory(runtime_dict_key="runtime-create-structure-features")
    def create_structure_features(
        self,
        query: Query,
        atom_array: AtomArray,
    ) -> tuple[dict, AtomArray | torch.Tensor]:
        """Creates the target structure features."""

        # Preprocess the raw AtomArray and add required IDs (e.g. do tokenization, add
        # component_id, token_position, ...)
        atom_array = preprocess_atom_array(
            atom_array=atom_array,
        )
        self.n_tokens = get_token_count(atom_array)

        # Set up the RDKit mols and conformers for everything in the query
        processed_reference_molecules = get_reference_conformer_data_inference_af3(
            atom_array=atom_array, query_data=query
        )

        # TODO: Refactor this function to make the GT optional, or alternatively
        # refactor this function to enable using single functions for both GT and pred
        # separately that achieve the same results (I believe this wrapper only adds
        # token_dim_index_map but padding is done automatically in the collator anyways,
        # so we could access the underlying primitives directly)
        target_structure_features = featurize_target_gt_structure_af3(
            atom_array=atom_array,
            atom_array_gt=None,
            n_tokens=self.n_tokens,
        )

        # NOTE: Conformer generation could be split into a create_conformer_features
        # function here. This is the separation we originally had in mind, but in
        # training we had a cross-dependency between structure and conformer features
        # because of some symmetry-expansion logic related to the permutation alignment.

        # Compute reference conformer features
        reference_conformer_features = featurize_reference_conformers_af3(
            processed_ref_mol_list=processed_reference_molecules
        )

        # Wrap up features and also return AtomArray
        # NOTE: It might be a bit cleaner to return the AtomArray separately to make it
        # more explicit that it is not a traditional feature, but then the function
        # signature "create_structure_features" is a bit misleading because every other
        # analogously named function in this class only returns a feature dict.
        structure_features = {
            "atom_array": atom_array,
            "target_structure_features": target_structure_features,
            "reference_conformer_features": reference_conformer_features,
        }

        return structure_features

    @log_runtime_memory(runtime_dict_key="runtime-create-msa-features")
    def create_msa_features(self, atom_array: AtomArray, *args, **kwargs) -> dict:
        """Creates the MSA features."""
        # NOTE: Only here for avoiding invalid syntax highlighting, but this will likely
        # not be an argument in the future
        # TODO: add the updated MSA pipeline here
        pdb_id = None

        # TODO: Implement a custom inference-adjusted function that returns
        # msa_array_collection
        msa_array_collection = process_msas_inference_af3(
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
            max_seq_counts=self.msa.max_seq_counts.model_dump(),
            aln_order=self.msa.aln_order,
            max_rows_paired=self.msa.max_rows_paired,
            min_chains_paired_partial=self.msa.min_chains_paired_partial,
            pairing_mask_keys=self.msa.pairing_mask_keys,
            moltypes=self.msa.moltypes,
        )
        msa_features = featurize_msa_af3(
            atom_array=atom_array,
            msa_array_collection=msa_array_collection,
            max_rows=self.msa.max_rows,
            max_rows_paired=self.msa.max_rows_paired,
            n_tokens=self.n_tokens,
            subsample_with_bands=self.msa.subsample_with_bands,
        )

        return msa_features

    @log_runtime_memory(runtime_dict_key="runtime-create-template-features")
    def create_template_features(self, atom_array: AtomArray, *args, **kwargs) -> dict:
        """Creates the template features."""
        # NOTE: Only here for avoiding invalid syntax highlighting, but this will likely
        # not be an argument in the future
        # TODO: add the updated template pipeline here
        pdb_id = None

        # TODO: Implement a custom inference-adjusted function that returns
        # template_slice_collection
        template_slice_collection = process_template_structures_inference_af3(
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
        )

        template_features = featurize_template_structures_af3(
            template_slice_collection=template_slice_collection,
            n_templates=self.template.n_templates,
            n_tokens=self.n_tokens,
            min_bin=self.template.distogram.min_bin,
            max_bin=self.template.distogram.max_bin,
            n_bins=self.template.distogram.n_bins,
        )

        return template_features

    @log_runtime_memory(runtime_dict_key="runtime-create-all-features")
    def create_all_features(
        self,
        query: Query,
    ) -> dict:
        """Creates all features for a single datapoint."""

        features = {}

        # Create initial AtomArray from query entry
        # NOTE: We could elevate this outside of the feature creation if we want, or we
        # could also make it part of create_structure_features. It initially felt like
        # this deserves an elevated role because the AtomArray is a central object to
        # all functions, but then what we actually use later is the preprocessed
        # AtomArray coming out of create_structure_features anyways.
        raw_atom_array = atom_array_from_query(query)

        # Target structure and conformer features
        structure_features = self.create_structure_features(
            query=query,
            atom_array=raw_atom_array,
        )
        features.update(structure_features)

        # Obtain the preprocessed AtomArray with all the additional necessary IDs
        # NOTE: Would be a bit cleaner to return the AtomArray as an elevated key in
        # sample_data outside of the regular features, but not sure if this would make
        # anything downstream more difficult. Although there is a separation between
        # features and any other sample data in the old inference draft, it ultimately
        # puts all the required information directly into the feature_dict in
        # __getitem__.
        preprocessed_atom_array = structure_features["atom_array"]

        # MSA features
        msa_features = self.create_msa_features(
            preprocessed_atom_array,
            ...,
        )
        features.update(msa_features)

        # Template features
        template_features = self.create_template_features(preprocessed_atom_array, ...)
        features.update(template_features)

        return features

    def __getitem__(
        self, index: int
    ) -> dict[str : torch.Tensor | dict[str, torch.Tensor]]:
        # Get query ID and seed information
        datapoint = self.datapoint_cache.iloc[index]
        query_id = datapoint["query_id"]
        query = self.query_cache[query_id]
        seed = datapoint["seed"]

        # TODO: Any particular seeding code could go here, e.g. seed_everything(seed).
        # See old inference dataset draft.
        do_seeding(seed)  # Added this to avoid Ruff complaint

        # TODO: Could wrap this in try/except
        features = self.create_all_features(query)

        return features

    def __len__(self):
        return len(self.datapoint_cache)
