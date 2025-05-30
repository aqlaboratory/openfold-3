"""
Inference class template for first inference pipeline prototype.
"""

import itertools
import logging
import math
from typing import Optional

import pandas as pd
import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from torch.utils.data import Dataset

from openfold3.core.data.framework.inference_query_format import (
    Query,
)
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
from openfold3.core.data.primitives.structure.tokenization import (
    get_token_count,
)

logger = logging.getLogger(__name__)


# TODO: Replace these with actual appropriate implementations
def atom_array_from_query(*args, **kwargs) -> AtomArray: ...


def preprocess_atom_array(*args, **kwargs) -> AtomArray: ...


def get_reference_conformer_data_inference_af3(*args, **kwargs) -> list: ...


def process_msas_inference_af3(*args, **kwargs): ...


def process_template_structures_inference_af3(*args, **kwargs): ...


def do_seeding(seed: int): ...


# NOTE: This is not subclassing SingleDataset for now and has no support for dataset
# registries
# TODO: Maybe register dataset?
@register_dataset
class InferenceDataset(Dataset):
    """Dataset class for running inference on a set of queries."""

    # TODO: Can accept a dataset_config here if we want
    def __init__(self, dataset_config, world_size: Optional[int] = None) -> None:
        """Initializes the InferenceDataset."""
        super().__init__()

        self.query_set = dataset_config.query_set

        # TODO: Any seeding configuration like in the old InferenceDataset could go here
        ...
        self.seeds: list = dataset_config.seeds
        self.world_size = world_size

        # TODO: Any other settings, e.g. from dataset_config, could go here
        ...

        self.query_cache = self.query_set.queries

        # Expose for notation convenience (not actually used for now)
        self._msa_directory_path = self.query_set.msa_directory_path

        # Parse CCD
        if self.query_set.ccd_file_path is not None:
            logger.debug("Parsing CCD file.")
            self.ccd = pdbx.CIFFile.read(self.query_set.ccd_file_path)
            logger.debug("Done parsing CCD file.")

        # Create individual datapoint cache (allows rerunning the same query with
        # different seeds)
        self.create_datapoint_cache()

    # TODO: Will pair datapoints with seeds and handle any required sorting (e.g. by
    # token_count)
    def create_datapoint_cache(self) -> None:
        qids = self.query_cache.keys()
        qid_values, seed_values = zip(
            *[(q, s) for q, s in itertools.product(qids, self.seeds)]
        )

        # To avoid the default DistributedSampler behavior of repeating samples
        # to match the world size, artificially inflate the dataset and flag the
        # repeated samples so that they are ignored in the metrics.
        repeated_samples = [False] * len(qid_values)
        if self.world_size is not None:
            extra_samples = (
                math.ceil(len(qid_values) / self.world_size) * self.world_size
            ) - len(qid_values)
            qid_values += qid_values[:extra_samples]
            seed_values += seed_values[:extra_samples]
            repeated_samples += [True] * extra_samples

        self.datapoint_cache = pd.DataFrame(
            {
                "query_id": qid_values,
                "seed": seed_values,
                "repeated_sample": repeated_samples,
            }
        )

    def get_atom_array(self, query: Query) -> AtomArray:
        """Creates a preprocessed AtomArray from the query."""
        # Creates a vanilla AtomArray without any additional custom IDs
        raw_atom_array = atom_array_from_query(query)

        # Preprocess the raw AtomArray and add required IDs (e.g. do tokenization, add
        # component_id, token_position, ...)
        atom_array = preprocess_atom_array(raw_atom_array)

        return atom_array

    def create_structure_features(
        self,
        query: Query,
        atom_array: AtomArray,
    ) -> dict[str, torch.Tensor]:
        """Creates the target structure features."""

        # TODO: Set up the RDKit mols and conformers for everything in the query
        processed_reference_molecules = get_reference_conformer_data_inference_af3(
            atom_array=atom_array, query_chains=query.chains
        )

        # TODO: Elevate token_dim_index_map to an importable variable and call
        # featurize_structure_af3 directly here
        target_structure_features = featurize_target_gt_structure_af3(
            atom_array=atom_array,
            atom_array_gt=None,
            n_tokens=self.n_tokens,
        )

        # Compute reference conformer features
        reference_conformer_features = featurize_reference_conformers_af3(
            processed_ref_mol_list=processed_reference_molecules
        )

        # Wrap up features
        structure_features = {
            "target_structure_features": target_structure_features,
            "reference_conformer_features": reference_conformer_features,
        }

        return structure_features

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
            max_seq_counts=self.msa.max_seq_counts.counts,
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

    def create_all_features(
        self,
        query: Query,
    ) -> dict:
        """Creates all features for a single datapoint."""

        features = {}

        # Create initial AtomArray from query entry
        preprocessed_atom_array = self.get_atom_array(query)
        self.n_tokens = get_token_count(preprocessed_atom_array)

        # TODO: At some point, think about a cleaner way to pass this through the model
        # runner than as a pseudo-feature
        features["atom_array"] = preprocessed_atom_array

        # Target structure and conformer features
        structure_features = self.create_structure_features(
            query=query,
            atom_array=preprocessed_atom_array,
        )
        features.update(structure_features)

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
