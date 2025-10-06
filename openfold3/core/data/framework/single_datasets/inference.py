"""
Inference class template for first inference pipeline prototype.
"""

import itertools
import logging
import traceback
from typing import Optional

import pandas as pd
import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from torch.utils.data import Dataset

from openfold3.core.config.msa_pipeline_configs import MsaSampleProcessorInputInference
from openfold3.core.data.framework.single_datasets.abstract_single import (
    register_dataset,
)
from openfold3.core.data.framework.single_datasets.dataset_utils import (
    pad_to_world_size,
)
from openfold3.core.data.pipelines.featurization.conformer import (
    featurize_reference_conformers_of3,
)
from openfold3.core.data.pipelines.featurization.msa import (
    MsaFeaturizerOF3,
    MsaFeaturizerOF3Config,
)
from openfold3.core.data.pipelines.featurization.structure import (
    TOKEN_DIM_INDEX_MAP,
    featurize_structure_of3,
)
from openfold3.core.data.pipelines.featurization.template import (
    featurize_template_structures_of3,
)
from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.pipelines.sample_processing.msa import (
    MsaSampleProcessorInference,
)
from openfold3.core.data.pipelines.sample_processing.template import (
    process_template_structures_of3,
)
from openfold3.core.data.primitives.structure.component import BiotiteCCDWrapper
from openfold3.core.data.primitives.structure.query import (
    StructureWithReferenceMolecules,
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import (
    add_token_positions,
    get_token_count,
    tokenize_atom_array,
)
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    DefaultDatasetConfigSection,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Query,
)

logger = logging.getLogger(__name__)


@register_dataset
class InferenceDataset(Dataset):
    """Dataset class for running inference on a set of queries."""

    # TODO: Can accept a dataset_config here if we want
    def __init__(
        self,
        dataset_config: DefaultDatasetConfigSection,
        world_size: Optional[int] = None,
    ) -> None:
        """Initializes the InferenceDataset."""
        super().__init__()
        self.query_set = dataset_config.query_set
        self.query_cache = self.query_set.queries

        self.seeds: list = dataset_config.seeds
        self.world_size = world_size

        # Main alignments
        self.msa_settings = dataset_config.msa
        self.msa_sample_processor_inference = MsaSampleProcessorInference(
            config=self.msa_settings
        )
        self.msa_featurizer_of3 = MsaFeaturizerOF3(
            config=MsaFeaturizerOF3Config(
                max_rows=self.msa_settings.max_rows,
                max_rows_paired=self.msa_settings.max_rows_paired,
                subsample_with_bands=self.msa_settings.subsample_with_bands,
            )
        )

        # Templates
        self.template_settings = dataset_config.template
        self.template_preprocessor_settings = (
            dataset_config.template_preprocessor_settings
        )
        if self.template_preprocessor_settings.preparse_structures:
            self.template_preprocessor_settings.structure_file_format = "npz"

        # Parse CCD
        if dataset_config.ccd_file_path is not None:
            logger.debug("Parsing CCD file.")
            self.ccd = pdbx.CIFFile.read(dataset_config.ccd_file_path)
        else:
            self.ccd = BiotiteCCDWrapper()

        # Create individual datapoint cache (allows rerunning the same query with
        # different seeds)
        self.create_datapoint_cache()

    def create_datapoint_cache(self) -> None:
        qids = self.query_cache.keys()

        # Order by total sequence length (excluding ligands) so that the run times
        # are more consistent across GPUs
        query_len_map = {}
        for k, v in self.query_cache.items():
            total_poly_seq_len = sum(
                [
                    len(c.sequence) * len(c.chain_ids) if c.sequence is not None else 0
                    for c in v.chains
                ]
            )
            query_len_map[k] = total_poly_seq_len

        qids = sorted(
            qids,
            key=lambda q: query_len_map[q],
        )

        qid_values, seed_values = zip(
            *[(q, s) for q, s in itertools.product(qids, self.seeds)]
        )

        _datapoint_cache = pd.DataFrame(
            {
                "query_id": qid_values,
                "seed": seed_values,
            }
        )

        self.datapoint_cache = pad_to_world_size(_datapoint_cache, self.world_size)

    @staticmethod
    def get_structure_with_ref_mols(query: Query) -> StructureWithReferenceMolecules:
        """Creates a preprocessed AtomArray and reference molecules from the query.

        Parses the Query object into a full AtomArray and processed reference molecules
        (RDKit mol objects with atom names and computed conformers) matching the
        molecule components in the query. The returned AtomArray follows the chain IDs
        given in the Query object. If a chain specifies multiple chain IDs, repeated
        identical chains with those IDs will be constructed and given the same entity
        ID. Residue names will be inferred from the sequence or CCD codes. If a ligand
        is specified through a SMILES string, it will be named as "LIG-X", where X
        starts at 1 and is incremented for each unnamed ligand entity found in the
        Query.

        Additionally, this method adds tokenization information (token IDs) and token
        positions to the AtomArray, which are required by other functions in the
        featurization pipeline.

        Args:
            query (Query):
                The Query object containing the chains to construct the structure from.

        Returns:
            StructureWithReferenceMolecules:
                A named tuple containing the tokenized AtomArray and a list of processed
                reference molecules.
        """
        # Gets AtomArray and processed reference molecules with conformers
        atom_array, processed_reference_molecules = structure_with_ref_mols_from_query(
            query=query,
        )

        # Add token-related IDs
        tokenize_atom_array(atom_array)
        add_token_positions(atom_array)

        return StructureWithReferenceMolecules(
            atom_array, processed_reference_molecules
        )

    def create_structure_features(
        self,
        atom_array: AtomArray,
        processed_reference_molecules: list[ProcessedReferenceMolecule],
        n_tokens: int,
    ) -> dict[str, torch.Tensor]:
        """Creates the target structure features."""

        target_structure_features = featurize_structure_of3(
            atom_array=atom_array,
            n_tokens=n_tokens,
            token_dim_index_map=TOKEN_DIM_INDEX_MAP,
            is_gt=False,
            add_perm_features=False,
        )

        # Compute reference conformer features
        reference_conformer_features = featurize_reference_conformers_of3(
            processed_ref_mol_list=processed_reference_molecules,
            add_ref_space_uid_to_perm=False,
        )

        # Wrap up features
        structure_features = target_structure_features | reference_conformer_features

        return structure_features

    def create_msa_features(self, query, atom_array, n_tokens) -> dict:
        """Creates the MSA features."""

        # Create MSA precursor input
        input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
            inference_query=query
        )
        msa_array_collection = self.msa_sample_processor_inference(input=input)

        msa_features = self.msa_featurizer_of3(
            atom_array=atom_array,
            msa_array_collection=msa_array_collection,
            n_tokens=n_tokens,
        )

        return msa_features

    def create_template_features(
        self, query: Query, atom_array: AtomArray, n_tokens: int, *args, **kwargs
    ) -> dict:
        """Creates the template features."""

        # Expand pre-chain template data
        assembly_data = {}
        for chain in query.chains:
            for chain_id in chain.chain_ids:
                assembly_data[chain_id] = {
                    "template_ids": chain.template_entry_chain_ids,
                    "cache_entry_file_path": chain.template_alignment_file_path,
                }

        # Sample processing
        template_slice_collection = process_template_structures_of3(
            atom_array=atom_array,
            n_templates=self.template_settings.n_templates,
            take_top_k=self.template_settings.take_top_k,
            template_cache_directory=None,
            assembly_data=assembly_data,
            template_structures_directory=self.template_preprocessor_settings.structure_directory,
            template_structure_array_directory=self.template_preprocessor_settings.structure_array_directory,
            template_file_format=self.template_preprocessor_settings.structure_file_format,
            ccd=self.ccd,
        )

        # Featurization
        template_features = featurize_template_structures_of3(
            template_slice_collection=template_slice_collection,
            n_templates=self.template_settings.n_templates,
            n_tokens=n_tokens,
            min_bin=self.template_settings.distogram.min_bin,
            max_bin=self.template_settings.distogram.max_bin,
            n_bins=self.template_settings.distogram.n_bins,
        )

        return template_features

    def create_all_features(
        self,
        query: Query,
    ) -> dict:
        """Creates all features for a single datapoint."""

        features = {}

        # Create initial AtomArray and ReferenceMolecules from query entry
        structure_objs = self.get_structure_with_ref_mols(
            query=query,
        )
        preprocessed_atom_array, processed_reference_molecules = structure_objs

        # TODO: At some point, think about a cleaner way to pass this through the model
        # runner than as a pseudo-feature
        features["atom_array"] = preprocessed_atom_array
        n_tokens = get_token_count(preprocessed_atom_array)

        # Target structure and conformer features
        structure_features = self.create_structure_features(
            atom_array=preprocessed_atom_array,
            processed_reference_molecules=processed_reference_molecules,
            n_tokens=n_tokens,
        )
        features.update(structure_features)

        # MSA features
        msa_features = self.create_msa_features(
            query, preprocessed_atom_array, n_tokens
        )
        features.update(msa_features)

        # Template features
        template_features = self.create_template_features(
            query, preprocessed_atom_array, n_tokens
        )
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
        is_repeated_sample = bool(datapoint["repeated_sample"])

        try:
            features = self.create_all_features(query)
            features["query_id"] = query_id
            features["seed"] = torch.tensor([seed])
            features["repeated_sample"] = torch.tensor(
                [is_repeated_sample], dtype=torch.bool
            )
            features["valid_sample"] = torch.tensor([True], dtype=torch.bool)

            return features
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning(
                "-" * 40
                + "\n"
                + f"Failed to process {query_id} with preferred"
                + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                + "-" * 40
            )
            features = {
                "query_id": query_id,
                "repeated_sample": torch.tensor([is_repeated_sample], dtype=torch.bool),
                "valid_sample": torch.tensor([False], dtype=torch.bool),
            }

            return features

    def __len__(self):
        return len(self.datapoint_cache)
