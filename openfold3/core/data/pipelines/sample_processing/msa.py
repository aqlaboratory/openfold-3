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

"""This module contains SampleProcessingPipelines for MSA features."""

from collections.abc import Sequence
from functools import partial

import numpy as np
import pandas as pd

from openfold3.core.config.config_utils import DirectoryPathOrNone
from openfold3.core.config.msa_pipeline_configs import (
    MsaSampleProcessorInput,
    MsaSampleProcessorInputInference,
    MsaSampleProcessorInputTrain,
)
from openfold3.core.data.io.sequence.msa import (
    MsaSampleParser,
    MsaSampleParserInference,
    MsaSampleParserTrain,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.sequence.msa import (
    MsaArray,
    MsaArrayCollection,
    calculate_profile,
    cap_seqs_per_species,
    count_species_per_rep,
    expand_paired_row_ids,
    extract_alignments_to_pair,
    find_monomer_homomer,
    find_pairing_indices,
    get_pairing_masks,
    map_row_ids_to_msa_arrays,
    map_to_paired_msa_row_id_per_rep,
    process_msa_pairing_metadata,
    sort_subsample_paired_row_ids,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-query")
def create_query_seqs(msa_array_collection: MsaArrayCollection) -> dict[int, MsaArray]:
    """Extracts and expands the query sequences and deletion matrices.

    Args:
        msa_array_collection (MsaArrayCollection):
            A collection of Msa objects and chain IDs for a single sample.

    Returns:
        dict[int, MsaArray]:
            Dict of MsaArray objects containing the query sequence and deletion matrix
            for each chain, indexed by chain id.
    """
    return {
        k: MsaArray(
            msa=msa_array_collection.rep_id_to_query_seq[v],
            deletion_matrix=np.zeros(
                msa_array_collection.rep_id_to_query_seq[v].shape, dtype=int
            ),
            metadata=pd.DataFrame(),
        )
        for (k, v) in msa_array_collection.chain_id_to_rep_id.items()
    }


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-paired")
def create_paired(
    msa_array_collection: MsaArrayCollection,
    max_rows_paired: int,
    min_chains_paired_partial: int,
    pairing_mask_keys: list[str],
    max_seq_per_species: int,
    msas_to_pair: Sequence[str] | None,
) -> dict[str, MsaArray]:
    """Creates paired MSA arrays from UniProt MSAs.

    Follows the AF2-Multimer strategy for pairing rows of UniProt MSAs based on species
    IDs and sequence similarity to the query sequence with added functionality to
    exclude all partially paired rows with less than a certain number of chains
    as suggested by the AF3 SI.

    Also crops the paired MSA along its rows to max_rows_paired.

    Args:
        msa_array_collection (MsaArrayCollection):
            A collection of Msa objects and chain IDs for a single sample.
        max_rows_paired (int):
            The maximum number of rows to keep from the paired rows.
        min_chains_paired_partial (int):
            The minimum allowed number of chains to partially pair. Can be at most
            the number of unique chains in the crop or assembly.
        pairing_mask_keys (list[str]):
            List of strings indicating which mask to add.
        max_seq_per_species (int):
            Max number of sequences to keep per species from each chain's MSA.
        msas_to_pair (list[str]):
            Msas to pair for online pairing

    Returns:
        dict[str, Msa]:
            Paired MSAs and deletion matrices for each chain.
    """
    # Get parsed uniprot hits
    msa_arrays_to_pair = extract_alignments_to_pair(msa_array_collection, msas_to_pair)

    # Ensure there are at least two chains with UniProt hits after filtering
    if len(msa_arrays_to_pair) <= 1:
        return {}

    # Process uniprot headers and sort by distance to query
    for rep_id in msa_arrays_to_pair:
        msa_arrays_to_pair[rep_id].metadata = process_msa_pairing_metadata(
            msa_arrays_to_pair[rep_id].metadata
        )

    if max_seq_per_species is not None:
        for rep_id, msa in msa_arrays_to_pair.items():
            msa_arrays_to_pair[rep_id] = cap_seqs_per_species(msa, max_seq_per_species)

    # Count species occurrences per chain
    count_array, species = count_species_per_rep(msa_arrays_to_pair)

    # Get pairing masks
    pairing_masks = get_pairing_masks(count_array, pairing_mask_keys)

    # No valid pairs, skip MSA pairing
    if not np.any(pairing_masks):
        return {}

    # Find species indices that pair rows
    paired_species_ids = find_pairing_indices(
        count_array,
        pairing_masks,
        max_rows_paired,
        min_chains_paired_partial,
    )
    if paired_species_ids.size == 0:
        return {}

    # Map species indices back to MSA row indices
    paired_msa_row_ids_per_rep = map_to_paired_msa_row_id_per_rep(
        msa_arrays_to_pair,
        paired_species_ids,
        species,
    )

    # Expand paired row IDs all chains
    paired_msa_row_ids_per_chain = expand_paired_row_ids(
        msa_array_collection, paired_msa_row_ids_per_rep, paired_species_ids
    )

    # Sort by row-products + subsample - only need to do the former if have more than
    # max_rows_paired, otherwise it doesn't matter due to the per-recycle subsampling
    paired_msa_row_ids_per_chain, paired_species_ids, n_rows_actual = (
        sort_subsample_paired_row_ids(
            paired_msa_row_ids_per_chain, paired_species_ids, max_rows_paired
        )
    )

    # Update row counts
    msa_array_collection.set_row_counts(n_rows_paired_subsampled=n_rows_actual)

    # Finally, map row IDs to actual MSA rows
    chain_id_to_paired_msa = map_row_ids_to_msa_arrays(
        msa_array_collection,
        msa_arrays_to_pair,
        paired_species_ids,
        paired_msa_row_ids_per_chain,
    )

    return chain_id_to_paired_msa


def create_paired_from_precomputed(
    msa_array_collection: MsaArrayCollection,
    max_rows_paired: int,
    paired_msa_order: list[str],
) -> dict[str, MsaArray]:
    """Creates per-chain paired MSA arrays in the expected format from precomputed
    paired MSAs.

    Args:
        msa_array_collection (MsaArrayCollection):
            A collection of Msa objects and chain IDs for a single sample.
        max_rows_paired (int):
            The maximum number of rows to keep from the paired rows.
        paired_msa_order (list[str]):
            The order in which to concatenate the paired MSA arrays vertically if
            multiple are provided. Alignments not in this list are not added to the
            paired MSA stack.

    Returns:
        dict[str, MsaArray]: _description_
    """

    # Process precomputed paired MSAs
    processed_prepaired_msas = {}
    for rep_id, paired_msa_dict in msa_array_collection.rep_id_to_paired_msa.items():
        # Flatten
        prepaired_msa = MsaArray.multi_concatenate(
            [
                paired_msa_dict[paired_msa_key]
                for paired_msa_key in paired_msa_order
                if paired_msa_key in paired_msa_dict
            ]
        )
        # Crop
        processed_prepaired_msas[rep_id] = prepaired_msa.truncate(max_rows_paired)

    msa_array_collection.rep_id_to_paired_msa = processed_prepaired_msas

    # Map to per-chain
    chain_id_to_paired_msa = {
        k: msa_array_collection.rep_id_to_paired_msa[v]
        for (k, v) in msa_array_collection.chain_id_to_rep_id.items()
    }
    return chain_id_to_paired_msa


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-main")
def create_main(
    msa_array_collection: MsaArrayCollection,
    chain_id_to_paired_msa: dict[str, MsaArray],
    max_rows: int,
    aln_order: list[str],
    subsample_main: bool,
    keep_subsampled_order: bool,
) -> dict[str, MsaArray]:
    """Creates main MSA arrays from non-UniProt MSAs.

    Note: this function also removes all sequences from the final main MSA that are
    present in the cropped paired MSA of the corresponding chain. Also creates the
    profile and deletion mean from the redundant main MSA before subsampling.

    Args:
        msa_array_collection (MsaArrayCollection):
            A collection of MsaArrays and chain IDs for a single sample.
        chain_id_to_paired_msa (dict[str, MsaArray]):
            Dict of paired Msa objects per chain.
        max_rows (int):
            Maximum number of sequence rows allowed for the MSA vstack of each chain.
        subsample_main (bool):
            Whether to apply per-chain main MSA subsampling.
        aln_order (list[str]):
            The order in which to concatenate the main MSA arrays vertically. Alignments
            not in this list are not added to the main MSA.
        keep_subsampled_order (bool):
            Whether to keep the order of sequences in the subsampled main MSA relative
            to the unsubsampled one.

    Returns:
        dict[str, MsaArray]:
            List of MsaArrays containing the main MSA arrays and deletion matrices for
            each chain.
    """

    # Iterate over representatives
    rep_id_to_main_msa = {}
    rep_id_to_profile = {}
    rep_id_to_del_mean = {}

    for rep_id, chain_data in msa_array_collection.rep_id_to_main_msa.items():
        chain_data = msa_array_collection.rep_id_to_main_msa[rep_id]

        # Get MSAs forming the main MSA and deletion matrices from all non-UniProt MSAs
        main_msa_redundant = np.concatenate(
            [chain_data[aln].msa for aln in aln_order if aln in chain_data],
            axis=0,
        )
        main_deletion_matrix_redundant = np.concatenate(
            [chain_data[aln].deletion_matrix for aln in aln_order if aln in chain_data],
            axis=0,
        )

        # Get paired MSAs if any and deduplicate
        if len(chain_id_to_paired_msa) > 0:
            # The relevant paired MSA for this representative
            paired_arr = chain_id_to_paired_msa[
                msa_array_collection.rep_id_to_chain_id[rep_id]
            ].msa
            arr = main_msa_redundant

            # 1) Convert each 2D array into a 1D "structured" view of type void This
            # way, each row is treated as one item.
            arr_view = arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
            paired_view = paired_arr.view(
                np.dtype((np.void, paired_arr.dtype.itemsize * paired_arr.shape[1]))
            )

            # 2) Vectorized membership check: is row in paired_msa? ~np.isin(...)
            # inverts the boolean array, so True -> "unique" row
            is_unique = np.squeeze(~np.isin(arr_view, paired_view), axis=-1)

            # Apply filtering with the boolean mask
            filtered_msa = main_msa_redundant[is_unique, :]
            filtered_deletion = main_deletion_matrix_redundant[is_unique, :]
        else:
            filtered_msa = main_msa_redundant
            filtered_deletion = main_deletion_matrix_redundant

        # Add to ID-MSA map
        rep_id_to_main_msa[rep_id] = MsaArray(
            msa=filtered_msa,
            deletion_matrix=filtered_deletion,
            metadata=pd.DataFrame(),
        )

        # Calculate profile and del mean from the redundant main MSA
        rep_id_to_profile[rep_id] = calculate_profile(
            msa_array=main_msa_redundant,
            molecule_type=msa_array_collection.rep_id_to_mol_type[rep_id],
            chunk_size=1000,
        )
        rep_id_to_del_mean[rep_id] = np.mean(main_deletion_matrix_redundant, axis=0)

    # Reindex dicts from representatives to chain IDs and subsample main MSAs
    chain_id_to_main_msa = {}
    chain_id_to_profile = {}
    chain_id_to_del_mean = {}
    n_rows_paired_subsampled = msa_array_collection.row_counts.n_rows_paired_subsampled
    n_rows_main_subsampled = {}
    max_n_rows_main_subsampled = 0
    # row upper limit for main MSAs
    n_rows_main_msa_lim = max(0, max_rows - n_rows_paired_subsampled - 1)
    for chain_id, rep_id in msa_array_collection.chain_id_to_rep_id.items():
        filtered_msa_array = rep_id_to_main_msa[rep_id]
        # actual number of rows in the unsubsampled main MSA for this chain
        n_rows_main_msa = filtered_msa_array.msa.shape[0]

        if subsample_main:
            # No main MSA or limit exhausted
            if n_rows_main_msa == 0 or n_rows_main_msa_lim == 0:
                idx = np.empty((0,), dtype=int)
            # Subsample otherwise
            else:
                k = np.random.randint(1, min(n_rows_main_msa, n_rows_main_msa_lim) + 1)
                idx = np.random.choice(n_rows_main_msa, size=k, replace=False)

            if keep_subsampled_order:
                idx.sort()
        else:
            # Keep up to the limit
            idx = np.arange(min(n_rows_main_msa, n_rows_main_msa_lim))

        main_msa = MsaArray(
            msa=filtered_msa_array.msa[idx, :],
            deletion_matrix=filtered_msa_array.deletion_matrix[idx, :],
            metadata=pd.DataFrame(),
        )
        chain_id_to_main_msa[chain_id] = main_msa
        chain_id_to_profile[chain_id] = rep_id_to_profile[rep_id]
        chain_id_to_del_mean[chain_id] = rep_id_to_del_mean[rep_id]

        main_msa_depth = main_msa.msa.shape[0]
        if main_msa_depth > max_n_rows_main_subsampled:
            max_n_rows_main_subsampled = main_msa_depth
        n_rows_main_subsampled[chain_id] = main_msa_depth

    # Update row counts
    n_rows_total = 1 + n_rows_paired_subsampled + max_n_rows_main_subsampled
    msa_array_collection.set_row_counts(
        n_rows_total=n_rows_total,
        n_rows_main_subsampled=n_rows_main_subsampled,
    )
    return chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean


class MsaSampleProcessor:
    """Base class for MSA sample processing."""

    def __init__(self, config: MSASettings):
        self.config = config
        self.msa_sample_parser = MsaSampleParser(config=config)
        self.query_seq_processor = create_query_seqs
        self.paired_msa_processor = partial(
            create_paired,
            max_rows_paired=config.max_rows_paired,
            min_chains_paired_partial=config.min_chains_paired_partial,
            pairing_mask_keys=config.pairing_mask_keys,
            max_seq_per_species=config.max_seq_per_species,
            msas_to_pair=config.msas_to_pair,
        )
        self.main_msa_processor = partial(
            create_main,
            max_rows=config.max_rows,
            aln_order=config.aln_order,
            subsample_main=config.subsample_main,
            keep_subsampled_order=config.keep_subsampled_order,
        )

    def create_query_seq(
        self,
        input: MsaSampleProcessorInput | None = None,
        msa_array_collection: MsaArrayCollection | None = None,
    ) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleProcessor directly. Subclass it and "
            "implement create_query_seq, paired_msa_processor and main_msa_processor "
            "methods to use it."
        )

    def create_paired_msa(
        self,
        input: MsaSampleProcessorInput | None = None,
        msa_array_collection: MsaArrayCollection | None = None,
    ) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleProcessor directly. Subclass it and "
            "implement create_query_seq, paired_msa_processor and main_msa_processor "
            "methods to use it."
        )

    def create_main_msa(
        self,
        input: MsaSampleProcessorInput | None = None,
        msa_array_collection: MsaArrayCollection | None = None,
        chain_id_to_paired_msa: dict[str, MsaArray] | None = None,
    ) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleProcessor directly. Subclass it and "
            "implement create_query_seq, paired_msa_processor and main_msa_processor "
            "methods to use it."
        )

    @log_runtime_memory(runtime_dict_key="runtime-msa-proc")
    def __call__(self, input: MsaSampleProcessorInput) -> MsaArrayCollection:
        # Parse MSAs
        msa_array_collection = self.msa_sample_parser(input=input)

        # Create dicts with the processed query, paired and main MSA data per chain
        chain_id_to_query_seq = self.create_query_seq(
            input=input, msa_array_collection=msa_array_collection
        )
        chain_id_to_paired_msa = self.create_paired_msa(
            input=input, msa_array_collection=msa_array_collection
        )
        chain_id_to_main_msa, chain_id_to_profile, chain_id_to_deletion_mean = (
            self.create_main_msa(
                input=input,
                msa_array_collection=msa_array_collection,
                chain_id_to_paired_msa=chain_id_to_paired_msa,
            )
        )

        # Update MsaArrayCollection with processed MSA data
        msa_array_collection.set_state_processed(
            chain_id_to_query_seq=chain_id_to_query_seq,
            chain_id_to_paired_msa=chain_id_to_paired_msa,
            chain_id_to_main_msa=chain_id_to_main_msa,
            chain_id_to_profile=chain_id_to_profile,
            chain_id_to_deletion_mean=chain_id_to_deletion_mean,
        )

        return msa_array_collection


# TODO: test
class MsaSampleProcessorTrain(MsaSampleProcessor):
    """Pipeline for MSA sample processing for training."""

    def __init__(
        self,
        config: MSASettings,
        *,
        alignment_array_directory: DirectoryPathOrNone = None,
        alignment_db_directory: DirectoryPathOrNone = None,
        alignment_index: dict | None = None,
        alignments_directory: DirectoryPathOrNone = None,
        use_roda_monomer_format: bool = False,
    ):
        super().__init__(config=config)
        self.msa_sample_parser = MsaSampleParserTrain(
            config=config,
            alignment_array_directory=alignment_array_directory,
            alignment_db_directory=alignment_db_directory,
            alignment_index=alignment_index,
            alignments_directory=alignments_directory,
            use_roda_monomer_format=use_roda_monomer_format,
        )

    @log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-query-bridge")
    def create_query_seq(
        self,
        input: MsaSampleProcessorInputTrain,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create query sequences from MSA arrays."""
        if len(msa_array_collection.rep_id_to_query_seq) > 0:
            chain_id_to_query_seq = self.query_seq_processor(
                msa_array_collection=msa_array_collection
            )
        else:
            chain_id_to_query_seq = {}
        return chain_id_to_query_seq

    @log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-paired-bridge")
    def create_paired_msa(
        self,
        input: MsaSampleProcessorInputTrain,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create paired MSAs from MSA arrays."""
        if len(msa_array_collection.rep_id_to_query_seq) > 0:
            # Determine whether to do pairing
            if not find_monomer_homomer(msa_array_collection):
                # Create paired UniProt MSA arrays
                chain_id_to_paired_msa = self.paired_msa_processor(
                    msa_array_collection=msa_array_collection,
                )
            else:
                chain_id_to_paired_msa = {}
        else:
            chain_id_to_paired_msa = {}
        return chain_id_to_paired_msa

    @log_runtime_memory(runtime_dict_key="runtime-msa-proc-create-main-bridge")
    def create_main_msa(
        self,
        input: MsaSampleProcessorInputTrain,
        msa_array_collection: MsaArrayCollection,
        chain_id_to_paired_msa: dict[str, MsaArray],
    ) -> dict[str, MsaArray]:
        """Create main MSAs from MSA arrays."""
        if len(msa_array_collection.rep_id_to_query_seq) > 0:
            # Create main MSA arrays
            chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean = (
                self.main_msa_processor(
                    msa_array_collection=msa_array_collection,
                    chain_id_to_paired_msa=chain_id_to_paired_msa,
                )
            )
        else:
            chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean = {}, {}, {}
        return chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean


class MsaSampleProcessorInference(MsaSampleProcessor):
    """Pipeline for MSA sample processing for inference."""

    def __init__(self, config: MSASettings):
        super().__init__(config=config)
        self.msa_sample_parser = MsaSampleParserInference(config=config)

    def create_query_seq(
        self,
        input: MsaSampleProcessorInputInference,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create query sequences from MSA arrays."""
        if (len(msa_array_collection.rep_id_to_query_seq) > 0) & input.use_msas:
            chain_id_to_query_seq = self.query_seq_processor(
                msa_array_collection=msa_array_collection
            )
        else:
            chain_id_to_query_seq = {}
        return chain_id_to_query_seq

    def create_paired_msa(
        self,
        input: MsaSampleProcessorInputInference,
        msa_array_collection: MsaArrayCollection,
    ) -> dict[str, MsaArray]:
        """Create paired MSAs from MSA arrays."""
        if (
            (len(msa_array_collection.rep_id_to_query_seq) > 0)
            & input.use_msas
            & input.use_paired_msas
        ):
            # Use precomputed paired MSAs
            # TODO modularize better
            if len(msa_array_collection.rep_id_to_paired_msa) > 0:
                chain_id_to_paired_msa = create_paired_from_precomputed(
                    msa_array_collection=msa_array_collection,
                    max_rows_paired=self.config.max_rows_paired,
                    paired_msa_order=self.config.paired_msa_order,
                )
            # Pair online from main MSAs
            elif not find_monomer_homomer(msa_array_collection):
                # Create paired UniProt MSA arrays
                chain_id_to_paired_msa = self.paired_msa_processor(
                    msa_array_collection=msa_array_collection,
                )
            else:
                chain_id_to_paired_msa = {}
        else:
            chain_id_to_paired_msa = {}
        return chain_id_to_paired_msa

    def create_main_msa(
        self,
        input: MsaSampleProcessorInputInference,
        msa_array_collection: MsaArrayCollection,
        chain_id_to_paired_msa: dict[str, MsaArray],
    ) -> dict[str, MsaArray]:
        """Create main MSAs from MSA arrays."""
        if (
            (len(msa_array_collection.rep_id_to_query_seq) > 0)
            & input.use_msas
            & input.use_main_msas
        ):
            # Create main MSA arrays
            chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean = (
                self.main_msa_processor(
                    msa_array_collection=msa_array_collection,
                    chain_id_to_paired_msa=chain_id_to_paired_msa,
                )
            )
        else:
            chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean = {}, {}, {}
        return chain_id_to_main_msa, chain_id_to_profile, chain_id_to_del_mean
