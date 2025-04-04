"""This module contains SampleProcessingPipelines for MSA features."""

from pathlib import Path
from typing import Any

from biotite.structure import AtomArray

from openfold3.core.data.io.sequence.msa import parse_msas_sample
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.sequence.msa import (
    MsaArrayCollection,
    create_main,
    create_paired,
    create_query_seqs,
    find_monomer_homomer,
)


@log_runtime_memory(runtime_dict_key="runtime-msa-proc")
def process_msas_af3(
    atom_array: AtomArray,
    assembly_data: dict[str, dict[str, Any]],
    alignments_directory: Path | None,
    alignment_db_directory: Path | None,
    alignment_index: dict | None,
    alignment_array_directory: Path | None,
    max_seq_counts: dict[str, int | float],
    aln_order: list[str],
    max_rows_paired: int,
    min_chains_paired_partial: int,
    pairing_mask_keys: list[str],
    moltypes: list[str],
) -> MsaArrayCollection:
    """Prepares the arrays needed to create MSA feature tensors.

    Follows the logic of the AF3 SI in sections 2.2 and 2.3.
    1. Query sequence
    2. Paired sequences from UniProt
        - only if n unique protein chains > 1
        - exclude block-diagonal unpaired sequences if min_chains_paired_partial = 2
        - only protein-protein chains are paired
    3. Main MSAs for each chain with unpaired sequences from non-UniProt databases

    Note: The returned MsaProcessedCollection contains None for the query_sequences
    if there are no protein or RNA chains in the crop.

    Args:
        atom_array (AtomArray):
            The cropped (training) or full (inference) atom array.
        assembly_data (dict[str, dict[str, Any]]):
            Dict containing the alignment representatives and molecule types for each
            chain.
        alignments_directory (Path | None):
            The path to the directory containing directories containing the alignment
            files per chain. Only used if alignment_db_directory is None.
        alignment_db_directory (Path | None):
            The path to the directory containing the alignment database or its shards
            AND the alignment database superindex file. If provided, it is used over
            alignments_directory.
        alignment_index (dict | None):
            Dictionary containing the alignment index for each chain ID. Only used if
            alignment_db_directory is provided.
        alignment_array_directory (Path | None):
            The path to the directory containing the preprocessed alignment arrays.
        max_seq_counts (dict):
            Dict of max number of sequences to keep from each parsed MSA. Also used to
            determine which MSAs to parse from each chain directory.
        aln_order (list[str]):
            A list of strings matching the alignment file names, indicating the order in
            which they should be concatenated to form the main MSA.
        max_rows_paired (int):
            The maximum number of rows to keep in the paired MSA.
        min_chains_paired_partial (int):
            The minimum number of chains to keep in the paired MSA.
        pairing_mask_keys (list[str]):
            List of keys indicating which types of masks to apply during pairing.
        moltypes (list[str]):
            List of molecule types to consider in the MSA.

    Returns:
        MsaArrayCollection:
            The collection of MsaArrays in the processed state.
    """

    if (alignment_db_directory is not None) and (alignment_index is None):
        raise ValueError(
            "Alignment index must be provided if alignment_db_directory is not None."
        )

    # Parse MSAs, deletion matrices into numpy arrays and metadata into dataframes
    msa_array_collection = parse_msas_sample(
        atom_array=atom_array,
        assembly_data=assembly_data,
        moltypes=moltypes,
        alignments_directory=alignments_directory,
        alignment_db_directory=alignment_db_directory,
        alignment_index=alignment_index,
        alignment_array_directory=alignment_array_directory,
        max_seq_counts=max_seq_counts,
    )

    # Create dicts with the processed query, paired and main MSA data per chain
    if len(msa_array_collection.rep_id_to_query_seq) > 0:
        # Create query
        chain_id_to_query_seq = create_query_seqs(msa_array_collection)

        # Determine whether to do pairing
        if not find_monomer_homomer(msa_array_collection):
            # Create paired UniProt MSA arrays
            chain_id_to_paired_msa = create_paired(
                msa_array_collection,
                max_rows_paired=max_rows_paired,
                min_chains_paired_partial=min_chains_paired_partial,
                pairing_mask_keys=pairing_mask_keys,
            )
        else:
            chain_id_to_paired_msa = {}

        # Create main MSA arrays
        chain_id_to_main_msa = create_main(
            msa_array_collection=msa_array_collection,
            chain_id_to_paired_msa=chain_id_to_paired_msa,
            aln_order=aln_order,
        )

    # Skip MSA processing if there are no protein or RNA chains
    else:
        chain_id_to_query_seq, chain_id_to_paired_msa, chain_id_to_main_msa = {}, {}, {}

    # Update MsaArrayCollection with processed MSA data
    msa_array_collection.set_state_processed(
        chain_id_to_query_seq=chain_id_to_query_seq,
        chain_id_to_paired_msa=chain_id_to_paired_msa,
        chain_id_to_main_msa=chain_id_to_main_msa,
    )

    return msa_array_collection
