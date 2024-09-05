"""This module contains SampleProcessingPipelines for MSA features."""

from pathlib import Path

from biotite.structure import AtomArray

from openfold3.core.data.io.sequence.msa import parse_msas_sample
from openfold3.core.data.primitives.sequence.msa import (
    MsaParsed,
    MsaProcessed,
    MsaProcessedCollection,
    MsaSlice,
    apply_crop_to_msa,
    create_crop_to_seq_map,
    create_main,
    create_paired,
    create_query_seqs,
    expand_paired_msas,
    find_monomer_homomer,
)


def process_msas_af3(
    alignments_directory: Path | None,
    alignment_db_directory: Path | None,
    alignment_index: dict | None,
    chain_rep_map: dict[str, str],
    max_seq_counts: dict[str, int | float],
) -> tuple[MsaParsed, MsaParsed, dict[str, MsaParsed]]:
    """Prepares the arrays needed to create MSA feature tensors.

    Follows the logic of the AF3 SI in sections 2.2 and 2.3.
    1. Query sequence
    2. Paired sequences from UniProt
        - only if n unique protein chains > 1
        - exclude block-diagonal unpaired sequences
    3. Main MSAs for each chain with unpaired sequences from non-UniProt databases

    Args:
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
        chain_rep_map (dict[str, str]):
            Dict mapping chain IDs to representative chain IDs to parse for a sample.
            The representative chain IDs are used to find the directory from which to
            parse the MSAs or are used to index the alignment database, so they need to
            match the corresponding directory/alignment index names.
        max_seq_counts (int | float):
            Max number of sequences to keep from each parsed MSA. Also used to determine
            which MSAs to parse from each chain directory.

    Returns:
        tuple[Msa, Msa, dict[int, Msa]]:
            Tuple containing
                - Msa object for the query sequence
                - paired Msa concatenated across all chains
                - dict mapping chain IDs to main Msa objects.
    """

    if (alignment_db_directory is not None) and (alignment_index is None):
        raise ValueError(
            "Alignment index must be provided if alignment_db_directory is not None."
        )

    # Parse MSAs for the cropped sample
    msa_collection = parse_msas_sample(
        alignments_directory=alignments_directory,
        alignment_db_directory=alignment_db_directory,
        alignment_index=alignment_index,
        chain_rep_map=chain_rep_map,
        max_seq_counts=max_seq_counts,
    )

    # Create query
    query_seqs = create_query_seqs(msa_collection)

    # Determine whether to do pairing
    is_monomer_homomer = find_monomer_homomer(msa_collection)

    if not is_monomer_homomer:
        # Create paired UniProt MSA arrays and Rfam
        paired_msa_per_chain = create_paired(msa_collection, paired_row_cutoff=8191)

        # Expand across duplicate chains and concatenate
        paired_msas = expand_paired_msas(msa_collection, paired_msa_per_chain)

    else:
        paired_msa_per_chain = None
        paired_msas = None

    # Create main MSA arrays
    main_msas = create_main(
        msa_collection=msa_collection,
        paired_msa_per_chain=paired_msa_per_chain,
        aln_order=[
            "uniref90_hits",
            "uniprot_hits",
            "bfd_uniclust_hits",
            "bfd_uniref_hits",
            "mgnify_hits",
            "rfam_hits",
            "rnacentral_hits",
            "nucleotide_collection_hits",
        ],
    )

    return MsaProcessedCollection(
        query_sequences=query_seqs, paired_msas=paired_msas, main_msas=main_msas
    )


def process_msas_cropped_af3(
    alignments_directory: Path | None,
    alignment_db_directory: Path | None,
    alignment_index: dict | None,
    atom_array: AtomArray,
    data_cache_entry_chains: dict[str, int | str],
    max_seq_counts: dict[str, int | float],
    token_budget: int,
    max_rows_paired: int,
) -> tuple[MsaProcessed, MsaSlice]:
    """Wraps the process_msas_af3 function with the crop-to-sequence logic.

    Args:
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
        atom_array (AtomArray):
            Cropped atom array.
        data_cache_entry_chains (dict[int, Union[int, str]]):
            Dictionary of chains to chain features from the data cache for a PDB
            assembly.
        max_seq_counts (int | float):
            Max number of sequences to keep from each parsed MSA. Also used to determine
            which MSAs to parse from each chain directory.
        atom_array (AtomArray):
            Cropped atom array.
        token_budget (int):
            Crop size.
        max_rows_paired (int):
            Max number of paired rows.

    Returns:
        tuple[MsaProcessed, MsaSlice]:
            Tuple containing
                - Msa object for the query sequence
                - paired Msa concatenated across all chains
                - dict mapping chain IDs to main Msa objects.
    """

    # Find representatives and token -> residue maps in crop
    msa_slice = create_crop_to_seq_map(atom_array, data_cache_entry_chains)

    # Parse and process MSAs
    msa_processed_collection = process_msas_af3(
        alignments_directory=alignments_directory,
        alignment_db_directory=alignment_db_directory,
        alignment_index=alignment_index,
        chain_rep_map=msa_slice.chain_rep_map,
        max_seq_counts=max_seq_counts,
    )

    # Apply slices to MSAs
    msa_processed = apply_crop_to_msa(
        atom_array=atom_array,
        msa_processed_collection=msa_processed_collection,
        msa_slice=msa_slice,
        token_budget=token_budget,
        max_rows_paired=max_rows_paired,
    )
    return msa_processed
