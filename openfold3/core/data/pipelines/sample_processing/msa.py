"""This module contains SampleProcessingPipelines for MSA features."""

from typing import Optional, Union

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
    chain_rep_map: dict[int, str],
    alignments_path: str,
    max_seq_counts: dict[str, Union[int, float]],
    use_alignment_database: bool,
    alignment_index: Optional[dict] = None,
) -> tuple[MsaParsed, MsaParsed, dict[str, MsaParsed]]:
    """Prepares the arrays needed to create MSA feature tensors.

    Follows the logic of the AF3 SI in sections 2.2 and 2.3.
    1. Query sequence
    2. Paired sequences from UniProt
        - only if n unique protein chains > 1
        - exclude block-diagonal unpaired sequences
    3. Main MSAs for each chain with unpaired sequences from non-UniProt databases

    Args:
        chain_rep_map (dict[int, str]):
            Dict mapping chain IDs to representative chain IDs to parse for a sample.
            The representative chain IDs are used to find the directory from which to
            parse the MSAs or is used to index the alignment database, so they need to
            match the corresponding directory names.
        alignments_path (str):
            The path to the directories containing the alignment files per chain.
        max_seq_counts (int):
            Max number of sequences to keep from each pared
        use_alignment_database (bool):
            Whether to use the alignment database.
        alignment_index (dict):
            Dictionary containing the alignment index for each chain ID.

    Returns:
        tuple[Msa, Msa, dict[int, Msa]]:
            Tuple containing
                - Msa object for the query sequence
                - paired Msa concatenated across all chains
                - dict mapping chain IDs to main Msa objects.
    """

    if use_alignment_database and alignment_index is None:
        raise ValueError(
            "Alignment index must be provided if use_alignment_database is True."
        )

    # Parse MSAs for the cropped sample
    msa_collection = parse_msas_sample(
        chain_rep_map=chain_rep_map,
        alignments_path=alignments_path,
        use_alignment_database=use_alignment_database,
        alignment_index=alignment_index,
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
    atom_array: AtomArray,
    data_cache_entry_chains: dict[int, Union[int, str]],
    alignments_path: str,
    max_seq_counts: dict[str, Union[int, float]],
    use_alignment_database: bool,
    token_budget: int,
    max_rows_paired: int,
    alignment_index: Optional[dict] = None,
) -> tuple[MsaProcessed, MsaSlice]:
    """Wraps the process_msas_af3 function with the crop-to-sequence logic.

    Args:
        atom_array (AtomArray):
            Cropped atom array.
        token_budget (int):
            Crop size.
        max_rows_paired (int):
            Max number of paired rows.
        See process_msas_af3 for the rest of the arguments.

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
        chain_rep_map=msa_slice.chain_rep_map,
        alignments_path=alignments_path,
        max_seq_counts=max_seq_counts,
        use_alignment_database=use_alignment_database,
        alignment_index=alignment_index,
    )

    # Apply slices to MSAs
    msa_processed = apply_crop_to_msa(
        atom_array=atom_array,
        msa_processed_collection=msa_processed_collection,
        msa_slice=msa_slice,
        token_budget=token_budget,
        max_rows_paired=max_rows_paired,
    )
    return msa_processed, msa_slice
