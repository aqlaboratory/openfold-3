"""This module contains SampleProcessingPipelines for MSA features."""

from typing import Optional, Union

from openfold3.core.data.primitives.sequence.msa import (
    create_paired,
    create_query,
    create_unpaired,
    find_monomer_homomer,
    merge_paired_msas,
    parse_msas_sample,
)


def process_msas_af3(
    chain_ids: list[list[str], list[str]],
    alignments_path: str,
    max_seq_counts: dict[str, Union[int, float]],
    use_alignment_database: bool,
    alignment_index: Optional[dict] = None,
):
    """Prepares the arrays needed to create MSA feature tensors.

    Follows the logic of the AF3 SI in sections 2.2 and 2.3.
    1. Query sequence
    2. Paired sequences from UniProt
        - only if n unique protein chains > 1
        - exclude block-diagonal unpaired sequences
    3. Unpaired sequences from non-UniProt databases

    Args:
        chain_ids (list[list[str], list[str]]):
            List of two lists of chain IDs, the first containing the query chain IDs
            the second the representative chain IDs mapping to directories in the
            path specified by alignments_path.
        alignments_path (str):
            The path to the directories containing the alignment files per chain.
        max_seq_counts (int):
            Max number of sequences to keep from each pared
        use_alignment_database (bool):
            Whether to use the alignment database.
        alignment_index (dict):
            Dictionary containing the alignment index for each chain ID.

    Returns:
        tuple[Msa, Msa, dict[str, Msa]]:
            Tuple containing
                - Msa object for the query sequence
                - paired Msa concatenated across all chains
                - dict mapping chain IDs to unpaired Msa objects.
    """

    if use_alignment_database and alignment_index is None:
        raise ValueError(
            "Alignment index must be provided if use_alignment_database is True."
        )

    # Parse MSAs for the cropped sample
    msa_collection = parse_msas_sample(
        chain_ids=chain_ids,
        alignments_path=alignments_path,
        use_alignment_database=use_alignment_database,
        alignment_index=alignment_index,
        max_seq_counts=max_seq_counts,
    )

    # TODO yet to add RNA parsing and pairing logic
    # Create query
    query_seq = create_query(msa_collection)

    # Determine whether to do pairing
    is_monomer_homomer = find_monomer_homomer(msa_collection)

    if not is_monomer_homomer:
        # Create paired UniProt MSA arrays and Rfam
        paired_msa_per_chain = create_paired(msa_collection, paired_row_cutoff=8191)

        # Expand across duplicate chains and concatenate
        paired_msa = merge_paired_msas(msa_collection, paired_msa_per_chain)

    else:
        paired_msa_per_chain = None
        paired_msa = None

    # Create unpaired non-UniProt MSA arrays
    unpaired_msas = create_unpaired(
        msa_collection=msa_collection,
        paired_msa_per_chain=paired_msa_per_chain,
        aln_order=[
            "uniref90_hits",
            "uniprot_hits",
            "bfd_uniclust_hits",
            "bfd_uniref_hits",
            "mgnify_hits",
        ],
    )

    return query_seq, paired_msa, unpaired_msas
