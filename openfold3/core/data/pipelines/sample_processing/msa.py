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


# def process_msas_msaserver(
#     alignments_directory: Path, assembly_data: dict[str, dict[str, Any]]
# ) -> MsaArrayCollection:
#     """Read in the output of the MSA server and prep

#     Args:
#         alignments_directory (Path | None): path to the msa directory generated
#             by the MSA server, ie path/to/{structure_id}/msas
#         assembly_data (dict[str, dict[str, Any]]): dict that
#             maps chain ID to relevant data for the chain:
#                 - `chain_id` (str): The chain ID.
#                 - `molecule_type` (str): The molecule type.
#                 - `alignment_representative_id` (str): The
#                   representative chain ID for the alignment.

#     Notes:
#     - Right now only support reading the direct output of the
#       MSA server function within the repo, so all we need for
#       input is the msa directory.
#     - pairing and squaring off happens externally so no need to do it
#       here can can treat the paired MSA as just another MSA from a db

#     TODO:
#      - use existing key for chain identifier instead of `chain_id`
#      - add truncation/max seq count logic
#     """
#     # cache output so we dont need to recompute
#     cached_output = alignments_directory / "cached_output.pkl"
#     if cached_output.exists():
#         with open(cached_output, "rb") as f:
#             msa_array_collection = pickle.load(f)
#         return msa_array_collection

#     ## setup mappings
#     chainid2reprid = {
#         v["chain_id"]: v["alignment_representative_id"] for
# v in assembly_data.values()
#     }
#     chainid2moltype = {
#         v["chain_id"]: v["molecule_type"] for v in assembly_data.values()
#     }
#     ## create MsaArrayCollection
#     msa_array_collection = MsaArrayCollection(
#         chain_id_to_rep_id=chainid2reprid, chain_id_to_mol_type=chainid2moltype
#     )

#     unique_representatives = list(set(chainid2reprid.values()))
#     load_paired = len(unique_representatives) > 1
#     ## hardcoding filenames for now since this is only going to support the direct
#     ## output of the MSA server
#     ### load paired and main msas - Note that mmseqs a3m are multi sample
#     ### so the output of parse_mmseqs_a3m is {chain_id: MsaArray},
#     ### as opposed to parse_a3m
#     if load_paired:
#         ### parse MSAs, returning mapping of mmseqs_chainid <> msa
#         with open(alignments_directory / "pair.a3m") as infl:
#             paired_mmseqs_chainid2msa = parse_mmseqs_a3m(infl.read())
#     else:
#         paired_mmseqs_chainid2msa = {}

#     with open(alignments_directory / "uniref.a3m") as infl:
#         uniref_mmseqs_chainid2msa = parse_mmseqs_a3m(infl.read())
#     with open(alignments_directory / "bfd.mgnify30.metaeuk30.smag30.a3m") as infl:
#         bfd_mmseqs_chainid2msa = parse_mmseqs_a3m(infl.read())

#     ## deduplicate main MSAs if paired and generate representative<>MSA mappings
#     reprid2_mainmsa_deduped = {}
#     reprid2_query_seq = {}
#     for repr_id in unique_representatives:
#         msa_uniref = uniref_mmseqs_chainid2msa[repr_id]
#         msa_bfd = bfd_mmseqs_chainid2msa[repr_id]
#         reprid2_query_seq[repr_id] = uniref_mmseqs_chainid2msa[repr_id].msa[0, :][
#             np.newaxis, :
#         ]
#         if load_paired:  # if paired, deduplicate
#             msa_paired = paired_mmseqs_chainid2msa[repr_id]
#             ids_uniref = np.array(msa_uniref.metadata)
#             ids_bfd = np.array(msa_bfd.metadata)
#             ids_paired = np.array(msa_paired.metadata)
#             msa_uniref_dedup = msa_uniref.subset(
#                 np.isin(ids_uniref, ids_paired, invert=True)
#             )
#             msa_bfd_dedup = msa_bfd.subset(np.isin(ids_bfd, ids_paired, invert=True))
#         else:
#             msa_uniref_dedup = msa_uniref
#             msa_bfd_dedup = msa_bfd

#         msa_main = msa_uniref_dedup.concatenate(msa_bfd_dedup, axis=0)
#         reprid2_mainmsa_deduped[repr_id] = msa_main

#     ## make chain<>MSA mappings
#     chainid2msa_main = {}
#     chainid2msa_paired = {}
#     chainid2query_seq = {}
#     for chain_id, repr_id in chainid2reprid.items():
#         chainid2msa_main[chain_id] = reprid2_mainmsa_deduped[repr_id]
#         chainid2query_seq[chain_id] = reprid2_query_seq[repr_id]
#         if load_paired:
#             chainid2msa_paired[chain_id] = paired_mmseqs_chainid2msa[repr_id]

#     ## finalize MsaArrayCollection
#     msa_array_collection.set_state_processed(
#         chain_id_to_query_seq=chainid2query_seq,
#         chain_id_to_paired_msa=chainid2msa_paired,
#         chain_id_to_main_msa=chainid2msa_main,
#     )
#     ## save output so we dont need to recompute
#     with open(cached_output, "wb") as f:
#         pickle.dump(msa_array_collection, f)
#     return msa_array_collection
