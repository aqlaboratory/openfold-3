"""This module contains IO functions for reading and writing MSA files."""

import os
import string
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np

from openfold3.core.data.io.sequence.fasta import parse_fasta
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.sequence.msa import (
    MsaCollection,
    MsaParsed,
    MsaSlice,
)


def _msa_list_to_np(msa: Sequence[str]) -> np.array:
    """Converts a list of sequences to a numpy array.

    Args:
        msa (Sequence[str]):
            list of ALIGNED sequences of equal length.

    Returns:
        np.array:
            2D num.seq.-by-seq.len. numpy array
    """
    sequence_length = len(msa[0])
    msa_array = np.empty((len(msa), sequence_length), dtype="<U1")
    for i, sequence in enumerate(msa):
        msa_array[i] = list(sequence)
    return msa_array


def parse_a3m(msa_string: str, max_seq_count: int | None = None) -> MsaParsed:
    """Parses sequences and deletion matrix from a3m format alignment.

    This function needs to be wrapped in a with open call to read the file.

    Args:
        msa_string (str):
            The string contents of a a3m file. The first sequence in the file
            should be the query sequence.
        max_seq_count (int | None):
            The maximum number of sequences to parse from the file.

    Returns:
        Msa: A Msa object containing the sequences, deletion matrix and metadata.
    """

    sequences, metadata = parse_fasta(msa_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    msa = [s.translate(deletion_table) for s in sequences]

    # Embed in numpy array
    msa = _msa_list_to_np(msa)
    deletion_matrix = np.array(deletion_matrix)

    parsed_msa = MsaParsed(msa=msa, deletion_matrix=deletion_matrix, metadata=metadata)

    # Crop the MSA
    if max_seq_count is not None:
        parsed_msa.truncate(max_seq_count)

    return parsed_msa


def parse_stockholm(msa_string: str, max_seq_count: int | None = None) -> MsaParsed:
    """Parses sequences and deletion matrix from stockholm format alignment.

    This function needs to be wrapped in a with open call to read the file.

    Args:
        msa_string (str):
            The string contents of a stockholm file. The first sequence in the file
            should be the query sequence.
        max_seq_count (int | None):
            The maximum number of sequences to parse from the file.

    Returns:
        Msa: A Msa object containing the sequences, deletion matrix and metadata.
    """

    # Parse each line into header: sequence dictionary
    name_to_sequence = OrderedDict()
    for line in msa_string.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        name, sequence = line.split()
        if name not in name_to_sequence:
            name_to_sequence[name] = ""
        name_to_sequence[name] += sequence

    msa = []
    deletion_matrix = []

    # Iterate over the header: sequence dictionary
    query = ""
    keep_columns = []
    for seq_index, sequence in enumerate(name_to_sequence.values()):
        if seq_index == 0:
            # Gather the columns with gaps from the query
            query = sequence
            keep_columns = [i for i, res in enumerate(query) if res != "-"]

        # Remove the columns with gaps in the query from all sequences.
        aligned_sequence = "".join([sequence[c] for c in keep_columns])

        msa.append(aligned_sequence)

        # Count the number of deletions w.r.t. query.
        deletion_vec = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query):
            if seq_res != "-" or query_res != "-":
                if query_res == "-":
                    deletion_count += 1
                else:
                    deletion_vec.append(deletion_count)
                    deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Embed in numpy array
    msa = _msa_list_to_np(msa)
    deletion_matrix = np.array(deletion_matrix)
    metadata = list(name_to_sequence.keys())

    parsed_msa = MsaParsed(msa=msa, deletion_matrix=deletion_matrix, metadata=metadata)

    # Crop the MSA
    if max_seq_count is not None:
        parsed_msa.truncate(max_seq_count)

    return parsed_msa


MSA_PARSER_REGISTRY = {".a3m": parse_a3m, ".sto": parse_stockholm}


def parse_msas_direct(
    folder_path: Path, max_seq_counts: dict[str, int] | None = None
) -> dict[str, MsaParsed]:
    """Parses a set of MSA files into a dictionary of Msa objects.

    This function is used to parse MSAs for a single chain.

    Args:
        folder_path (Path):
            Path to folder containing the MSA files to parse.
        max_seq_counts (dict[str, int] | None):
            A map from file names to maximum sequences to keep from the corresponding
            MSA file. The set of keys in this dict is also used to parse only a subset
            of the files in the folder with the corresponding names.

    Returns:
        dict[str: Msa]: A dict containing the parsed MSAs.
    """
    # Get all msa filepaths, filenames and extensions for a specific chain
    file_list = list(folder_path.iterdir())
    msas = {}

    if len(file_list) == 0:
        raise RuntimeError(
            f"No alignments found in {folder_path}. Folders for chains"
            "without any aligned sequences need to contain at least one"
            ".sto file with only the query sequence."
        )
    else:
        for aln_file in file_list:
            # Split extensions from the filenames
            basename, ext = aln_file.stem, aln_file.suffix
            if ext not in [".sto", ".a3m"]:
                raise NotImplementedError(
                    "Currently only .sto and .a3m file parsing is supported for"
                    f"alignment parsing, not {ext}."
                )

            # Only include files with specified max values in the max_seq_counts dict
            if max_seq_counts is not None and basename not in max_seq_counts:
                continue

            # Parse the MSAs with the appropriate parser
            with open(aln_file.absolute()) as f:
                msas[basename] = MSA_PARSER_REGISTRY[ext](
                    f.read(), max_seq_counts[basename]
                )

    return msas


def parse_msas_alignment_database(
    alignment_index_entry: dict,
    alignment_database_path: Path,
    max_seq_counts: dict[str, int] | None = None,
) -> dict[str, MsaParsed]:
    """Parses an entry from an alignment database into a dictionary of Msa objects.

    This function is used to parse MSAs for a single chain.

    Args:
        alignment_index_entry (dict):
            A subdictionary of the alignment index dictionary, indexing a specific
            chain.
        alignment_database_path (Path):
            Path to the lowest-level directory containing the alignment databases.
        max_seq_count (dict[str, int] | None):
            A map from file names to maximum sequences to keep from the corresponding
            MSA file. The set of keys in this dict is also used to parse only a subset
            of the files in the folder with the corresponding names.

    Returns:
        dict[str: Msa]: A dict containing the parsed MSAs.
    """
    msas = {}

    with open(
        (alignment_database_path.absolute() / Path(alignment_index_entry["db"])), "rb"
    ) as f:

        def read_msa(start, size):
            """Helper function to parse an alignment database file."""
            f.seek(start)
            msa = f.read(size).decode("utf-8")
            return msa

        for file_name, start, size in alignment_index_entry["files"]:
            # Split extensions from the filenames
            basename, ext = os.path.splitext(file_name)
            if ext not in [".sto", ".a3m"]:
                raise NotImplementedError(
                    "Currently only .sto and .a3m file parsing is supported for"
                    f"alignment parsing, not {ext}."
                )

            # Only include files with specified max values in the max_seq_counts dict
            if max_seq_counts is not None and basename not in max_seq_counts:
                continue

            # Parse the MSAs with the appropriate parser
            msas[basename] = MSA_PARSER_REGISTRY[ext](
                read_msa(start, size), max_seq_counts[basename]
            )
    return msas


@log_runtime_memory(runtime_dict_key="runtime-msa-proc-parse")
def parse_msas_sample(
    alignments_directory: Path | None,
    alignment_db_directory: Path | None,
    alignment_index: dict | None,
    msa_slice: MsaSlice,
    max_seq_counts: dict[str, int | float] | None = None,
) -> MsaCollection:
    """Parses MSA(s) for a training sample.

    This function is used to parse MSAs for a one or multiple chains, depending on the
    number of chains in the parsed PDB file and crop during training.

    Args:
        alignments_directory (Path | None):
            Path to the lowest-level directory containing the directories of MSAs
            per chain ID.
        alignment_db_directory (Path | None):
            Path to the directory containing the alignment database or its shards AND
            the alignment database superindex file. If provided, it is used over
            alignments_directory.
        alignment_index (Optional[dict], optional):
            Dictionary containing the alignment index.
        msa_slice (MsaSlice):
            Object containing the mappings from the crop to the MSA sequences.
        max_seq_counts (Optional[dict[str, int]], optional):
            Dictionary mapping the sequence database from which sequence hits were
            returned to the max number of sequences to parse from all the hits. See
            Section 2.2, Tables 1 and 2 in the AlphaFold3 SI for more details. This
            dict, when provided, is used to specify a) which alignment files to parse
            and b) the maximum number of sequences to parse.

    Returns:
        MsaCollection:
            A collection of Msa objects and chain IDs for a single sample.
    """
    chain_rep_map = msa_slice.chain_rep_map

    # Parse MSAs for each representative ID
    # This requires parsing MSAs for duplicate chains only once
    representative_chain_ids = list(set(chain_rep_map.values()))
    representative_msas = {}
    for rep_id in representative_chain_ids:
        if alignment_db_directory is not None:
            representative_msas[rep_id] = parse_msas_alignment_database(
                alignment_index_entry=alignment_index[rep_id],
                alignment_database_path=alignment_db_directory,
                max_seq_counts=max_seq_counts,
            )
        else:
            representative_msas[rep_id] = parse_msas_direct(
                folder_path=(alignments_directory / Path(rep_id)),
                max_seq_counts=max_seq_counts,
            )

    # Reindex the parsed MSAs to the original chain IDs and calculate Msa length and
    # pull out the query sequence
    rep_msa_map, rep_seq_map, num_cols = {}, {}, {}

    for _, rep_id in chain_rep_map.items():
        all_msas_per_chain = representative_msas[rep_id]
        example_msa = all_msas_per_chain[next(iter(all_msas_per_chain))].msa
        if rep_id not in rep_msa_map:
            rep_msa_map[rep_id] = all_msas_per_chain
            rep_seq_map[rep_id] = example_msa[0, :]
            num_cols[rep_id] = example_msa.shape[1]

    return MsaCollection(
        rep_msa_map=rep_msa_map,
        rep_seq_map=rep_seq_map,
        chain_rep_map=chain_rep_map,
        chain_to_molecule_type=msa_slice.chain_to_molecule_type,
        num_cols=num_cols,
    )
