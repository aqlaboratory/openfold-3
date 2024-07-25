import os
import string
import textwrap
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np

from openfold3.core.data.preprocessing.msa_primitives import Msa


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


def parse_fasta(
    fasta_string: str, max_seq_count: Optional[int] = None
) -> tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA file.

    This function needs to be wrapped in a with open call to read the file.

    Arguments:
        fasta_string:
            The string contents of a fasta file. The first sequence in the file
            should be the query sequence.
        max_seq_count:
            The maximum number of sequences to parse from the file.

    Returns:
        tuple[Sequence[str], Sequence[str]]:
            A list of sequences and a list of headers.
    """

    sequences = []
    headers = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            headers.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif line.startswith("#"):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line
        # Break if we have enough sequences
        if (max_seq_count is not None) & (len(sequences) == max_seq_count):
            break

    return sequences, headers


def parse_a3m(msa_string: str, max_seq_count: Optional[int] = None) -> Msa:
    """Parses sequences and deletion matrix from a3m format alignment.

    This function needs to be wrapped in a with open call to read the file.

    Args:
        msa_string:
            The string contents of a a3m file. The first sequence in the file
            should be the query sequence.
        max_seq_count:
            The maximum number of sequences to parse from the file.

    Returns:
        Msa: A Msa object containing the sequences, deletion matrix and headers.
    """

    sequences, headers = parse_fasta(msa_string, max_seq_count)
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

    return Msa(msa=msa, deletion_matrix=deletion_matrix, headers=headers)


def parse_stockholm(msa_string: str, max_seq_count: Optional[int] = None) -> Msa:
    """Parses sequences and deletion matrix from stockholm format alignment.

    This function needs to be wrapped in a with open call to read the file.

    Args:
        msa_string:
            The string contents of a stockholm file. The first sequence in the file
            should be the query sequence.
        max_seq_count:
            The maximum number of sequences to parse from the file.

    Returns:
        Msa: A Msa object containing the sequences, deletion matrix and headers.
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
        # Break if we have enough sequences
        if (max_seq_count is not None) & (len(name_to_sequence) == max_seq_count):
            break

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
    headers = list(name_to_sequence.keys())

    return Msa(msa=msa, deletion_matrix=deletion_matrix, headers=headers)


MSA_PARSER_REGISTRY = {".a3m": parse_a3m, ".sto": parse_stockholm}


def parse_msas_direct(
    folder_path: Sequence[str], max_seq_counts: Optional[dict[str, int]] = None
) -> dict[str, Msa]:
    """Parses a set of MSA files into a dictionary of Msa objects.

    This function is used to parse MSAs for a single chain.

    Args:
        folder_path:
            Path to folder containing the MSA files to parse.
        max_seq_count:
            A map from file names to maximum sequences to keep from the corresponding
            MSA file. The set of keys in this dict is also used to parse only a subset
            of the files in the folder with the corresponding names.

    Returns:
        dict[str: Msa]: A dict containing the parsed MSAs.
    """
    # Get all msa filepaths, filenames and extensions for a specific chain
    file_names = os.listdir(folder_path)
    msas = {}

    if len(file_names) == 0:
        raise RuntimeError(
            textwrap.dedent(f"""
                           No alignments found in {folder_path}. Folders for chains 
                           without any aligned sequences need to contain at least one
                           .sto file with only the query sequence.
                           """)
        )
    else:
        for file_name in file_names:
            # Split extensions from the filenames
            basename, ext = os.path.splitext(file_name)
            if ext not in [".sto", ".a3m"]:
                raise RuntimeError(
                    "All files in the alignments folder must be in .sto or .a3m format."
                )

            # Only include files with specified max values in the max_seq_counts dict
            if max_seq_counts is not None and basename not in max_seq_counts:
                continue

            # Parse the MSAs with the appropriate parser
            with open(os.path.join(folder_path, file_name)) as msa_string:
                msas[basename] = MSA_PARSER_REGISTRY[ext](
                    msa_string.read(), max_seq_counts[basename]
                )

    return msas


def parse_msas_alignment_database(
    alignment_index_entry: dict,
    alignment_database_path: str,
    max_seq_counts: Optional[dict[str, int]] = None,
) -> dict[str, Msa]:
    """Parses an entry from an alignment database into a dictionary of Msa objects.

    This function is used to parse MSAs for a single chain.

    Args:
        alignment_index_entry:
            A subdictionary of the alignment index dictionary, indexing a specific
            chain.
        alignment_database_path:
            Path to the lowest-level directory containing the alignment databases.
        max_seq_count:
            A map from file names to maximum sequences to keep from the corresponding
            MSA file. The set of keys in this dict is also used to parse only a subset
            of the files in the folder with the corresponding names.

    Returns:
        dict[str: Msa]: A dict containing the parsed MSAs.
    """
    msas = {}

    with open(
        os.path.join(alignment_database_path, alignment_index_entry["db"]), "rb"
    ) as fp:

        def read_msa(start, size):
            """Helper function to parse an alignment database file."""
            fp.seek(start)
            msa = fp.read(size).decode("utf-8")
            return msa

        alignments_to_parse = []
        for file_name, start, size in alignment_index_entry["files"]:
            # Split extensions from the filenames
            basename, ext = os.path.splitext(file_name)
            if ext not in [".sto", ".a3m"]:
                raise RuntimeError(
                    "All files in the alignments folder must be in .sto or .a3m format."
                )
            
            # Only include files with specified max values in the max_seq_counts dict
            if max_seq_counts is not None and basename not in max_seq_counts:
                continue
            alignments_to_parse.append([basename, ext, start, size])

            # Parse the MSAs with the appropriate parser
            msas[basename] = MSA_PARSER_REGISTRY[ext](
                read_msa(start, size), max_seq_counts[basename]
            )
    return msas


def parse_msas_assembly():
    

    return