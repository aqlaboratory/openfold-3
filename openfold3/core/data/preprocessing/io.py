import os
import string
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
    file_path: str, max_seq_count: Optional[int] = None
) -> tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA file.

    Arguments:
        file_path:
            Path to fasta file.
        max_seq_count:
            The maximum number of sequences to parse from the file.

    Returns:
        tuple[Sequence[str], Sequence[str]]:
            A list of sequences and a list of headers.
    """
    with open(file_path) as infile:
        fasta_string = infile.read()
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


def parse_a3m(file_path: str, max_seq_count: Optional[int] = None) -> Msa:
    """Parses sequences and deletion matrix from a3m format alignment.

    Args:
        file_path:
            Path to a3m file.
        max_seq_count:
            The maximum number of sequences to parse from the file.

    Returns:
        Msa: A Msa object containing the sequences, deletion matrix and headers.
    """

    sequences, headers = parse_fasta(file_path, max_seq_count)
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


def parse_stockholm(file_path: str, max_seq_count: Optional[int] = None) -> Msa:
    """Parses sequences and deletion matrix from stockholm format alignment.

    Args:
        file_path:
            The string contents of a stockholm file. The first sequence in the file
            should be the query sequence.
        max_seq_count:
            The maximum number of sequences to parse from the file.

    Returns:
        Msa: A Msa object containing the sequences, deletion matrix and headers.
    """

    with open(file_path) as infile:
        # Parse each line into header: sequence dictionary
        name_to_sequence = OrderedDict()
        for line in infile.read().splitlines():
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


def parse_msas(
    folder_path: Sequence[str], max_seq_counts: Optional[dict[str, int]] = None
) -> dict[str:Msa]:
    """Parses a set of MSA files into a dictionary of Msa objects.

    Args:
        folder_path:
            Path to folder containing the MSA files to parse.
        max_seq_count:
            A map from file names to maximum sequences to keep from the corresponding
            MSA file.

    Returns:
        dict[str: Msa]: A dict containing the parsed MSAs.
    """
    # Get all msa filepaths, filenames and extensions for a specific chain
    file_names = os.listdir(folder_path)
    
    # Only include files with specified max values in the max_seq_counts dict
    if max_seq_counts is not None:
        file_names = [
            file_name for file_name in file_names 
            if os.path.splitext(file_name)[0] in max_seq_counts
        ]

    file_paths = [os.path.join(folder_path, p) for p in file_names]
    file_names = [os.path.splitext(p) for p in file_names]

    if not all([ext in [".sto", ".a3m"] for _, ext in file_names]):
        raise RuntimeError(
            "All files in the alignments folder must be in .sto or .a3m format."
            )
    
    # Parse the MSAs with the appropriate parser
    msas = {
        file_names[i][0]: MSA_PARSER_REGISTRY[file_names[i][1]](
            file_path, max_seq_counts[file_names[i][0]]
        )
        for i, file_path in enumerate(file_paths)
    }

    return msas
