from collections import OrderedDict
from typing import Sequence

import biotite.sequence.io.fasta as fasta
import numpy as np

from openfold3.core.data.preprocessing.msa_primitives import Msa


def _msa_list_to_np(msa: Sequence[str]) -> np.array:
    """Converts a list of sequences to a numpy array.

    Args:
        msa (Sequence[str]): list of ALIGNED sequences of equal length.

    Returns:
        np.array: 2D num.seq.-by-seq.len. numpy array
    """
    sequence_length = len(msa[0])
    msa_array = np.empty((len(msa), sequence_length), dtype="<U1")
    for i, sequence in enumerate(msa):
        msa_array[i] = list(sequence)
    return msa_array


def parse_fasta(file_path: str) -> tuple[np.array, np.array, Sequence[str]]:
    """Parses fasta file into a list of sequences and a list of descriptions.

    Args:
        file_path (str): Path to fasta file.
    """

    raise NotImplementedError("Fasta and a3m parsing is not yet implemented.")
    fasta_file = fasta.FastaFile.read(file_path)
    headers = []
    msa = []
    for header, string in fasta_file.items():
        headers.append(header)
        msa.append(string)

    msa = _msa_list_to_np(msa)

    return msa, headers


def parse_a3m(file_path: str) -> tuple[np.array, np.array, Sequence[str]]:
    """Parses a3m file into a list of sequences and a list of descriptions.

    Args:
        file_path (str): Path to a3m file.
    """
    return parse_fasta(file_path)


def parse_stockholm(file_path: str,
                    max_seq_count: int) -> tuple[np.array, np.array, Sequence[str]]:
    """Parses sequences and deletion matrix from stockholm format alignment.

    Args:
        file_path: 
            The string contents of a stockholm file. The first
            sequence in the file should be the query sequence.
        max_seq_count:
            The maximum number of sequences to parse from the file.

    Returns:
        np.array: A numpy array of the sequences in the alignment.
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
            if len(name_to_sequence) == max_seq_count:
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


