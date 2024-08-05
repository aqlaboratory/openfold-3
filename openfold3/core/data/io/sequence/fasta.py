"""This module contains IO functions for reading and writing fasta files."""

from typing import Sequence


def parse_fasta(fasta_string: str) -> tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA file.

    This function needs to be wrapped in a with open call to read the file.

    Arguments:
        fasta_string:
            The string contents of a fasta file. The first sequence in the file
            should be the query sequence.

    Returns:
        tuple[Sequence[str], Sequence[str]]:
            A list of sequences and a list of metadata.
    """

    sequences = []
    metadata = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            metadata.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif line.startswith("#"):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, metadata
