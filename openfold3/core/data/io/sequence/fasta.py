"""This module contains IO functions for reading and writing fasta files."""

from pathlib import Path
from typing import Sequence


def write_multichain_fasta(
    output_path: Path,
    chain_to_sequence: dict,
) -> Path:
    """Writes a FASTA file from a dictionary of chain IDs to sequences.

    The output FASTA will follow the format:
    >{chain_id}
    {sequence}

    Args:
        output_path:
            Path to write the FASTA file to.
        chain_to_sequence:
            Dictionary mapping chain IDs to sequences.

    Returns:
        Path to the written FASTA file.
    """
    with open(output_path, "w") as file:
        file.writelines(
            f">{chain_id}\n{seq}\n" for chain_id, seq in chain_to_sequence.items()
        )

    return output_path


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
