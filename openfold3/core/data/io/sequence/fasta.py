"""This module contains IO functions for reading and writing fasta files."""

from pathlib import Path


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
