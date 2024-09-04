"""This module contains IO functions for reading and writing fasta files."""

import contextlib
from pathlib import Path

from tqdm import tqdm


def read_multichain_fasta(input_path: Path) -> dict[str, str]:
    """Reads a FASTA file into a dictionary of chain IDs to sequences.

    The input FASTA should follow the format:
    >{id}
    {sequence}
    >{id}
    {sequence}
    >...

    Args:
        input_path:
            Path to the FASTA file to read.

    Returns:
        Dictionary mapping chain IDs to sequences.
    """
    chain_to_sequence = {}
    with open(input_path) as file, contextlib.suppress(StopIteration):
        while True:
            chain = next(file).replace(">", "").strip()
            seq = next(file).strip()

            chain_to_sequence[chain] = seq

    return chain_to_sequence


def consolidate_preprocessed_fastas(preprocessed_dir: Path) -> dict[str, str]:
    """Reads all FASTA files in a preprocessed directory into a single dictionary.

    This is meant to be used on the output directory of the preprocessing metadata
    extraction script, which is formatted like this:

    pdb_id:
        pdb_id.fasta
        ...
    pdb_id:
        pdb_id.fasta
        ...

    Where individual fasta files are formatted like this:
    >{chain_id}
    {sequence}
    >{chain_id}
    {sequence}
    ...

    Args:
        preprocessed_dir:
            Path to the directory containing preprocessed FASTA files.

    Returns:
        A dictionary mapping IDs to sequences. IDs follow the format
        {pdb_id}_{chain_id}.
    """
    ids_to_seq = {}

    for pdb_dir in tqdm(preprocessed_dir.iterdir()):
        pdb_id = pdb_dir.name

        chain_id_to_seq = read_multichain_fasta(pdb_dir / f"{pdb_id}.fasta")

        for chain_id, seq in chain_id_to_seq.items():
            ids_to_seq[f"{pdb_id}_{chain_id}"] = seq

    return ids_to_seq


def write_multichain_fasta(
    output_path: Path,
    id_to_sequence: dict[str, str],
) -> Path:
    """Writes a FASTA file from a dictionary of IDs to sequences.

    The output FASTA will follow the format:
    >{id}
    {sequence}

    Args:
        output_path:
            Path to write the FASTA file to.
        chain_to_sequence:
            Dictionary mapping IDs to sequences.

    Returns:
        Path to the written FASTA file.
    """
    with open(output_path, "w") as file:
        file.writelines(f">{id_}\n{seq}\n" for id_, seq in id_to_sequence.items())

    return output_path
