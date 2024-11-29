"""This module contains IO functions for reading and writing fasta files."""

import contextlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            chain = next(file)
            assert chain.startswith(">"), "Invalid FASTA format"
            chain = chain.replace(">", "").strip()

            seq = next(file).strip()
            assert not seq.startswith(">"), "Invalid FASTA format"

            chain_to_sequence[chain] = seq

    return chain_to_sequence


def consolidate_preprocessed_fastas(preprocessed_dir: Path) -> dict[str, str]:
    """Reads all FASTA files in a preprocessed directory into a single dictionary.

    Note that this uses threading to speed up the process.

    Args:
        preprocessed_dir:
            Path to the directory of preprocessed files created during the preprocessing
            scripts. The directory is expected to be structured like this:
            4h1w/
                4h1w.fasta
                [...]
            1nag/
                1nag.fasta
                [...]
            [...]


    Returns:
        A dictionary mapping IDs to sequences. IDs follow the format
        {pdb_id}_{chain_id}.
    """
    ids_to_seq = {}

    # Function to read FASTA for a single directory
    def process_pdb_dir(pdb_dir: Path):
        pdb_id = pdb_dir.name
        chain_id_to_seq = read_multichain_fasta(pdb_dir / f"{pdb_id}.fasta")
        return {
            f"{pdb_id}_{chain_id}": seq for chain_id, seq in chain_id_to_seq.items()
        }

    # Collect all directories
    pdb_dirs = list(preprocessed_dir.iterdir())

    # Use ThreadPoolExecutor for threading
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_pdb_dir, pdb_dir): pdb_dir for pdb_dir in pdb_dirs
        }

        # Use tqdm to track progress
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Consolidating FASTAs"
        ):
            result = future.result()
            ids_to_seq.update(result)

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
