# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains IO functions for reading and writing fasta files."""

import contextlib
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_chain_id_to_seq_from_fasta(input_path: Path) -> dict[str, str]:
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
    line_count = 0

    with open(input_path) as file, contextlib.suppress(StopIteration):
        while True:
            chain = next(file)
            line_count += 1
            assert chain.startswith(">"), f"Invalid FASTA format on line {line_count}"
            chain = chain.replace(">", "").strip()

            seq = next(file).strip()
            line_count += 1
            assert not seq.startswith(">"), f"Invalid FASTA format on line {line_count}"

            if chain in chain_to_sequence:
                logger.warning(
                    f"Duplicate header {chain} in line {line_count - 1}, skipping."
                )
                continue
            else:
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
        fasta_path = pdb_dir / f"{pdb_id}.fasta"

        if not fasta_path.exists():
            logger.warning(f"FASTA file not found for {pdb_id}")
            return {}

        chain_id_to_seq = get_chain_id_to_seq_from_fasta(fasta_path)
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
    sort: bool = False,
) -> Path:
    """Writes a FASTA file from a dictionary of IDs to sequences.

    The output FASTA will follow the format:
    >{id}
    {sequence}

    Args:
        output_path:
            Path to write the FASTA file to.
        id_to_sequence:
            Dictionary mapping IDs to sequences.
        sort:
            Whether to sort by ID before writing. Defaults to False.

    Returns:
        Path to the written FASTA file.
    """
    if sort:
        id_to_sequence = dict(sorted(id_to_sequence.items()))

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
