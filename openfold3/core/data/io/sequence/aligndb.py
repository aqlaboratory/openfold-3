"""This module contains IO functions for reading and writing the alignment database
format of OpenFold."""

import os
from typing import Optional

from openfold3.core.data.io.sequence.msa import MSA_PARSER_REGISTRY
from openfold3.core.data.primitives.sequence.msa import Msa


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
