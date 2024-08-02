import dataclasses
import math
import os
import string
import textwrap
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np


@dataclasses.dataclass(frozen=False)
class Msa:
    """Class representing a parsed MSA file.

    The metadata attrubute gets updated in certain functions of the MSA preparation.

    Attributes:
        msa: np.array
            A 2D numpy array containing the aligned sequences.
        deletion_matrix: np.array
            A 2D numpy array containing the cumulative deletion counts up to each
            position for each row in the MSA.
        metadata: Optional[Sequence[str]]
            A list of metadata persed from sequence headers of the MSA."""

    msa: np.array
    deletion_matrix: np.array
    metadata: Optional[Sequence[str]]

    def __len__(self):
        return self.msa.shape[0]

    def truncate(self, max_seq_count: int) -> None:
        """Truncate the MSA to a maximum number of sequences.

        Args:
            max_seq_count (int): Number of sequences to keep in the MSA.

        Returns:
            None
        """

        if not isinstance(max_seq_count, int) | (max_seq_count == math.inf):
            raise ValueError("max_seq_count should be an integer or math.inf.")

        if self.__len__() > max_seq_count:
            if max_seq_count == math.inf:
                max_seq_count = self.__len__()

            self.msa = self.msa[:max_seq_count, :]
            self.deletion_matrix = self.deletion_matrix[:max_seq_count, :]
            self.metadata = (
                self.metadata[:max_seq_count]
                if isinstance(self.metadata, list)
                else self.metadata.iloc[: (max_seq_count - 1)]
            )


@dataclasses.dataclass(frozen=False)
class MsaCollection:
    """Class representing a collection MSAs for a single sample.

    Attributes:
        rep_msa_map: dict[str, dict[str, Msa]]
            Dictionary mapping representative chain IDs to dictionaries of Msa objects.
        rep_seq_map: dict[str, np.ndarray[np.str_]]
            Dictionary mapping representative chain IDs to numpy arrays of their
            corresponding query sequences.
        chain_rep_map: dict[str, str]
            Dictionary mapping chain IDs to representative chain IDs.
        num_cols: dict[str, int]
            Dict mapping representative chain ID to the number of columns in the MSA.
    """

    rep_msa_map: dict[str, dict[str, Msa]]
    rep_seq_map: dict[str, np.ndarray[np.str_]]
    chain_rep_map: dict[str, str]
    num_cols: dict[str, int]


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
        Msa: A Msa object containing the sequences, deletion matrix and metadata.
    """

    sequences, metadata = parse_fasta(msa_string)
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

    parsed_msa = Msa(msa=msa, deletion_matrix=deletion_matrix, metadata=metadata)

    # Crop the MSA
    if max_seq_count is not None:
        parsed_msa.truncate(max_seq_count)

    return parsed_msa


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
        Msa: A Msa object containing the sequences, deletion matrix and metadata.
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
    metadata = list(name_to_sequence.keys())

    parsed_msa = Msa(msa=msa, deletion_matrix=deletion_matrix, metadata=metadata)

    # Crop the MSA
    if max_seq_count is not None:
        parsed_msa.truncate(max_seq_count)

    return parsed_msa


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
            textwrap.dedent(
                "No alignments found in {folder_path}. Folders for chains"
                "without any aligned sequences need to contain at least one"
                ".sto file with only the query sequence."
            )
        )
    else:
        for file_name in file_names:
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
            with open(os.path.join(folder_path, file_name)) as f:
                msas[basename] = MSA_PARSER_REGISTRY[ext](
                    f.read(), max_seq_counts[basename]
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


def parse_msas_sample(
    chain_ids: list[list[str], list[str]],
    alignments_path: str,
    use_alignment_database: bool,
    alignment_index: Optional[dict] = None,
    max_seq_counts: Optional[dict[str, int]] = None,
) -> MsaCollection:
    """Parses MSA(s) for a training sample.

    This function is used to parse MSAs for a one or multiple chains, depending on the
    number of chains in the parsed PDB file and crop during training.

    Args:
        chain_ids (list[tuple[str, str]]):
            Two lists of chain IDs and representative chain IDs to parse for a sample.
        alignments_path (str):
            Path to the lowest-level directory containing either the directories of MSAs
            per chain ID or the alignment databases.
        use_alignment_database (bool):
            Whether to use the alignment database instead of directly parsing the
            alignment files.
        alignment_index (Optional[dict], optional):
            Dictionary containing the alignment index.
        max_seq_counts (Optional[dict[str, int]], optional):
            Dictionary mapping the sequence database from which sequence hits were
            returned to the max number of sequences to parse from all the hits. See
            Section 2.2, Tables 1 and 2 in the AlphaFold3 SI for more details. This
            dict, when provided, is used to specific a) which alignment files to parse
            and b) the maximum number of sequences to parse.

    Returns:
        MsaCollection:
            A collection of Msa objects and chain IDs for a single sample.
    """
    # Parse MSAs for each representative ID
    # This requires parsing MSAs for duplicate chains only once
    representative_chain_ids = list(set(chain_ids[1]))
    representative_msas = {}
    for rep_id in representative_chain_ids:
        if use_alignment_database:
            representative_msas[rep_id] = parse_msas_alignment_database(
                alignment_index_entry=alignment_index[rep_id],
                alignment_database_path=alignments_path,
                max_seq_counts=max_seq_counts,
            )
        else:
            representative_msas[rep_id] = parse_msas_direct(
                folder_path=os.path.join(alignments_path, rep_id),
                max_seq_counts=max_seq_counts,
            )

    # Reindex the parsed MSAs to the original chain IDs and calculate Msa length and
    # pull out the query sequence
    rep_msa_map, rep_seq_map, chain_rep_map, num_cols= {}, {}, {}, {}

    for chain_id, rep_id in zip(chain_ids[0], chain_ids[1]):
        chain_rep_map[chain_id] = rep_id
        all_msas_per_chain = representative_msas[rep_id]
        example_msa = all_msas_per_chain[next(iter(all_msas_per_chain))].msa
        if rep_id not in rep_msa_map:
            rep_msa_map[rep_id] = all_msas_per_chain
            rep_seq_map[rep_id] = example_msa[0, :]
            num_cols[rep_id] = example_msa.shape[1]

    return MsaCollection(
        rep_msa_map=rep_msa_map,
        rep_seq_map=rep_seq_map,
        chain_rep_map=chain_rep_map,
        num_cols=num_cols,
    )
