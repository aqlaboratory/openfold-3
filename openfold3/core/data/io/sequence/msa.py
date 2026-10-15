# Copyright 2026 AlQuraishi Laboratory
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

"""This module contains IO functions for reading and writing MSA files."""

import os
import string
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path

import numpy as np

from openfold3.core.config.config_utils import DirectoryPathOrNone
from openfold3.core.config.msa_pipeline_configs import (
    MsaKind,
    MsaSampleProcessorInput,
    MsaSampleProcessorInputInference,
    MsaSampleProcessorInputTrain,
)
from openfold3.core.data.io.sequence.fasta import parse_fasta
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.quality_control.msa_issues import (
    MsaIssue,
    MsaIssueKind,
    report_msa_issues,
)
from openfold3.core.data.primitives.sequence.msa import (
    MsaArray,
    MsaArrayCollection,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings


def standardize_filepaths(input_path: Path | list[Path]) -> list[Path]:
    """Standardizes and expands input paths.

    Converts
        - a directory path to a list of file paths the directory contains
        - a file path to a list containing the file path itself
        - a list of directory paths to a list of file paths the directories contain
          at depth=1
        - a list of file paths to a list of file paths
    """
    # DirPath -> [FilePaths]
    if isinstance(input_path, Path) and input_path.is_dir():
        return [p for p in list(input_path.iterdir()) if p.is_file()]
    # FilePath -> [FilePath]
    elif isinstance(input_path, Path) and input_path.is_file():
        return [input_path]
    # [DirPaths, FilePaths] -> [FilePaths]
    elif isinstance(input_path, list):
        input_path_files = []
        for p in input_path:
            if p.is_file():
                input_path_files.append(p)
            elif p.is_dir():
                input_path_files.extend(
                    [p_i for p_i in list(p.iterdir()) if p_i.is_file()]
                )
        return input_path_files
    else:
        raise ValueError(
            f"Input path {input_path} must be a Path to a file, a directory or a list "
            f"of file/directory Paths but got {type(input_path)}: {input_path}."
        )


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


def parse_a3m(msa_string: str, max_seq_count: int | None = None) -> MsaArray:
    """Parses sequences and deletion matrix from a3m format alignment.

    This function needs to be wrapped in a with open call to read the file.

    Args:
        msa_string (str):
            The string contents of a a3m file. The first sequence in the file
            should be the query sequence.
        max_seq_count (int | None):
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

    parsed_msa = MsaArray.from_parsed(
        msa=msa, deletion_matrix=deletion_matrix, metadata=metadata
    )

    # Crop the MSA
    if max_seq_count is not None:
        parsed_msa.truncate(max_seq_count, inplace=True)

    return parsed_msa


def parse_stockholm(
    msa_string: str, max_seq_count: int | None = None, gap_symbols: set | None = None
) -> MsaArray:
    """Parses sequences and deletion matrix from stockholm format alignment.

    This function needs to be wrapped in a with open call to read the file.

    Args:
        msa_string (str):
            The string contents of a stockholm file. The first sequence in the file
            should be the query sequence.
        max_seq_count (int | None):
            The maximum number of sequences to parse from the file.
        gap_symbols (set | None):
            Set of symbols that are considered as gaps in the alignment. When None,
            defaults to {"-", "."}.

    Returns:
        Msa: A Msa object containing the sequences, deletion matrix and metadata.
    """

    if gap_symbols is None:
        gap_symbols = set(["-", "."])

    # Parse each line into header: sequence dictionary
    name_to_sequence = OrderedDict()
    for line in msa_string.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        name, sequence = line.split()
        if name not in name_to_sequence:
            # Add header to dictionary
            name_to_sequence[name] = ""
        # Extend sequence
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
            keep_columns = [i for i, res in enumerate(query) if res not in gap_symbols]

        # Remove the columns with gaps in the query from all sequences.
        aligned_sequence = "".join([sequence[c] for c in keep_columns])

        msa.append(aligned_sequence)

        # Count the number of deletions w.r.t. query.
        deletion_vec = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query, strict=True):
            if seq_res not in gap_symbols or query_res not in gap_symbols:
                if query_res in gap_symbols:
                    deletion_count += 1
                else:
                    deletion_vec.append(deletion_count)
                    deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Embed in numpy array
    msa = _msa_list_to_np(msa)
    deletion_matrix = np.array(deletion_matrix)
    metadata = list(name_to_sequence.keys())

    parsed_msa = MsaArray.from_parsed(
        msa=msa, deletion_matrix=deletion_matrix, metadata=metadata
    )

    # Crop the MSA
    if max_seq_count is not None:
        parsed_msa.truncate(max_seq_count, inplace=True)

    return parsed_msa


MSA_PARSER_REGISTRY = {".a3m": parse_a3m, ".sto": parse_stockholm}


def parse_msas_direct(
    file_list: list[Path], max_seq_counts: dict[str, int] | None = None
) -> dict[str, MsaArray]:
    """Parses a set of MSA files (a3m or sto) into a dictionary of Msa objects.

    This function is used to parse MSAs for a single chain.

    Args:
        folder_path (Path | list[Path]):
            A list of paths to the MSA files to parse.
        max_seq_counts (dict[str, int] | None):
            A map from file names to maximum sequences to keep from the corresponding
            MSA file. The set of keys in this dict is also used to parse only a subset
            of the files in the folder with the corresponding names.

    Returns:
        dict[str: Msa]: A dict containing the parsed MSAs.
    """

    msas = {}

    if len(file_list) == 0:
        raise RuntimeError(
            f"No alignments found in {file_list}. Folders for chains"
            "without any aligned sequences need to contain at least one"
            ".sto file with only the query sequence."
        )
    else:
        for aln_file in file_list:
            if aln_file.is_dir():
                warnings.warn(
                    f"Skipping directory {aln_file} in {file_list}. When a list of "
                    "paths is provided, only files are parsed. If you want to parse "
                    "all files in a directory, use the parse_msas_direct function "
                    "with a directory path instead of a list of file paths.",
                    stacklevel=2,
                )

            # Split extensions from the filenames
            basename, ext = aln_file.stem, aln_file.suffix
            if ext not in [".sto", ".a3m"]:
                warnings.warn(
                    f"Found file {basename}.{ext} with an unsupported extension in "
                    f"{file_list}. Only .sto and .a3m files are supported for direct "
                    "MSA parsing.",
                    stacklevel=2,
                )
                continue

            # Only include files with specified max values in the max_seq_counts dict
            if (max_seq_counts is not None) and (basename not in max_seq_counts):
                continue

            # Parse the MSAs with the appropriate parser
            limit = None if max_seq_counts is None else max_seq_counts.get(basename)
            with open(aln_file.absolute()) as f:
                msas[basename] = MSA_PARSER_REGISTRY[ext](f.read(), limit)

    return msas


def parse_msas_alignment_database(
    alignment_index_entry: dict,
    alignment_database_path: Path,
    max_seq_counts: dict[str, int] | None = None,
) -> dict[str, MsaArray]:
    """Parses an entry from an alignment database into a dictionary of Msa objects.

    This function is used to parse MSAs for a single chain.

    Args:
        alignment_index_entry (dict):
            A subdictionary of the alignment index dictionary, indexing a specific
            chain.
        alignment_database_path (Path):
            Path to the lowest-level directory containing the alignment databases.
        max_seq_count (dict[str, int] | None):
            A map from file names to maximum sequences to keep from the corresponding
            MSA file. The set of keys in this dict is also used to parse only a subset
            of the files in the folder with the corresponding names.

    Returns:
        dict[str: Msa]: A dict containing the parsed MSAs.
    """
    msas = {}

    with open(
        (alignment_database_path.absolute() / Path(alignment_index_entry["db"])), "rb"
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
                warnings.warn(
                    f"Found unsupported file type {ext} in {alignment_database_path}. "
                    "Only .sto and .a3m files are supported for alignment database "
                    "parsing.",
                    stacklevel=2,
                )
                continue

            # Only include files with specified max values in the max_seq_counts dict
            if max_seq_counts is not None and basename not in max_seq_counts:
                continue

            # Parse the MSAs with the appropriate parser
            limit = None if max_seq_counts is None else max_seq_counts.get(basename)
            msas[basename] = MSA_PARSER_REGISTRY[ext](read_msa(start, size), limit)
    return msas


def parse_msas_preparsed(
    file_list: list[Path],
) -> dict[str, MsaArray]:
    """Parses a pre-parsed .npz file into a dictionary of Msa objects.

    This function is used to parse MSAs for a single chain. If a list of npz files is
    provided, where each file contains a dict of MSA arrays, the function will
    concatenate the MSAs from all dicts into a single dict, so for repeating keys, only
    the last MSA will be kept.

    Args:
        file_list (list[Path]):
            Path a list of npz files pre-parsed using
            openfold3.scripts.data_preprocessing.preparse_alginments_of3.

    Returns:
        dict[str, MsaArray]:
            A dict containing the parsed MSAs.
    """
    msas = {}

    for aln_file in file_list:
        # Parse npz file
        with np.load(aln_file, allow_pickle=True) as pre_parsed_msas:
            # Unpack the pre-parsed MSA arrays into a dict of MsaArrays
            for k in list(pre_parsed_msas.keys()):
                unpacked_msas = pre_parsed_msas[k].item()
                if k in msas:
                    warnings.warn(
                        f"Found duplicate key {k} in {aln_file}. Only the last "
                        "MSA will be kept.",
                        stacklevel=2,
                    )
                msas[k] = MsaArray.from_parsed(
                    msa=unpacked_msas["msa"],
                    deletion_matrix=unpacked_msas["deletion_matrix"],
                    metadata=unpacked_msas["metadata"],
                )

    return msas


class MsaSampleParserMapper:
    """Data class to hold mappings between chain IDs, representative IDs, and MSA
    objects for MSA parsing."""

    map_keys: tuple = (
        "chain_id_to_rep_id",
        "chain_id_to_mol_type",
        "rep_id_to_chain_id",
        "rep_id_to_mol_type",
        "rep_id_to_main_msa_paths",
        "rep_id_to_paired_msa_paths",
        "rep_id_to_query_seq",
        "rep_id_to_main_msa",
        "rep_id_to_paired_msa",
    )

    def __init__(self) -> None:
        for name in self.map_keys:
            setattr(self, name, {})


class MsaSampleParser:
    """Base MSA sample parser class"""

    def __init__(
        self,
        config: MSASettings,
        *,
        alignment_array_directory: DirectoryPathOrNone = None,
        alignment_db_directory: DirectoryPathOrNone = None,
        alignment_index: dict | None = None,
        alignments_directory: DirectoryPathOrNone = None,
        use_roda_monomer_format: bool = False,
    ):
        self.config = config
        self.alignment_array_directory = alignment_array_directory
        self.alignment_db_directory = alignment_db_directory
        self.alignment_index = alignment_index
        self.alignments_directory = alignments_directory
        self.use_roda_monomer_format = use_roda_monomer_format

    def create_maps(self) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleParser directly. Subclass it and "
            "implement create_maps and parse_msas methods to use it."
        )

    def parse_msas(self) -> None:
        raise NotImplementedError(
            "You are trying to use the MsaSampleParser directly. Subclass it and "
            "implement create_maps and parse_msas methods to use it."
        )

    def create_msa_array_collection(
        self, maps: MsaSampleParserMapper
    ) -> MsaArrayCollection:
        # Set msa collection to parsed, will be empty if no protein or RNA chains
        msa_array_collection = MsaArrayCollection(
            chain_id_to_rep_id=maps.chain_id_to_rep_id,
            chain_id_to_mol_type=maps.chain_id_to_mol_type,
            rep_id_to_chain_id=maps.rep_id_to_chain_id,
            rep_id_to_mol_type=maps.rep_id_to_mol_type,
        )
        msa_array_collection.set_state_parsed(
            rep_id_to_query_seq=maps.rep_id_to_query_seq,
            rep_id_to_paired_msa=maps.rep_id_to_paired_msa,
            rep_id_to_main_msa=maps.rep_id_to_main_msa,
        )
        return msa_array_collection

    @log_runtime_memory(runtime_dict_key="runtime-msa-proc-parse")
    def __call__(self, input: MsaSampleProcessorInput) -> MsaArrayCollection:
        # Create maps between chain IDs, representative IDs, molecule types
        maps = self.create_maps(input=input)

        # Parse MSAs for each representative ID
        maps = self.parse_msas(maps=maps)

        # Collect data into MsaArrayCollection
        msa_array_collection = self.create_msa_array_collection(maps)

        # Init row counts
        msa_array_collection.set_row_counts(0, 0, {})

        return msa_array_collection


class MsaSampleParserTrain(MsaSampleParser):
    """Training MSA sample parser class"""

    def __init__(self, config: MSASettings, **paths_kwargs):
        super().__init__(config=config, **paths_kwargs)

    def create_maps(self, input: MsaSampleProcessorInputTrain) -> MsaSampleParserMapper:
        """Populates the chain_id_to_rep_id and chain_id_to_mol_type maps.

        Updates the following attributes:
            - chain_id_to_rep_id: dict
                Maps chain IDs to representative IDs.
            - chain_id_to_mol_type: dict
                Maps chain IDs to molecule types.
            - rep_id_to_chain_id: dict
                Maps representative IDs to an example chain ID.
            - rep_id_to_mol_type: dict
                Maps representative IDs to molecule types.
        """
        maps = MsaSampleParserMapper()
        for chain_id, chain_data in input.msa_chain_data.items():
            if chain_data.molecule_type in self.config.moltypes:
                rep_id = chain_data.alignment_representative_id
                maps.chain_id_to_rep_id[chain_id] = rep_id
                maps.chain_id_to_mol_type[chain_id] = chain_data.molecule_type
                if rep_id not in maps.rep_id_to_chain_id:
                    maps.rep_id_to_chain_id[rep_id] = chain_id
                if rep_id not in maps.rep_id_to_mol_type:
                    maps.rep_id_to_mol_type[rep_id] = chain_data.molecule_type
        return maps

    def parse_msas(self, maps: MsaSampleParserMapper) -> MsaSampleParserMapper:
        """Parses MSAs for each representative chain.


        Updates the following attributes:
            - rep_id_to_query_seq: dict
                Maps representative IDs to query sequences.
            - rep_id_to_main_msa: dict
                Maps representative IDs to main MSA objects.
            - rep_id_to_paired_msa: dict
                Maps representative IDs to paired MSA objects.
        """
        # Parse MSAs for each representative ID
        if len(maps.chain_id_to_rep_id) > 0:
            # Parse MSAs for each representative ID
            representative_chain_ids = sorted(set(maps.chain_id_to_rep_id.values()))
            for rep_id in representative_chain_ids:
                if self.alignment_array_directory is not None:
                    if self.use_roda_monomer_format:
                        pre_parsed = (
                            self.alignment_array_directory / rep_id / "alignment.npz"
                        )
                    else:
                        pre_parsed = self.alignment_array_directory / f"{rep_id}.npz"
                    file_list = standardize_filepaths(pre_parsed)
                    all_msas_per_chain = parse_msas_preparsed(file_list=file_list)
                elif self.alignment_db_directory is not None:
                    all_msas_per_chain = parse_msas_alignment_database(
                        alignment_index_entry=self.alignment_index[rep_id],
                        alignment_database_path=self.alignment_db_directory,
                        max_seq_counts=self.config.max_seq_counts,
                    )
                else:
                    file_list = standardize_filepaths(
                        self.alignments_directory / Path(rep_id)
                    )
                    all_msas_per_chain = parse_msas_direct(
                        file_list=file_list,
                        max_seq_counts=self.config.max_seq_counts,
                    )
                maps.rep_id_to_main_msa[rep_id] = all_msas_per_chain

                # Create query sequence from the first row
                maps.rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                    sorted(all_msas_per_chain.keys())[0]
                ].msa[0, :][np.newaxis, :]

            # Parsing precomputed paired MSAs is not currently supported for training

        return maps


def _find_alignment_names(paths: list[Path]) -> dict[str, Path]:
    """Finds the alignments stored at a chain's MSA paths, without parsing them.

    Alignments are named after the file stem for .sto/.a3m files and after the
    array name for .npz files. Files with other extensions are skipped.

    Args:
        paths (list[Path]):
            MSA files, or directories containing them.

    Returns:
        dict[str, Path]:
            Maps alignment names to the file they are stored in.
    """
    names = {}
    for path in standardize_filepaths(paths):
        if path.suffix in MSA_PARSER_REGISTRY:
            names[path.stem] = path
        elif path.suffix == ".npz":
            # Only reads the archive's index, not the arrays
            with np.load(path, allow_pickle=True) as npz:
                names.update(dict.fromkeys(npz.files, path))
    return names


def select_query_alignment(msas: dict[str, MsaArray], preferred: list[str]) -> str:
    """Selects the alignment whose first row is used as a chain's query sequence:
    the first of `preferred` present in `msas`, else the alphabetically first.

    Args:
        msas (dict[str, MsaArray]):
            A chain's parsed alignments, by name.
        preferred (list[str]):
            Names of the alignments the MSA settings use, in order of preference.
    """
    return next((name for name in preferred if name in msas), min(msas))


def chains_of_rep(chain_id_to_rep_id: dict[str, str], rep_id: str) -> tuple[str, ...]:
    """The IDs of the chains that share representative `rep_id`."""
    return tuple(
        chain_id
        for chain_id, chain_rep_id in chain_id_to_rep_id.items()
        if chain_rep_id == rep_id
    )


@dataclass(frozen=True)
class _AlignmentSelection:
    """The MSA settings that select the alignments of one kind of MSA.

    Attributes:
        required_setting (str):
            The name of the setting listing the alignments this kind of MSA is made
            from, at least one of which must be read when it is used.
        required_names (list[str]):
            The alignment names listed by `required_setting`.
        optional_setting (str | None):
            The name of a setting listing further alignments this kind of MSA
            uses, if any.
        optional_names (list[str]):
            The alignment names listed by `optional_setting`.
    """

    required_setting: str
    required_names: list[str]
    optional_setting: str | None = None
    optional_names: list[str] = field(default_factory=list)


class _AlignmentRole(Enum):
    """What an alignment is used for in one kind of MSA."""

    # Named in the setting the MSA is made from, e.g. aln_order
    REQUIRED = "required"
    # Only named in a setting the MSA also uses, e.g. msas_to_pair
    OPTIONAL = "optional"
    # Named in neither, so not used
    UNSELECTED = "unselected"


class _UnreadReason(Enum):
    """Why an alignment found in a chain's MSA files is not read."""

    # Raw .sto/.a3m files are only read if named in max_seq_counts
    NOT_IN_MAX_SEQ_COUNTS = "not_in_max_seq_counts"
    # A max_seq_counts entry of 0 disables an alignment in any format
    DISABLED = "disabled"


@dataclass(frozen=True)
class _AlignmentStatus:
    """What happens to one alignment found in a chain's MSA files, and why.

    Attributes:
        name (str):
            The alignment name, see `_find_alignment_names`.
        role (_AlignmentRole):
            What the alignment is used for.
        unread_reason (_UnreadReason | None):
            Why the alignment is not read, or None if it is read.
    """

    name: str
    role: _AlignmentRole
    unread_reason: _UnreadReason | None

    @property
    def is_read(self) -> bool:
        return self.unread_reason is None


def _why_ignored(
    status: _AlignmentStatus, selection: _AlignmentSelection
) -> str | None:
    """Why an alignment found in a chain's MSA files is not used, or None if it is
    used or was deliberately disabled."""
    if status.unread_reason is _UnreadReason.DISABLED:
        return None
    if status.role is _AlignmentRole.UNSELECTED:
        settings = " or ".join(
            setting
            for setting in [selection.required_setting, selection.optional_setting]
            if setting is not None
        )
        return f"not in {settings}"
    if status.unread_reason is _UnreadReason.NOT_IN_MAX_SEQ_COUNTS:
        return "not in max_seq_counts"
    return None


def _unread_notes(statuses: list[_AlignmentStatus], roles: set[_AlignmentRole]) -> str:
    """Explains why alignments with one of `roles` are not read, for an error
    message."""
    not_in_max_seq_counts = sorted(
        status.name
        for status in statuses
        if status.unread_reason is _UnreadReason.NOT_IN_MAX_SEQ_COUNTS
        and status.role in roles
    )
    disabled = sorted(
        status.name
        for status in statuses
        if status.unread_reason is _UnreadReason.DISABLED and status.role in roles
    )
    notes = ""
    if not_in_max_seq_counts:
        notes += (
            "; .sto/.a3m alignments must also be named in max_seq_counts, which "
            f"{not_in_max_seq_counts} are not"
        )
    if disabled:
        notes += f"; {disabled} are disabled by a max_seq_counts of 0"
    return notes


class MsaSampleParserInference(MsaSampleParser):
    """Inference MSA sample parser class"""

    def __init__(self, config: MSASettings, **paths_kwargs):
        super().__init__(config=config, **paths_kwargs)

    def create_maps(
        self, input: MsaSampleProcessorInputInference
    ) -> MsaSampleParserMapper:
        """Creates maps between chain IDs, representative IDs, and molecule types.

        Chains with the same molecule type and sequence share one representative,
        named after the first such chain, and are featurized from the MSA files of
        that chain. MSA file and directory names play no part in this, so they can
        be named arbitrarily. A chain without a sequence is its own representative.

        Updates the following attributes:
            - chain_id_to_rep_id: dict
                Maps chain IDs to representative IDs.
            - chain_id_to_mol_type: dict
                Maps chain IDs to molecule types.
            - rep_id_to_main_msa_paths: dict
                Maps representative IDs to main MSA file paths.
            - rep_id_to_paired_msa_paths: dict
                Maps representative IDs to paired MSA file paths.

        Problems with the chains' MSA files, including those found by
        `check_alignment_names`, are warned about or raised together.

        Raises:
            MsaValidationError:
                If any representative's MSA files are unusable, see
                `check_alignment_names`.
        """
        # Create maps
        maps = MsaSampleParserMapper()
        issues: list[MsaIssue] = []
        sequence_to_rep_id: dict[tuple[MoleculeType, str], str] = {}
        chain_ids_without_msas: list[str] = []
        for chain_id, chain_data in input.msa_chain_data.items():
            if chain_data.molecule_type in self.config.moltypes:
                main_msa_file_paths = (
                    sorted(chain_data.main_msa_file_paths)
                    if chain_data.main_msa_file_paths
                    else []
                )
                paired_msa_file_paths = (
                    sorted(chain_data.paired_msa_file_paths)
                    if chain_data.paired_msa_file_paths
                    else []
                )

                if len(main_msa_file_paths) == 0 and len(paired_msa_file_paths) == 0:
                    # Warned about below, once all chains with MSAs are known
                    chain_ids_without_msas.append(chain_id)
                    continue

                # Fetch representative ID
                if chain_data.sequence is None:
                    rep_id = chain_id
                else:
                    rep_id = sequence_to_rep_id.setdefault(
                        (chain_data.molecule_type, chain_data.sequence), chain_id
                    )

                # Identical sequences pointing at different MSA files: only the
                # representative's files are read. The same files given differently,
                # e.g. as a directory or as relative paths, are not different.
                for msa_kind, rep_id_to_paths, paths in [
                    (MsaKind.MAIN, maps.rep_id_to_main_msa_paths, main_msa_file_paths),
                    (
                        MsaKind.PAIRED,
                        maps.rep_id_to_paired_msa_paths,
                        paired_msa_file_paths,
                    ),
                ]:
                    rep_paths = rep_id_to_paths.get(rep_id)
                    if (
                        rep_paths is not None
                        and paths
                        and not self._same_alignment_files(paths, rep_paths)
                    ):
                        issues.append(
                            MsaIssue(
                                kind=MsaIssueKind.DIFFERENT_FILES_SAME_SEQUENCE,
                                query_name=input.query_name,
                                chain_ids=(rep_id, chain_id),
                                msa_kind=msa_kind,
                                details={
                                    "rep_chain": rep_id,
                                    "chain": chain_id,
                                    "rep_paths": [str(p) for p in rep_paths],
                                    "paths": [str(p) for p in paths],
                                },
                            )
                        )

                maps.chain_id_to_rep_id[chain_id] = rep_id
                maps.chain_id_to_mol_type[chain_id] = chain_data.molecule_type
                if (rep_id not in maps.rep_id_to_chain_id) & (
                    len(main_msa_file_paths) > 0
                ):
                    maps.rep_id_to_chain_id[rep_id] = chain_id
                if (rep_id not in maps.rep_id_to_mol_type) & (
                    len(main_msa_file_paths) > 0
                ):
                    maps.rep_id_to_mol_type[rep_id] = chain_data.molecule_type
                if (rep_id not in maps.rep_id_to_main_msa_paths) & (
                    len(main_msa_file_paths) > 0
                ):
                    maps.rep_id_to_main_msa_paths[rep_id] = main_msa_file_paths
                if (rep_id not in maps.rep_id_to_paired_msa_paths) & (
                    len(paired_msa_file_paths) > 0
                ):
                    maps.rep_id_to_paired_msa_paths[rep_id] = paired_msa_file_paths

        for chain_id in chain_ids_without_msas:
            chain_data = input.msa_chain_data[chain_id]
            # A copy of a sequence that has MSAs elsewhere in the query is most
            # likely missing its paths by mistake
            copy_with_msas = (
                sequence_to_rep_id.get((chain_data.molecule_type, chain_data.sequence))
                if chain_data.sequence is not None
                else None
            )
            copy_note = (
                f" Chain {copy_with_msas} has the same sequence and does have MSA "
                f"files; to use them for chain {chain_id} too, list both chain IDs "
                "in one chain entry or give both entries the same MSA file paths."
                if copy_with_msas is not None
                else ""
            )
            issues.append(
                MsaIssue(
                    kind=MsaIssueKind.CHAIN_WITHOUT_MSAS,
                    query_name=input.query_name,
                    chain_ids=(chain_id,),
                    details={
                        "molecule_type": chain_data.molecule_type.name,
                        "copy_note": copy_note,
                    },
                )
            )

        issues += self.check_alignment_names(input=input, maps=maps)
        report_msa_issues(input.query_name, issues)

        return maps

    def check_alignment_names(
        self, input: MsaSampleProcessorInputInference, maps: MsaSampleParserMapper
    ) -> list[MsaIssue]:
        """Checks that each representative's MSA files hold alignments that the MSA
        settings use, so misnamed files fail early with an actionable message.

        Alignments are selected by name (see `_find_alignment_names`): .sto/.a3m
        files are only read if named in `max_seq_counts`, a `max_seq_counts` entry
        of 0 disables an alignment, main MSAs are stacked from the alignments named
        in `aln_order`, online pairing uses those named in `msas_to_pair`, and
        precomputed paired MSAs are stacked from those named in `paired_msa_order`.
        Each alignment found is classified once, see `_classify_alignments`, and
        the rules below follow from that:

            - NO_USABLE_ALIGNMENT: for each kind of MSA the query uses, at least
              one alignment the MSA is made from must be read.
            - IGNORED_ALIGNMENTS: for each kind of MSA the query uses, alignments
              that are found but not used are reported.
            - NO_READABLE_ALIGNMENT: for a kind of MSA the query does not use, any
              alignment must be read if a representative's query sequence is taken
              from it.
            - UNPAIRABLE_CHAINS: protein chains of a heteromer that online pairing
              cannot pair are reported.

        Args:
            input (MsaSampleProcessorInputInference):
                The query's MSA input, whose `use_*` flags say which kinds of MSA
                are used.
            maps (MsaSampleParserMapper):
                Maps from `create_maps`, with each representative's MSA paths.

        Returns:
            list[MsaIssue]:
                The problems found, to be passed to `report_msa_issues`.
        """
        issues = []
        rep_id_to_read_main: dict[str, set[str]] = {}
        rep_id_to_chain_ids: dict[str, tuple[str, ...]] = {}
        for msa_kind, rep_id_to_paths in [
            (MsaKind.MAIN, maps.rep_id_to_main_msa_paths),
            (MsaKind.PAIRED, maps.rep_id_to_paired_msa_paths),
        ]:
            selection = self._alignment_selection(msa_kind)
            uses_msa_kind = self._uses(input, msa_kind)
            for rep_id, paths in rep_id_to_paths.items():
                chain_ids = rep_id_to_chain_ids.setdefault(
                    rep_id, chains_of_rep(maps.chain_id_to_rep_id, rep_id)
                )
                statuses = self._classify_alignments(paths, selection)
                if msa_kind is MsaKind.MAIN:
                    rep_id_to_read_main[rep_id] = {
                        status.name for status in statuses if status.is_read
                    }
                problems = self._alignment_name_problems(
                    statuses,
                    selection,
                    uses_msa_kind=uses_msa_kind,
                    is_query_source=self._is_query_source(maps, rep_id, msa_kind),
                )
                issues += [
                    MsaIssue(
                        kind=kind,
                        query_name=input.query_name,
                        chain_ids=chain_ids,
                        msa_kind=msa_kind,
                        details={
                            "paths": [str(p) for p in paths],
                            "found": [status.name for status in statuses],
                            **details,
                        },
                    )
                    for kind, details in problems
                ]

        # Online pairing only runs without precomputed paired MSAs
        if self._uses(input, MsaKind.PAIRED) and not maps.rep_id_to_paired_msa_paths:
            issues += self._unpairable_chains(
                input, maps, rep_id_to_read_main, rep_id_to_chain_ids
            )

        return issues

    @staticmethod
    def _uses(input: MsaSampleProcessorInputInference, msa_kind: MsaKind) -> bool:
        """Whether the query uses MSAs of `msa_kind`."""
        use_msa_kind = (
            input.use_main_msas if msa_kind is MsaKind.MAIN else input.use_paired_msas
        )
        return input.use_msas and use_msa_kind

    @staticmethod
    def _is_query_source(
        maps: MsaSampleParserMapper, rep_id: str, msa_kind: MsaKind
    ) -> bool:
        """Whether a representative's query sequence is taken from its `msa_kind`
        MSAs: from the main MSAs, or the paired ones if there are no main MSAs."""
        return msa_kind is MsaKind.MAIN or rep_id not in maps.rep_id_to_main_msa_paths

    def _alignment_selection(self, msa_kind: MsaKind) -> _AlignmentSelection:
        """The MSA settings that select the alignments of `msa_kind` MSAs."""
        if msa_kind is MsaKind.MAIN:
            return _AlignmentSelection(
                required_setting="aln_order",
                required_names=self.config.aln_order,
                optional_setting="msas_to_pair",
                optional_names=self.config.msas_to_pair,
            )
        return _AlignmentSelection(
            required_setting="paired_msa_order",
            required_names=self.config.paired_msa_order,
        )

    def _classify_alignments(
        self, paths: list[Path], selection: _AlignmentSelection
    ) -> list[_AlignmentStatus]:
        """Finds the alignments at a representative's MSA paths of one kind, and
        what happens to each of them.

        Args:
            paths (list[Path]):
                MSA files, or directories containing them.
            selection (_AlignmentSelection):
                The settings selecting the alignments of this kind of MSA.

        Returns:
            list[_AlignmentStatus]:
                One status per alignment found, sorted by name.
        """
        statuses = []
        for name, path in sorted(_find_alignment_names(paths).items()):
            if name in selection.required_names:
                role = _AlignmentRole.REQUIRED
            elif name in selection.optional_names:
                role = _AlignmentRole.OPTIONAL
            else:
                role = _AlignmentRole.UNSELECTED

            if self._is_disabled_alignment(name):
                unread_reason = _UnreadReason.DISABLED
            elif path.suffix != ".npz" and name not in self.config.max_seq_counts:
                unread_reason = _UnreadReason.NOT_IN_MAX_SEQ_COUNTS
            else:
                unread_reason = None
            statuses.append(_AlignmentStatus(name, role, unread_reason))
        return statuses

    @staticmethod
    def _alignment_name_problems(
        statuses: list[_AlignmentStatus],
        selection: _AlignmentSelection,
        uses_msa_kind: bool,
        is_query_source: bool,
    ) -> list[tuple[MsaIssueKind, dict[str, object]]]:
        """Applies the rules of `check_alignment_names` to the alignments of one
        representative's MSA files of one kind.

        Args:
            statuses (list[_AlignmentStatus]):
                The alignments found in the files, see `_classify_alignments`.
            selection (_AlignmentSelection):
                The settings selecting the alignments of this kind of MSA.
            uses_msa_kind (bool):
                Whether the query uses this kind of MSA.
            is_query_source (bool):
                Whether the representative's query sequence is taken from these
                files.

        Returns:
            list[tuple[MsaIssueKind, dict[str, object]]]:
                The kind of each problem found, with the details of its message.
        """
        read = [status for status in statuses if status.is_read]
        if uses_msa_kind:
            if not any(status.role is _AlignmentRole.REQUIRED for status in read):
                notes = _unread_notes(
                    statuses,
                    roles={_AlignmentRole.REQUIRED, _AlignmentRole.OPTIONAL},
                )
                pairing_only = sorted(
                    status.name
                    for status in read
                    if status.role is _AlignmentRole.OPTIONAL
                )
                if pairing_only:
                    notes += (
                        f". {pairing_only} are only used as named in "
                        f"{selection.optional_setting}, e.g. for species pairing"
                    )
                return [
                    (
                        MsaIssueKind.NO_USABLE_ALIGNMENT,
                        {
                            "required_setting": selection.required_setting,
                            "required_names": selection.required_names,
                            "notes": notes,
                        },
                    )
                ]
            ignored = {
                status.name: reason
                for status in statuses
                if (reason := _why_ignored(status, selection)) is not None
            }
            if ignored:
                return [
                    (
                        MsaIssueKind.IGNORED_ALIGNMENTS,
                        {
                            "ignored": ", ".join(
                                f"{name} ({reason})"
                                for name, reason in sorted(ignored.items())
                            )
                        },
                    )
                ]
        elif is_query_source and not read:
            return [
                (
                    MsaIssueKind.NO_READABLE_ALIGNMENT,
                    {"notes": _unread_notes(statuses, roles=set(_AlignmentRole))},
                )
            ]
        return []

    def _unpairable_chains(
        self,
        input: MsaSampleProcessorInputInference,
        maps: MsaSampleParserMapper,
        rep_id_to_read_main: dict[str, set[str]],
        rep_id_to_chain_ids: dict[str, tuple[str, ...]],
    ) -> list[MsaIssue]:
        """Finds the protein chains of a heteromer that online pairing cannot pair,
        because none of their main alignments is named in msas_to_pair.

        Args:
            input (MsaSampleProcessorInputInference):
                The query's MSA input.
            maps (MsaSampleParserMapper):
                Maps from `create_maps`.
            rep_id_to_read_main (dict[str, set[str]]):
                The main alignments read for each representative.
            rep_id_to_chain_ids (dict[str, tuple[str, ...]]):
                The chains of each representative.

        Returns:
            list[MsaIssue]:
                An UNPAIRABLE_CHAINS issue if there are such chains, else nothing.
        """
        protein_rep_ids = [
            rep_id
            for rep_id in rep_id_to_read_main
            if maps.rep_id_to_mol_type[rep_id] == MoleculeType.PROTEIN
        ]
        if len(protein_rep_ids) < 2:
            return []
        unpairable = [
            rep_id
            for rep_id in protein_rep_ids
            if not rep_id_to_read_main[rep_id] & set(self.config.msas_to_pair)
        ]
        if not unpairable:
            return []
        consequence = (
            "so no species-paired MSA rows are generated for this complex"
            if len(protein_rep_ids) - len(unpairable) < 2
            else "so these chains get no species-paired MSA rows"
        )
        return [
            MsaIssue(
                kind=MsaIssueKind.UNPAIRABLE_CHAINS,
                query_name=input.query_name,
                chain_ids=tuple(
                    chain_id
                    for rep_id in unpairable
                    for chain_id in rep_id_to_chain_ids[rep_id]
                ),
                msa_kind=MsaKind.MAIN,
                details={
                    "msas_to_pair": self.config.msas_to_pair,
                    "consequence": consequence,
                },
            )
        ]

    def _is_disabled_alignment(self, name: str) -> bool:
        """Whether a max_seq_counts entry of 0 disables the alignment `name`.

        This applies to raw and preparsed .npz alignments alike, although other
        max_seq_counts entries only limit raw ones.
        """
        return self.config.max_seq_counts.get(name) == 0

    def _drop_disabled_alignments(
        self, msas: dict[str, MsaArray]
    ) -> dict[str, MsaArray]:
        """Drops the parsed alignments disabled by a max_seq_counts entry of 0,
        which would otherwise be left without rows."""
        return {
            name: msa
            for name, msa in msas.items()
            if not self._is_disabled_alignment(name)
        }

    @staticmethod
    def _same_alignment_files(paths: list[Path], other_paths: list[Path]) -> bool:
        """Whether two lists of MSA paths hold the same alignments in the same
        files, however the paths are given: as files or the directories
        containing them, in any order, and as relative or absolute paths.

        Args:
            paths (list[Path]):
                MSA files, or directories containing them.
            other_paths (list[Path]):
                MSA files, or directories containing them.
        """

        def alignment_files(paths: list[Path]) -> dict[str, Path]:
            return {
                name: path.resolve()
                for name, path in _find_alignment_names(paths).items()
            }

        return alignment_files(paths) == alignment_files(other_paths)

    def parse_msas(self, maps: MsaSampleParserMapper) -> MsaSampleParserMapper:
        """Parses MSAs for each representative chain and representative chain set.

        Updates the following attributes:
            - rep_id_to_query_seq: dict
                Maps representative IDs to query sequences.
            - rep_id_to_main_msa: dict
                Maps representative IDs to main MSA objects.
            - rep_id_to_paired_msa: dict
                Maps representative IDs to paired MSA objects.

        Raises:
            ValueError:
                If the MSA file paths are not in the expected format or if the MSA
                file paths are not supported.
        """

        if len(maps.chain_id_to_rep_id) > 0:
            # Parse MSAs for each representative
            representative_chain_ids = sorted(set(maps.chain_id_to_rep_id.values()))
            for rep_id in representative_chain_ids:
                # Parse main MSAs
                if rep_id in maps.rep_id_to_main_msa_paths:
                    example_path = maps.rep_id_to_main_msa_paths[rep_id][0]
                    if example_path.is_dir() or (
                        example_path.suffix in [".sto", ".a3m"]
                    ):
                        file_list = standardize_filepaths(
                            maps.rep_id_to_main_msa_paths[rep_id],
                        )
                        chain_msa_parser = partial(
                            parse_msas_direct,
                            max_seq_counts=self.config.max_seq_counts,
                        )
                    elif example_path.suffix == ".npz":
                        file_list = standardize_filepaths(
                            maps.rep_id_to_main_msa_paths[rep_id]
                        )
                        chain_msa_parser = parse_msas_preparsed
                    else:
                        raise ValueError(
                            f"Unsupported MSA path found {example_path}. Needs to be "
                            "one of the following: \n"
                            " - an .a3m or .sto file \n"
                            " - a directory containing .a3m or .sto files \n"
                            " - a .npz file \n"
                        )

                    # Parse MSAs into a dict of MsaArrays
                    all_msas_per_chain = self._drop_disabled_alignments(
                        chain_msa_parser(file_list=file_list)
                    )
                    maps.rep_id_to_main_msa[rep_id] = all_msas_per_chain

                    # Create query sequence from the first row of a used alignment
                    query_alignment = select_query_alignment(
                        all_msas_per_chain,
                        [*self.config.aln_order, *self.config.msas_to_pair],
                    )
                    maps.rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                        query_alignment
                    ].msa[0, :][np.newaxis, :]

                # Parse paired MSAs
                if rep_id in maps.rep_id_to_paired_msa_paths:
                    example_path = maps.rep_id_to_paired_msa_paths[rep_id][0]
                    if example_path.is_dir() or (
                        example_path.suffix in [".sto", ".a3m"]
                    ):
                        file_list = standardize_filepaths(
                            maps.rep_id_to_paired_msa_paths[rep_id],
                        )
                        chain_msa_parser = partial(
                            parse_msas_direct,
                            max_seq_counts=self.config.max_seq_counts,
                        )
                    elif example_path.suffix == ".npz":
                        file_list = standardize_filepaths(
                            maps.rep_id_to_paired_msa_paths[rep_id]
                        )
                        chain_msa_parser = parse_msas_preparsed
                    else:
                        raise ValueError(
                            f"Unsupported MSA path found {example_path}. Needs to be "
                            "one of the following: \n"
                            " - an .a3m or .sto file \n"
                            " - a directory containing .a3m or .sto files \n"
                            " - a .npz file \n"
                        )

                    # Parse MSAs into a dict of MsaArrays
                    all_msas_per_chain = self._drop_disabled_alignments(
                        chain_msa_parser(file_list=file_list)
                    )
                    maps.rep_id_to_paired_msa[rep_id] = all_msas_per_chain

                    if rep_id not in maps.rep_id_to_query_seq:
                        # Create query sequence from the first row of a used
                        # alignment
                        query_alignment = select_query_alignment(
                            all_msas_per_chain, self.config.paired_msa_order
                        )
                        maps.rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                            query_alignment
                        ].msa[0, :][np.newaxis, :]

        return maps
