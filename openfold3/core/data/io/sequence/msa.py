"""This module contains IO functions for reading and writing MSA files."""

import os
import string
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from biotite.structure import AtomArray

from openfold3.core.config.config_utils import DirectoryPathOrNone
from openfold3.core.config.msa_pipeline_configs import (
    MsaSampleProcessorInputInference,
    MsaSampleProcessorInputTrain,
)
from openfold3.core.data.io.sequence.fasta import parse_fasta
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
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

    parsed_msa = MsaArray(msa=msa, deletion_matrix=deletion_matrix, metadata=metadata)

    # Crop the MSA
    if max_seq_count is not None:
        parsed_msa.truncate(max_seq_count)

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

    parsed_msa = MsaArray(msa=msa, deletion_matrix=deletion_matrix, metadata=metadata)

    # Crop the MSA
    if max_seq_count is not None:
        parsed_msa.truncate(max_seq_count)

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
            with open(aln_file.absolute()) as f:
                msas[basename] = MSA_PARSER_REGISTRY[ext](
                    f.read(), max_seq_counts[basename]
                )

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
            msas[basename] = MSA_PARSER_REGISTRY[ext](
                read_msa(start, size), max_seq_counts[basename]
            )
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
        pre_parsed_msas = np.load(aln_file, allow_pickle=True)

        # Unpack the pre-parsed MSA arrays into a dict of MsaArrays
        for k in list(pre_parsed_msas.keys()):
            unpacked_msas = pre_parsed_msas[k].item()
            if k in msas:
                warnings.warn(
                    f"Found duplicate key {k} in {aln_file}. Only the last MSA will "
                    "be kept.",
                    stacklevel=2,
                )
            msas[k] = MsaArray(
                msa=unpacked_msas["msa"],
                deletion_matrix=unpacked_msas["deletion_matrix"],
                metadata=unpacked_msas["metadata"],
            )

    return msas


# Functional MSA parser for training - TMP: here for backwards compatibility but should
# be replaced by MsaSampleParserTrain
@log_runtime_memory(runtime_dict_key="runtime-msa-proc-parse")
def parse_msas_sample(
    atom_array: AtomArray,
    assembly_data: dict[str, dict[str, Any]],
    moltypes: list[str],
    alignments_directory: Path | None,
    alignment_db_directory: Path | None,
    alignment_index: dict | None,
    alignment_array_directory: Path | None,
    max_seq_counts: dict[str, int] | None,
    use_roda_monomer_format: bool = False,
) -> MsaArrayCollection:
    """Parses MSA(s) for a training sample.

    This function is used to parse MSAs for a single sample, which may be a single chain
    for monomers or multiple chains for assemblies.

    If multiple paths are provided (not set to None), the following is the priority
    order:
        1. alignment_array_directory (pre-parsed npz)
        2. alignment_db_directory + alignment_index (alignment database files)
        3. alignments_directory (raw a3m or sto files)

    Args:
        atom_array (AtomArray):
            The cropped (training) or full (inference) atom array.
        assembly_data (dict[str, dict[str, Any]]):
            Dict containing the alignment representatives and molecule types for each
            chain.
        moltypes (list[str]):
            List of molecule type strings to consider for MSA parsing.
        alignments_directory (Path | None):
            Path to the lowest-level directory containing the directories of MSAs per
            chain ID.
        alignment_db_directory (Path | None):
            Path to the directory containing the alignment database or its shards AND
            the alignment database superindex file. If provided, it is used over
            alignments_directory.
        alignment_index (dict | None):
            Dictionary containing the alignment index.
        alignment_array_directory (Path | None):
            Path to the directory containing the preprocessed alignment arrays.
        max_seq_counts (dict[str, int] | None):
            Dictionary mapping filename strings (without extension) to the max number of
            sequences to parse from the corresponding MSA file. Only alignment files
            whose names are keys in this dict will be parsed.
        use_roda_monomer_format (bool):
            Whether input data is expected to be in the s3 RODA monomer
            format: <aln_dir>/<mgy_id>/alignment.npz
    Returns:
        MsaArrayCollection:
            A collection of Msa objects and chain IDs for a single sample.
    """
    # Get subset of atom array for which alignments are supposed to be provided
    # based on the dataset config
    # TODO refactor to get the list of chains that need MSAs from the assembly_data
    # dictionary instead of the atom array - won't need atom_array as input
    atom_array_with_alignments = atom_array[
        np.isin(
            atom_array.molecule_type_id,
            moltypes,
        )
    ]

    # Map chain IDs to representative IDs and molecule types
    chain_id_to_rep_id, chain_id_to_mol_type = {}, {}
    for chain_id_in_atom_array in sorted(set(atom_array_with_alignments.chain_id)):
        chain_data = assembly_data[chain_id_in_atom_array]
        chain_id_to_rep_id[chain_id_in_atom_array] = chain_data[
            "alignment_representative_id"
        ]
        chain_id_to_mol_type[chain_id_in_atom_array] = MoleculeType[
            chain_data["molecule_type"]
        ]

    # Parse MSAs for each representative ID
    rep_id_to_msa, rep_id_to_query_seq = {}, {}
    if len(chain_id_to_rep_id) > 0:
        # Parse MSAs for each representative ID
        # This requires parsing MSAs for duplicate chains only once
        representative_chain_ids = sorted(set(chain_id_to_rep_id.values()))
        representative_msas = {}
        for rep_id in representative_chain_ids:
            if alignment_array_directory is not None:
                if use_roda_monomer_format:
                    pre_parsed_msas = (
                        alignment_array_directory / rep_id / "alignment.npz"
                    )
                else:
                    pre_parsed_msas = alignment_array_directory / f"{rep_id}.npz"

                file_list = standardize_filepaths(pre_parsed_msas)
                representative_msas[rep_id] = parse_msas_preparsed(
                    file_list=file_list,
                )
            elif alignment_db_directory is not None:
                representative_msas[rep_id] = parse_msas_alignment_database(
                    alignment_index_entry=alignment_index[rep_id],
                    alignment_database_path=alignment_db_directory,
                    max_seq_counts=max_seq_counts,
                )
            else:
                file_list = standardize_filepaths(alignments_directory / Path(rep_id))
                representative_msas[rep_id] = parse_msas_direct(
                    file_list=file_list,
                    max_seq_counts=max_seq_counts,
                )

        # Pull out the query sequence from the first row of the first MSA of each chain
        for rep_id, all_msas_per_chain in representative_msas.items():
            example_msa = all_msas_per_chain[sorted(all_msas_per_chain.keys())[0]].msa
            rep_id_to_msa[rep_id] = all_msas_per_chain
            rep_id_to_query_seq[rep_id] = example_msa[0, :][np.newaxis, :]

    # Set msa collection to parsed, will be empty if no protein or RNA chains
    msa_array_collection = MsaArrayCollection(
        chain_id_to_rep_id=chain_id_to_rep_id,
        chain_id_to_mol_type=chain_id_to_mol_type,
    )
    msa_array_collection.set_state_parsed(
        rep_id_to_main_msa=rep_id_to_msa, rep_id_to_query_seq=rep_id_to_query_seq
    )
    return msa_array_collection


# Functional MSA parser for inference - TMP
def parse_msas_sample_inference(
    input: MsaSampleProcessorInputInference,
    moltypes: list[str],
    max_seq_counts: dict[str, int] | None,
) -> MsaArrayCollection:
    # Create maps
    chain_id_to_rep_id, chain_id_to_mol_type = {}, {}
    rep_id_to_main_msa_paths, rep_id_to_paired_msa_paths = {}, {}
    for chain_id, chain_data in input.msa_data.items():
        if chain_data["molecule_type"] in moltypes:
            main_msa_file_paths = (
                sorted(chain_data["main_msa_file_paths"])
                if chain_data["main_msa_file_paths"]
                else []
            )
            paired_msa_file_paths = (
                sorted(chain_data["paired_msa_file_paths"])
                if chain_data["paired_msa_file_paths"]
                else []
            )

            # Fetch representative ID
            rep_ids = set()
            # from paired if no main MSAs
            paths = (
                main_msa_file_paths
                if len(main_msa_file_paths) > 0
                else paired_msa_file_paths
            )
            for msa_file_path in paths:
                if msa_file_path.is_dir() or msa_file_path.suffix == ".npz":
                    rep_ids.add(msa_file_path.stem)
                elif msa_file_path.suffix in [".sto", ".a3m"]:
                    rep_ids.add(msa_file_path.parent.stem)

            rep_id = sorted(rep_ids)[0]

            if len(rep_ids) > 1:
                warnings.warn(
                    f"Found multiple representative IDs {rep_ids} for chain ID "
                    f"{chain_id}. Only the first representative ID will be used:"
                    f" {rep_id}.",
                    stacklevel=2,
                )

            chain_id_to_rep_id[chain_id] = rep_id
            chain_id_to_mol_type[chain_id] = chain_data["molecule_type"]
            if rep_id not in rep_id_to_main_msa_paths:
                rep_id_to_main_msa_paths[rep_id] = main_msa_file_paths
            if rep_id not in rep_id_to_paired_msa_paths:
                rep_id_to_paired_msa_paths[rep_id] = paired_msa_file_paths

    # Parse MSAs for each representative ID
    rep_id_to_query_seq, rep_id_to_main_msa, rep_id_to_paired_msa = {}, {}, {}
    if len(chain_id_to_rep_id) > 0:
        # Parse MSAs for each representative
        representative_chain_ids = sorted(set(chain_id_to_rep_id.values()))
        for rep_id in representative_chain_ids:
            # Parse main MSAs
            if rep_id in rep_id_to_main_msa_paths:
                example_path = rep_id_to_main_msa_paths[rep_id][0]
                if example_path.is_dir() or (example_path.suffix in [".sto", ".a3m"]):
                    file_list = standardize_filepaths(
                        rep_id_to_main_msa_paths[rep_id],
                    )
                    chain_msa_parser = partial(
                        parse_msas_direct,
                        max_seq_counts=max_seq_counts,
                    )
                elif example_path.suffix == ".npz":
                    file_list = standardize_filepaths(rep_id_to_main_msa_paths[rep_id])
                    chain_msa_parser = parse_msas_preparsed
                else:
                    raise ValueError(
                        f"Unsupported MSA path found {example_path}. Needs to be one "
                        "of the following: \n"
                        " - an .a3m or .sto file \n"
                        " - a directory containing .a3m or .sto files \n"
                        " - a .npz file \n"
                    )

                # Parse MSAs into a dict of MsaArrays
                all_msas_per_chain = chain_msa_parser(file_list=file_list)
                rep_id_to_main_msa[rep_id] = all_msas_per_chain

                # Create query sequence from the first row
                rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                    sorted(all_msas_per_chain.keys())[0]
                ].msa[0, :][np.newaxis, :]

            # Parse paired MSAs
            if rep_id in rep_id_to_paired_msa_paths:
                example_path = rep_id_to_paired_msa_paths[rep_id][0]
                if example_path.is_dir() or (example_path.suffix in [".sto", ".a3m"]):
                    file_list = standardize_filepaths(
                        rep_id_to_paired_msa_paths[rep_id]
                    )
                    chain_msa_parser = partial(
                        parse_msas_direct,
                        max_seq_counts=max_seq_counts,
                    )
                elif example_path.suffix == ".npz":
                    file_list = standardize_filepaths(
                        rep_id_to_paired_msa_paths[rep_id]
                    )
                    chain_msa_parser = parse_msas_preparsed
                else:
                    raise ValueError(
                        f"Unsupported MSA path found {example_path}. Needs to be one "
                        "of the following: \n"
                        " - an .a3m or .sto file \n"
                        " - a directory containing .a3m or .sto files \n"
                        " - a .npz file \n"
                    )

                # Parse MSAs into a dict of MsaArrays
                all_msas_per_chain = chain_msa_parser(file_list=file_list)
                rep_id_to_paired_msa[rep_id] = all_msas_per_chain

                if rep_id not in rep_id_to_query_seq:
                    # Create query sequence from the first row
                    rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                        sorted(all_msas_per_chain.keys())[0]
                    ].msa[0, :][np.newaxis, :]

    # Set msa collection to parsed, will be empty if no protein or RNA chains
    msa_array_collection = MsaArrayCollection(
        chain_id_to_rep_id=chain_id_to_rep_id,
        chain_id_to_mol_type=chain_id_to_mol_type,
    )
    msa_array_collection.set_state_parsed(
        rep_id_to_query_seq=rep_id_to_query_seq,
        rep_id_to_paired_msa=rep_id_to_paired_msa,
        rep_id_to_main_msa=rep_id_to_main_msa,
    )
    return msa_array_collection


class MsaSampleParserMapper:
    """Data class to hold mappings between chain IDs, representative IDs, and MSA
    objects for MSA parsing."""

    map_keys: tuple = (
        "chain_id_to_rep_id",
        "chain_id_to_mol_type",
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
        alignment_array_directory: DirectoryPathOrNone = None,
        alignment_db_directory: DirectoryPathOrNone = None,
        alignment_index: dict | None = None,
        alignments_directory: DirectoryPathOrNone = None,
    ):
        self.config = config
        self.alignment_array_directory = alignment_array_directory
        self.alignment_db_directory = alignment_db_directory
        self.alignment_index = alignment_index
        self.alignments_directory = alignments_directory

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
        )
        msa_array_collection.set_state_parsed(
            rep_id_to_query_seq=maps.rep_id_to_query_seq,
            rep_id_to_paired_msa=maps.rep_id_to_paired_msa,
            rep_id_to_main_msa=maps.rep_id_to_main_msa,
        )
        return msa_array_collection

    def __call__(self, input: MsaSampleProcessorInputInference) -> MsaArrayCollection:
        # Create maps between chain IDs, representative IDs, molecule types
        maps = self.create_maps(input=input)

        # Parse MSAs for each representative ID
        self.parse_msas(maps=maps)

        # Collect data into MsaArrayCollection
        msa_array_collection = self.create_msa_array_collection(maps)

        return msa_array_collection


class MsaSampleParserTrain(MsaSampleParser):
    """Training MSA sample parser class"""

    def create_maps(self, input: MsaSampleProcessorInputTrain) -> MsaSampleParserMapper:
        """Populates the chain_id_to_rep_id and chain_id_to_mol_type maps.

        Updates the following attributes:
            - chain_id_to_rep_id: dict
                Maps chain IDs to representative IDs.
            - chain_id_to_mol_type: dict
                Maps chain IDs to molecule types."""
        maps = MsaSampleParserMapper()
        for chain_id, chain_data in input.msa_chain_data.items():
            if chain_data.molecule_type in self.config.moltypes:
                maps.chain_id_to_rep_id[chain_id] = (
                    chain_data.alignment_representative_id
                )
                maps.chain_id_to_mol_type[chain_id] = chain_data.molecule_type
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
                    file_list = standardize_filepaths(
                        self.alignment_array_directory / f"{rep_id}.npz"
                    )
                    all_msas_per_chain = parse_msas_preparsed(file_list=file_list)
                elif self.alignment_db_directory is not None:
                    all_msas_per_chain = parse_msas_alignment_database(
                        alignment_index_entry=self.alignment_index[rep_id],
                        alignment_database_path=self.alignment_db_directory,
                        max_seq_counts=self.config.max_seq_counts,
                    )
                else:
                    file_list = standardize_filepaths(
                        self.config.alignments_directory / Path(rep_id)
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


class MsaSampleParserInference(MsaSampleParser):
    """Inference MSA sample parser class"""

    def create_maps(
        self, input: MsaSampleProcessorInputInference
    ) -> MsaSampleParserMapper:
        """Creates maps between chain IDs, representative IDs, and molecule types.

        Updates the following attributes:
            - chain_id_to_rep_id: dict
                Maps chain IDs to representative IDs.
            - chain_id_to_mol_type: dict
                Maps chain IDs to molecule types.
            - rep_id_to_main_msa_paths: dict
                Maps representative IDs to main MSA file paths.
            - rep_id_to_paired_msa_paths: dict
                Maps representative IDs to paired MSA file paths.
        """
        # Create maps
        maps = MsaSampleParserMapper()
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

                # Fetch representative ID
                rep_ids = set()
                # from paired if no main MSAs
                paths = (
                    main_msa_file_paths
                    if len(main_msa_file_paths) > 0
                    else paired_msa_file_paths
                )

                if len(paths) == 0:
                    warnings.warn(
                        (
                            f"Expected MSA file for chain {chain_id} of type "
                            f"{chain_data.molecule_type.name} in query "
                            f"{input.query_name}, but no MSA files found. No MSA "
                            "features will be computed for this chain."
                        ),
                        stacklevel=2,
                    )
                    continue

                for msa_file_path in paths:
                    if msa_file_path.is_dir() or msa_file_path.suffix == ".npz":
                        rep_ids.add(msa_file_path.stem)
                    elif msa_file_path.suffix in [".sto", ".a3m"]:
                        rep_ids.add(msa_file_path.parent.stem)

                rep_id = sorted(rep_ids)[0]

                if len(rep_ids) > 1:
                    warnings.warn(
                        f"Found multiple representative IDs {rep_ids} for chain ID "
                        f"{chain_id}. Only the first representative ID will be used:"
                        f" {rep_id}.",
                        stacklevel=2,
                    )

                maps.chain_id_to_rep_id[chain_id] = rep_id
                maps.chain_id_to_mol_type[chain_id] = chain_data.molecule_type
                if (rep_id not in maps.rep_id_to_main_msa_paths) & (
                    len(main_msa_file_paths) > 0
                ):
                    maps.rep_id_to_main_msa_paths[rep_id] = main_msa_file_paths
                if (rep_id not in maps.rep_id_to_paired_msa_paths) & (
                    len(paired_msa_file_paths) > 0
                ):
                    maps.rep_id_to_paired_msa_paths[rep_id] = paired_msa_file_paths

        return maps

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
                    all_msas_per_chain = chain_msa_parser(file_list=file_list)
                    maps.rep_id_to_main_msa[rep_id] = all_msas_per_chain

                    # Create query sequence from the first row
                    maps.rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                        sorted(all_msas_per_chain.keys())[0]
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
                    all_msas_per_chain = chain_msa_parser(file_list=file_list)
                    maps.rep_id_to_paired_msa[rep_id] = all_msas_per_chain

                    if rep_id not in maps.rep_id_to_query_seq:
                        # Create query sequence from the first row
                        maps.rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                            sorted(all_msas_per_chain.keys())[0]
                        ].msa[0, :][np.newaxis, :]

        return maps
