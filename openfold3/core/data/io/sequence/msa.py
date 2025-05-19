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

from openfold3.core.data.format.msa import (
    MsaSampleParserConfig,
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
        for seq_res, query_res in zip(sequence, query):
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
    input_path: Path | list[Path], max_seq_counts: dict[str, int] | None = None
) -> dict[str, MsaArray]:
    """Parses a set of MSA files (a3m or sto) into a dictionary of Msa objects.

    This function is used to parse MSAs for a single chain.

    Args:
        folder_path (Path | list[Path]):
            Either path to directory containing the MSA files to parse, path to a single
            MSA file or a list of paths to the MSA files to parse.
        max_seq_counts (dict[str, int] | None):
            A map from file names to maximum sequences to keep from the corresponding
            MSA file. The set of keys in this dict is also used to parse only a subset
            of the files in the folder with the corresponding names.

    Returns:
        dict[str: Msa]: A dict containing the parsed MSAs.
    """
    # Convert to list of Paths to MSA files
    if isinstance(input_path, Path) & input_path.is_dir():
        file_list = list(input_path.iterdir())
    elif isinstance(input_path, Path) & input_path.is_file():
        file_list = [input_path]

    msas = {}

    if len(file_list) == 0:
        raise RuntimeError(
            f"No alignments found in {input_path}. Folders for chains"
            "without any aligned sequences need to contain at least one"
            ".sto file with only the query sequence."
        )
    else:
        for aln_file in file_list:
            if aln_file.is_dir():
                warnings.warn(
                    f"Skipping directory {aln_file} in {input_path}. When a list of "
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
                    "{input_path}. Only .sto and .a3m files are supported for direct "
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
    input_path: Path | list[Path],
) -> dict[str, MsaArray]:
    """Parses a pre-parsed .npz file into a dictionary of Msa objects.

    This function is used to parse MSAs for a single chain. If a list of npz files is
    provided, where each file contains a dict of MSA arrays, the function will
    concatenate the MSAs from all dicts into a single dict, so for repeating keys, only
    the last MSA will be kept.

    Args:
        input_path (Path):
            Path to an npz file or list of npz files pre-parsed using
            openfold3.scripts.data_preprocessing.preparse_alginments_af3.

    Returns:
        dict[str, MsaArray]:
            A dict containing the parsed MSAs.
    """
    msas = {}

    if isinstance(input_path, Path):
        file_list = [input_path]
    elif isinstance(input_path, list):
        file_list = input_path

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

    Returns:
        MsaArrayCollection:
            A collection of Msa objects and chain IDs for a single sample.
    """
    # Get subset of atom array for which alignments are supposed to be provided
    # based on the dataset config
    # TODO refactor to get the list of chains that need MSAs from the assembly_data
    # dictionary instead of the atom array - won't need atom_array as input
    moltypes_local = [MoleculeType[moltype.upper()].value for moltype in moltypes]
    atom_array_with_alignments = atom_array[
        np.isin(
            atom_array.molecule_type_id,
            moltypes_local,
        )
    ]

    # Map chain IDs to representative IDs and molecule types
    chain_id_to_rep_id, chain_id_to_mol_type = {}, {}
    for chain_id_in_atom_array in sorted(set(atom_array_with_alignments.chain_id)):
        chain_data = assembly_data[chain_id_in_atom_array]
        chain_id_to_rep_id[chain_id_in_atom_array] = chain_data[
            "alignment_representative_id"
        ]
        chain_id_to_mol_type[chain_id_in_atom_array] = chain_data["molecule_type"]

    # Parse MSAs for each representative ID
    rep_id_to_msa, rep_id_to_query_seq = {}, {}
    if len(chain_id_to_rep_id) > 0:
        # Parse MSAs for each representative ID
        # This requires parsing MSAs for duplicate chains only once
        representative_chain_ids = sorted(set(chain_id_to_rep_id.values()))
        representative_msas = {}
        for rep_id in representative_chain_ids:
            if alignment_array_directory is not None:
                representative_msas[rep_id] = parse_msas_preparsed(
                    input_path=alignment_array_directory / f"{rep_id}.npz",
                )
            elif alignment_db_directory is not None:
                representative_msas[rep_id] = parse_msas_alignment_database(
                    alignment_index_entry=alignment_index[rep_id],
                    alignment_database_path=alignment_db_directory,
                    max_seq_counts=max_seq_counts,
                )
            else:
                representative_msas[rep_id] = parse_msas_direct(
                    input_path=(alignments_directory / Path(rep_id)),
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
        rep_id_to_unpaired_msa=rep_id_to_msa, rep_id_to_query_seq=rep_id_to_query_seq
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
                    chain_msa_parser = partial(
                        parse_msas_direct,
                        max_seq_counts=max_seq_counts,
                    )
                elif example_path.suffix == ".npz":
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
                all_msas_per_chain = chain_msa_parser(
                    input_path=rep_id_to_main_msa_paths[rep_id],
                )
                rep_id_to_main_msa[rep_id] = all_msas_per_chain

                # Create query sequence from the first row
                rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                    sorted(all_msas_per_chain.keys())[0]
                ].msa[0, :][np.newaxis, :]

            # Parse paired MSAs
            if rep_id in rep_id_to_paired_msa_paths:
                example_path = rep_id_to_paired_msa_paths[rep_id][0]
                if example_path.is_dir() or (example_path.suffix in [".sto", ".a3m"]):
                    chain_msa_parser = partial(
                        parse_msas_direct,
                        max_seq_counts=max_seq_counts,
                    )
                elif example_path.suffix == ".npz":
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
                all_msas_per_chain = chain_msa_parser(
                    input_path=rep_id_to_paired_msa_paths[rep_id],
                )
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


class MsaSampleParser:
    """Base MSA sample parser class"""

    def __init__(self, config: MsaSampleParserConfig):
        self.config = config

    def empty_attributes(self) -> None:
        self.chain_id_to_rep_id = {}
        self.chain_id_to_mol_type = {}
        self.rep_id_to_main_msa_paths = {}
        self.rep_id_to_paired_msa_paths = {}
        self.rep_id_to_query_seq = {}
        self.rep_id_to_main_msa = {}
        self.rep_id_to_paired_msa = {}

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

    def create_msa_array_collection(self) -> MsaArrayCollection:
        # Set msa collection to parsed, will be empty if no protein or RNA chains
        msa_array_collection = MsaArrayCollection(
            chain_id_to_rep_id=self.chain_id_to_rep_id,
            chain_id_to_mol_type=self.chain_id_to_mol_type,
        )
        msa_array_collection.set_state_parsed(
            rep_id_to_query_seq=self.rep_id_to_query_seq,
            rep_id_to_paired_msa=self.rep_id_to_paired_msa,
            rep_id_to_main_msa=self.rep_id_to_main_msa,
        )
        return msa_array_collection

    def forward(self, input: MsaSampleProcessorInputInference) -> MsaArrayCollection:
        self.empty_attributes()

        # Create maps between chain IDs, representative IDs, molecule types
        self.create_maps(input=input)

        # Parse MSAs for each representative ID
        self.parse_msas()

        # Collect data into MsaArrayCollection
        return self.create_msa_array_collection()

    def __call__(self, input: MsaSampleProcessorInputInference) -> MsaArrayCollection:
        return self.forward(input=input)


class MsaSampleParserTrain(MsaSampleParser):
    """Training MSA sample parser class"""

    def create_maps(self, input: MsaSampleProcessorInputTrain) -> None:
        # Create maps
        for chain_id, chain_data in input.msa_chain_data.items():
            if chain_data.molecule_type in self.config.moltypes:
                self.chain_id_to_rep_id[chain_id] = (
                    chain_data.alignment_representative_id
                )
                self.chain_id_to_mol_type[chain_id] = chain_data.molecule_type

    def parse_msas(self) -> None:
        # Parse MSAs for each representative ID
        if len(self.chain_id_to_rep_id) > 0:
            # Parse MSAs for each representative ID
            representative_chain_ids = sorted(set(self.chain_id_to_rep_id.values()))
            for rep_id in representative_chain_ids:
                if self.config.alignment_array_directory is not None:
                    all_msas_per_chain = parse_msas_preparsed(
                        input_path=self.config.alignment_array_directory
                        / f"{rep_id}.npz",
                    )
                elif self.config.alignment_db_directory is not None:
                    all_msas_per_chain = parse_msas_alignment_database(
                        alignment_index_entry=self.config.alignment_index[rep_id],
                        alignment_database_path=self.config.alignment_db_directory,
                        max_seq_counts=self.config.max_seq_counts,
                    )
                else:
                    all_msas_per_chain = parse_msas_direct(
                        input_path=(self.config.alignments_directory / Path(rep_id)),
                        max_seq_counts=self.config.max_seq_counts,
                    )
                self.rep_id_to_main_msa[rep_id] = all_msas_per_chain

                # Create query sequence from the first row
                self.rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                    sorted(all_msas_per_chain.keys())[0]
                ].msa[0, :][np.newaxis, :]

            # Parsing precomputed paired MSAs is not currently supported for training


class MsaSampleParserInference(MsaSampleParser):
    """Inference MSA sample parser class"""

    def create_maps(self, input: MsaSampleProcessorInputInference) -> None:
        """_summary_

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
        for chain_id, chain_data in input.msa_chain_data.items():
            print(chain_data.molecule_type)
            print(self.config.moltypes)
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

                self.chain_id_to_rep_id[chain_id] = rep_id
                self.chain_id_to_mol_type[chain_id] = chain_data.molecule_type
                if (rep_id not in self.rep_id_to_main_msa_paths) & (
                    len(main_msa_file_paths) > 0
                ):
                    self.rep_id_to_main_msa_paths[rep_id] = main_msa_file_paths
                if (rep_id not in self.rep_id_to_paired_msa_paths) & (
                    len(paired_msa_file_paths) > 0
                ):
                    self.rep_id_to_paired_msa_paths[rep_id] = paired_msa_file_paths

        print(self.chain_id_to_rep_id)
        print(self.chain_id_to_mol_type)
        print(self.rep_id_to_main_msa_paths)
        print(self.rep_id_to_paired_msa_paths)

    def parse_msas(self) -> None:
        """_summary_

        Updates the following attributes:
            - rep_id_to_query_seq: dict
                Maps representative IDs to query sequences.
            - rep_id_to_main_msa: dict
                Maps representative IDs to main MSA objects.
            - rep_id_to_paired_msa: dict
                Maps representative IDs to paired MSA objects.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """

        if len(self.chain_id_to_rep_id) > 0:
            # Parse MSAs for each representative
            representative_chain_ids = sorted(set(self.chain_id_to_rep_id.values()))
            for rep_id in representative_chain_ids:
                # Parse main MSAs
                if rep_id in self.rep_id_to_main_msa_paths:
                    example_path = self.rep_id_to_main_msa_paths[rep_id][0]
                    if example_path.is_dir() or (
                        example_path.suffix in [".sto", ".a3m"]
                    ):
                        chain_msa_parser = partial(
                            parse_msas_direct,
                            max_seq_counts=self.config.max_seq_counts,
                        )
                    elif example_path.suffix == ".npz":
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
                    all_msas_per_chain = chain_msa_parser(
                        input_path=self.rep_id_to_main_msa_paths[rep_id],
                    )
                    self.rep_id_to_main_msa[rep_id] = all_msas_per_chain

                    # Create query sequence from the first row
                    self.rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                        sorted(all_msas_per_chain.keys())[0]
                    ].msa[0, :][np.newaxis, :]

                # Parse paired MSAs
                if rep_id in self.rep_id_to_paired_msa_paths:
                    example_path = self.rep_id_to_paired_msa_paths[rep_id][0]
                    if example_path.is_dir() or (
                        example_path.suffix in [".sto", ".a3m"]
                    ):
                        chain_msa_parser = partial(
                            parse_msas_direct,
                            max_seq_counts=self.config.max_seq_counts,
                        )
                    elif example_path.suffix == ".npz":
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
                    all_msas_per_chain = chain_msa_parser(
                        input_path=self.rep_id_to_paired_msa_paths[rep_id],
                    )
                    self.rep_id_to_paired_msa[rep_id] = all_msas_per_chain

                    if rep_id not in self.rep_id_to_query_seq:
                        # Create query sequence from the first row
                        self.rep_id_to_query_seq[rep_id] = all_msas_per_chain[
                            sorted(all_msas_per_chain.keys())[0]
                        ].msa[0, :][np.newaxis, :]

        print(self.rep_id_to_query_seq)
        print(self.rep_id_to_main_msa)
        print(self.rep_id_to_paired_msa)
