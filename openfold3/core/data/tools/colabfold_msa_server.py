import json
import logging
import os
import random
import shutil
import tarfile
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, model_validator
from pydantic_core import Url
from tqdm import tqdm

from openfold3.core.data.io.sequence.msa import parse_a3m
from openfold3.core.data.primitives.sequence.hash import get_sequence_hash
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)

logger = logging.getLogger(__name__)


TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)


class MsaServerPairingStrategy(IntEnum):
    """Enum for MSA server pairing strategy."""

    GREEDY = 0
    COMPLETE = 1

    def __str__(self) -> str:
        return self.name.lower()


def query_colabfold_msa_server(
    x: list[str],
    prefix: Path,
    user_agent: str,
    use_templates: bool = False,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    use_env: bool = True,
    use_filter: bool = True,
    filter: bool | None = None,
    host_url: str = "https://api.colabfold.com",
) -> list[str] | tuple[list[str], list[str]]:
    """Submints a single query to the colabfold MSA server.

    Adapted from Colabfold run_mmseqs2 https://github.com/sokrypton/ColabFold/blob/main/colabfold/colabfold.py#L69

    Args:
        x (list[str]):
            List of amino acid sequences to query the MSA server with.
        prefix (Path):
            Output directory to save the results to.
        user_agent (str):
            User associated with API call.
        use_templates (bool, optional):
            Whether to run template search. Defaults to False. If use_pairing is True,
            this internally gets set to False.
        use_pairing (bool, optional):
            Whether to generate a single paired MSA Defaults to False.
        pairing_strategy (str, optional):
            Pairing method, one of ["complete", "greedy"]. For pairing of more than 2
            chains, "complete" requires a taxonomic group to be present in all chains,
            where greedy requires it to be in only 2
        use_env (bool, optional):
            Whether to align against env db(BFD/cfdb) Defaults to True.
        use_filter (bool, optional):
            Whether to apply diversity filter. Defaults to True.
        filter (bool | None, optional):
            Legacy option to enable diversity filter. Defaults to None.
        host_url (str, optional):
            host url for MSA server. Defaults to "https://api.colabfold.com".

    Returns:
        list[str] | tuple[list[str], list[str]]:
            List of MSA strings in a3m format, one per query sequence. If use_templates
            is True, also returns a list of template paths for each sequence.
    """

    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    headers = {}
    if user_agent != "":
        headers["User-Agent"] = user_agent
    else:
        logger.warning(
            "No user agent specified. Please set a user agent"
            "(e.g., 'toolname/version contact@email') to help"
            "us debug in case of problems. This warning will become an error"
            "in the future."
        )

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        while True:
            error_count = 0
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger
                # than a multiple of 3"
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
            except requests.exceptions.Timeout:
                logger.warning("Timeout while submitting to MSA server. Retrying...")
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server."
                    f"Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        while True:
            error_count = 0
            try:
                res = requests.get(
                    f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching status from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server."
                    f"Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        error_count = 0
        while True:
            try:
                res = requests.get(
                    f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers
                )
            except requests.exceptions.Timeout:
                logger.warning(
                    "Timeout while fetching result from MSA server. Retrying..."
                )
                continue
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server."
                    f"Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                time.sleep(5)
                if error_count > 5:
                    raise
                continue
            break
        with open(path, "wb") as out:
            out.write(res.content)

    seqs = [x] if isinstance(x, str) else x

    # Compatibility to old option
    if filter is not None:
        use_filter = filter

    # Setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"
    # TODO move to config construction
    pairing_strategy = MsaServerPairingStrategy[pairing_strategy.upper()]
    if use_pairing:
        use_templates = False
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == MsaServerPairingStrategy.GREEDY:
            mode = "pairgreedy"
        elif pairing_strategy == MsaServerPairingStrategy.COMPLETE:
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    # Put everything in the same dir
    path = f"{prefix}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # Call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # Deduplicate and keep track of order
    seqs_unique = []
    # TODO this might be slow for large sets - see main MSA deduplication code for a
    # faster option
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    # Run query
    # TODO add warning or msg if using existing files
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise Exception(
                        "MMseqs2 API is giving errors."
                        "Please confirm your input is a valid protein sequence."
                        "If error persists, please try again an hour later."
                    )

                if out["status"] == "MAINTENANCE":
                    raise Exception(
                        "MMseqs2 API is undergoing maintenance."
                        "Please try again in a few minutes."
                    )

                # Wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    raise Exception(
                        "MMseqs2 API is giving errors."
                        "Please confirm your input is a valid protein sequence."
                        "If error persists, please try again an hour later."
                    )

            # Download results
            download(ID, tar_gz_file)

    # Prepare list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # Extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # Process templates
    if use_templates:
        templates = {}
        with open(f"{path}/pdb70.m8") as f:
            for line in f:
                p = line.rstrip().split()
                M, pdb, _, _ = p[0], p[1], p[2], p[10]  # M, pdb, qid, e_value
                M = int(M)
                if M not in templates:
                    templates[M] = []
                templates[M].append(pdb)

        template_paths = {}
        for k, TMPL in templates.items():
            TMPL_PATH = f"{prefix}/templates_{k}"
            if not os.path.isdir(TMPL_PATH):
                os.mkdir(TMPL_PATH)
                TMPL_LINE = ",".join(TMPL[:20])
                response = None
                while True:
                    error_count = 0
                    try:
                        # "good practice to set connect timeouts to slightly
                        # larger than a multiple of 3"
                        response = requests.get(
                            f"{host_url}/template/{TMPL_LINE}",
                            stream=True,
                            timeout=6.02,
                            headers=headers,
                        )
                    except requests.exceptions.Timeout:
                        logger.warning(
                            "Timeout while submitting to template server. Retrying..."
                        )
                        continue
                    except Exception as e:
                        error_count += 1
                        logger.warning(
                            f"Error while fetching result from template server."
                            f"Retrying... ({error_count}/5)"
                        )
                        logger.warning(f"Error: {e}")
                        time.sleep(5)
                        if error_count > 5:
                            raise
                        continue
                    break
                with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
                    tar.extractall(path=TMPL_PATH)
                os.symlink("pdb70_a3m.ffindex", f"{TMPL_PATH}/pdb70_cs219.ffindex")
                with open(f"{TMPL_PATH}/pdb70_cs219.ffdata", "w") as f:
                    f.write("")
            template_paths[k] = TMPL_PATH

        template_paths_ = []
        for n in Ms:
            if n not in template_paths:
                template_paths_.append(None)
            else:
                template_paths_.append(template_paths[n])
        template_paths = template_paths_

    # Gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        with open(a3m_file) as f:
            for line in f:
                if len(line) > 0:
                    if "\x00" in line:
                        line = line.replace("\x00", "")
                        update_M = True
                    if line.startswith(">") and update_M:
                        M = int(line[1:].rstrip())
                        update_M = False
                        if M not in a3m_lines:
                            a3m_lines[M] = []
                    a3m_lines[M].append(line)

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    return (a3m_lines, template_paths) if use_templates else a3m_lines


class ChainInput(NamedTuple):
    """A query name and chain ID tuple.

    Attributes:
        query_name (str): The name of the query.
        chain_id (str): The chain ID.
        sequence (str): sequence for the chain.
    """

    query_name: str
    chain_id: str
    sequence: str

    def __str__(self) -> str:
        return self.name()

    @property
    def name(self) -> str:
        """Joins the query name and chain ID with a delimiter into a string."""
        delimiter = "-"
        return f"{self.query_name}{delimiter}{self.chain_id}"

    @property
    def rep_id(self) -> str:
        """Returns the representative ID for this chain ID."""
        return get_sequence_hash(self.sequence)


class ComplexGroup(list[str]):
    """Wrapper around standard list to provide a a hash of concatenated sequences."""

    @property
    def rep_id(self):
        seqs = sorted(s for s in self)
        return get_sequence_hash("".join(seqs))


# TODO rename
@dataclass
class ColabFoldMapper:
    """Data class to hold mappings for colabfold MSA server.

    Used identifiers:
        query_name:
            name of the query structure in the input query cache.
        chain_id:
            a (query_name, chain identifier) tuple based on the query
        rep_id:
            a hash of the sequence id
        seq:
            the actual protein sequence.
        complex_id:
            - An identifier associated with all protein sequences in the
            same query, constructed based on a hash of the sequences.
            Only used for queries with more than 2 unique protein sequences.
            - Can be constructed using `ComplexGroup` class wrapped around the
            set of sequences

    Attributes:
        seq_to_rep_id (dict[str, str]):
            Sequence to representative ID mapping (hash of input sequecne).
        rep_id_to_seq (dict[str, str]):
            Representative ID to sequence mapping.
        chain_id_to_rep_id (dict[str, str]):
            Chain ID to representative ID mapping.
        query_name_to_complex_id (dict[str, str]):
            Query name to complex ID mapping.
        complex_id_to_complex_group (dict[str, ComplexGroup]):
            Complex id identifier mapped to sequences that constructed the id
        seqs (list[str]):
            List of unique sequences.
        rep_ids (list[ChainID]):
            List of representative IDs.
    """

    seq_to_rep_id: dict[str, ChainInput] = field(default_factory=dict)
    rep_id_to_m: dict[int, ChainID] = field(default_factory=dict)
    rep_id_to_seq: dict[ChainInput, str] = field(default_factory=dict)
    chain_id_to_rep_id: dict[ChainInput, ChainInput] = field(default_factory=dict)
    query_name_to_complex_id: dict[str, str] = field(default_factory=dict)
    complex_id_to_complex_group: dict[str, ComplexGroup] = field(default_factory=dict)
    seqs: list[str] = field(default_factory=list)
    rep_ids: list[ChainInput] = field(default_factory=list)


def collect_colabfold_msa_data(
    inference_query_set: InferenceQuerySet,
) -> ColabFoldMapper:
    """Parses the protein sequences from the query cache and creates mappings.

    Args:
        inference_query_set (InferenceQuerySet):
            The inference query set containing the queries and chains.

    Returns:
        ColabFoldMapper:
            Data class containing the mappings for colabfold MSA server.
    """

    colabfold_mapper = ColabFoldMapper()
    m_i = 101
    # Get unique set of sequences for main MSAs
    for query_name, query in inference_query_set.queries.items():
        chain_inputs_seen = set()
        rep_ids_query = []

        for chain in query.chains:
            if chain.molecule_type == MoleculeType.PROTEIN:
                seq = chain.sequence
                chain_inputs = [
                    ChainInput(query_name, str(c_id), seq) for c_id in chain.chain_ids
                ]

                # Make sure there are no duplicates in the chain IDs across chains of
                # the same query TODO: could move to pydantic model validation
                if len(set(chain_inputs) & chain_inputs_seen) > 0:
                    raise RuntimeError(
                        f"Duplicate chain IDs found in query {query_name}: "
                        f"{chain.chain_ids}"
                    )

                chain_inputs_seen.update(chain_inputs)

                # Collect mapping data and sequences for main MSAs
                if seq not in colabfold_mapper.seq_to_rep_id:
                    rep_id = chain_inputs[0].rep_id
                    colabfold_mapper.seq_to_rep_id[seq] = rep_id
                    colabfold_mapper.rep_id_to_seq[rep_id] = seq
                    colabfold_mapper.rep_id_to_m[rep_id] = m_i
                    m_i += 1
                    for chain_input in chain_inputs:
                        colabfold_mapper.chain_id_to_rep_id[chain_input.name] = rep_id
                    colabfold_mapper.seqs.append(seq)
                    colabfold_mapper.rep_ids.append(rep_id)
                else:
                    for chain_input in chain_inputs:
                        colabfold_mapper.chain_id_to_rep_id[chain_input.name] = (
                            colabfold_mapper.seq_to_rep_id[seq]
                        )

                # Collect paired MSA data
                for chain_input in chain_inputs:
                    rep_ids_query.append(
                        colabfold_mapper.chain_id_to_rep_id[chain_input.name]
                    )

        # Only do pairing if number of unique protein sequences is > 1
        if len(set(rep_ids_query)) > 1:
            sequences = [colabfold_mapper.rep_id_to_seq[r] for r in set(rep_ids_query)]
            complex_group = ComplexGroup(sequences)
            complex_id = complex_group.rep_id
            if complex_id not in colabfold_mapper.complex_id_to_complex_group:
                colabfold_mapper.complex_id_to_complex_group[complex_id] = complex_group
            colabfold_mapper.query_name_to_complex_id[query_name] = complex_id

    return colabfold_mapper


def _save_mapping(mapping, save_filepath, append=False):
    if os.path.exists(save_filepath) and append:
        logger.warning(
            f"Mapping file {save_filepath} already exists. Appending new sequences."
        )
        with open(save_filepath) as f:
            old_mapping = json.load(f)
            mapping.update(old_mapping)

    with open(save_filepath, "w") as f:
        json.dump(mapping, f, indent=4)


def save_colabfold_mappings(
    colabfold_msa_input: ColabFoldMapper, output_directory: Path
) -> None:
    """Saves the mappings for colabfold MSA server to JSON files.

    Args:
        colabfold_msa_input (ColabFoldMapper):
            The ColabFoldMapper object containing the mappings.
        output_directory (Path):
            The output directory to save the JSON files.
    """

    mapping_files_dir = output_directory / "mappings"
    mapping_files_dir.mkdir(parents=True, exist_ok=True)

    _save_mapping(
        colabfold_msa_input.seq_to_rep_id,
        mapping_files_dir / "seq_to_rep_id.json",
        append=True,
    )
    _save_mapping(
        colabfold_msa_input.rep_id_to_seq,
        mapping_files_dir / "rep_id_to_seq.json",
        append=True,
    )
    _save_mapping(
        colabfold_msa_input.chain_id_to_rep_id,
        mapping_files_dir / "chain_id_to_rep_id.json",
        append=False,
    )
    _save_mapping(
        colabfold_msa_input.query_name_to_complex_id,
        mapping_files_dir / "query_name_to_complex_id.json",
        append=False,
    )

    with open(mapping_files_dir / "README.md", "w") as f:
        f.write(
            "# Mapping files\n"
            "These files contain mappings between the following entities:\n"
            "  query_name: name of the query complex in the input query cache\n"
            "  chain_id: a (query_name, chain identifier) tuple, indicating a unique "
            "instantiation of a protein chain.\n"
            "  rep_id: a chain_id associated with a unique protein sequence, selected "
            "upon first occurrence of that specific sequence; all subsequent chain_ids"
            " with the same sequence will have this chain_id as the representative\n"
            "  seq: the actual protein sequence\n"
            "  complex_id: an identifier associated with a unique SET of protein"
            " sequences in the same query, consisting of the sorted representative IDs"
            " of ALL chains in the complex; only used for queries with more than 2 "
            "unique protein sequences\n"
        )


class ColabFoldQueryRunner:
    """Class to run queries on the ColabFold MSA server.

    Attributes:
        colabfold_mapper (ColabFoldMapper):
            The ColabFoldMapper object containing the mappings.
        output_directory (Path):
            The output directory to save the results to.
        msa_file_format (str | list[str]):
            The file format for the MSA files. Can be a single format or a list of
            formats. Elements can be "a3m" or "npz".
        user_agent (str):
            The user agent to use for the API calls.
    """

    def __init__(
        self,
        colabfold_mapper: ColabFoldMapper,
        output_directory: Path,
        msa_file_format: str | list[str],
        user_agent: str,
        host_url: Url = "https://api.colabfold.com",
    ):
        self.colabfold_mapper = colabfold_mapper
        self.output_directory = output_directory
        self.msa_file_format = (
            msa_file_format if isinstance(msa_file_format, list) else [msa_file_format]
        )
        self.user_agent = user_agent
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.host_url = host_url
        for subdir in ["raw", "main", "paired"]:
            (self.output_directory / subdir).mkdir(parents=True, exist_ok=True)
            if subdir == "raw":
                for subsubdir in ["main", "paired"]:
                    (self.output_directory / subdir / subsubdir).mkdir(
                        parents=True, exist_ok=True
                    )

    def query_format_main(self):
        """Submits queries and formats the outputs for main MSAs."""
        # Submit query for main MSAs
        # TODO: add template alignments fetching code here by setting use_templates=True
        # TODO: replace prints with proper logging
        print(
            f"Submitting {len(self.colabfold_mapper.seqs)} sequences to the Colabfold"
            " MSA server for main MSAs..."
        )
        # TODO: chunking
        # TODO: warn if too many sequences maybe?
        a3m_lines_main = query_colabfold_msa_server(
            self.colabfold_mapper.seqs,
            prefix=self.output_directory / "raw/main",
            use_templates=False,
            use_pairing=False,
            user_agent=self.user_agent,
            host_url=self.host_url,
        )

        main_alignments_path = self.output_directory / "main"
        main_alignments_path.mkdir(parents=True, exist_ok=True)
        template_alignments_path = self.output_directory / "template"
        template_alignments_path.mkdir(parents=True, exist_ok=True)

        template_alignments = pd.read_csv(
            self.output_directory / "raw/main/pdb70.m8", sep="\t", header=None
        )
        m_with_templates = set(template_alignments[0])

        for rep_id, aln in zip(self.colabfold_mapper.rep_ids, a3m_lines_main):
            rep_dir = main_alignments_path / str(rep_id)
            template_rep_dir = template_alignments_path / str(rep_id)

            # TODO: add code for which format to save the MSA in
            # If save as a3m...
            if "a3m" in self.msa_file_format:
                rep_dir.mkdir(parents=True, exist_ok=True)
                a3m_file = rep_dir / "colabfold_main.a3m"
                with open(a3m_file, "w") as f:
                    f.write(aln)

            # If save as npz...
            if "npz" in self.msa_file_format:
                npz_file = Path(f"{rep_dir}.npz")
                msas = {"colabfold_main": parse_a3m(aln)}
                msas_preparsed = {}
                for k, v in msas.items():
                    msas_preparsed[k] = v.to_dict()
                np.savez_compressed(npz_file, **msas_preparsed)

            # Format template alignments
            m_i = self.colabfold_mapper.rep_id_to_m[rep_id]
            if m_i in m_with_templates:
                template_rep_dir.mkdir(parents=True, exist_ok=True)
                template_alignment_file = template_rep_dir / "colabfold_template.m8"
                template_alignment = template_alignments[template_alignments[0] == m_i]
                template_alignment.to_csv(
                    template_alignment_file,
                    sep="\t",
                    header=False,
                    index=False,
                )

    def query_format_paired(self):
        """Submits queries and formats the outputs for paired MSAs."""
        paired_alignments_directory = self.output_directory / "paired"
        paired_alignments_directory.mkdir(parents=True, exist_ok=True)
        # Submit queries for paired MSAss
        num_complexes = len(self.colabfold_mapper.complex_id_to_complex_group)
        print(
            f"Submitting {num_complexes} paired MSA queries"
            " to the Colabfold MSA server..."
        )
        for complex_id, complex_group in tqdm(
            self.colabfold_mapper.complex_id_to_complex_group.items(),
            total=num_complexes,
            desc="Computing paired MSAs",
        ):
            # Submit the query to the Colabfold MSA server
            (self.output_directory / "raw/paired").mkdir(parents=True, exist_ok=True)
            a3m_lines_paired = query_colabfold_msa_server(
                complex_group,
                prefix=self.output_directory / f"raw/paired/{complex_id}",
                use_templates=False,
                use_pairing=True,
                user_agent=self.user_agent,
                host_url=self.host_url,
            )

            # TODO: process the returned MSAs - save per representative ID
            complex_directory = paired_alignments_directory / str(complex_group.rep_id)
            complex_directory.mkdir(parents=True, exist_ok=True)
            for seq, aln in zip(complex_group, a3m_lines_paired):
                rep_dir = complex_directory / str(get_sequence_hash(seq))

                # If save as a3m...
                if "a3m" in self.msa_file_format:
                    rep_dir.mkdir(parents=True, exist_ok=True)
                    a3m_file = rep_dir / "colabfold_paired.a3m"
                    with open(a3m_file, "w") as f:
                        f.write(aln)

                # If save as npz...
                if "npz" in self.msa_file_format:
                    npz_file = Path(f"{rep_dir}.npz")
                    msas = {"colabfold_paired": parse_a3m(aln)}
                    msas_preparsed = {}
                    for k, v in msas.items():
                        msas_preparsed[k] = v.to_dict()
                    np.savez_compressed(npz_file, **msas_preparsed)

    def cleanup(self):
        """Remove raw colabfold MSA directory.

        If the same MSA output directory is used, this directory must be removed
        to avoid using old MSAs for new inputs.
        """
        shutil.rmtree(self.output_directory / "raw", ignore_errors=True)


def add_msa_paths_to_iqs(
    inference_query_set: InferenceQuerySet,
    colabfold_mapper: ColabFoldMapper,
    output_directory: Path,
) -> InferenceQuerySet:
    """Adds the paths to the MSA files to the inference query set.

    Args:
        inference_query_set (InferenceQuerySet):
            The inference query set containing the queries and chains.
        colabfold_mapper (ColabFoldMapper):
            The ColabFoldMapper object containing the mappings.
        output_directory (Path):
            The output directory to save the results to.

    Returns:
        InferenceQuerySet:
            The updated inference query set with the MSA file paths added.
    """
    for query_name, query in inference_query_set.queries.items():
        for chain in query.chains:
            if chain.molecule_type == MoleculeType.PROTEIN:
                # Add main MSA file paths to the chain field
                rep_id = colabfold_mapper.seq_to_rep_id[chain.sequence]

                # Use npz if available, otherwise use a3m
                main_msa_file_path = output_directory / "main" / f"{str(rep_id)}.npz"
                if not main_msa_file_path.exists():
                    main_msa_file_path = (
                        output_directory / "main" / str(rep_id) / "colabfold_main.a3m"
                    )

                if chain.main_msa_file_paths is not None:
                    warnings.warn(
                        f"Query {query_name} chain {chain} already has "
                        "main_msa_file_paths set. These are now overwritten "
                        "with path(s) to the ColabFold MSAs.",
                        stacklevel=2,
                    )
                chain.main_msa_file_paths = [main_msa_file_path]

                # Add paired MSA file paths to the chain field
                if query_name in colabfold_mapper.query_name_to_complex_id:
                    complex_id = colabfold_mapper.query_name_to_complex_id[query_name]

                    # Use npz if available, otherwise use a3m
                    paired_msa_file_paths = (
                        output_directory
                        / "paired"
                        / str(complex_id)
                        / f"{str(rep_id)}.npz"
                    )
                    if not paired_msa_file_paths.exists():
                        paired_msa_file_paths = (
                            output_directory
                            / "paired"
                            / str(complex_id)
                            / str(rep_id)
                            / "colabfold_paired.a3m"
                        )

                    if chain.paired_msa_file_paths is not None:
                        warnings.warn(
                            f"Query {query_name} chain {chain} already has "
                            "paired_msa_file_paths set. These are now "
                            "overwritten with path(s) to the ColabFold MSAs.",
                            stacklevel=2,
                        )
                    chain.paired_msa_file_paths = [paired_msa_file_paths]

                # Add template alignment file paths
                template_alignment_file_path = (
                    output_directory
                    / "template"
                    / str(rep_id)
                    / "colabfold_template.m8"
                )
                if template_alignment_file_path.exists():
                    if chain.template_alignment_file_path is not None:
                        warnings.warn(
                            f"Query {query_name} chain {chain} already has its"
                            "template_alignment_file_path set. This are now "
                            "overwritten with a path to the template alignment file"
                            "from the ColabFold MSA server.",
                            stacklevel=2,
                        )
                    chain.template_alignment_file_path = template_alignment_file_path

    return inference_query_set


class MsaComputationSettings(BaseModel):
    """Settings to run ColabFold MSA server.

    See preprocess_colabfold_msas for details on the parameters"""

    msa_file_format: Literal["npz", "a3m"] = "npz"
    server_user_agent: str = "openfold"
    server_url: Url = Url("https://api.colabfold.com")
    save_mappings: bool = True
    msa_output_directory: Path = Path(tempfile.gettempdir()) / "of3_colabfold_msas"
    cleanup_msa_dir: bool = True

    @model_validator(mode="after")
    def create_dif(self) -> "MsaComputationSettings":
        """Creates the output directory if it does not exist."""
        if not self.msa_output_directory.exists():
            self.msa_output_directory.mkdir(parents=True, exist_ok=True)
        return self


def preprocess_colabfold_msas(
    inference_query_set: InferenceQuerySet,
    compute_settings: MsaComputationSettings,
) -> InferenceQuerySet:
    """Gathers sequences, runs the ColabFold MSA server queries, updates MSA paths.

    Args:
        inference_query_set (InferenceQuerySet):
            The inference query set containing the queries and chains.
        compute_settings: pydantic model with server settings, contains:
            msa_file_format (str):
                The format of the MSA files to save.
                Can be "a3m" (unprocessed MSAs for inspectable but slower parsing)
                or "npz" (processed MSAs for faster parsing).
            user_agent (str):
                The user agent to use for the API calls.
            save_mappings (bool, optional):
                Whether to save the mappings to JSON files. Defaults to False.

    Returns:
        InferenceQuerySet:
            The updated inference query set with the MSA file paths added.

    Mapping:
        First, maps each sequence to a representative ID (deduplicate repeated
        sequences). Then maps each set of sequences in a structure to a representative
        set = complex ID (deduplicate repeated complexes with the exact same
        stoichiometry). Only for complexes with at least 2 different protein sequences.

    Server query:
        First, submits all unique sequences to the ColabFold MSA server as one query
        with pairing=False. Then submits each unqiue set of sequences in the same
        complex to the ColabFold MSA server as separate queries with pairing=True. Total
        number of queries = 1 + number of unique complexes with at least 2 different
        protein sequences.

    Output formatting:
        The raw MSA files are stored per server query at
            unpaired:
                output_directory/raw/main
            paired:
                output_directory/raw/paired
        The OpenFold3 online MSA pipeline requires per-chain MSAs.
        The unpaired per-chain MSA files are stored per representative ID at
            unparsed:
                output_directory/main/<representative_id>/colabfold_main.a3m
            pre-parsed:
                output_directory/main/<representative_id>.npz`
        The pre-paired per-chain MSA files are stored per complex ID at
            unparsed:
                output_directory/paired/<complex_id>/<representative_id>/colabfold_paired.a3m
            pre-parsed:
                output_directory/paired/<complex_id>/<representative_id>.npz

    Inference query set update:
        By default, uses the npz file paths if available, otherwise uses the a3m file
        paths.
    """
    # Gather MSA data
    colabfold_mapper = collect_colabfold_msa_data(inference_query_set)
    output_directory = compute_settings.msa_output_directory
    logger.warning(f"Using output directory: {output_directory} for ColabFold MSAs.")

    # Save mappings to file
    if compute_settings.save_mappings:
        save_colabfold_mappings(colabfold_mapper, output_directory)

    # Run batch queries for main and paired MSAs
    colabfold_query_runner = ColabFoldQueryRunner(
        colabfold_mapper=colabfold_mapper,
        output_directory=output_directory,
        msa_file_format=compute_settings.msa_file_format,
        user_agent=compute_settings.server_user_agent,
        host_url=compute_settings.server_url,
    )
    colabfold_query_runner.query_format_main()
    colabfold_query_runner.query_format_paired()

    # Add paths to the IQS
    inference_query_set = add_msa_paths_to_iqs(
        inference_query_set=inference_query_set,
        colabfold_mapper=colabfold_mapper,
        output_directory=output_directory,
    )

    # Remove raw MSA directory
    colabfold_query_runner.cleanup()

    return inference_query_set
