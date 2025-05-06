import json
import logging
import os
import random
import tarfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import requests
from tqdm import tqdm

from openfold3.core.data.io.sequence.msa import parse_a3m
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.af3_all_atom.config.inference_query_format import (
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


"""
TODOS:
- clear up all other TODOs
- add docstrings/typehints
- add tests
- add code that adds paths to the query cache for each chains MSA
"""


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
    # TODO this might be slow for large sets - see unpaired MSA deduplication code for a
    # faster option
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    # Run query
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


class ChainID(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """

    query_name: str
    chain_id: str

    def __str__(self) -> str:
        return self.stringify()

    def stringify(self, delimiter: str = "-") -> str:
        return f"{self.query_name}{delimiter}{self.chain_id}"


class ComplexID(tuple[ChainID, ...]):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """

    def __new__(cls, *chain_ids: ChainID) -> "ComplexID":
        for c in chain_ids:
            if not isinstance(c, ChainID):
                raise TypeError(f"Expected ChainID, got {type(c)}")
        return super().__new__(cls, chain_ids)

    def __iter__(self) -> Iterator[ChainID]:
        return super().__iter__()

    def stringify(
        self, inner_delimiter: str = "-", outer_delimiter: str = ".", sort: bool = False
    ) -> str:
        return outer_delimiter.join(c.stringify(inner_delimiter) for c in self)

    def __str__(self) -> str:
        return self.stringify()


# TODO rename
@dataclass
class ColabFoldMapper:
    """_summary_

    Attributes:
        seq_to_rep_id (dict[str, ChainID]): _description_
        rep_id_to_seq (dict[ChainID, str]): _description_
        chain_id_to_rep_id (dict[ChainID, ChainID]): _description_
        query_name_to_complex_id (dict[str, ComplexID]): _description_
        complex_ids (set[ComplexID]): _description_
        seqs (list[str]): _description_
        rep_ids (list[ChainID]): _description_
    """

    seq_to_rep_id: dict[str, ChainID] = field(default_factory=dict)
    rep_id_to_seq: dict[ChainID, str] = field(default_factory=dict)
    chain_id_to_rep_id: dict[ChainID, ChainID] = field(default_factory=dict)
    query_name_to_complex_id: dict[str, ComplexID] = field(default_factory=dict)
    complex_ids: set[ComplexID] = field(default_factory=set)
    seqs: list[str] = field(default_factory=list)
    rep_ids: list[ChainID] = field(default_factory=list)


def collect_colabfold_msa_data(
    inference_set: InferenceQuerySet,
) -> ColabFoldMapper:
    """_summary_

    Args:
        inference_set (InferenceQuerySet): _description_

    Raises:
        RuntimeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        InferenceQueryCacheMsaData: _description_
    """

    colabfold_mapper = ColabFoldMapper()
    # Get unique set of sequences for unpaired MSAs
    for query_name, query in inference_set.queries.items():
        chain_ids_seen = set()
        rep_ids_query = []

        for chain in query.chains:
            if chain.molecule_type == MoleculeType.PROTEIN:
                seq = chain.sequence
                chain_ids = []
                for chain_id in chain.chain_ids:
                    chain_id = str(chain_id)

                    chain_ids.append(ChainID(query_name, chain_id))

                # Make sure there are no duplicates in the chain IDs across chains of
                # the same query TODO: could move to pydantic model validation
                if len(set(chain_ids) & chain_ids_seen) > 0:
                    raise RuntimeError(
                        f"Duplicate chain IDs found in query {query_name}: "
                        f"{chain.chain_ids}"
                    )

                chain_ids_seen.update(chain_ids)

                # Collect mapping data and sequences for unpaired MSAs
                if seq not in colabfold_mapper.seq_to_rep_id:
                    colabfold_mapper.seq_to_rep_id[seq] = chain_ids[0]
                    colabfold_mapper.rep_id_to_seq[chain_ids[0]] = seq
                    colabfold_mapper.chain_id_to_rep_id[chain_ids[0]] = chain_ids[0]
                    colabfold_mapper.seqs.append(seq)
                    colabfold_mapper.rep_ids.append(chain_ids[0])
                else:
                    for chain_id in chain_ids:
                        colabfold_mapper.chain_id_to_rep_id[chain_id] = (
                            colabfold_mapper.seq_to_rep_id[seq]
                        )

                # Collect paired MSA data
                for chain_id in chain_ids:
                    rep_ids_query.append(colabfold_mapper.chain_id_to_rep_id[chain_id])

        # Only do pairing if number of unique protein sequences is > 1
        if len(set(rep_ids_query)) > 1:
            complex_id = ComplexID(*sorted(rep_ids_query, key=lambda c: c.stringify()))
            if complex_id not in colabfold_mapper.complex_ids:
                colabfold_mapper.complex_ids.add(complex_id)
            colabfold_mapper.query_name_to_complex_id[query_name] = complex_id

    return colabfold_mapper


def save_colabfold_mappings(
    colabfold_msa_input: ColabFoldMapper, output_directory: Path
) -> None:
    """_summary_

    Args:
        colabfold_msa_input (InferenceQueryCacheMsaData): _description_
        output_directory (Path): _description_
    """

    mapping_files_directory_path = output_directory / "mappings"
    mapping_files_directory_path.mkdir(parents=True, exist_ok=True)
    for mapping_name, mapping in zip(
        [
            "seq_to_rep_id",
            "rep_id_to_seq",
            "chain_id_to_rep_id",
            "query_name_to_complex_id",
        ],
        [
            colabfold_msa_input.seq_to_rep_id,
            colabfold_msa_input.rep_id_to_seq,
            colabfold_msa_input.chain_id_to_rep_id,
            colabfold_msa_input.query_name_to_complex_id,
        ],
    ):
        mapping_file_path = mapping_files_directory_path / f"{mapping_name}.json"
        with open(mapping_file_path, "w") as f:
            json.dump(mapping, f, indent=4)
    with open(mapping_files_directory_path / "README.md", "w") as f:
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
    """_summary_

    Attributes:
        colabfold_mapper (ColabFoldMapper): _description_
        output_directory (Path): _description_
        user_agent (str): _description_
    """

    def __init__(
        self, colabfold_mapper: ColabFoldMapper, output_directory: Path, user_agent: str
    ):
        """_summary_

        Args:
            colabfold_mapper (ColabFoldMapper): _description_
            output_directory (Path): _description_
            user_agent (str): _description_
        """
        self.colabfold_mapper = colabfold_mapper
        self.output_directory = output_directory
        self.user_agent = user_agent
        self.output_directory.mkdir(parents=True, exist_ok=True)
        for subdir in ["raw", "unpaired", "paired"]:
            (self.output_directory / subdir).mkdir(parents=True, exist_ok=True)
            if subdir == "raw":
                for subsubdir in ["unpaired", "paired"]:
                    (self.output_directory / subdir / subsubdir).mkdir(
                        parents=True, exist_ok=True
                    )

    def query_format_unpaired(self):
        """_summary_"""
        # Submit query for unpaired MSAs
        # TODO: add template alignments fetching code here by setting use_templates=True
        # TODO: replace prints with proper logging
        print(
            f"Submitting {len(self.colabfold_mapper.seqs)} sequences to the Colabfold"
            " MSA server for unpaired MSAs..."
        )
        # TODO: chunking
        # TODO: warn if too many sequences maybe?
        a3m_lines_unpaired = query_colabfold_msa_server(
            self.colabfold_mapper.seqs,
            prefix=self.output_directory / "raw/unpaired",
            use_templates=False,
            use_pairing=False,
            user_agent=self.user_agent,
        )

        unpaired_alignments_path = self.output_directory / "unpaired"
        unpaired_alignments_path.mkdir(parents=True, exist_ok=True)

        for rep_id, aln in zip(self.colabfold_mapper.rep_ids, a3m_lines_unpaired):
            rep_dir = unpaired_alignments_path / rep_id.stringify()
            rep_dir.mkdir(parents=True, exist_ok=True)

            # TODO: add code for which format to save the MSA in
            # If save as a3m...
            a3m_file = rep_dir / "colabfold_unpaired.a3m"
            with open(a3m_file, "w") as f:
                f.write(aln)

            # If save as npz...
            npz_file = rep_dir / "colabfold_unpaired.npz"
            npz_object = parse_a3m(aln)
            np.savez_compressed(npz_file, npz_object)

    def query_format_paired(self):
        """_summary_"""
        paired_alignments_directory = self.output_directory / "paired"
        paired_alignments_directory.mkdir(parents=True, exist_ok=True)
        # Submit queries for paired MSAss
        print(
            f"Submitting {len(self.colabfold_mapper.complex_ids)} paired MSA queries"
            " to the Colabfold MSA server..."
        )
        for complex_id in tqdm(
            self.colabfold_mapper.complex_ids,
            total=len(self.colabfold_mapper.complex_ids),
            desc="Computing paired MSAs",
        ):
            # Get the representative sequences for the query
            seqs_query = [
                self.colabfold_mapper.rep_id_to_seq[rep_id] for rep_id in complex_id
            ]

            # Submit the query to the Colabfold MSA server
            (self.output_directory / "raw/paired").mkdir(parents=True, exist_ok=True)
            a3m_lines_paired = query_colabfold_msa_server(
                seqs_query,
                prefix=self.output_directory / f"raw/paired/{complex_id.stringify()}",
                use_templates=False,
                use_pairing=True,
                user_agent=self.user_agent,
            )

            # TODO: process the returned MSAs - save per representative ID
            complex_directory = paired_alignments_directory / complex_id.stringify()
            for rep_id, aln in zip(complex_id, a3m_lines_paired):
                rep_directory = complex_directory / rep_id.stringify()
                rep_directory.mkdir(parents=True, exist_ok=True)

                # If save as a3m...
                a3m_file = rep_directory / "colabfold_paired.a3m"
                with open(a3m_file, "w") as f:
                    f.write(aln)

                # If save as npz...
                npz_file = rep_directory / "colabfold_paired.npz"
                npz_object = parse_a3m(aln)
                np.savez_compressed(npz_file, npz_object)

    def cleanup(self):
        """_summary_"""
        # TODO add code to optionally clean up the raw MSA files


# TODO use pydantic object as inputs
def preprocess_colabfold_msas(
    inference_set: InferenceQuerySet,
    output_directory: Path,
    user_agent: str,
    save_mappings: bool = False,
):
    """_summary_

    Args:
        inference_set (InferenceQuerySet): _description_
        output_directory (Path): _description_
        user_agent (str): _description_
    """
    # Gather MSA data
    colabfold_mapper = collect_colabfold_msa_data(inference_set)

    # Save mappings to file
    if save_mappings:
        save_colabfold_mappings(colabfold_mapper, output_directory)

    # Run batch queries for unpaired and paired MSAs
    colabfold_query_runner = ColabFoldQueryRunner(
        colabfold_mapper=colabfold_mapper,
        output_directory=output_directory,
        user_agent=user_agent,
    )
    colabfold_query_runner.query_format_unpaired()
    colabfold_query_runner.query_format_paired()

    # Add paths to the IQS
