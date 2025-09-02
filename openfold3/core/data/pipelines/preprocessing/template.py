"""Preprocessing pipelines for template data ran before training/evaluation."""

import hashlib
import multiprocessing as mp
import os
import random
import tempfile
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Annotated, Literal, Optional

import numpy as np
import pandas as pd
from biotite.database.rcsb import fetch
from biotite.structure.io import pdbx
from biotite.structure.io.pdbx import CIFFile
from pydantic import (
    BaseModel,
    BeforeValidator,
    DirectoryPath,
    model_validator,
)
from tqdm import tqdm

from openfold3.core.config.config_utils import (
    _convert_molecule_type,
    _ensure_list,
)
from openfold3.core.data.io.dataset_cache import read_datacache, write_datacache_to_json
from openfold3.core.data.io.s3 import open_local_or_s3
from openfold3.core.data.io.sequence.template import (
    A3mParser,
    TemplateData,
    parse_entry_chain_id,
    parse_hmmsearch_sto,
    parse_template_alignment,
)
from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz
from openfold3.core.data.io.structure.cif import _load_ciffile, parse_mmcif
from openfold3.core.data.primitives.caches.format import DatasetCache
from openfold3.core.data.primitives.quality_control.logging_utils import (
    PDB_ID,
    TEMPLATE_PROCESS_LOGGER,
    configure_template_logger,
)
from openfold3.core.data.primitives.sequence.template import (
    TemplateHitCollection,
    _TemplateQueryEntry,
    check_release_date_diff,
    check_release_date_max,
    check_sequence,
    create_residue_idx_map,
    match_query_chain_and_sequence,
    match_template_chain_and_sequence,
    parse_representatives,
)
from openfold3.core.data.primitives.structure.component import BiotiteCCDWrapper
from openfold3.core.data.primitives.structure.metadata import (
    get_asym_id_to_canonical_seq_dict,
    get_cif_block,
    get_release_date,
)
from openfold3.core.data.primitives.structure.template import clean_template_atom_array
from openfold3.core.data.resources.residues import (
    MoleculeType,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)


# TODO: rename variables to be PDB-agnostic
# --- Template alignment preprocessing ---
# Step 1/3: Create sequence cache for template structures
def create_seq_cache_for_template(
    template_pdb_id: str,
    template_structures_directory: Path,
    template_cache_directory: Path,
    template_file_format: str,
) -> None:
    """Creates a sequence cache for a template structure.

    Args:
        template_pdb_id (str):
            The PDB ID of the template structure.
        template_structures_directory (Path):
            Path to the directory containing template structures in mmCIF format.
        template_cache_directory (Path):
            Path to directory where the sequence cache will be saved for the template.
        template_file_format (str):
            File format of the template structures.
    """

    template_process_logger = TEMPLATE_PROCESS_LOGGER.get()

    try:
        cif_file = _load_ciffile(
            template_structures_directory / f"{template_pdb_id}.{template_file_format}"
        )
        chain_id_seq_map = get_asym_id_to_canonical_seq_dict(cif_file)
        release_date = get_release_date(get_cif_block(cif_file)).strftime("%Y-%m-%d")

        template_precache = {
            "release_date": release_date,
            "chain_id_seq_map": chain_id_seq_map,
        }

        np.savez_compressed(
            template_cache_directory.parent
            / Path(f"template_precache/{template_pdb_id}.npz"),
            **template_precache,
        )
        template_process_logger.info(f"Sequence cache for {template_pdb_id} saved.")
    except FileNotFoundError:
        template_process_logger.info(
            f"Template structure {template_pdb_id} not found in "
            f"{template_structures_directory}. Skipping this template."
        )
        return


class _OF3TemplateSequenceCacheConstructor:
    def __init__(
        self,
        template_structures_directory,
        template_cache_directory,
        template_file_format,
        log_level,
        log_to_file,
        log_to_console,
        log_dir,
    ):
        self.template_structures_directory = template_structures_directory
        self.template_cache_directory = template_cache_directory
        self.template_file_format = template_file_format
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_dir = log_dir

    @wraps(create_seq_cache_for_template)
    def __call__(self, template_pdb_id: str) -> None:
        try:
            # Create logger and set it as the context logger for the process
            TEMPLATE_PROCESS_LOGGER.set(
                configure_template_logger(
                    log_level=self.log_level,
                    log_to_file=self.log_to_file,
                    log_to_console=self.log_to_console,
                    log_dir=self.log_dir,
                )
            )
            # Create sequence cache for template
            create_seq_cache_for_template(
                template_pdb_id=template_pdb_id,
                template_structures_directory=self.template_structures_directory,
                template_cache_directory=self.template_cache_directory,
                template_file_format=self.template_file_format,
            )
        except Exception as e:
            TEMPLATE_PROCESS_LOGGER.get().info(
                f"Failed to process template {template_pdb_id}:\n{e}\n"
            )


def create_template_seq_cache_of3(
    template_structures_directory: Path,
    template_cache_directory: Path,
    template_file_format: str,
    num_workers: int,
    log_level: str,
    log_to_file: bool,
    log_to_console: bool,
    log_dir: Path,
):
    """Creates the sequence cache for all template structures.

    Args:
        template_structures_directory (Path):
            Path to the directory containing template structures in mmCIF format.
        template_cache_directory (Path):
            Path to directory where the sequence cache will be saved for the template.
        template_file_format (str):
            File format of the template structures.
        num_workers (int):
            Number of workers to use for multiprocessing.
        log_level (str):
            Log level for the logger.
        log_to_file (bool):
            Whether to log to file.
        log_to_console (bool):
            Whether to log to console.
        log_dir (Path):
            Directory where the log file will be saved.
    """

    (template_cache_directory.parent / Path("template_precache")).mkdir(
        parents=True, exist_ok=True
    )
    (template_cache_directory.parent / Path("template_seq_logs")).mkdir(
        parents=True, exist_ok=True
    )
    template_pdb_ids = [
        f.stem
        for f in list(template_structures_directory.glob(f"*.{template_file_format}"))
    ]
    wrapped_template_seq_cache_constructor = _OF3TemplateSequenceCacheConstructor(
        template_structures_directory,
        template_cache_directory,
        template_file_format,
        log_level,
        log_to_file,
        log_to_console,
        log_dir,
    )
    with mp.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                wrapped_template_seq_cache_constructor,
                template_pdb_ids,
                chunksize=1,
            ),
            total=len(template_pdb_ids),
            desc="1/3: Creating template sequence cache",
        ):
            pass


# TODO: clean up this function!
# Step 2/3: Create template cache for query chains
def create_template_cache_for_query(
    query_pdb_chain_id: str,
    rep_pdb_chain_id: str,
    template_alignment_file: Path,
    template_structures_directory: Path,
    template_cache_directory: Path,
    query_structures_directory: Path,
    max_templates_construct: int,
    query_structures_filename: str,
    query_file_format: str,
    query_seq_load_logic: str,
    s3_client_config: dict | None,
) -> None:
    """Creates a json cache of filtered template hits for a query.

    A query denotes a single protein chain for which template hits are to be filtered,
    i.e. corresponding to a protein chain whose structure needs to be predicted during
    training, evaluation or inference.

    Note that a template is skipped if:
        - no CIF file is provided for the QUERY against which the template was aligned;
        the PDB IDs of the QUERY need to match between the alignment and the CIF file -
        there is a mismatch between the author chain IDs for the QUERY against which the
        template was aligned AND the sequence provided in the alignment file cannot be
        remapped to an exact subsequence of any chains in the QUERY CIF file - no CIF
        file is provided for the TEMPLATE; the PDB IDs of the TEMPLATE need to match
        between the alignment and the CIF file - there is a mismatch between the author
        chain IDs for the TEMPLATE AND the sequence provided in the alignment file
        cannot be remapped to an exact subsequence of any chains in the TEMPLATE CIF
        file - the sequence of the template does not pass the AF3 sequence filters

    Another note: the template alignment is parsed from the directory indicated by the
    representative ID of a query chain, whereas the query structure is parsed using the
    PDB ID-chain pair of the query chain.

    The cache contains the e-value and mapping from the query residues to the template
    hit residues.

    Args:
        query_pdb_chain_id (str):
            The PDB ID and chain ID of the query chain.
        rep_pdb_chain_id (str):
            The PDB ID and chain ID of the representative chain for the query chain.
        template_alignment_file (Path):
            Path to the template alignment stockholm file. Currently only the output of
            hmmsearch is accepted.
        template_structures_directory (Path):
            Path to the directory containing template structures in mmCIF format. The
            PDB IDs of the template CIF files need to match the PDB IDs in the alignment
            file for a template to be used.
        template_cache_directory (Path):
            Path to directory where the template cache will be saved for the query.
        query_structures_directory (Path):
            Path to the directory containing query structures in mmCIF format. The PDB
            IDs of the query CIF files need to match the PDB IDs for the query (1st row)
            in the alignment file for it to have any templates.
        max_templates_construct (int):
            Maximum number of templates to keep per query chain during template cache
            construction.
        query_structures_filename (str):
            Name of the query structure file within each query structure subdirectory.
            Uses the the subdir name if set to "None".
        query_file_format (str):
            File format of the query structures.
        query_seq_load_logic (str):
            Whether to derive the structure-associated query sequence from a structure
            file or a preprocessed fasta file. Should be one of "structure" or "fasta".
        s3_client_config (dict | None):
            Configuration for the S3 client. Should contain the profile name for the S3
            client.
    """
    template_process_logger = TEMPLATE_PROCESS_LOGGER.get()

    data_log = {
        "query_pdb_id": query_pdb_chain_id.split("_")[0],
        "query_chain_id": query_pdb_chain_id.split("_")[1],
        "can_load_aln_file": False,
        "template_cache_already_computed": False,
        "query_cif_exists": False,
        "query_seq_match": False,
        "n_total_templates_in_aln": 0,
        "n_unique_templates_in_aln": 0,
        "n_templates_pass_seq_filters": 0,
        "n_templates_has_cif": 0,
        "n_template_chain_match": 0,
        "n_valid_templates_prefilter": 0,
    }
    if s3_client_config is not None:
        profile = s3_client_config["profile"]
    else:
        profile = None

    # Parse alignment
    try:
        with open_local_or_s3(template_alignment_file, profile=profile, mode="r") as f:
            hits = parse_hmmsearch_sto(f.read())
    except Exception as e:
        template_process_logger.info(
            f"Failed to parse alignment file {template_alignment_file}, skipping. Make"
            " sure that an hmmsearch output globally aligned with hmmalign was "
            f"provided. \nError: \n{e}\nTraceback: \n{traceback.format_exc()}"
        )
        data_log_to_tsv(
            data_log,
            template_cache_directory.parent
            / Path(f"data_log/data_log_{os.getpid()}.tsv"),
        )
        return
    template_process_logger.info(f"Alignment file {template_alignment_file} parsed.")
    data_log["can_load_aln_file"] = True

    # Filter queries
    query = hits[0]
    query_pdb_id, query_chain_id = parse_entry_chain_id(query_pdb_chain_id)
    query_pdb_id_t, query_chain_id_t = parse_entry_chain_id(query.name)
    template_cache_path_rep = template_cache_directory / Path(f"{rep_pdb_chain_id}.npz")
    if template_cache_path_rep.exists():
        template_process_logger.info(
            f"Template cache for {query.name} already exists. Skipping templates for "
            "this structure."
        )
        data_log["template_cache_already_computed"] = True
        data_log_to_tsv(
            data_log,
            template_cache_directory.parent
            / Path(f"data_log/data_log_{os.getpid()}.tsv"),
        )
        return
    # 1. Parse fasta of the structure
    # the query and all its templates are skipped if the structure identified by the PDB
    # ID of the first hit in the alignments file is not provided in
    # query_structures_directory
    query_structures_filename = (
        query_pdb_id
        if query_structures_filename == "None"
        else query_structures_filename
    )
    qp = query_structures_directory / Path(
        f"{query_structures_filename}.{query_file_format}"
    )
    # Currently only checking existence of local files
    if (not str(qp).startswith("s3:/")) and (not (qp).exists()):
        template_process_logger.info(
            f"Query .{query_file_format} structure {query_structures_filename} not "
            f"found in  {query_structures_directory}. Skipping templates for this "
            "structure."
        )
        data_log_to_tsv(
            data_log,
            template_cache_directory.parent
            / Path(f"data_log/data_log_{os.getpid()}.tsv"),
        )
        return
    data_log["query_cif_exists"] = True
    # 2. Parse query chain and sequence
    # the query and all its templates are skipped if its HMM sequence cannot be mapped
    # exactly to a subsequence of the MATCHING chain in the CIF/PDB file provided in
    # query_structures_directory (no chain/sequence remapping is done)
    if match_query_chain_and_sequence(
        query_structures_directory,
        query,
        query_pdb_id,
        query_chain_id,
        query_seq_load_logic,
        query_file_format,
        query_structures_filename,
        s3_profile=profile,
    ):
        template_process_logger.info(
            f"The query sequences in the structure (query {query_pdb_id} chain "
            f"{query_chain_id}) and template alignment (query {query_pdb_id_t} chain "
            f"{query_chain_id_t}) don't match. Skipping templates for this structure."
        )
        data_log_to_tsv(
            data_log,
            template_cache_directory.parent
            / Path(f"data_log/data_log_{os.getpid()}.tsv"),
        )
        return
    else:
        template_process_logger.info(
            f"Query {query_pdb_id} chain {query_chain_id} sequence matches "
            "alignment sequence."
        )
        data_log["query_seq_match"] = True

    # Filter template hits
    data_log["n_total_templates_in_aln"] = len(hits)
    data_log["n_unique_templates_in_aln"] = len(
        set(hit.hit_sequence for hit in hits.values())
    )
    filtered_seq = set()
    template_hits_filtered = {}
    for idx, hit in hits.items():
        # Skip query
        if idx == 0:
            continue

        hit_pdb_id, hit_chain_id = hit.name.split("_")

        # Skip hits if sequence alignment already used
        if hit.hit_sequence in filtered_seq:
            template_process_logger.info(
                f"Template {hit.name} sequence alignment is a duplicate. "
                "Skipping this template."
            )
            continue

        # 1. Apply sequence filters: AF3 SI Section 2.4
        if check_sequence(query_seq=query.hit_sequence.replace("-", ""), hit=hit):
            template_process_logger.info(
                f"Template {hit_pdb_id} sequence does not pass sequence"
                " filters. Skipping this template."
            )
            continue
        else:
            template_process_logger.info(
                f"Template {hit_pdb_id} sequence passes sequence filters."
            )
            data_log["n_templates_pass_seq_filters"] += 1

        # 2. Parse structure
        # The template is skipped if the structure identified by the PDB ID of the
        # corresponding hit in the alignment file is not provided in
        # template_structures_path
        try:
            template_precache = np.load(
                template_cache_directory.parent
                / Path(f"template_precache/{hit_pdb_id}.npz"),
                allow_pickle=True,
            )
            chain_id_seq_map = template_precache["chain_id_seq_map"].item()
            release_date = template_precache["release_date"].item()
        except FileNotFoundError:
            chain_id_seq_map = None
            release_date = None

        if chain_id_seq_map is None:
            template_process_logger.info(
                f"Template structure {hit_pdb_id} not found in "
                f"{template_structures_directory}. Skipping this template."
            )
            continue
        else:
            template_process_logger.info(f"Template structure {hit_pdb_id} parsed.")
            data_log["n_templates_has_cif"] += 1
        # 3. Parse template chain and sequence
        # the template is skipped if its HMM sequence cannot be mapped
        # exactly to a subsequence of ANY chain in the CIF file provided in
        # template_structures_path with a PDB ID matching the the hit's PDB ID
        # in the alignment file
        # !!! Note that the chain ID-sequence map for this step is derived from the
        # unprocessed CIF file provided in the template_structures_directory
        hit_chain_id_matched = match_template_chain_and_sequence(chain_id_seq_map, hit)
        if hit_chain_id_matched is None:
            template_process_logger.info(
                f"Could not match template {hit_pdb_id} chain {hit_chain_id} "
                f"sequence in {chain_id_seq_map}. Skipping this template."
            )
            continue
        else:
            data_log["n_template_chain_match"] += 1

        # Create residue index map
        idx_map = create_residue_idx_map(query, hit)

        # Store as filtered hit
        # hmmsearch is sorted in descending e-value order so index is enough to sort
        template_hits_filtered[f"{hit_pdb_id}_{hit_chain_id_matched}"] = {
            "index": hit.index,
            "release_date": release_date,
            "idx_map": idx_map,
        }

        # Store sequence alignment for hit as already used
        filtered_seq.add(hit.hit_sequence)

        # Break if max templates reached
        if len(template_hits_filtered) == max_templates_construct:
            template_process_logger.info(
                f"Max number of templates ({max_templates_construct}) reached."
            )
            break

    # Save data log
    data_log["n_valid_templates_prefilter"] = len(template_hits_filtered)
    data_log_to_tsv(
        data_log,
        template_cache_directory.parent / Path(f"data_log/data_log_{os.getpid()}.tsv"),
    )

    # Save filtered hits to json using the representative ID
    if len(template_hits_filtered) > 0:
        np.savez_compressed(template_cache_path_rep, **template_hits_filtered)
        template_process_logger.info(
            f"Template cache for {query.name} saved with "
            f"{len(template_hits_filtered)} valid hits."
        )
    else:
        # TODO optimize to not recompute template empty template caches multiple times
        template_process_logger.info(f"0 valid templates found for {query.name}.")


class _OF3TemplateCacheConstructor:
    def __init__(
        self,
        template_alignment_directory: Path,
        template_alignment_filename: str,
        template_structures_directory: Path,
        template_cache_directory: Path,
        query_structures_directory: Path,
        max_templates_construct: int,
        query_structures_filename: str,
        query_file_format: str,
        query_seq_load_logic: str,
        log_level: str,
        log_to_file: bool,
        log_to_console: bool,
        log_dir: Path,
        s3_client_config: dict | None,
    ) -> None:
        """Wrapper class for creating the template cache.

        This wrapper around `create_template_cache_for_query` is needed for
        multiprocessing, so that we can pass the constant arguments in a convenient way
        catch any errors that would crash the workers, and change the function call to
        accept a single Iterable.

        The wrapper is written as a class object because multiprocessing doesn't support
        decorator-like nested functions.

        Attributes:
            template_alignment_directory (Path):
                Directory containing directories per query chain, with each subdirectory
                  containing hhsearch alignments per chain.
            template_alignment_filename (str):
                Name of the hhsearch aligment file within each query chain subdirectory.
                Needs to be identical for all query chains.
            template_structures_directory (Path):
                Directory containing the template CIF files.
            template_cache_directory (Path):
                Directory where template cache jsons per chain will be saved.
            query_structures_directory (Path):
                Directory containing the query CIF files.
            max_templates_construct (int):
                Maximum number of templates to keep per query chain during template
                cache construction.
            query_structures_filename (str):
                Name of the query structure file within each query structure
                subdirectory. Uses the the subdir name if set to "None".
            query_file_format (str):
                File format of the query structures.
            query_seq_load_logic (str):
                Whether to load the query sequence from the structure file or a
                preprocessed fasta file. Should be one of "structure" or "fasta".
            log_level (str):
                Log level for the logger.
            log_to_file (bool):
                Whether to log to file.
            log_dir (Path):
                Directory where the log file will be saved.
            s3_client_config (dict | None):
                Configuration for the S3 client.

        """
        self.template_alignment_directory = template_alignment_directory
        self.template_alignment_filename = template_alignment_filename
        self.template_structures_directory = template_structures_directory
        self.template_cache_directory = template_cache_directory
        self.query_structures_directory = query_structures_directory
        self.max_templates_construct = max_templates_construct
        self.query_structures_filename = query_structures_filename
        self.query_file_format = query_file_format
        self.query_seq_load_logic = query_seq_load_logic
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_dir = log_dir
        self.s3_client_config = s3_client_config

    @wraps(create_template_cache_for_query)
    def __call__(self, input: _TemplateQueryEntry) -> None:
        try:
            # Create logger and set it as the context logger for the process
            TEMPLATE_PROCESS_LOGGER.set(
                configure_template_logger(
                    log_level=self.log_level,
                    log_to_file=self.log_to_file,
                    log_to_console=self.log_to_console,
                    log_dir=self.log_dir,
                )
            )

            # Parse query and representative IDs
            query_pdb_chain_id, rep_pdb_chain_id = (
                input.dated_query.query_pdb_chain_id,
                input.rep_pdb_chain_id,
            )
            query_pdb_id = query_pdb_chain_id.split("_")[0]
            template_alignment_file = (
                self.template_alignment_directory
                / Path(rep_pdb_chain_id)
                / self.template_alignment_filename
            )
            # Create template cache for query
            create_template_cache_for_query(
                query_pdb_chain_id=query_pdb_chain_id,
                rep_pdb_chain_id=rep_pdb_chain_id,
                template_alignment_file=template_alignment_file,
                template_structures_directory=self.template_structures_directory,
                template_cache_directory=self.template_cache_directory,
                query_structures_directory=self.query_structures_directory
                / Path(query_pdb_id),
                max_templates_construct=self.max_templates_construct,
                query_structures_filename=self.query_structures_filename,
                query_file_format=self.query_file_format,
                query_seq_load_logic=self.query_seq_load_logic,
                s3_client_config=self.s3_client_config,
            )
        except Exception as e:
            TEMPLATE_PROCESS_LOGGER.get().info(
                f"Failed to process templates for query {query_pdb_chain_id}:\n{e}\n"
            )


def create_template_cache_of3(
    dataset_cache_file: Path,
    template_alignment_directory: Path,
    template_alignment_filename: str,
    template_structures_directory: Path,
    template_cache_directory: Path,
    query_structures_directory: Path,
    max_templates_construct: int,
    query_structures_filename: str,
    query_file_format: str,
    query_seq_load_logic: str,
    single_moltype: str | None,
    num_workers: int,
    log_level: str,
    log_to_file: bool,
    log_to_console: bool,
    log_dir: Path,
    s3_client_config: dict | None,
) -> None:
    """Creates the full template cache for all query chains.

    Uses

    Args:
        dataset_cache_file (Path):
            Path to the metadata cache json file.
        template_alignment_directory (Path):
            Directory containing directories per query chain, with each subdirectory
            containing template alignments per chain.
        template_alignment_filename (str):
            Name of the template alignment file within each query chain subdirectory.
            Needs to be identical for all query chains.
        template_structures_directory (Path):
            Directory containing the template structures in mmCIF format.
        template_cache_directory (Path):
            Directory where the template cache jsons per chain will be saved.
        query_structures_directory (Path):
            Directory containing the query structures in mmCIF format.
        max_templates_construct (int):
            Maximum number of templates to keep per query chain during template cache
            construction.
        query_structures_filename (str):
            Name of the query structure file within each query structure subdirectory.
            Uses the the subdir name if set to "None".
        query_file_format (str):
            File format of the query structures.
        query_seq_load_logic (str):
            Whether to load the query sequence from the structure file or a preprocessed
            fasta file. Should be one of "structure" or "fasta".
        single_moltype (str | None):
            Constant molecule type to use if the dataset contains only a single molecule
            type and the dataset cache does not contain per-chain molecule type field.
            If None, the molecule type is inferred from the dataset cache.
        num_workers (int):
            Number of workers to use for multiprocessing.
        log_level (str):
            Log level for the logger.
        log_to_file (bool):
            Whether to log to file.
        log_to_console (bool):
            Whether to log to console.
        log_dir (Path):
            Directory where the log file will be saved.
        s3_client_config (dict | None):
            Configuration for the S3 client.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    template_cache_directory.mkdir(parents=True, exist_ok=True)
    (template_cache_directory.parent / Path("data_log")).mkdir(
        parents=True, exist_ok=True
    )
    # Parse list of chains from metadata cache
    dataset_cache = read_datacache(dataset_cache_file)
    template_query_iterator = parse_representatives(
        dataset_cache, True, single_moltype
    ).entries
    # Create template cache for each query chain
    wrapped_template_cache_constructor = _OF3TemplateCacheConstructor(
        template_alignment_directory,
        template_alignment_filename,
        template_structures_directory,
        template_cache_directory,
        query_structures_directory,
        max_templates_construct,
        query_structures_filename,
        query_file_format,
        query_seq_load_logic,
        log_level,
        log_to_file,
        log_to_console,
        log_dir,
        s3_client_config,
    )
    with mp.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                wrapped_template_cache_constructor,
                template_query_iterator,
                chunksize=1,
            ),
            total=len(template_query_iterator),
            desc="2/3: Creating template cache",
        ):
            pass
    # Collate data logs
    collate_data_logs(template_cache_directory, "full_data_log_constructed_cache.tsv")


# Step 3/3: Filter template cache for query chains and add to dataset cache
def filter_template_cache_for_query(
    input_data: _TemplateQueryEntry,
    template_cache_directory: Path,
    max_templates: int,
    is_core_train: bool,
    max_release_date: Optional[datetime | str] = None,
    min_release_date_diff: Optional[int] = None,
) -> TemplateHitCollection:
    """Filters the template cache for a query chain.

    Note: returns an empty dict if template_cache_directory does not contain a json file
    for the alignment representative of a query chain.

    Args:
        input_data (_TemplateQueryEntry):
            Tuple containing the representative ID - query PDB - chain ID pair with
            query release dates or the representative ID and a list of query PDB -
            chain ID release date pairs.
        template_cache_directory (Path):
            Directory containing template cache jsons per chain.
        max_templates (int):
            Maximum number of templates to keep per query chain.
        is_core_train (bool):
            Whether the dataset is core train or not.
        max_release_date (Optional[datetime], optional):
            Maximum release date for templates. Defaults to None.
        min_release_date_diff (Optional[int], optional):
            Minimum release date difference for core train templates. Defaults to None.

    Returns:
        TemplateHitCollection:
            Dict mapping a query PDB - chain ID pair to a list of valid template
            representative IDs.
    """
    template_process_logger = TEMPLATE_PROCESS_LOGGER.get()

    # Unpack input and format release dates
    if is_core_train:
        rep_id, (query_pdb_chain_id, query_release_date) = (
            input_data.rep_pdb_chain_id,
            (
                input_data.dated_query.query_pdb_chain_id,
                datetime.strptime(
                    input_data.dated_query.query_release_date, "%Y-%m-%d"
                ),
            ),
        )
    else:
        rep_id, query_pdb_chain_ids_release_dates = (
            input_data.rep_pdb_chain_id,
            input_data.dated_query,
        )
        query_pdb_chain_id = query_pdb_chain_ids_release_dates[0].query_pdb_chain_id
        if isinstance(max_release_date, str):
            max_release_date = datetime.strptime(max_release_date, "%Y-%m-%d")
    data_log = {
        "query_pdb_id": query_pdb_chain_id.split("_")[0],
        "query_chain_id": query_pdb_chain_id.split("_")[1],
        "can_load_template_cache": False,
        "n_valid_templates_prefilter": 0,
        "n_dropped_due_to_release_date": 0,
        "n_valid_templates_postfilter": 0,
    }

    # Parse template cache of the representative if available
    template_cache_file = template_cache_directory / Path(f"{rep_id}.npz")
    if template_cache_file.exists():
        template_cache = np.load(template_cache_file, allow_pickle=True)
    else:
        template_process_logger.info(
            f"Template cache for representative {rep_id} not found. Returning no valid "
            "templates."
        )
        data_log_to_tsv(
            data_log,
            template_cache_directory.parent
            / Path(f"data_log/data_log_{os.getpid()}.tsv"),
        )
        if is_core_train:
            return TemplateHitCollection({tuple(query_pdb_chain_id.split("_")): []})
        else:
            return TemplateHitCollection(
                {
                    tuple(query_pdb_chain_id[0].split("_")): []
                    for query_pdb_chain_id in query_pdb_chain_ids_release_dates
                }
            )

    # Sort by index/e-value
    unpacked_template_cache = {
        key: value.item() for key, value in template_cache.items()
    }
    sorted_template_cache = sorted(
        unpacked_template_cache.items(), key=lambda x: x[1]["index"]
    )
    ids_dates = [
        (template_id, datetime.strptime(template_data["release_date"], "%Y-%m-%d"))
        for template_id, template_data in sorted_template_cache
    ]
    data_log["n_valid_templates_prefilter"] = len(ids_dates)

    # Filter templates
    filtered_templates = []
    for template_id, template_date in ids_dates:
        # Apply release date filters
        if is_core_train:
            if not check_release_date_diff(
                query_release_date=query_release_date,
                template_release_date=template_date,
                min_release_date_diff=min_release_date_diff,
            ):
                data_log["n_dropped_due_to_release_date"] += 1
                continue
        else:
            if not check_release_date_max(
                template_release_date=template_date, max_release_date=max_release_date
            ):
                data_log["n_dropped_due_to_release_date"] += 1
                continue
        # Add to list of filtered templates if pass
        filtered_templates.append(template_id)
        data_log["n_valid_templates_postfilter"] += 1

        # Break if max templates reached
        if len(filtered_templates) == max_templates:
            break

    data_log_to_tsv(
        data_log,
        template_cache_directory.parent / Path(f"data_log/data_log_{os.getpid()}.tsv"),
    )
    template_process_logger.info(
        f"Successfully filtered {len(filtered_templates)} templates for "
        f"{query_pdb_chain_id}."
    )
    if is_core_train:
        return TemplateHitCollection(
            {tuple(query_pdb_chain_id.split("_")): filtered_templates}
        )
    else:
        return TemplateHitCollection(
            {
                tuple(query_pdb_chain_id[0].split("_")): filtered_templates
                for query_pdb_chain_id in query_pdb_chain_ids_release_dates
            }
        )


class _OF3TemplateCacheFilter:
    def __init__(
        self,
        template_cache_directory: Path,
        max_templates_filter: int,
        is_core_train: bool,
        max_release_date: datetime | None,
        min_release_date_diff: int | None,
        log_level: str,
        log_to_file: bool,
        log_to_console: bool,
        log_dir: Path,
    ) -> None:
        """Wrapper class for filtering the template cache and updating the dataset cache

        This wrapper around `filter_template_cache_for_query` is needed for
        multiprocessing, so that we can pass the constant arguments in a convenient way
        catch any errors that would crash the workers, and change the function call to
        accept a single Iterable.

        The wrapper is written as a class object because multiprocessing doesn't support
        decorator-like nested functions.

        Attributes:
            template_cache_directory (Path):
                Directory containing template cache jsons per chain.
            max_templates_filter (int):
                Maximum number of templates to keep per query chain.
            is_core_train (bool):
                Whether the dataset is core train or not.
            max_release_date (datetime | None):
                Maximum release date for templates.
            min_release_date_diff (int | None):
                Minimum release date difference for core train templates.
            log_level (str):
                Log level for the logger.
            log_to_file (bool):
                Whether to log to file.
            log_to_console (bool):
                Whether to log to console.
            log_dir (Path):
                Directory where the log file will be saved.

        """
        self.template_cache_path = template_cache_directory
        self.max_templates_filter = max_templates_filter
        self.is_core_train = is_core_train
        self.max_release_date = max_release_date
        self.min_release_date_diff = min_release_date_diff
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_dir = log_dir

    @wraps(filter_template_cache_for_query)
    def __call__(self, input: _TemplateQueryEntry) -> TemplateHitCollection:
        try:
            # Create logger and set it as the context logger for the process
            TEMPLATE_PROCESS_LOGGER.set(
                configure_template_logger(
                    log_level=self.log_level,
                    log_to_file=self.log_to_file,
                    log_to_console=self.log_to_console,
                    log_dir=self.log_dir,
                )
            )

            # Filter templates for query
            valid_templates = filter_template_cache_for_query(
                input,
                self.template_cache_path,
                self.max_templates_filter,
                self.is_core_train,
                self.max_release_date,
                self.min_release_date_diff,
            )
            return valid_templates
        except Exception as e:
            if self.is_core_train:
                TEMPLATE_PROCESS_LOGGER.get().info(
                    f"Failed to filter templates for query {input[0][0]}: \n{e}\n"
                )
                return {input[0][0]: []}
            else:
                query_pdb_chain_ids = [
                    query_pdb_chain_id.query_pdb_chain_id
                    for query_pdb_chain_id in input.dated_query
                ]
                TEMPLATE_PROCESS_LOGGER.get().info(
                    "Failed to filter templates for queries "
                    f"{query_pdb_chain_ids}: "
                    f"\n{e}\n"
                )
                return {
                    query_pdb_chain_id: [] for query_pdb_chain_id in query_pdb_chain_ids
                }


def filter_template_cache_of3(
    dataset_cache_file: Path,
    updated_dataset_cache_file: Path,
    template_cache_directory: Path,
    max_templates_filter: int,
    single_moltype: str | None,
    is_core_train: bool,
    num_workers: int,
    log_level: str,
    log_to_file: bool,
    log_to_console: bool,
    log_dir: Path,
    max_release_date: datetime | None = None,
    min_release_date_diff: int | None = None,
) -> None:
    """Filters the template cache and updates the dataset cache with valid template IDs.

    Args:
        dataset_cache_file (Path):
            Path to the dataset cache json file.
        updated_dataset_cache_file (Path):
            Path to the updated dataset cache json file containing valid template
            representative IDs.
        template_cache_directory (Path):
            Path to the directory containing template cache jsons per chain.
        max_templates_filter (int):
            Maximum number of templates to keep per query chain.
        single_moltype (str | None):
            Constant molecule type to use if the dataset contains only a single molecule
            type and the dataset cache does not contain per-chain molecule type field.
            If None, the molecule type is inferred from the dataset cache.
        is_core_train (bool):
            Whether the dataset is core train or not.
        num_workers (int):
            Number of workers to use for multiprocessing.
        log_level (str):
            Log level for the logger.
        log_to_file (bool):
            Whether to log to file.
        log_to_console (bool):
            Whether to log to console.
        log_dir (Path):
            Directory where the log file will be saved.
        max_release_date (datetime | None):
            Maximum release date for templates. Defaults to None.
        min_release_date_diff (int | None):
            Minimum release date difference for core train templates. Defaults to None.
    """
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # Parse list of chains from metadata cache
    dataset_cache = read_datacache(dataset_cache_file)
    template_query_iterator = parse_representatives(
        dataset_cache, is_core_train, single_moltype
    ).entries
    data_iterator_len = len(template_query_iterator)

    # Filter template cache for each query chain
    wrapped_template_cache_filter = _OF3TemplateCacheFilter(
        template_cache_directory,
        max_templates_filter,
        is_core_train,
        max_release_date,
        min_release_date_diff,
        log_level,
        log_to_file,
        log_to_console,
        log_dir,
    )
    with mp.Pool(num_workers) as pool:
        for _, valid_templates in tqdm(
            enumerate(
                pool.imap_unordered(
                    wrapped_template_cache_filter,
                    template_query_iterator,
                    chunksize=30,
                )
            ),
            total=data_iterator_len,
            desc="3/3: Filtering template cache",
        ):
            # Update dataset cache with list of valid template representative IDs
            for grp in valid_templates.items():
                try:
                    (pdb_id, chain_id), valid_template_list = grp
                    dataset_cache.structure_data[pdb_id].chains[
                        chain_id
                    ].template_ids = valid_template_list
                except Exception as e:
                    print(f"Failed to update dataset cache for {grp}: \n{e}\n")

            # if (idx + 1) % save_frequency == 0:
            #     with open(updated_dataset_cache_file, "w") as f:
            #         json.dump(dataset_cache, f, indent=4)

    # Save final complete dataset cache
    write_datacache_to_json(dataset_cache, updated_dataset_cache_file)

    # Collate data logs
    collate_data_logs(template_cache_directory, "full_data_log_filtered_cache.tsv")


def data_log_to_tsv(data_log: dict, tsv_file: Path) -> None:
    """Writes the data log to a tsv file.

    Args:
        data_log (dict):
            Dictionary containing the data log.
        tsv_file (Path):
            Path to the tsv file where the data log will be saved.
    """
    file_exists = tsv_file.exists()
    with open(tsv_file, "a") as f:
        data_string = ""
        header_string = ""
        for key, value in data_log.items():
            header_string += f"{key}\t"
            data_string += f"{value}\t"
        # Remove final tab
        header_string = header_string[:-1] + "\n"
        data_string = data_string[:-1] + "\n"
        if not file_exists:
            f.write(header_string)
        f.write(data_string)
    return


def collate_data_logs(template_cache_directory, fname):
    files = [
        f
        for f in list(
            (template_cache_directory.parent / Path("data_log")).glob("data_log_*")
        )
        if f.is_file()
    ]
    df_all = pd.DataFrame()
    for f in files:
        df_all = pd.concat(
            [
                df_all,
                pd.read_csv(f, sep="\t", na_values=["NaN"]),
            ]
        )
        f.unlink()
    df_all.to_csv(
        template_cache_directory.parent / Path(f"{fname}"), sep="\t", index=False
    )


# --- Template structure preprocessing ---
# Step 1/1: Preprocess template structures
def preprocess_template_structure_for_template_old(
    template_pdb_id: str,
    template_structures_directory: Path,
    template_file_format: str,
    template_structure_array_directory: Path,
    ccd: CIFFile,
    moltypes_included: np.ndarray[int],
    log_dir: Path,
):
    """Preparse and process a template structure.

    Args:
        template_pdb_id (str):
            PDB ID of the template structure.
        template_structures_directory (Path):
            Path to the directory containing template structures in mmCIF format.
        template_file_format (str):
            File format of the template structures.
        template_structure_array_directory (Path):
            Path to the directory where the template structure arrays will be saved.
        ccd (CIFFile):
            The Chemical Component Dictionary.
        moltypes_included (np.ndarray[int]):
            Array of molecule types to include in the template structure arrays.
        log_dir (Path):
            Directory where the log file are saved. This dir will also contain the per-
            worker process logs for which samples have been successfully processed.
    """
    template_assembly_directory = template_structure_array_directory / Path(
        f"{template_pdb_id}"
    )

    # Parse template structure
    cif_file, atom_array_template_assembly = parse_mmcif(
        template_structures_directory
        / Path(f"{template_pdb_id}.{template_file_format}")
    )

    # Sanitize template structure
    atom_array_template = clean_template_atom_array(
        atom_array_template_assembly, cif_file, None, ccd
    )

    # Save template structure for each chain separately
    # TODO: can switch back to atom_array_template.label_asym_id once the dummy chain ID
    #  addition bug is resolved
    chain_ids = np.unique(atom_array_template.chain_id)
    chains_included = []
    for chain_id in chain_ids:
        # Find molecule type of chain and save if included
        atom_array_chain = atom_array_template[atom_array_template.chain_id == chain_id]
        chain_mol_type = np.array(list(set(atom_array_chain.molecule_type_id)))
        if len(chain_mol_type) != 1:
            raise ValueError(
                f"Multiple molecule types found in chain {chain_id} of "
                f"template {template_pdb_id}."
            )
        if np.isin(chain_mol_type, moltypes_included).all():
            chains_included.append(chain_id)
            write_atomarray_to_npz(
                atom_array=atom_array_chain,
                output_file=template_assembly_directory
                / f"{template_pdb_id}_{chain_id}.npz",
            )

    # Log as completed
    pid = os.getpid()
    line = "{}\t{}\t{}\n".format(template_pdb_id, pid, ",".join(chains_included))
    with open(log_dir / f"completed_{pid}.tsv", "a") as f:
        f.write(line)


class _OF3TemplateStructurePreprocessor:
    def __init__(
        self,
        template_structures_directory: Path,
        template_file_format: str,
        template_structure_array_directory: Path,
        ccd: CIFFile,
        moltypes_included: np.ndarray[int],
        log_level: str,
        log_to_file: bool,
        log_to_console: bool,
        log_dir: Path,
    ) -> None:
        """Wrapper class for preprocessing template structures.

        This wrapper around `preprocess_structure_for_template` is needed for
        multiprocessing, so that we can pass the constant arguments in a convenient way
        catch any errors that would crash the workers, and change the function call to
        accept a single Iterable.

        The wrapper is written as a class object because multiprocessing doesn't support
        decorator-like nested functions.

        Attributes:
            template_structures_directory (Path):
                Path to the directory containing template structures in mmCIF format.
            template_file_format (str):
                File format of the template structures.
            template_structure_array_directory (Path):
                Path to the directory where the template structure arrays will be saved.
            ccd (CIFFile):
                The parsed Chemical Component Dictionary.
            moltypes_included (np.ndarray[int]):
                Array of molecule types to include in the template structure arrays.
            log_level (str):
                Log level for the logger.
            log_to_file (bool):
                Whether to log to file.
            log_to_console (bool):
                Whether to log to console.
            log_dir (Path):
                Directory where the log file will be saved.
        """
        self.template_structures_directory = template_structures_directory
        self.template_file_format = template_file_format
        self.template_structure_array_directory = template_structure_array_directory
        self.ccd = ccd
        self.moltypes_included = moltypes_included
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_dir = log_dir

    @wraps(preprocess_template_structure_for_template_old)
    def __call__(self, template_pdb_id: str) -> None:
        try:
            # Create logger and set it as the context logger for the process
            TEMPLATE_PROCESS_LOGGER.set(
                configure_template_logger(
                    log_level=self.log_level,
                    log_to_file=self.log_to_file,
                    log_to_console=self.log_to_console,
                    log_dir=self.log_dir,
                )
            )
            PDB_ID.set(template_pdb_id)
            # Preprocess template structure
            preprocess_template_structure_for_template_old(
                template_pdb_id=template_pdb_id,
                template_structures_directory=self.template_structures_directory,
                template_file_format=self.template_file_format,
                template_structure_array_directory=self.template_structure_array_directory,
                ccd=self.ccd,
                moltypes_included=self.moltypes_included,
                log_dir=self.log_dir,
            )
            TEMPLATE_PROCESS_LOGGER.get().info(
                f"Successfully preprocessed template structure {template_pdb_id}."
            )
        except Exception as e:
            TEMPLATE_PROCESS_LOGGER.get().info(
                "Failed to preprocess template structure "
                f"{template_pdb_id}:"
                f"\n\nException:\n{str(e)}"
                f"\n\nType:\n{type(e).__name__}"
                f"\n\nTraceback:\n{traceback.format_exc()}"
            )


def preprocess_template_structures(
    template_structures_directory: Path,
    template_file_format: str,
    template_structure_array_directory: Path,
    ccd_file: Path,
    completed_entries_file: Path | None,
    moltypes_included: str,
    num_workers: int,
    chunksize: int,
    log_level: str,
    log_to_file: bool,
    log_to_console: bool,
    log_dir: Path,
) -> None:
    """Preprocesses the template structures.

    Args:
        template_structures_directory (Path):
            Path to the directory containing template structures.
        template_file_format (str):
            File format of the template structures.
        template_structure_array_directory (Path):
            Path to the directory where the template structure arrays will be saved.
        ccd_file (Path):
            Path to the Chemical Component Dictionary file.
        completed_entries_file (Path | None):
            Path to the file containing the list of completed entries.
        moltypes_included (str):
            Comma-separated str of molecule types to include in the template structure
            arrays.
        num_workers (int):
            Number of workers to use for multiprocessing.
        chunksize (int):
            Number of tasks per worker.
        log_level (str):
            Log level for the logger.
        log_to_file (bool):
            Whether to log to file.
        log_to_console (bool):
            Whether to log to console.
        log_dir (Path):
            Directory where the log file will be saved.
    """
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # Parse molecule types to include
    moltypes_included = np.array(
        [MoleculeType[t.strip().upper()] for t in moltypes_included.split(",")]
    )

    # Get list of unique template PDB IDs, create per-entry dirs
    template_pdb_ids = []
    for f in list(template_structures_directory.glob(f"*.{template_file_format}")):
        template_pdb_ids.append(f.stem.split(".")[0])

    # Make sure there is no bias in ordering which could potentially affect
    # preprocessing times
    random.shuffle(template_pdb_ids)
    print(f"Found {len(template_pdb_ids)} template structures.")

    # Subset the template PDB IDs to only those that have not been preprocessed
    if completed_entries_file is not None:
        completed_entries = pd.read_csv(completed_entries_file, sep="\t", header=0)
        template_pdb_ids = list(
            set(template_pdb_ids) - set(completed_entries["entry"].tolist())
        )
        print(f"Preprocessing {len(template_pdb_ids)} template structures.")

    # Pre-create directories for template structure arrays, otherwise IO issues
    for template_pdb_id in tqdm(
        template_pdb_ids, desc="1/2: Creating directories", total=len(template_pdb_ids)
    ):
        template_assembly_dir = template_structure_array_directory / Path(
            f"{template_pdb_id}"
        )
        template_assembly_dir.mkdir(parents=True, exist_ok=True)

    # Parse the CCD
    ccd = pdbx.CIFFile.read(ccd_file)

    # Preprocess template structures
    wrapped_template_structure_preprocessor = _OF3TemplateStructurePreprocessor(
        template_structures_directory,
        template_file_format,
        template_structure_array_directory,
        ccd,
        moltypes_included,
        log_level,
        log_to_file,
        log_to_console,
        log_dir,
    )
    with mp.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                wrapped_template_structure_preprocessor,
                template_pdb_ids,
                chunksize=chunksize,
            ),
            total=len(template_pdb_ids),
            desc="2/2: Preprocessing template structures",
        ):
            pass


# New template preprocessing pipelines
# TODO: replace old versions from above with these new ones
class TemplatePreprocessorInputTrain(BaseModel):
    pass


class TemplatePreprocessorInputInference(BaseModel):
    aln_path: Path
    query_seq_str: str
    template_entry_chain_ids: list[str] | None = None


class TemplatePreprocessorSettings(BaseModel):
    """Settings for template preprocessing.

    See AF3 SI Section 2.4. for details on some of these settings.

    Attributes:
        mode (Literal["train", "inference"]):
            Whether templates are preprocessed for training or inference.
        moltypes (list[MoleculeType]):
            List of molecule types to preprocess templates for.
        max_sequences_parse (int):
            Maximum number of align sequences to parse from the template alignments
            before filtering.
        max_seq_id (float | None):
            Maximum allowed sequence identity of the template relative to the query for
            the template to pass filtering.
        min_align (float | None):
            Minimum required alignment coverage of the the query by the template for it
            to pass filtering.
        min_len (int | None):
            Minimum required number of aligned template residues for the template to
            pass filtering.
        max_release_date (str | None):
            Maximum allowed release date of the template structure for it to pass
            filtering.
        min_release_date_diff (int | None):
            Minimum number of days required between the query and template release dates
            for the template to pass filtering. Equivalently, the minimum number of days
            that a template structure needed to have been released before the query
            structure it is provided for as a template.
        max_templates (int):
            Maximum number of valid templates to keep per query chain after filtering.
        fetch_missing_template_structures (bool):
            Whether to fetch missing template structures from the PDB. Requires internet
            access.
        create_precache (bool):
            Whether to cache of the template structure data (release date and sequence
            information) for template filtering.
        preparse_structures (bool):
            Whether to preparse the template structures into per-chain AtomArray .npz
            files for faster subsequent online template processing.
        n_processes (int):
            Number of processes to use template preprocessing.
        chunksize (int):
            Number of tasks per worker in multiprocessing.
        structure_directory (DirectoryPath):
            Directory containing raw template structures or where template structures
            are to be downloaded.
        structure_file_format (str):
            File format of the template structures. One of "cif", "pdb".
        precache_directory (DirectoryPath | None):
            Directory containing precomputed template structure pre-caches or where new
            ones are to be saved.
        structure_array_directory (DirectoryPath | None):
            Directory containing preparsed template structures or where new ones will be
            saved.
        cache_directory (DirectoryPath | None):
            Directory containing template cache entry .npz files or where new ones will
            be saved.
        ccd_file_path (FilePath | None):
            Path to the Chemical Component Dictionary file. Only required if
            `preparse_structures` is True.
    """

    mode: Literal["train", "predict"] = "predict"
    moltypes: Annotated[
        list[MoleculeType],
        BeforeValidator(lambda v: _convert_molecule_type(_ensure_list(v))),
    ] = [MoleculeType.PROTEIN]
    max_sequences_parse: int = 200
    max_seq_id: float | None = None
    min_align: float | None = None
    min_len: int | None = None
    max_release_date: str | None = None
    min_release_date_diff: int | None = None
    max_templates: int = 20

    fetch_missing_structures: bool = True
    create_precache: bool = False
    preparse_structures: bool = False
    n_processes: int = 4
    chunksize: int = 1

    structure_directory: DirectoryPath | None = None
    structure_file_format: str = "cif"
    output_directory: Path | None = None

    precache_directory: Path | None = None
    structure_array_directory: Path | None = None
    cache_directory: Path | None = None

    ccd_file_path: Path | None = None

    @model_validator(mode="after")
    def _prepare_output_directories(self) -> "TemplatePreprocessorSettings":
        # TODO: add .pdb support
        if self.structure_file_format != "cif":
            raise NotImplementedError(
                f"structure_file_format {self.structure_file_format} was provided but "
                "currently, only cif file format is supported for template structure "
                "preprocessing due to metadata requirements of the template pipeline."
            )

        if self.output_directory is not None:
            base = Path(self.output_directory)
        else:
            base = Path(tempfile.gettempdir()) / "of3_template_data"

        # only set these if the user did not give them explicitly
        self.structure_directory = self.structure_directory or (
            base / "template_structures"
        )
        self.cache_directory = self.cache_directory or (base / "template_cache")
        if self.create_precache:
            self.precache_directory = self.precache_directory or (
                base / "template_precache"
            )
        if self.preparse_structures:
            self.structure_array_directory = self.structure_array_directory or (
                base / "template_structure_arrays"
            )

        for d in (
            base,
            self.output_directory,
            self.structure_directory,
            self.cache_directory,
            self.precache_directory,
            self.structure_array_directory,
        ):
            if d is not None:
                os.makedirs(d, exist_ok=True)

        return self


class TemplatePreprocessor:
    """Template preprocessing pipeline for OF3.

    Prepares template alignments before model training or inference by parsing and
    filtering them down to the set of valid templates for each query chain. Optionally,
    it can preparse template structures into per-chain AtomArray .npz files for
    faster subsequent online template processing.
    """

    def __init__(
        self,
        input_set: DatasetCache | InferenceQuerySet,
        config: TemplatePreprocessorSettings,
    ) -> None:
        self.input_set = input_set

        self.moltypes = config.moltypes
        self.max_sequences_parse = config.max_sequences_parse
        self.max_seq_id = config.max_seq_id
        self.min_align = config.min_align
        self.min_len = config.min_len
        self.max_release_date = config.max_release_date
        self.min_release_date_diff = config.min_release_date_diff
        self.max_templates = config.max_templates

        self.fetch_missing_structures = config.fetch_missing_structures
        self.create_precache = config.create_precache
        self.preparse_structures = config.preparse_structures
        self.n_processes = config.n_processes
        self.chunksize = config.chunksize

        self.structure_directory = config.structure_directory
        self.structure_file_format = config.structure_file_format
        self.precache_directory = config.precache_directory
        self.structure_array_directory = config.structure_array_directory
        self.cache_directory = config.cache_directory

        # TODO: update with set_ccd biotite method instead
        if config.ccd_file_path is not None:
            self.ccd = pdbx.CIFFile.read(config.ccd_file_path)
        else:
            self.ccd = BiotiteCCDWrapper()

        self.inputs = []  # replaced below by the parsers
        self.seq_hash_map = {}  # replaced in call by a manager dict
        self.hash_template_id_map = {}  # replaced in call by a manager dict
        if isinstance(input_set, DatasetCache):
            self._parse_dataset_cache()
        elif isinstance(input_set, InferenceQuerySet):
            self._parse_inference_query_set()
        else:
            raise ValueError(
                "Input set must be either DatasetCache or InferenceQuerySet"
            )

    def _parse_dataset_cache(self) -> None:
        raise NotImplementedError

    def _parse_inference_query_set(self) -> None:
        paths_seen = set()
        inputs = []
        for query in self.input_set.queries.values():
            for chain in query.chains:
                if chain.molecule_type not in self.moltypes:
                    continue

                template_alignment_path = Path(chain.template_alignment_file_path)

                if template_alignment_path not in paths_seen:
                    paths_seen.add(template_alignment_path)
                    inputs.append(
                        TemplatePreprocessorInputInference(
                            aln_path=template_alignment_path,
                            query_seq_str=chain.sequence,
                            template_entry_chain_ids=chain.template_entry_chain_ids,
                        )
                    )
        self.inputs = inputs

    def _update_dataset_cache(self) -> None:
        raise NotImplementedError

    def _update_inference_query_set(self) -> None:
        for query_name, query in self.input_set.queries.items():
            for idx, chain in enumerate(query.chains):
                if chain.molecule_type not in self.moltypes:
                    continue
                # Add new npz file path to the chain
                query_seq_hash = get_sequence_hash(chain.sequence)
                template_cache_entry_file = (
                    self.cache_directory / f"{query_seq_hash}.npz"
                )
                # No templates for chains whose preprocessing fails
                if template_cache_entry_file.exists():
                    new_path = Path(template_cache_entry_file)
                else:
                    new_path = None
                self.input_set.queries[query_name].chains[
                    idx
                ].template_alignment_file_path = new_path

                # Add the template entry chain IDs to the chain
                self.input_set.queries[query_name].chains[
                    idx
                ].template_entry_chain_ids = list(
                    self.hash_template_id_map.get(query_seq_hash, [])
                )

    def __call__(self) -> None:
        # Preprocess template alignments into template cache entries
        if len(self.inputs) == 1:
            print("Preprocessing templates...")
            self._preprocess_templates_for_query(self.inputs[0])
        elif len(self.inputs) > 1:
            manager = mp.Manager()
            self.seq_hash_map = manager.dict()
            self.hash_template_id_map = manager.dict()
            with mp.Pool(self.n_processes) as pool:
                for _ in tqdm(
                    pool.imap_unordered(
                        self._preprocess_templates_for_query,
                        self.inputs,
                        chunksize=self.chunksize,
                    ),
                    total=len(self.inputs),
                    desc="Preprocessing templates",
                ):
                    pass

        else:
            return

        # Update the dataset cache/inference query set with the preprocessed template
        # data
        if isinstance(self.input_set, DatasetCache):
            self._update_dataset_cache()
        elif isinstance(self.input_set, InferenceQuerySet):
            self._update_inference_query_set()

    def _preprocess_templates_for_query(
        self,
        input_data: TemplatePreprocessorInputTrain | TemplatePreprocessorInputInference,
    ) -> None:
        # preprocess templates for a single chain
        # 1. Parse template alignment file
        templates = parse_template_alignment(
            input_data.aln_path, input_data.query_seq_str, self.max_sequences_parse
        )

        # TODO: 2. if template entry and chain ID list provided in the IQS - skip some
        # below:

        # TODO: 3. TRAIN: match  query sequence in aln to query sequence in structure

        # 4. Representative mapping For training - core weighted PDB set, need to index
        # by entry ID, and cannot index by sequence hash due to the way filtering is
        # done
        query_seq_hash = get_sequence_hash(input_data.query_seq_str)
        # skip template preprocessing for chain if already done
        if input_data.query_seq_str in self.seq_hash_map:
            return
        self.seq_hash_map[input_data.query_seq_str] = query_seq_hash
        template_cache_entry_file = self.cache_directory / f"{query_seq_hash}.npz"
        cache_entry_available = template_cache_entry_file.exists()

        # 5. Template consistency checks and filtering
        if not cache_entry_available:  # !!! cannot do this for training!
            template_cache_entry = {}
            template_ids = []
            for _, template in templates.items():
                # A. Sequence checks
                if run_template_sequence_checks(
                    template, self.max_seq_id, self.min_align, self.min_len
                ):
                    continue

                # B. Get which files are available
                template_structure_file = (
                    self.structure_directory
                    / f"{template.entry_id}.{self.structure_file_format}"
                )
                structure_available = template_structure_file.exists()
                if self.precache_directory is not None:
                    precache_entry_file = (
                        self.precache_directory / f"{template.entry_id}.npz"
                    )
                    precache_entry_available = precache_entry_file.exists()
                else:
                    precache_entry_file = None
                    precache_entry_available = False
                if self.structure_array_directory is not None:
                    template_structure_array_subdirectory = (
                        self.structure_array_directory / template.entry_id
                    )
                    structure_arrays_available = (
                        template_structure_array_subdirectory.exists()
                        and any(template_structure_array_subdirectory.iterdir())
                    )
                else:
                    template_structure_array_subdirectory = None
                    structure_arrays_available = False

                # C. Fetch template structure if needed
                # We need
                # - either the raw template structure
                # - or the precache entry and the structure arrays both
                if not structure_available & (
                    (not precache_entry_available) | (not structure_arrays_available)
                ):
                    if not self.fetch_missing_structures:
                        # TODO: add warning - missing template structure data but
                        # fetching turned off
                        continue
                    else:
                        fetch(
                            pdb_ids=template.entry_id,
                            format="cif",
                            target_path=self.structure_directory,
                        )

                # D. Load template structure
                # i. from precache if available
                if precache_entry_available:
                    # load precached structure
                    precache_entry = np.load(precache_entry_file, allow_pickle=True)
                    chain_id_seq_map = precache_entry["chain_id_seq_map"].item()
                    release_date = precache_entry["release_date"].item()
                # ii. from raw structure if not precached
                else:
                    # Preprocess into per-chain arrays if prompted
                    if self.preparse_structures & (not structure_arrays_available):
                        cif_file, _ = preprocess_template_structure_for_template(
                            template_structure_file,
                            template_structure_array_subdirectory,
                            self.ccd,
                            self.moltypes,
                        )
                    else:
                        cif_file = _load_ciffile(template_structure_file)

                    chain_id_seq_map = get_asym_id_to_canonical_seq_dict(cif_file)
                    release_date = get_release_date(get_cif_block(cif_file)).strftime(
                        "%Y-%m-%d"
                    )

                    if self.create_precache:
                        np.savez_compressed(
                            precache_entry_file,
                            **{
                                "release_date": release_date,
                                "chain_id_seq_map": chain_id_seq_map,
                            },
                        )

                # E. Realign with the sequence from the template structure file if the
                # template aligment did not contain the sequence.
                if not all(
                    [
                        template.seq,
                        template.query_aln_pos,
                        template.aln_pos,
                        template.q_cov,
                    ]
                ):
                    template_sequence = chain_id_seq_map.get(template.chain_id)
                    if template_sequence is None:
                        # TODO: add warning - the chain ID from the alignment is not
                        # present in the structure file
                        continue
                    parser = A3mParser(max_sequences=None)
                    template = parser(
                        f">query_X/1-{len(input_data.query_seq_str)}\n{input_data.query_seq_str}\n>{template.entry_id}_{template.chain_id}/{1}-{len(template_sequence)}\n{template_sequence}\n",
                        input_data.query_seq_str,
                        realign=True,
                    )[1]

                # F. Apply release date checks
                if not isinstance(release_date, datetime):
                    release_date = datetime.strptime(release_date, "%Y-%m-%d")
                if run_template_release_date_checks(
                    template_release_date=release_date,
                    query_release_date=None,  # TODO: add for training logic
                    max_template_release_date=self.max_release_date,
                    min_release_date_diff=self.min_release_date_diff,
                ):
                    continue

                # G. Match template sequence from alignment to template sequence in
                # structure and attempt to remap chain ID if needed
                chain_id_matched = match_template_seq_from_aln_to_struc(
                    template, chain_id_seq_map
                )
                if chain_id_matched is None:
                    # TODO: add warning - could not match template sequence from
                    # alignment to structure sequence
                    continue

                # H. Add to cache entry
                template_cache_entry[f"{template.entry_id}_{chain_id_matched}"] = {
                    "index": template.index,
                    "release_date": release_date,
                    "idx_map": np.concatenate(
                        [
                            template.query_aln_pos[:, np.newaxis],
                            template.aln_pos[:, np.newaxis],
                        ],
                        axis=1,
                    ),
                }
                template_ids.append(f"{template.entry_id}_{chain_id_matched}")

                # I. Break if max templates reached
                if len(template_cache_entry) == self.max_templates:
                    break

            # 6. Save template cache entry and update shared dict of template IDs
            if len(template_cache_entry) > 0:
                np.savez_compressed(template_cache_entry_file, **template_cache_entry)
                self.hash_template_id_map[query_seq_hash] = template_ids

        # Load the existing template cache entry if available to add the processed
        # template ids into the shared hash_template_id_map and then input set
        else:
            if query_seq_hash in self.hash_template_id_map:
                return
            template_cache_entry = np.load(template_cache_entry_file, allow_pickle=True)
            template_cache_entry = {
                key: value.item() for key, value in template_cache_entry.items()
            }

            self.hash_template_id_map[query_seq_hash] = list(
                template_cache_entry.keys()
            )


class TemplatePrecachePreprocessor:
    """Preprocessing pipeline for extracting template structure metadata."""

    def __init__(
        self,
        config: TemplatePreprocessorSettings,
    ) -> None:
        self.n_processes = config.n_processes
        self.chunksize = config.chunksize

        self.structure_directory = config.structure_directory
        self.structure_file_format = config.structure_file_format
        self.precache_directory = config.precache_directory

        # Get list of template entry IDs
        self.template_entry_ids = [
            f.stem
            for f in list(
                self.structure_directory.glob(f"*.{self.structure_file_format}")
            )
        ]

        if not self.template_entry_ids:
            print(
                f"No template structure files found in {self.structure_directory} "
                f"with format {self.structure_file_format}"
            )
            return

        print(f"Found {len(self.template_entry_ids)} template structures to process.")

    def __call__(self) -> None:
        with mp.Pool(self.n_processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    self._preprocess_template_precache_entry,
                    self.template_entry_ids,
                    chunksize=self.chunksize,
                ),
                total=len(self.template_entry_ids),
                desc="Preprocessing template structures into precache entries",
            ):
                pass

    def _preprocess_template_precache_entry(
        self,
        template_entry_id: str,
    ) -> None:
        """Create a precache entry for a single template structure.

        Args:
            template_entry_id (str):
                Entry ID of the template structure, typically, PDB ID.
        """
        precache_entry_file = self.precache_directory / f"{template_entry_id}.npz"

        if precache_entry_file.exists():
            return

        cif_file = _load_ciffile(
            self.structure_directory
            / f"{template_entry_id}.{self.structure_file_format}"
        )
        chain_id_seq_map = get_asym_id_to_canonical_seq_dict(cif_file)
        release_date = get_release_date(get_cif_block(cif_file)).strftime("%Y-%m-%d")

        np.savez_compressed(
            precache_entry_file,
            **{
                "release_date": release_date,
                "chain_id_seq_map": chain_id_seq_map,
            },
        )


class TemplateStructurePreprocessor:
    """Preprocessing pipeline for template structures.

    Pre-parses and cleans up structure files for downstream template processing.
    Saves per-chain AtomArray .npz files for each template structure.
    """

    def __init__(self, config: TemplatePreprocessorSettings) -> None:
        self.moltypes = config.moltypes

        self.n_processes = config.n_processes
        self.chunksize = config.chunksize

        self.structure_directory = config.structure_directory
        self.structure_file_format = config.structure_file_format
        self.structure_array_directory = config.structure_array_directory

        if config.ccd_file_path is not None:
            self.ccd = pdbx.CIFFile.read(config.ccd_file_path)
        else:
            self.ccd = BiotiteCCDWrapper()

        self.template_entry_ids = [
            f.stem
            for f in list(
                self.structure_directory.glob(f"*.{self.structure_file_format}")
            )
        ]

        if not self.template_entry_ids:
            print(
                f"No template structure files found in {self.structure_directory} "
                f"with format {self.structure_file_format}"
            )
            return

        print(f"Found {len(self.template_entry_ids)} template structures to process.")

    def __call__(self) -> None:
        with mp.Pool(self.n_processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    self._preprocess_template_structure_for_template,
                    self.template_entry_ids,
                    chunksize=self.chunksize,
                ),
                total=len(self.template_entry_ids),
                desc="Preprocessing template structures into structure arrays",
            ):
                pass

    def _preprocess_template_structure_for_template(
        self,
        template_entry_id: str,
    ) -> None:
        preprocess_template_structure_for_template(
            self.structure_directory
            / f"{template_entry_id}.{self.structure_file_format}",
            self.structure_array_directory / f"{template_entry_id}",
            self.ccd,
            self.moltypes,
        )


# New primitives: TODO move to primitives
def get_sequence_hash(sequence_str: str) -> str:
    """Generates a SHA-256 hash for the given sequence string."""
    hasher = hashlib.sha256()
    hasher.update(sequence_str.encode("utf-8"))
    return hasher.hexdigest()


def run_template_sequence_checks(
    template: TemplateData,
    max_seq_id: float | None,
    min_align: float | None,
    min_len: int | None,
) -> bool:
    fails = False
    if max_seq_id is not None:
        fails |= template.seq_id > max_seq_id
    if min_align is not None:
        fails |= template.q_cov < min_align
    if min_len is not None:
        fails |= len(template.seq) < min_len
    return fails


def run_template_release_date_checks(
    template_release_date: datetime,
    query_release_date: datetime | None,
    max_template_release_date: datetime | None,
    min_release_date_diff: int | None,
) -> bool:
    fails = False
    if min_release_date_diff is not None:
        if query_release_date is None:
            raise ValueError(
                "Query release date not provided but needed if min_release_date_diff "
                "is specified."
            )
        fails |= (
            query_release_date - template_release_date
        ).days < min_release_date_diff

    if max_template_release_date is not None:
        fails |= template_release_date > max_template_release_date

    return fails


def remap_template_chain_id(
    original_chain_id: str,
    seq_from_aln: str,
    chain_id_seq_map: dict[str, str],
) -> str | None:
    chain_id_matched = None
    for k, v in chain_id_seq_map.items():
        # Skip the original hit chain
        if k == original_chain_id:
            continue
        # Remap hit chain ID if found in another chain
        if seq_from_aln in v:
            chain_id_matched = k
            break

    return chain_id_matched


def match_template_seq_from_aln_to_struc(
    template: TemplateData, chain_id_seq_map: dict[str, str]
) -> str | None:
    seq_from_struc = chain_id_seq_map.get(template.chain_id)

    # A) If chain ID not in CIF file, attempt to find sequence in other chains
    if seq_from_struc is None:  # noqa: SIM114 will add different logs
        chain_id_matched = remap_template_chain_id(
            original_chain_id=template.chain_id,
            seq_from_aln=template.seq,
            chain_id_seq_map=chain_id_seq_map,
        )
    # B) If chain ID is in CIF file but HMM sequence does not match CIF sequence,
    # attempt to find in other chains
    elif (seq_from_struc is not None) & (template.seq not in seq_from_struc):
        chain_id_matched = remap_template_chain_id(
            original_chain_id=template.chain_id,
            seq_from_aln=template.seq,
            chain_id_seq_map=chain_id_seq_map,
        )
    # C) If HMM sequence matches CIF sequence, use original chain ID
    else:
        chain_id_matched = template.chain_id

    return chain_id_matched


def preprocess_template_structure_for_template(
    template_structure_file: Path,
    template_structure_array_subdirectory: Path,
    ccd: CIFFile,
    moltypes: np.ndarray[int],
) -> tuple[CIFFile, np.ndarray]:
    """Preparse and process a template structure."""
    # Parse template structure
    cif_file, atom_array = parse_mmcif(template_structure_file)

    # Sanitize template structure
    atom_array = clean_template_atom_array(atom_array, cif_file, None, ccd)

    # Save template structure for each chain separately
    chain_ids = np.unique(atom_array.chain_id)
    for chain_id in chain_ids:
        # Find molecule type of chain and save if included
        atom_array_chain = atom_array[atom_array.chain_id == chain_id]
        chain_mol_type = list(set(atom_array_chain.molecule_type_id))
        if len(chain_mol_type) > 1:
            raise ValueError(
                f"Multiple molecule types found in chain {chain_id} of "
                f"template {template_structure_file.stem}."
            )
        if chain_mol_type[0] in moltypes:
            if not template_structure_array_subdirectory.exists():
                os.makedirs(template_structure_array_subdirectory)
            write_atomarray_to_npz(
                atom_array=atom_array_chain,
                output_file=template_structure_array_subdirectory
                / f"{template_structure_file.stem}_{chain_id}.npz",
            )
    return cif_file, atom_array
