"""Preprocessing pipelines for template data ran before training/evaluation."""

import json
import multiprocessing as mp
import os
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from openfold3.core.data.io.dataset_cache import read_datacache
from openfold3.core.data.io.sequence.template import parse_hmmsearch_sto
from openfold3.core.data.primitives.quality_control.logging_utils import (
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
    parse_release_date,
    parse_representatives,
    parse_structure,
)


# Metadata cache creation
def create_template_cache_for_query(
    query_pdb_chain_id: str,
    template_alignment_file: Path,
    template_structures_directory: Path,
    template_cache_directory: Path,
    query_structures_directory: Path,
    max_templates_construct: int,
    query_file_format: str,
    template_file_format: str,
) -> None:
    """Creates a json cache of filtered template hits for a query.

    A query denotes a single protein chain for which template hits are to be filtered,
    i.e. corresponding to a protein chain whose structure needs to be predicted during
    training, evaluation or inference.

    Note that a template is skipped if:
        - no CIF file is provided for the QUERY against which the template was aligned;
        the PDB IDs of the QUERY need to match between the alignment and the CIF file
        - there is a mismatch between the author chain IDs for the QUERY against which
        the template was aligned AND the sequence provided in the alignment file cannot
        be remapped to an exact subsequence of any chains in the QUERY CIF file
        - no CIF file is provided for the TEMPLATE; the PDB IDs of the TEMPLATE need to
        match between the alignment and the CIF file
        - there is a mismatch between the author chain IDs for the TEMPLATE AND the
        sequence provided in the alignment file cannot be remapped to an exact
        subsequence of any chains in the TEMPLATE CIF file
        - the sequence of the template does not pass the AF3 sequence filters

    Another note: the template alignment is parsed from the directory indicated by the
    representative ID of a query chain, whereas the query structure is parsed using the
    PDB ID-chain pair of the query chain.

    The cache contains the e-value and mapping from the query residues to the template
    hit residues.

    Args:
        query_pdb_chain_id (str):
            The PDB ID and chain ID of the query chain.
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
            Path to the directory containing query structures in mmCIF format. The
            PDB IDs of the query CIF files need to match the PDB IDs for the query (1st
            row) in the alignment file for it to have any templates.
        max_templates_construct (int):
            Maximum number of templates to keep per query chain during template cache
            construction.
        query_file_format (str):
            File format of the query structures.
        template_file_format (str):
            File format of the template structures.
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

    # Parse alignment
    try:
        with open(template_alignment_file) as f:
            hits = parse_hmmsearch_sto(f.read())
    except Exception as e:
        template_process_logger.info(
            "Failed to parse alignment file, skipping. Make sure that"
            " an hmmsearch output globally aligned with hmmalign was provided. "
            f"\nError: \n{e}\nTraceback: \n{traceback.format_exc()}"
        )
        data_log_to_tsv(
            data_log,
            template_cache_directory.parent / Path(f"data_log_{os.getpid()}.tsv"),
        )
        return
    template_process_logger.info(f"Alignment file {template_alignment_file} parsed.")
    data_log["can_load_aln_file"] = True

    # Filter queries
    query = hits[0]
    query_pdb_id, query_chain_id = query_pdb_chain_id.split("_")
    query_pdb_id_t, query_chain_id_t = query.name.split("_")
    template_cache_path_rep = template_cache_directory / Path(f"{query.name}.npz")
    if template_cache_path_rep.exists():
        template_process_logger.info(
            f"Template cache for {query.name} already exists. Skipping templates for "
            "this structure."
        )
        data_log["template_cache_already_computed"] = True
        data_log_to_tsv(
            data_log,
            template_cache_directory.parent / Path(f"data_log_{os.getpid()}.tsv"),
        )
        return
    # 1. Parse fasta of the structure
    # the query and all its templates are skipped if the structure identified by the PDB
    # ID of the first hit in the alignments file is not provided in
    # query_structures_directory
    qp = query_structures_directory / Path(f"{query_pdb_id}.{query_file_format}")
    if not (qp).exists():
        template_process_logger.info(
            f"Query .cif structure {query_pdb_id} not found in "
            f"{query_structures_directory}. Skipping templates for this structure."
        )
        data_log_to_tsv(
            data_log,
            template_cache_directory.parent / Path(f"data_log_{os.getpid()}.tsv"),
        )
        return
    data_log["query_cif_exists"] = True
    # 2. Parse query chain and sequence
    # the query and all its templates are skipped if its HMM sequence cannot be mapped
    # exactly to a subsequence of the MATCHING chain in the CIF file provided in
    # query_structures_directory (no chain/sequence remapping is done)
    # !!! Note that the chain ID-sequence map for this step is derived from the
    # preprocessed fasta file, not from the CIF file
    if match_query_chain_and_sequence(
        query_structures_directory, query, query_pdb_id, query_chain_id
    ):
        template_process_logger.info(
            f"The query sequences in the structure (query {query_pdb_id} chain "
            f"{query_chain_id}) and template alignment (query {query_pdb_id_t} chain "
            f"{query_chain_id_t}) don't match. Skipping templates for this structure."
        )
        data_log_to_tsv(
            data_log,
            template_cache_directory.parent / Path(f"data_log_{os.getpid()}.tsv"),
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
                f"Template {hit_pdb_id} sequence passes sequence " "filters."
            )
            data_log["n_templates_pass_seq_filters"] += 1

        # 2. Parse structure
        # The template is skipped if the structure identified by the PDB ID of the
        # corresponding hit in the alignment file is not provided in
        # template_structures_path
        cif_file, atom_array = parse_structure(
            template_structures_directory,
            hit_pdb_id,
            file_format=f"{template_file_format}",
        )
        if cif_file is None:
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
        hit_chain_id_matched = match_template_chain_and_sequence(
            cif_file, atom_array, hit
        )
        if hit_chain_id_matched is None:
            template_process_logger.info(
                f"Could not match template {hit_pdb_id} chain {hit_chain_id} "
                f"sequence in {cif_file}. Skipping this template."
            )
            continue
        else:
            data_log["n_template_chain_match"] += 1

        # Parse release date
        release_date = parse_release_date(cif_file)

        # Create residue index map
        idx_map = create_residue_idx_map(query, hit)

        # Store as filtered hit
        # hmmsearch is sorted in descending e-value order so index is enough to sort
        template_hits_filtered[f"{hit_pdb_id}_{hit_chain_id_matched}"] = {
            "index": hit.index,
            "release_date": release_date.strftime("%Y-%m-%d"),
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
        data_log, template_cache_directory.parent / Path(f"data_log_{os.getpid()}.tsv")
    )

    # Save filtered hits to json using the representative ID
    if len(template_hits_filtered) > 0:
        np.savez(template_cache_path_rep, **template_hits_filtered)
        template_process_logger.info(
            f"Template cache for {query.name} saved with "
            f"{len(template_hits_filtered)} valid hits."
        )
    else:
        # TODO optimize to not recompute template empty template caches multiple times
        template_process_logger.info(f"0 valid templates found for {query.name}.")


class _AF3TemplateCacheConstructor:
    def __init__(
        self,
        template_alignment_directory: Path,
        template_alignment_filename: str,
        template_structures_directory: Path,
        template_cache_directory: Path,
        query_structures_directory: Path,
        max_templates_construct: int,
        query_file_format: str,
        template_file_format: str,
        log_level: str,
        log_to_file: bool,
        log_to_console: bool,
        log_dir: Path,
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
            query_file_format (str):
                File format of the query structures.
            template_file_format (str):
                File format of the template structures.
            log_level (str):
                Log level for the logger.
            log_to_file (bool):
                Whether to log to file.
            log_dir (Path):
                Directory where the log file will be saved.

        """
        self.template_alignment_directory = template_alignment_directory
        self.template_alignment_filename = template_alignment_filename
        self.template_structures_directory = template_structures_directory
        self.template_cache_directory = template_cache_directory
        self.query_structures_directory = query_structures_directory
        self.max_templates_construct = max_templates_construct
        self.query_file_format = query_file_format
        self.template_file_format = template_file_format
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_dir = log_dir

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

            # Create template cache for query
            create_template_cache_for_query(
                query_pdb_chain_id=query_pdb_chain_id,
                template_alignment_file=(
                    self.template_alignment_directory
                    / Path(rep_pdb_chain_id)
                    / self.template_alignment_filename
                ),
                template_structures_directory=self.template_structures_directory,
                template_cache_directory=self.template_cache_directory,
                query_structures_directory=self.query_structures_directory
                / Path(query_pdb_id),
                max_templates_construct=self.max_templates_construct,
                query_file_format=self.query_file_format,
                template_file_format=self.template_file_format,
            )
        except Exception as e:
            TEMPLATE_PROCESS_LOGGER.get().info(
                "Failed to process templates for query " f"{query_pdb_chain_id}:\n{e}\n"
            )


def create_template_cache_af3(
    dataset_cache_file: Path,
    template_alignment_directory: Path,
    template_alignment_filename: str,
    template_structures_directory: Path,
    template_cache_directory: Path,
    query_structures_directory: Path,
    max_templates_construct: int,
    query_file_format: str,
    template_file_format: str,
    num_workers: int,
    log_level: str,
    log_to_file: bool,
    log_to_console: bool,
    log_dir: Path,
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
        query_file_format (str):
            File format of the query structures.
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
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # Parse list of chains from metadata cache
    dataset_cache = read_datacache(dataset_cache_file)
    template_query_iterator = parse_representatives(dataset_cache, True).entries

    # Create template cache for each query chain
    wrapped_template_cache_constructor = _AF3TemplateCacheConstructor(
        template_alignment_directory,
        template_alignment_filename,
        template_structures_directory,
        template_cache_directory,
        query_structures_directory,
        max_templates_construct,
        query_file_format,
        template_file_format,
        log_level,
        log_to_file,
        log_to_console,
        log_dir,
    )
    with mp.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                wrapped_template_cache_constructor,
                template_query_iterator,
                chunksize=1,
            ),
            total=len(template_query_iterator),
            desc="Creating template cache",
        ):
            pass
    # Collate data logs
    collate_data_logs(template_cache_directory, "full_data_log_constructed_cache.tsv")


# Dataset cache update
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
        data_log = {
            "query_pdb_id": query_pdb_chain_id.split("_")[0],
            "query_chain_id": query_pdb_chain_id.split("_")[1],
            "can_load_template_cache": False,
            "n_valid_templates_prefilter": 0,
            "n_dropped_due_to_release_date": 0,
            "n_valid_templates_postfilter": 0,
        }
    else:
        rep_id, query_pdb_chain_ids_release_dates = (
            input_data.rep_pdb_chain_id,
            input_data.dated_query,
        )
        if isinstance(max_release_date, str):
            max_release_date = datetime.strptime(max_release_date, "%Y-%m-%d")

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
            template_cache_directory.parent / Path(f"data_log_{os.getpid()}.tsv"),
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
            if check_release_date_diff(
                query_release_date=query_release_date,
                template_release_date=template_date,
                min_release_date_diff=min_release_date_diff,
            ):
                data_log["n_dropped_due_to_release_date"] += 1
                continue
        else:
            if check_release_date_max(
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
        data_log, template_cache_directory.parent / Path(f"data_log_{os.getpid()}.tsv")
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


class _AF3TemplateCacheFilter:
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
                    "Failed to filter templates for query " f"{input[0][0]}: \n{e}\n"
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


def filter_template_cache_af3(
    dataset_cache_file: Path,
    updated_dataset_cache_file: Path,
    template_cache_directory: Path,
    max_templates_filter: int,
    is_core_train: bool,
    num_workers: int,
    save_frequency: int,
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
        is_core_train (bool):
            Whether the dataset is core train or not.
        num_workers (int):
            Number of workers to use for multiprocessing.
        save_frequency (int):
            Frequency at which to save the updated dataset cache.
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
        dataset_cache, is_core_train
    ).entries
    data_iterator_len = len(template_query_iterator)

    # Filter template cache for each query chain
    wrapped_template_cache_filter = _AF3TemplateCacheFilter(
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
        for idx, valid_templates in tqdm(
            enumerate(
                pool.imap_unordered(
                    wrapped_template_cache_filter,
                    template_query_iterator,
                    chunksize=30,
                )
            ),
            total=data_iterator_len,
            desc="Filtering template cache",
        ):
            # Update dataset cache with list of valid template representative IDs
            for (pdb_id, chain_id), valid_template_list in valid_templates.items():
                dataset_cache.structure_data[pdb_id].chains[
                    chain_id
                ].template_ids = valid_template_list

            if (idx + 1) % save_frequency == 0:
                with open(updated_dataset_cache_file, "w") as f:
                    json.dump(dataset_cache, f, indent=4)

    # Save final complete dataset cache
    dataset_cache.to_json(updated_dataset_cache_file)

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
        for f in list(template_cache_directory.parent.glob("data_log_*"))
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
        template_cache_directory.parent / Path(f"{fname}.tsv"), sep="\t", index=False
    )
