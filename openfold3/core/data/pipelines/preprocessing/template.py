"""Preprocessing pipelines for template data ran before training/evaluation."""

import json
import logging
import multiprocessing as mp
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from openfold3.core.data.io.sequence.template import parse_hmmsearch_sto
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

logger = logging.getLogger(__name__)


# Metadata cache creation
def create_template_cache_for_query(
    query_pdb_chain_id: str,
    template_alignment_file: Path,
    template_structures_directory: Path,
    template_cache_directory: Path,
    query_structures_directory: Path,
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
    """
    pid = os.getpid()

    # Parse alignment
    try:
        with open(template_alignment_file) as f:
            hits = parse_hmmsearch_sto(f.read())
    except Exception as e:
        logger.info(
            f"{pid} - Failed to parse alignment file, skipping. Make sure that"
            f" an hhsearch output stockholm was provided. Error: {e}"
        )
        return
    logger.debug(f"{pid} - Alignment file {template_alignment_file} parsed.")

    # Filter queries
    query = hits[0]
    query_pdb_id, query_chain_id = query_pdb_chain_id.split("_")
    query_pdb_id_t, query_chain_id_t = query.name.split("_")
    # 1. Parse fasta of the structure
    # the query and all its templates are skipped if the structure identified by the PDB
    # ID of the first hit in the alignments file is not provided in
    # query_structures_directory
    if not (query_structures_directory / Path(f"{query_pdb_id}.bcif")).exists():
        logger.debug(
            f"{pid} - Query .cif structure {query_pdb_id} not found in "
            f"{query_structures_directory}. Skipping templates for this structure."
        )
        return
    # 2. Parse query chain and sequence
    # the query and all its templates are skipped if its HMM sequence cannot be mapped
    # exactly to a subsequence of the MATCHING chain in the CIF file provided in
    # query_structures_directory (no chain/sequence remapping is done)
    # !!! Note that the chain ID-sequence map for this step is derived from the
    # preprocessed fasta file, not from the CIF file
    if match_query_chain_and_sequence(
        query_structures_directory, query, query_pdb_id, query_chain_id
    ):
        logger.debug(
            f"{pid} - The query sequences in the structure (query {query_pdb_id} chain "
            f"{query_chain_id}) and template alignment (query {query_pdb_id_t} chain "
            f"{query_chain_id_t}) don't match. Skipping templates for this structure."
        )
        return
    else:
        logger.debug(
            f"{pid} - Query {query_pdb_id} chain {query_chain_id} sequence matches "
            "alignment sequence."
        )

    # Filter template hits
    template_hits_filtered = {}
    for idx, hit in hits.items():
        # Skip query
        if idx == 0:
            continue
        hit_pdb_id, hit_chain_id = hit.name.split("_")

        # 1. Apply sequence filters: AF3 SI Section 2.4
        if check_sequence(query_seq=query.hit_sequence.replace("-", ""), hit=hit):
            logger.debug(
                f"{pid} - Template {hit_pdb_id} sequence does not pass sequence"
                " filters. Skipping this template."
            )
            continue
        else:
            logger.debug(
                f"{pid} - Template {hit_pdb_id} sequence passes sequence " "filters."
            )

        # 2. Parse structure
        # The template is skipped if the structure identified by the PDB ID of the
        # corresponding hit in the alignment file is not provided in
        # template_structures_path
        cif_file, atom_array = parse_structure(
            template_structures_directory, hit_pdb_id, file_format="cif"
        )
        if cif_file is None:
            logger.debug(
                f"{pid} - Template structure {hit_pdb_id} not found in "
                f"{template_structures_directory}. Skipping this template."
            )
            continue
        else:
            logger.debug(f"{pid} - Template structure {hit_pdb_id} parsed.")
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
            logger.debug(
                f"{pid} - Could not match template {hit_pdb_id} chain {hit_chain_id} "
                f"sequence in {cif_file}. Skipping this template."
            )
            continue

        # Parse release date
        release_date = parse_release_date(cif_file)

        # Create residue index map
        idx_map = create_residue_idx_map(query, hit)

        # Store as filtered hit
        # Note: since hmmer outputs hits in ascending order of e-value, the index of the
        # hit in the hits dict is used as the e-value
        template_hits_filtered[f"{hit_pdb_id}_{hit_chain_id_matched}"] = {
            "e_value": hit.index,
            "release_date": release_date.strftime("%Y-%m-%d"),
            "idx_map": idx_map,
        }

    # Save filtered hits to json using the representative ID
    if len(template_hits_filtered) > 0:
        template_cache_path_rep = template_cache_directory / Path(f"{query.name}.json")
        if not os.path.exists(template_cache_path_rep):
            with open(template_cache_path_rep, "w") as f:  # TODO check mp safe
                json.dump(template_hits_filtered, f, indent=4)
        logging.info(
            f"{pid} - Template cache for {query.name} saved with "
            f"{len(template_hits_filtered)} valid hits."
        )
    else:
        logging.info(f"{pid} - 0 valid templates found for {query.name}.")


class _AF3TemplateCacheConstructor:
    def __init__(
        self,
        template_alignment_directory: Path,
        template_alignment_filename: str,
        template_structures_directory: Path,
        template_cache_directory: Path,
        query_structures_directory: Path,
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

        """
        self.template_alignment_directory = template_alignment_directory
        self.template_alignment_filename = template_alignment_filename
        self.template_structures_directory = template_structures_directory
        self.template_cache_directory = template_cache_directory
        self.query_structures_directory = query_structures_directory

    @wraps(create_template_cache_for_query)
    def __call__(self, input: _TemplateQueryEntry) -> None:
        try:
            query_pdb_chain_id, rep_pdb_chain_id = (
                input.dated_query.query_pdb_chain_id,
                input.rep_pdb_chain_id,
            )
            query_pdb_id = query_pdb_chain_id.split("_")[0]
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
            )
        except Exception as e:
            logger.info(
                "Failed to process templates for query " f"{query_pdb_chain_id}: {e}"
            )


def create_template_cache_af3(
    dataset_cache_file: Path,
    template_alignment_directory: Path,
    template_alignment_filename: str,
    template_structures_directory: Path,
    template_cache_directory: Path,
    query_structures_directory: Path,
    num_workers: int,
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
        num_workers (int):
            Number of workers to use for multiprocessing.
    """
    # Parse list of chains from metadata cache
    with open(dataset_cache_file) as f:
        dataset_cache = json.load(f)
    template_query_iterator = parse_representatives(dataset_cache, True).entries

    # Create template cache for each query chain
    wrapped_template_cache_constructor = _AF3TemplateCacheConstructor(
        template_alignment_directory,
        template_alignment_filename,
        template_structures_directory,
        template_cache_directory,
        query_structures_directory,
    )
    with mp.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                wrapped_template_cache_constructor,
                template_query_iterator,
                chunksize=30,
            ),
            total=len(template_query_iterator),
            desc="Creating template cache",
        ):
            pass
    return


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
        if isinstance(max_release_date, str):
            max_release_date = datetime.strptime(max_release_date, "%Y-%m-%d")

    # Parse template cache of the representative if available
    template_cache_file = template_cache_directory / Path(f"{rep_id}.json")
    if template_cache_file.exists():
        with open(template_cache_directory / Path(f"{rep_id}.json")) as f:
            template_cache = json.load(f)
    else:
        logger.info(
            f"Template cache for representative {rep_id} not found. Returning no valid "
            "templates."
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

    # Sort by e-value
    sorted_templates = sorted(template_cache.items(), key=lambda x: x[1]["e_value"])
    ids_dates = [
        (template_id, datetime.strptime(template_data["release_date"], "%Y-%m-%d"))
        for template_id, template_data in sorted_templates
    ]

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
                continue
        else:
            if check_release_date_max(
                template_release_date=template_date, max_release_date=max_release_date
            ):
                continue
        # Add to list of filtered templates if pass
        filtered_templates.append(template_id)

        # Break if max templates reached
        if len(filtered_templates) == max_templates:
            break

    logging.info(
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
        template_cache_directory,
        max_templates,
        is_core_train,
        max_release_date,
        min_release_date_diff,
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
            max_templates (int):
                Maximum number of templates to keep per query chain.
            is_core_train (bool):
                Whether the dataset is core train or not.
            max_release_date (datetime | None):
                Maximum release date for templates.
            min_release_date_diff (int | None):
                Minimum release date difference for core train templates.

        """
        self.template_cache_path = template_cache_directory
        self.max_templates = max_templates
        self.is_core_train = is_core_train
        self.max_release_date = max_release_date
        self.min_release_date_diff = min_release_date_diff

    @wraps(filter_template_cache_for_query)
    def __call__(self, input: _TemplateQueryEntry) -> TemplateHitCollection:
        try:
            valid_templates = filter_template_cache_for_query(
                input,
                self.template_cache_path,
                self.max_templates,
                self.is_core_train,
                self.max_release_date,
                self.min_release_date_diff,
            )
            return valid_templates
        except Exception as e:
            if self.is_core_train:
                logger.info(
                    "Failed to filter templates for query " f"{input[0][0]}: {e}"
                )
                return {input[0][0]: []}
            else:
                query_pdb_chain_ids = [
                    query_pdb_chain_id.query_pdb_chain_id
                    for query_pdb_chain_id in input.dated_query
                ]
                logger.info(
                    "Failed to filter templates for queries "
                    f"{query_pdb_chain_ids}: "
                    f"{e}."
                )
                return {
                    query_pdb_chain_id: [] for query_pdb_chain_id in query_pdb_chain_ids
                }


def filter_template_cache_af3(
    dataset_cache_file: Path,
    updated_dataset_cache_file: Path,
    template_cache_directory: Path,
    max_templates: int,
    is_core_train: bool,
    num_workers: int,
    save_frequency: int,
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
        max_templates (int):
            Maximum number of templates to keep per query chain.
        is_core_train (bool):
            Whether the dataset is core train or not.
        num_workers (int):
            Number of workers to use for multiprocessing.
        save_frequency (int):
            Frequency at which to save the updated dataset cache.
        max_release_date (datetime | None):
            Maximum release date for templates. Defaults to None.
        min_release_date_diff (int | None):
            Minimum release date difference for core train templates. Defaults to None.
    """
    # Parse list of chains from metadata cache
    with open(dataset_cache_file) as f:
        dataset_cache = json.load(f)
    template_query_iterator = parse_representatives(
        dataset_cache, is_core_train
    ).entries
    data_iterator_len = len(template_query_iterator)

    # Filter template cache for each query chain
    wrapped_template_cache_filter = _AF3TemplateCacheFilter(
        template_cache_directory,
        max_templates,
        is_core_train,
        max_release_date,
        min_release_date_diff,
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
                dataset_cache["structure_data"][pdb_id]["chains"][chain_id][
                    "template_ids"
                ] = valid_template_list

            if (idx + 1) % save_frequency == 0:
                with open(updated_dataset_cache_file, "w") as f:
                    json.dump(dataset_cache, f, indent=4)

    # Save final complete dataset cache
    with open(updated_dataset_cache_file, "w") as f:
        json.dump(dataset_cache, f, indent=4)
