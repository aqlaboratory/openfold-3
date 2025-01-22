import json
import logging
import multiprocessing as mp
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from openfold3.core.data.primitives.caches.format import (
    DisorderedPreprocessingDataCache,
    DisorderedPreprocessingStructureData,
    PreprocessingDataCache,
    PreprocessingStructureData,
)


# TODO: now this module contains pipelines for both disordered metadata and dataset
# cache creation -> separate when refactoring other datacache creation modules
def find_parent_metadata_cache_subset(
    parent_metadata_cache: PreprocessingDataCache,
    pred_structures_directory: Path,
    gt_structures_directory: Path,
    ost_aln_output_directory: Path,
    subset_file: Path | None,
    logger: logging.Logger,
) -> list[str]:
    """Finds the subset of PDB IDs that have all necessary data available.

    Args:
        parent_metadata_cache (PreprocessingDataCache): _description_
        pred_structures_directory (Path): _description_
        gt_structures_directory (Path): _description_
        ost_aln_output_directory (Path): _description_
        subset_file (Path | None): _description_
        logger (logging.Logger): _description_

    Returns:
        list[str]: _description_
    """

    logger.info(
        "Loaded parent metadata cache with "
        f"{len(parent_metadata_cache.structure_data)} entries."
    )

    # - structures available
    pred_pdb_ids = [i.stem for i in list(pred_structures_directory.iterdir())]
    gt_pdb_ids = [i.stem for i in list(pred_structures_directory.iterdir())]

    shared_pdb_ids = sorted(
        set(pred_pdb_ids)
        & set(gt_pdb_ids)
        & set(parent_metadata_cache.structure_data.keys())
    )
    logger.info(
        f"{len(shared_pdb_ids)} metadata keys have both GT and predicted structures "
        f"in {gt_structures_directory} and {pred_structures_directory}, respectively."
    )

    # - alignment successful
    # TODO: remove once OST is integrated into this script
    aln_pdb_ids = [i.stem for i in list(ost_aln_output_directory.iterdir())]

    shared_pdb_ids = sorted(set(shared_pdb_ids) & set(aln_pdb_ids))
    logger.info(
        f"{len(shared_pdb_ids)} have precomputed alignments available in "
        f"{ost_aln_output_directory}."
    )

    # - subset file
    if subset_file is not None:
        subset_pdb_ids = list(pd.read_csv(subset_file, header=None, sep="\t")[0])
        shared_pdb_ids = sorted(set(shared_pdb_ids) & set(subset_pdb_ids))
        logger.info(f"{len(shared_pdb_ids)} are in the subset file {subset_file}.")
    else:
        logger.info(f"No subset file, using all {len(shared_pdb_ids)} entries.")

    return shared_pdb_ids


# TODO: add support for computing GDT with OST on the fly here
def build_provisional_disordered_metadata_cache(
    parent_metadata_cache: PreprocessingDataCache,
    pdb_id_list: list[str],
    ost_aln_output_directory: Path,
    num_workers: int,
    chunksize: int,
    logger: logging.Logger,
) -> DisorderedPreprocessingDataCache:
    """
    Creates the disorder metadata cache from a parent metadata cache.

    Args:
        parent_metadata_cache (PreprocessingDataCache):
            The parent metadata cache from which to derive the disordered metadata
            cache.
        pdb_id_list (list[str]):
            A list of PDB IDs to subset the disordered metadata cache to.
        ost_aln_output_directory (Path):
            The directory where the OST structural alignment output files are stored.
        num_workers (int):
            The number of workers to parallelize the structure data entry updates to.
        chunksize (int):
            The chunksize for the parallelization.
        logger (logging.Logger):
            The logger object.

    Returns (DisorderedPreprocessingDataCache):
        The disordered metadata cache with populated gdt and chain_map fields.
    """

    # Update the structure data
    structure_data = {}
    wrapped_builder = _DisorderedMetadataCacheBuilder(ost_aln_output_directory, logger)
    input_data = [
        (pdb_id, parent_metadata_cache.structure_data[pdb_id]) for pdb_id in pdb_id_list
    ]

    with mp.Pool(num_workers) as pool:
        for pdb_id, provisional_entry in tqdm(
            pool.imap_unordered(
                wrapped_builder,
                input_data,
                desc="1/x: Building provisional disordered metadata cache",
                chunksize=chunksize,
            ),
            total=len(pdb_id_list),
        ):
            structure_data[pdb_id] = provisional_entry

    provisional_metadata_cache = DisorderedPreprocessingDataCache(
        structure_data=structure_data,
        reference_molecule_data=parent_metadata_cache.reference_molecule_data,
    )

    return provisional_metadata_cache


def _create_provisional_disordered_structure_data_entry(
    pdb_id: str,
    structure_data_entry: PreprocessingStructureData,
    ost_aln_output_directory: Path,
) -> DisorderedPreprocessingStructureData:
    """Selects the model with the highest GDT to GT and creates its structure data field

    Args:
        pdb_id (str): _description_
        structure_data_entry (PreprocessingStructureData): _description_
        ost_aln_output_directory (Path): _description_

    Returns:
        DisorderedPreprocessingStructureData: _description_
    """
    # Parse the pred-GT results for all models
    ost_aln_output_directory_i = ost_aln_output_directory / pdb_id
    aln_filenames = [i.stem for i in list(ost_aln_output_directory_i.iterdir())]
    ost_aln_data = []
    gdts = np.zeros(len(aln_filenames))
    for idx, aln_filename in enumerate(aln_filenames):
        with open(ost_aln_output_directory / f"{aln_filename}.json") as f:
            ost_aln_data.append(json.load(f))
            gdts[idx] = float(ost_aln_data["oligo_gdtts"])

    # Find the one with the highest GDT
    best_idx = np.argmax(gdts)

    # Add to the entry
    return DisorderedPreprocessingStructureData(
        status=structure_data_entry.status,
        release_date=structure_data_entry.release_date,
        resolution=structure_data_entry.resolution,
        chains=structure_data_entry.chains,
        interfaces=structure_data_entry.interfaces,
        token_count=structure_data_entry.token_count,
        gdt=gdts[best_idx],
        chain_map=ost_aln_data[best_idx]["chain_mapping"],
        best_model_filename=aln_filenames[best_idx],
        has_clash=None,
    )


class _DisorderedMetadataCacheBuilder:
    def __init__(self, ost_aln_output_directory: Path, logger: logging.Logger):
        self.ost_aln_output_directory = ost_aln_output_directory
        self.logger = logger

    @wraps(_create_provisional_disordered_structure_data_entry)
    def __call__(
        self, input_data: tuple[str, PreprocessingStructureData]
    ) -> DisorderedPreprocessingStructureData:
        try:
            pdb_id, structure_data_entry = input_data
            provisional_entry = _create_provisional_disordered_structure_data_entry(
                pdb_id, structure_data_entry, self.ost_aln_output_directory
            )
            return pdb_id, provisional_entry
        except Exception as e:
            self.logger.info(f"Error processing {pdb_id}: {e}")

            failed_provisional_entry = DisorderedPreprocessingStructureData(
                status="failed",
                release_date=None,
                resolution=None,
                chains=None,
                interfaces=None,
                token_count=None,
                gdt=None,
                chain_map=None,
                has_clash=None,
            )

            return pdb_id, failed_provisional_entry
