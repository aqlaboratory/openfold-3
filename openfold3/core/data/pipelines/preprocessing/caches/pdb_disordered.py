import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from openfold3.core.data.primitives.caches.format import (
    DisorderedPreprocessingDataCache,
    DisorderedPreprocessingStructureData,
    PreprocessingDataCache,
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


# TODO: add support for computing GDT with OST on the fly
def build_provisional_disordered_metadata_cache(
    parent_metadata_cache: PreprocessingDataCache,
    pdb_id_list: list[str],
    ost_aln_output_directory: Path,
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

    Returns (DisorderedPreprocessingDataCache):
        The disordered metadata cache with populated gdt and chain_map fields.
    """

    # Update the structure data
    structure_data = {}
    for pdb_id in tqdm(
        pdb_id_list,
        desc="1/x: Building disordered metadata cache",
        total=len(pdb_id_list),
    ):
        structure_data_entry = parent_metadata_cache.structure_data[pdb_id]

        with open(ost_aln_output_directory / f"{pdb_id}.json") as f:
            ost_aln_data = json.load(f)

        structure_data[pdb_id] = DisorderedPreprocessingStructureData(
            status=structure_data_entry.status,
            release_date=structure_data_entry.release_date,
            resolution=structure_data_entry.resolution,
            chains=structure_data_entry.chains,
            interfaces=structure_data_entry.interfaces,
            token_count=structure_data_entry.token_count,
            gdt=ost_aln_data["oligo_gdtts"],
            chain_map=ost_aln_data["chain_mapping"],
            has_clash=None,
        )

    provisional_metadata_cache = DisorderedPreprocessingDataCache(
        structure_data=structure_data,
        reference_molecule_data=parent_metadata_cache.reference_molecule_data,
    )

    return provisional_metadata_cache
