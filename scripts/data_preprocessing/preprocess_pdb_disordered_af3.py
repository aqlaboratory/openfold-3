"""Script for creating a metadata cache for the disordered distillation set."""

import logging
from pathlib import Path
from typing import Literal

import click

from openfold3.core.data.pipelines.preprocessing.structure import (
    preprocess_pdb_disordered_af3,
)


# TODO: rename to make it more clear this script is for metadata cache creation
@click.command()
@click.option(
    "--metadata_cache_file",
    required=True,
    help=(
        "Metadata cache JSON file created by preprocessing the PDB using "
        "scripts/data_preprocessing/preprocess_pdb_af3.py."
    ),
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--gt_structures_directory",
    required=True,
    help="Flat directory of cif files of ground truth PDB structures.",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--pred_structures_directory",
    required=True,
    help=(
        "Directory containing one subdir per PDB entry, with each subdir."
        " containing one or multiple cif files of predicted structures. The input "
        "metadata cache is always subset to the set of PDB IDs for which a predicted"
        " structure can be found in the pred_structures_directory."
    ),
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--pred_file_name",
    required=True,
    help="Name of the predicted structure file to choose from each per-entry subdir.",
    type=str,
)
@click.option(
    "--output_directory",
    required=True,
    help="Output directory for the disordered metadata cache.",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
# TODO: add option to run OpenStructure inside this script - requires OpenStructure as
# a non-conflicting dependency of openfold3
@click.option(
    "--ost_aln_output_directory",
    required=True,
    help=(
        "Directory where precomputed structural aligment results can be provided."
        "Structural alignments can be precomputed using "
        "scripts/data_preprocessing/compare_structures_with_ost.py, which requires "
        "OpenStructure to be installed. Currently, this is necessary to run the "
        "the disordered distillation preprocessing pipeline."
    ),
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--subset_file",
    required=False,
    help=(
        "A tsv file containing a single column of PDB IDs to subset the metadata "
        "cache to. If not provided, all PDB IDs from the metadata cache will be "
        "used to create the disordered metadata cache. The input metadata cache is"
        "always subset to the set of PDB IDs for which a predicted structure can"
        "be found in the pred_structures_directory."
    ),
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--log_level",
    default="INFO",
    help="Logging level.",
    type=Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
)
def main(
    metadata_cache_file: Path,
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    pred_file_name: str,
    output_directory: Path,
    ost_aln_output_directory: Path,
    subset_file: Path,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
) -> None:
    """Creates a metadata cache for the disordered distillation set.

    Args:
        metadata_cache_file (Path):
            _description_
        gt_structures_directory (Path):
            _description_
        pred_structures_directory (Path):
            _description_
        pred_file_name (str):
            _description_
        output_directory (Path):
            _description_
        ost_aln_output_directory (Path):
            _description_
        subset_file (Path):
            _description_
        log_level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]):
            _description_
    """

    logger = logging.getLogger("openfold3")
    logger.setLevel(getattr(logging, log_level))
    logger.addHandler(logging.StreamHandler())

    preprocess_pdb_disordered_af3(
        metadata_cache_file=metadata_cache_file,
        gt_structures_directory=gt_structures_directory,
        pred_structures_directory=pred_structures_directory,
        pred_file_name=pred_file_name,
        output_directory=output_directory,
        ost_aln_output_directory=ost_aln_output_directory,
        subset_file=subset_file,
    )


if __name__ == "__main__":
    main()
