"""Script for comparing ground truth and AF2-predicted structures with OpenStructure.

Output used for preprocessing datapoints for the disordered dataset.

This script is currently separate from preprocess_pdb_disordered_af3.py as it requires
OpenStructure, but will be combined with it in a future release."""

import logging
import multiprocessing as mp
import subprocess
import warnings
from functools import wraps
from pathlib import Path
from typing import Literal

import click
import pandas as pd
from tqdm import tqdm


@click.command()
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
        " containing one or multiple pdb files of predicted structures. The input "
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
    "--gt_structure_file_format",
    required=True,
    help="File format of the ground truth structures.",
    type=click.Choice(["cif", "pdb"], case_sensitive=True),
)
@click.option(
    "--output_directory",
    required=True,
    help="Output directory for the structural alignment result.",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--gt_biounit_id",
    required=False,
    default=None,
    help=(
        "ID of the bioassembly to use from the GT. This parameter only works if "
        "gt_structure_file_format is 'cif' and the corresponding cif files "
        "contains the specified bioassembly."
    ),
    type=str,
)
@click.option(
    "--pred_biounit_id",
    required=False,
    default=None,
    help=(
        "ID of the bioassembly to use from the prediction. This parameter only "
        "works if gt_structure_file_format is 'cif' and the corresponding cif files "
        "contains the specified bioassembly."
    ),
    type=str,
)
@click.option(
    "--log_file",
    required=True,
    help="File to where the output logs are saved.",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--num_workers",
    required=False,
    default=1,
    help="Number of workers to use for parallel processing.",
    type=int,
)
@click.option(
    "--chunksize",
    required=False,
    default=1,
    help="Number of workers to use for parallel processing.",
    type=int,
)
def main(
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    gt_structure_file_format: Literal["cif", "pdb"],
    output_directory: Path,
    gt_biounit_id: str | None,
    pred_biounit_id: str | None,
    log_file: Path,
    num_workers: int,
    chunksize: int,
) -> None:
    """Run OpenStructure structure alignment for all GT-pred pairs."""

    # Configure the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # Set the logging level for the file handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if (gt_structure_file_format == "pdb") and (gt_biounit_id is not None):
        msg = (
            "Bioassembly ID is only supported for GT structures in mmcif format."
            " The specified bioassembly will be ignored."
        )
        logger.info(msg)
        warnings.warn(msg, stacklevel=2)

    # Get list of predicted structures with GT structures available
    pred_pdb_ids = [i.stem for i in list(pred_structures_directory.iterdir())]
    logger.info(
        f"Found dirs for {len(pred_pdb_ids)} predicted structures in "
        f"{pred_structures_directory}."
    )
    gt_pdb_ids = [i.stem for i in list(pred_structures_directory.iterdir())]

    pred_pdb_ids = sorted(set(pred_pdb_ids) & set(gt_pdb_ids))
    logger.info(
        f"{len(pred_pdb_ids)} predicted structures have corresponding GT."
        f"structure at {gt_structures_directory}."
    )

    pd.DataFrame(pred_pdb_ids).to_csv(
        output_directory / "preds_with_gt.tsv", index=False, header=False, sep="\t"
    )

    # Pre-create directories
    alignment_output_directory = output_directory / "alignment_results"
    alignment_output_directory.mkdir(exist_ok=True)
    for pdb_id in tqdm(
        pred_pdb_ids, desc="1/3: Creating output directories", total=len(pred_pdb_ids)
    ):
        (alignment_output_directory / f"{pdb_id}").mkdir(exist_ok=True)

    # Run OST for each GT-pred pair
    wrapped_ost_aligner = _OSTCompareStructuresWrapper(
        gt_structures_directory=gt_structures_directory,
        pred_structures_directory=pred_structures_directory,
        gt_structure_file_format=gt_structure_file_format,
        alignment_output_directory=alignment_output_directory,
        gt_biounit_id=gt_biounit_id,
        pred_biounit_id=pred_biounit_id,
        logger=logger,
    )

    with mp.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                wrapped_ost_aligner,
                pred_pdb_ids,
                chunksize=chunksize,
            ),
            total=len(pred_pdb_ids),
            desc="2/3: Aligning structures",
        ):
            pass

    # Collect failed entries
    aln_pdb_ids = [i.stem for i in list(alignment_output_directory.iterdir())]
    failed_pdb_ids = sorted(set(pred_pdb_ids) - set(aln_pdb_ids))

    logger.info(f"{len(failed_pdb_ids)} failed to be aligned.")

    if len(failed_pdb_ids) > 0:
        pd.DataFrame(failed_pdb_ids).to_csv(
            output_directory / "failed.tsv", index=False, header=False, sep="\t"
        )

        for pdb_id in tqdm(
            failed_pdb_ids, desc="3/3: Removing failed dirs", total=len(failed_pdb_ids)
        ):
            (alignment_output_directory / f"{pdb_id}").rmdir()

    else:
        logger.info("3/3: No failed dirs to remove.")


def compare_pred_to_gt(
    pdb_id: str,
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    gt_structure_file_format: Literal["cif", "pdb"],
    alignment_output_directory: Path,
    gt_biounit_id: str | None,
    pred_biounit_id: str | None,
) -> None:
    """Runs OpenStructure structure alignment for a single GT-pred pair.

    See for more details: https://openstructure.org/docs/2.9.0/actions/.

    Args:
        pdb_id (str):
            PDB ID of the structure to compare.
        gt_structures_directory (Path):
            Flat directory of cif files of ground truth PDB structures.
        pred_structures_directory (Path):
            Directory containing one subdir per PDB entry, with each subdir
            containing one or multiple pdb files of predicted structures.
        gt_structure_file_format (Literal["cif", "pdb"]):
            File format of the ground truth structures.
        alignment_output_directory (Path):
            Output directory for OST.
        gt_biounit_id (str | None):
            Bioassembly ID of the GT structure to compare.
        pred_biounit_id (str | None):
            Bioassembly ID of the predicted structure to compare.
    """
    # Make sure the reference format is mmcif so that the specified bioassembly is used
    rf = "mmcif" if gt_structure_file_format == "cif" else "pdb"

    # Run OST on each model file
    model_files = list((pred_structures_directory / pdb_id).iterdir())
    for model_file in model_files:
        ost_command = [
            "ost",
            "compare-structures",
            "-m",
            f"{model_file}",
            "-r",
            f"{str(gt_structures_directory)}/{pdb_id}.{gt_structure_file_format}",
            "-o",
            f"{str(alignment_output_directory)}/{pdb_id}.json",
            "-rf",
            f"{rf}",
            "--rigid-scores",
        ]
        if gt_biounit_id is not None:
            ost_command.extend(["-rb", gt_biounit_id])
        if pred_biounit_id is not None:
            ost_command.extend(["-mb", pred_biounit_id])

        subprocess.run(ost_command)


class _OSTCompareStructuresWrapper:
    def __init__(
        self,
        gt_structures_directory: Path,
        pred_structures_directory: Path,
        gt_structure_file_format: Literal["cif", "pdb"],
        alignment_output_directory: Path,
        gt_biounit_id: str | None,
        pred_biounit_id: str | None,
        logger: logging.Logger,
    ) -> None:
        """Wrapper class for aligning PDB structures to AF2-predicted models.

        Used for calculating GDT and chain mapping to get the correct chain alignment
        between the ground truth and predicted structures.

        This wrapper around `compare_pred_to_gt` is needed for multiprocessing, so that
        we can pass the constant arguments in a convenient way catch any errors that
        would crash the workers, and change the function call to accept a single
        Iterable.

        The wrapper is written as a class object because multiprocessing doesn't support
        decorator-like nested functions.

        Args:
            gt_structures_directory (Path):
                Flat directory of cif files of ground truth PDB structures.
            pred_structures_directory (Path):
                Directory containing one subdir per PDB entry, with each subdir
                containing one or multiple pdb files of predicted structures
            gt_structure_file_format (Literal["cif", "pdb"]):
                File format of the ground truth structures.
            alignment_output_directory (Path):
                Output directory for OST.
            gt_biounit_id (str | None):
                Bioassembly ID of the GT structure to compare.
            pred_biounit_id (str | None):
                Bioassembly ID of the predicted structure to compare.
            logger (logging.Logger):
                The configured logger object.
        """
        self.gt_structures_directory = gt_structures_directory
        self.pred_structures_directory = pred_structures_directory
        self.gt_structure_file_format = gt_structure_file_format
        self.alignment_output_directory = alignment_output_directory
        self.gt_biounit_id = gt_biounit_id
        self.pred_biounit_id = pred_biounit_id
        self.logger = logger

    @wraps(compare_pred_to_gt)
    def __call__(self, pdb_id: str) -> None:
        try:
            compare_pred_to_gt(
                pdb_id=pdb_id,
                gt_structures_directory=self.gt_structures_directory,
                pred_structures_directory=self.pred_structures_directory,
                gt_structure_file_format=self.gt_structure_file_format,
                alignment_output_directory=self.alignment_output_directory,
                gt_biounit_id=self.gt_biounit_id,
                pred_biounit_id=self.pred_biounit_id,
            )
        except Exception as e:
            self.logger.info(f"Failed to preprocess entry {pdb_id}:\n{e}\n")


if __name__ == "__main__":
    main()
