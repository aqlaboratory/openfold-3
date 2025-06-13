"""Script for comparing ground truth and AF2-predicted structures with OpenStructure.

Output used for preprocessing datapoints for the disordered dataset.

This script is currently separate from preprocess_pdb_disordered_of3.py as it requires
OpenStructure, but will be combined with it in a future release."""

import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import warnings
from functools import wraps
from pathlib import Path
from typing import Literal, Optional

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option(
    "--gt_structures_directory",
    required=True,
    help=(
        "Directory containing one subdir per PDB entry, with each subdir."
        " containing one or multiple structure files of ground truth structures"
    ),
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
    "--subset_file",
    required=False,
    default=None,
    help=(
        "A tsv file of a single column containing IDs from the set of predicted"
        "structures to subset the alignment to."
    ),
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
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
    gt_biounit_id: Optional[str],
    pred_biounit_id: Optional[str],
    subset_file: Optional[Path],
    log_file: Path,
    num_workers: int,
    chunksize: int,
) -> None:
    """Run OpenStructure structure alignment for all GT-pred pairs."""

    # Configure the logger
    logger_main = logging.getLogger(__name__)
    logger_main.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    logger_main.addHandler(file_handler)

    if (gt_structure_file_format == "pdb") and (gt_biounit_id is not None):
        msg = (
            "Bioassembly ID is only supported for GT structures in mmcif format."
            " The specified bioassembly will be ignored."
        )
        logger_main.info(msg)
        warnings.warn(msg, stacklevel=2)

    # Get list of predicted structures with GT structures available
    pred_pdb_ids = [i.stem for i in list(pred_structures_directory.iterdir())]
    logger_main.info(
        f"Found dirs for {len(pred_pdb_ids)} predicted structures in "
        f"{pred_structures_directory}."
    )
    gt_pdb_ids = [i.stem for i in list(pred_structures_directory.iterdir())]

    pred_pdb_ids = sorted(set(pred_pdb_ids) & set(gt_pdb_ids))
    logger_main.info(
        f"{len(pred_pdb_ids)} predicted structures have corresponding GT"
        f"structure at {gt_structures_directory}."
    )

    pd.DataFrame(pred_pdb_ids).to_csv(
        output_directory / "preds_with_gt.tsv", index=False, header=False, sep="\t"
    )

    if subset_file is not None:
        subset_pdb_ids = pd.read_csv(subset_file, header=None, sep="\t")[0].tolist()
        logger_main.info(f"{len(subset_pdb_ids)} IDs are in the subset file.")
        pred_pdb_ids = sorted(set(pred_pdb_ids) & set(subset_pdb_ids))
        logger_main.info(
            f"{len(pred_pdb_ids)} predicted structures are in the subset file."
        )

    try:
        # Pre-create directories
        alignment_output_directory = output_directory / "alignment_results"
        alignment_output_directory.mkdir(exist_ok=True)
        worker_log_directory = output_directory / "worker_logs"
        worker_log_directory.mkdir(exist_ok=True)
        for pdb_id in tqdm(
            pred_pdb_ids,
            desc="1/4: Creating output directories",
            total=len(pred_pdb_ids),
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
        )

        with mp.Pool(num_workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    wrapped_ost_aligner,
                    pred_pdb_ids,
                    chunksize=chunksize,
                ),
                total=len(pred_pdb_ids),
                desc="2/4: Aligning structures",
            ):
                pass
    except Exception as e:
        logger_main.info(f"Failed to align structures:\n{e}\n")
    finally:
        # Collate logs
        with log_file.open("a") as out_file:
            for worker_log in tqdm(
                worker_log_directory.iterdir(),
                desc="3/4: Collating worker logs",
                total=len(list(worker_log_directory.iterdir())),
            ):
                out_file.write(f"Log file: {worker_log.name}\n")
                out_file.write(worker_log.read_text())
                worker_log.unlink()

            if not list(worker_log_directory.iterdir()):
                worker_log_directory.rmdir()

        # Collect failed entries
        successful_pdb_ids, failed_pdb_ids = parse_success_status(log_file)

        logger_main.info(f"{len(successful_pdb_ids)} are successfully aligned.")
        logger_main.info(f"{len(failed_pdb_ids)} failed to be aligned.")

        if len(successful_pdb_ids) > 0:
            pd.DataFrame(successful_pdb_ids).to_csv(
                output_directory / "successful.tsv", index=False, header=False, sep="\t"
            )

        if len(failed_pdb_ids) > 0:
            pd.DataFrame(failed_pdb_ids).to_csv(
                output_directory / "failed.tsv", index=False, header=False, sep="\t"
            )

            for pdb_id in tqdm(
                failed_pdb_ids,
                desc="4/4: Removing failed dirs",
                total=len(failed_pdb_ids),
            ):
                shutil.rmtree(alignment_output_directory / f"{pdb_id}")

        else:
            logger_main.info("4/4: No failed dirs to remove.")


def compare_pred_to_gt(
    pdb_id: str,
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    gt_structure_file_format: Literal["cif", "pdb"],
    alignment_output_directory: Path,
    gt_biounit_id: Optional[str],
    pred_biounit_id: Optional[str],
    logger: logging.Logger,
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
        model_filename = model_file.stem
        ost_command = [
            "ost",
            "compare-structures",
            "-m",
            f"{model_file}",
            "-r",
            f"{gt_structures_directory}/{pdb_id}/{pdb_id}.{gt_structure_file_format}",
            "-o",
            f"{alignment_output_directory}/{pdb_id}/{model_filename}.json",
            "-rf",
            f"{rf}",
            "--rigid-scores",
            "-v",
            "0",
        ]
        if gt_biounit_id is not None:
            ost_command.extend(["-rb", gt_biounit_id])
        if pred_biounit_id is not None:
            ost_command.extend(["-mb", pred_biounit_id])

        result = subprocess.run(ost_command, capture_output=True, text=True)
        if result.stdout:
            logger.info("STDOUT:\n%s", result.stdout.strip())
        if result.stderr:
            logger.error("STDERR:\n%s", result.stderr.strip())
        if result.returncode != 0:
            logger.error("Command failed with return code %d", result.returncode)


class _OSTCompareStructuresWrapper:
    def __init__(
        self,
        gt_structures_directory: Path,
        pred_structures_directory: Path,
        gt_structure_file_format: Literal["cif", "pdb"],
        alignment_output_directory: Path,
        gt_biounit_id: Optional[str],
        pred_biounit_id: Optional[str],
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
        """
        self.gt_structures_directory = gt_structures_directory
        self.pred_structures_directory = pred_structures_directory
        self.gt_structure_file_format = gt_structure_file_format
        self.alignment_output_directory = alignment_output_directory
        self.gt_biounit_id = gt_biounit_id
        self.pred_biounit_id = pred_biounit_id

    @wraps(compare_pred_to_gt)
    def __call__(self, pdb_id: str) -> None:
        try:
            # Setup worker logger
            logger = logging.getLogger(f"worker_{os.getpid()}")
            logger.setLevel(logging.INFO)
            logger.handlers = []
            logger.propagate = False
            handler = logging.FileHandler(
                self.alignment_output_directory.parent
                / f"worker_logs/worker_{os.getpid()}.log"
            )
            logger.addHandler(handler)

            logger.info(f"Processing {pdb_id}.")

            compare_pred_to_gt(
                pdb_id=pdb_id,
                gt_structures_directory=self.gt_structures_directory,
                pred_structures_directory=self.pred_structures_directory,
                gt_structure_file_format=self.gt_structure_file_format,
                alignment_output_directory=self.alignment_output_directory,
                gt_biounit_id=self.gt_biounit_id,
                pred_biounit_id=self.pred_biounit_id,
                logger=logger,
            )
        except Exception as e:
            logger.info(f"Failed to preprocess entry {pdb_id}:\n{e}\n")


def parse_success_status(log_file: Path):
    """
    Parse the log and return two sorted lists:
      - success_pdb_ids: PDBs that were processed and did not fail
      - failed_pdb_ids: PDBs that encountered a failure

    A PDB is considered 'failed' if either:
      1) The log contains "Failed to preprocess entry <PDB_ID>:"
      2) The log shows "Command failed with return code X"
         after a line "Processing <PDB_ID>."

    Otherwise, if we see "Processing <PDB_ID>." and no corresponding fail,
    that PDB is 'successful'.
    """

    # Regex patterns for relevant lines
    re_failed_preprocess = re.compile(r"^Failed to preprocess entry (\S+):")
    re_processing = re.compile(r"^Processing (\S+)\.$")
    re_command_failed = re.compile(r"^Command failed with return code \d+")

    # Keep track of all PDBs we see, and which ones fail
    all_pdb_ids = set()
    failed_pdb_ids = set()

    # Tracks the current PDB being processed for associating "Command failed ..."
    current_pdb = None

    lines = log_file.read_text().splitlines()
    for line in lines:
        # 1) "Processing <PDB_ID>." => track it as seen
        match_proc = re_processing.match(line)
        if match_proc:
            current_pdb = match_proc.group(1)
            all_pdb_ids.add(current_pdb)
            continue

        # 2) "Failed to preprocess entry <PDB_ID>:"
        match_failp = re_failed_preprocess.match(line)
        if match_failp:
            pdb_id = match_failp.group(1)
            all_pdb_ids.add(pdb_id)  # ensure it's in our set of 'seen' PDBs
            failed_pdb_ids.add(pdb_id)
            continue

        # 3) "Command failed with return code X" => last processed PDB is considered
        #    failed
        match_cmdfail = re_command_failed.match(line)
        if match_cmdfail and current_pdb:
            failed_pdb_ids.add(current_pdb)
            # Optionally reset current_pdb = None if you want to avoid a repeated
            # assignment if subsequent lines belong to a different PDB, but it's safe to
            # leave.

    success_pdb_ids = sorted(all_pdb_ids - failed_pdb_ids)
    failed_pdb_ids = sorted(failed_pdb_ids)

    return success_pdb_ids, failed_pdb_ids


if __name__ == "__main__":
    main()
