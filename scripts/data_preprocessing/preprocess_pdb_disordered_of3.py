"""Script for creating a metadata cache for the disordered distillation set."""

import json
from pathlib import Path
from typing import Any

import click

from openfold3.core.data.pipelines.preprocessing.structure import (
    preprocess_pdb_disordered_of3,
)


# TODO: rename to make it more clear this script is for metadata cache creation
@click.command()
@click.option(
    "--metadata_cache_file",
    required=True,
    help=(
        "Metadata cache JSON file created by preprocessing the PDB using "
        "scripts/data_preprocessing/preprocess_pdb_of3.py."
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
    help=(
        "Directory of preprocessed GT PDB structures. It should contain one subdir "
        "per PDB entry, with each subdir containing one or multiple structure files "
        "of ground truth structures preprocessed using preprocess_pdb_of3.py."
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
        "Directory of PDB structures predicted using AF2. It should contain one subdir "
        "per PDB entry, with each subdir containing one or multiple cif files of "
        "predicted structures. The input metadata cache is always subset to the set "
        "of PDB IDs for which a predicted structure can be found in the "
        "pred_structures_directory."
    ),
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--gt_file_format",
    required=True,
    help="File format of the structure file to use from gt_structures_directory.",
    type=str,
)
@click.option(
    "--pred_file_format",
    required=True,
    help="File format of the predicted file to use from pred_structures_directory.",
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
    "--ccd_file",
    required=True,
    help=("Path to a CCD file."),
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--pocket_distance_threshold",
    required=True,
    help=(
        "The distance in A between any non-protein atom and protein backbone atom."
        "below which a corresponding protein residue qualfies as a pocket residue."
    ),
    type=float,
)
@click.option(
    "--clash_distance_thresholds",
    required=True,
    help=(
        "Comma-delimited floats indicating distances below which protein-"
        "non-protein atom pairs are considered to be clashing."
    ),
    type=str,
)
@click.option(
    "--transfer_annot_dict",
    required=True,
    help=(
        "A json string encoding a dict of annotations to transfer from the GT to "
        "the predicted structure. Keys are annotation names, values are default "
        "values to initialize the annotation before transfer."
    ),
    type=str,
)
@click.option(
    "--delete_annot_list",
    required=False,
    help=(
        "Comma-delimited list of annotations to delete from the predicted structure."
    ),
    type=str,
    default="",
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
@click.option(
    "--log_file",
    required=True,
    help=("A log file where the output logs are saved."),
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
def main(
    metadata_cache_file: Path,
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    gt_file_format: str,
    pred_file_format: str,
    output_directory: Path,
    ost_aln_output_directory: Path,
    subset_file: Path,
    ccd_file: Path,
    pocket_distance_threshold: float,
    clash_distance_thresholds: str,
    transfer_annot_dict: str,
    delete_annot_list: str,
    num_workers: int,
    chunksize: int,
    log_file: Path,
) -> None:
    """Creates a metadata cache for the disordered distillation set.

    Args:
        metadata_cache_file (Path):
            Parent metadata cache file created by preprocessing the PDB using
            scripts/data_preprocessing/preprocess_pdb_of3.py.
        gt_structures_directory (Path):
            Directory of preprocessed GT PDB structures. It should contain one subdir
            per PDB entry, with each subdir containing one or multiple structure files
            of ground truth structures preprocessed using preprocess_pdb_of3.py.
        pred_structures_directory (Path):
            Directory of PDB structures predicted using AF2. It should contain one
            subdir per PDB entry, with each subdir containing one or multiple files
            of predicted structures. The input metadata cache is always subset to the
            set of PDB IDs for which a predicted structure can be found in the
            pred_structures_directory.
        output_directory (Path):
            Output directory for the disordered metadata cache and processed structures.
        ost_aln_output_directory (Path):
            Directory where precomputed structural aligment results can be provided.
        subset_file (Path):
            A tsv file containing a single column of PDB IDs to subset the metadata
            cache to. If not provided, all PDB IDs from the metadata cache will be
            used to create the disordered metadata cache. The input metadata cache is
            always subset to the set of PDB IDs for which a predicted structure can
            be found in the pred_structures_directory.
        ccd_file (Path):
            Path to a CCD file.
        pocket_distance_threshold (float):
            The distance in A between any non-protein atom and protein backbone atom
            below which a corresponding protein residue qualfies as a pocket residue.
        clash_distance_thresholds (str):
            Comma-delimited floats indicating distances below which protein-
            non-protein atom pairs are considered to be clashing.
        transfer_annot_dict (str):
            A json string encoding a dict of annotations to transfer from the GT to
            the predicted structure. Keys are annotation names, values are default
            values to initialize the annotation before transfer.
        delete_annot_list (str):
            Comma-delimited list of annotations to delete from the predicted structure.
        num_workers (int):
            Number of workers to use for parallel processing.
        chunksize (int):
            Chunksize for parallel processing.
        log_file (Path):
            A log file where the output logs are saved.
    """
    # Parse input args
    clash_distance_thresholds, transfer_annot_dict, delete_annot_list = (
        parse_input_args(
            clash_distance_thresholds, transfer_annot_dict, delete_annot_list
        )
    )

    preprocess_pdb_disordered_of3(
        metadata_cache_file=metadata_cache_file,
        gt_structures_directory=gt_structures_directory,
        pred_structures_directory=pred_structures_directory,
        gt_file_format=gt_file_format,
        pred_file_format=pred_file_format,
        output_directory=output_directory,
        ost_aln_output_directory=ost_aln_output_directory,
        subset_file=subset_file,
        ccd_file=ccd_file,
        pocket_distance_threshold=pocket_distance_threshold,
        clash_distance_thresholds=clash_distance_thresholds,
        transfer_annot_dict=transfer_annot_dict,
        delete_annot_list=delete_annot_list,
        num_workers=num_workers,
        chunksize=chunksize,
        log_file=log_file,
    )


def parse_input_args(
    clash_distance_thresholds: str,
    transfer_annot_dict: str,
    delete_annot_list: str,
) -> tuple[list[float], dict[str, Any], list[str]]:
    """Parse input arguments from str to the appropriate data types."""
    try:
        clash_distance_thresholds = [
            float(threshold.strip())
            for threshold in clash_distance_thresholds.split(",")
        ]
    except Exception as e:
        print(f"Invalid clash_distance_thresholds string: {e}")
        exit()
    try:
        transfer_annot_dict = json.loads(transfer_annot_dict)
    except json.JSONDecodeError:
        print("Invalid transfer_annot_dict JSON string!")
        exit()
    try:
        if delete_annot_list != "":
            delete_annot_list = [
                annot.strip() for annot in delete_annot_list.split(",")
            ]
        else:
            delete_annot_list = []
    except Exception as e:
        print(f"Invalid delete_annot_list string: {e}")
        exit()
    return clash_distance_thresholds, transfer_annot_dict, delete_annot_list


if __name__ == "__main__":
    main()
