"""Script for comparing ground truth and AF2-predicted structures with OpenStructure.

Output used for preprocessing datapoints for the disordered dataset.

This script is currently separate from preprocess_pdb_disordered_af3.py as it requires
OpenStructure, but will be combined with it in a future release."""

import subprocess
from functools import wraps
from pathlib import Path

import click


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
def main(
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    pred_file_name: str,
    output_directory: Path,
) -> None:
    pass


def compare_pred_to_gt(
    pdb_id: str,
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    pred_file_name: str,
    gt_structure_file_format: str,
    pred_structure_file_format: str,
    output_directory: Path,
) -> None:
    """Runs OpenStructure structure alignment for a single GT-pred pair.

    See for more details: https://openstructure.org/docs/2.9.0/actions/.

    Args:
        pdb_id (str):
            _description_
        gt_structures_directory (Path):
            _description_
        pred_structures_directory (Path):
            _description_
        pred_file_name (str):
            _description_
        gt_structure_file_format (str):
            _description_
        pred_structure_file_format (str):
            _description_
        output_directory (Path):
            _description_
    """
    ost_command = [
        "ost",
        "compare-structures",
        "-m",
        f"{str(pred_structures_directory)}/{pdb_id}/{pred_file_name}.{pred_structure_file_format}",
        "-r",
        f"{str(gt_structures_directory)}/{pdb_id}.{gt_structure_file_format}",
        "-rb",
        "1",
        "--rigid-scores",
        "-o",
        f"{str(output_directory)}/{pdb_id}.json",
    ]
    subprocess.run(ost_command)


class _OSTCompareStructuresWrapper:
    def __init__(
        self,
        data_directory: Path,
        structure_filename: str,
        structure_file_format: str,
        output_dir: Path,
    ) -> None:
        """Wrapper class for pre-parsing protein mononer files into .pkl."""
        self.data_directory = data_directory
        self.structure_filename = structure_filename
        self.structure_file_format = structure_file_format
        self.output_dir = output_dir

    @wraps(compare_pred_to_gt)
    def __call__(self, entry_id: str) -> None:
        try:
            compare_pred_to_gt(
                entry_id,
                self.data_directory,
                self.structure_filename,
                self.structure_file_format,
                self.output_dir,
            )
        except Exception as e:
            print(f"Failed to preparse monomer {entry_id}:\n{e}\n")


if __name__ == "__main__":
    main()
