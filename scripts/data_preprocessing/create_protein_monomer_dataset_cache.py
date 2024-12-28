# %%
import json
from pathlib import Path

import click

from openfold3.core.data.pipelines.preprocessing.dataset_cache import (
    create_protein_monomer_dataset_cache_af3,
)


@click.command()
@click.option(
    "--data_directory",
    required=True,
    help="Directory containing per-monomer folders.If the directory lives in an"
    " S3 bucket, the path should be 's3:/<bucket>/<prefix>'.",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--protein_reference_molecule_data_file",
    required=True,
    help=(
        "Path to a reference molecule data file containing the unique set of all CCD"
        "reference molecules that occur in the entries. An example file is"
        "available in openfold3/core/data/resources."
    ),
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--dataset_name",
    required=True,
    type=str,
    help="The name of the dataset to create.",
)
@click.option(
    "--output_path",
    required=True,
    help="Path to where the dataset cache json is saved",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--s3_client_config",
    required=False,
    default=None,
    type=str,
    help="The argument s3_client_config "
    "input as a JSON string with keys 'profile' and 'max_keys'.",
)
@click.option(
    "--check_filename_exists",
    required=False,
    default=None,
    type=str,
    help="This enables a recursive search of the specified s3 directory"
    "This is an expensive operation and should only be used when necessary."
    "The passed filename should be a basename of the file to search for,"
    "for exampl, 'best_structure_relaxed.pdb'; the associated ID of the "
    "file will only be added if the file exists."
    "You should specify a large value for --num_workers to speed up the search.",
)
@click.option(
    "--num_workers",
    required=False,
    default=1,
    type=int,
    help="The number of workers to use for parallel processing."
    "Only used if --target_filename is specified.",
)
def main(
    data_directory: Path,
    protein_reference_molecule_data_file: Path,
    dataset_name: str,
    output_path: Path,
    s3_client_config: dict | None = None,
    check_filename_exists: str | None = None,
    num_workers: int = 1,
):
    # Parse S3 config
    if s3_client_config is not None:
        try:
            s3_client_config = json.loads(s3_client_config)
        except json.JSONDecodeError:
            click.echo("Invalid max_seq_counts JSON string!")
            return
    else:
        s3_client_config = {}

    # Create cache
    create_protein_monomer_dataset_cache_af3(
        data_directory=data_directory,
        protein_reference_molecule_data_file=protein_reference_molecule_data_file,
        dataset_name=dataset_name,
        output_path=output_path,
        s3_client_config=s3_client_config,
        check_filename_exists=check_filename_exists,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
