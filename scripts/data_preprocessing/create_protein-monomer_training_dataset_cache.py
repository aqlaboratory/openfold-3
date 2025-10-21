# %%
import logging
from pathlib import Path

import click

from openfold3.core.data.io.s3 import parse_s3_config
from openfold3.core.data.pipelines.preprocessing.caches.protein_monomer import (
    create_protein_monomer_dataset_cache_of3,
)
from openfold3.core.data.pipelines.preprocessing.structure import (
    preparse_protein_monomer_structures,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data_directory",
    required=True,
    help="Directory containing per-monomer folders. If the directory lives in an"
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
    "--output_dir",
    required=True,
    help="Path to where the dataset cache json and preparsed structures are saved",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--structure_filename",
    required=False,
    default=None,
    type=str,
    help="Shared name of structure files to pre-parse across all monomers."
    "If None, the structure file is assumed to have the same name as the monomer ID.",
)
@click.option(
    "--structure_file_format",
    required=True,
    type=str,
    help="Format of the structure files to pre-parse.",
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
    "for example, 'best_structure_relaxed.pdb'; the associated ID of the "
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
    output_dir: Path,
    structure_filename: str,
    structure_file_format: str,
    s3_client_config: dict | None = None,
    check_filename_exists: str | None = None,
    num_workers: int = 1,
    chunksize: int = 1,
):
    """Create a dataset cache and preparsed structures for a protein monomer dataset.

    Args:
        data_directory (Path):
            Directory containing per-monomer folders. If the directory lives in an S3
            bucket, the path should be 's3:/<bucket>/<prefix>'.
        protein_reference_molecule_data_file (Path):
            Path to a reference molecule data file containing the unique set of all CCD
            reference molecules that occur in the entries. An example file is available
            in openfold3/core/data/resources.
        dataset_name (str):
            The name of the dataset to create.
        output_dir (Path):
            Path to where the dataset cache json is saved.
        structure_filename (str):
            Shared name of structure files to pre-parse across all monomers. If None,
            the structure file is assumed to have the same name as the monomer ID.
        s3_client_config (dict | None, optional):
            The argument s3_client_config input as a JSON string with keys 'profile' and
            'max_keys'. Defaults to None.
        check_filename_exists (str | None, optional):
            This enables a recursive search of the specified s3 directory This is an
            expensive operation and should only be used when necessary. The passed
            filename should be a basename of the file to search for, for example,
            'best_structure_relaxed.pdb'; the associated ID of the file will only be
            added if the file exists. You should specify a large value for --num_workers
            to speed up the search. Defaults to None.
        num_workers (int, optional):
            The number of workers to use for parallel processing. Only used for cache
            creation from S3 if --target_filename is specified. Always used for
            structure pre-parsing. Defaults to 1.
        chunksize (int, optional):
            The number of entries to process in each chunk. Defaults to 1.
    """
    # Parse S3 config
    if s3_client_config is not None:
        s3_client_config = parse_s3_config(s3_client_config)

    # Create cache
    logger.info("Creating dataset cache...")
    dataset_cache = create_protein_monomer_dataset_cache_of3(
        data_directory=data_directory,
        protein_reference_molecule_data_file=protein_reference_molecule_data_file,
        dataset_name=dataset_name,
        output_dir=output_dir,
        s3_client_config=s3_client_config,
        check_filename_exists=check_filename_exists,
        num_workers=num_workers,
    )

    # Pre-parse structures into pkl/npz files
    logger.info("Pre-parsing structures...")
    preparse_protein_monomer_structures(
        dataset_cache=dataset_cache,
        data_directory=data_directory,
        structure_filename=structure_filename,
        structure_file_format=structure_file_format,
        output_dir=output_dir,
        num_workers=num_workers,
        chunksize=chunksize,
    )


if __name__ == "__main__":
    main()
