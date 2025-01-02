import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import click

from openfold3.core.data.pipelines.preprocessing.dataset_cache import (
    create_pdb_val_dataset_cache_af3,
)


# TODO: Does the disordered dataset also need to be an input to this?
@click.command()
@click.option(
    "--train-cache",
    "train_cache_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the structure train_cache.json created in preprocessing.",
)
@click.option(
    "--metadata-cache",
    "metadata_cache_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the structure metadata_cache.json created in preprocessing.",
)
@click.option(
    "--preprocessed-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to directory of directories containing preprocessed mmCIF files.",
)
@click.option(
    "--alignment-representatives-fasta",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the alignment representatives FASTA file.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Output path the dataset_cache.json will be written to.",
)
@click.option(
    "--dataset-name",
    type=str,
    required=True,
    help="Name of the dataset, e.g. 'PDB-weighted'.",
)
@click.option(
    "--max-release-date",
    type=str,
    required=True,
    default="2023-01-13",
    help="Maximum release date for included structures, formatted as 'YYYY-MM-DD'.",
)
@click.option(
    "--min-release-date",
    type=str,
    required=True,
    default="2021-09-30",
    help="Minimum release date for included structures, formatted as 'YYYY-MM-DD'.",
)
@click.option(
    "--max-resolution",
    type=float,
    default=4.5,
    help="Maximum resolution for structures in the dataset in Å.",
)
@click.option(
    "--max-polymer-chains",
    type=int,
    default=1000,
    help="Maximum number of polymer chains for included structures.",
)
@click.option(
    "--random-seed",
    type=int | None,
    default=None,
    help="Random seed for reproducibility.",
)
@click.option(
    "--write-no-alignment-repr-entries",
    is_flag=True,
    help=(
        "Whether to write out entries with no alignment representative explicitly to "
        "a no_alignment_representative_entries.json."
    ),
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="WARNING",
    help="Set the logging level.",
)
@click.option(
    "--log-file",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to write the log file to.",
    default=None,
)
def main(  # TODO: Docstring
    pdb_weighted_cache_path: Path,
    metadata_cache_path: Path,
    preprocessed_dir: Path,
    alignment_representatives_fasta: Path,
    output_path: Path,
    dataset_name: str,
    max_release_date: str = "2023-01-13",
    min_release_date: str = "2021-10-01",
    max_resolution: float = 4.5,
    max_polymer_chains: int = 1000,
    random_seed: int | None = None,
    write_no_alignment_repr_entries: bool = False,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
    log_file: Path | None = None,
) -> None:
    """Create a validation dataset cache using AF3 filtering procedures.

    This follows the validation set creation outlined in the AF3 SI Section 5.8.
    
    Args:
        pdb_weighted_cache_path (Path):
            Path to the PDB-weighted training set cache created in preprocessing.
        metadata_cache_path (Path):
            Path to the structure metadata_cache.json created in preprocessing.
        preprocessed_dir (Path):
            Path to directory of directories containing files related to preprocessed
            structures (in particular the .fasta files created by the preprocessing
            pipeline).
        alignment_representatives_fasta (Path):
            Path to the alignment representatives FASTA file.
        output_path (Path):
            Output path the validation dataset cache JSON will be written to.
        dataset_name (str):
            Name of the dataset, e.g. 'PDB-validation'.
        max_release_date (str):
            Maximum release date for included structures, formatted as 'YYYY-MM-DD'.
        min_release_date (str):
            Minimum release date for included structures, formatted as 'YYYY-MM-DD'.
        max_resolution (float):
            Maximum resolution for structures in the dataset in Å.
        max_polymer_chains (int):
            Maximum number of polymer chains a structure can have to be included as a
            target.
        random_seed (int | None):
            Random seed for reproducibility.
        write_no_alignment_repr_entries (bool):
            Whether to write out entries with no alignment representative explicitly to
            a no_alignment_representative_entries.json.
        log_level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL]):
            Set the logging level.
        log_file (Path | None):
            Path to write the log file to.
    """
    max_release_date = datetime.strptime(max_release_date, "%Y-%m-%d").date()
    min_release_date = datetime.strptime(min_release_date, "%Y-%m-%d").date()
    # Set up logger
    logger = logging.getLogger("openfold3")
    logger.setLevel(getattr(logging, log_level))
    logger.addHandler(logging.StreamHandler())

    # Add file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        logger.addHandler(file_handler)

    create_pdb_val_dataset_cache_af3(
        train_cache_path=pdb_weighted_cache_path,
        metadata_cache_path=metadata_cache_path,
        preprocessed_dir=preprocessed_dir,
        alignment_representatives_fasta=alignment_representatives_fasta,
        output_path=output_path,
        dataset_name=dataset_name,
        max_release_date=max_release_date,
        min_release_date=min_release_date,
        max_resolution=max_resolution,
        max_polymer_chains=max_polymer_chains,
        random_seed=random_seed,
        write_no_alignment_repr_entries=write_no_alignment_repr_entries,
    )


if __name__ == "__main__":
    main()
