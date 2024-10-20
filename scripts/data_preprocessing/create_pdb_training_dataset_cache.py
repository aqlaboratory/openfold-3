import logging
from pathlib import Path
from typing import Literal

import click

from openfold3.core.data.pipelines.preprocessing.dataset_cache import (
    create_pdb_training_dataset_cache_af3,
)


@click.command()
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
    help="Maximum release date for included structures, formatted as 'YYYY-MM-DD'.",
)
@click.option(
    "--max-resolution",
    type=float,
    default=9.0,
    help="Maximum resolution for structures in the dataset in Å.",
)
@click.option(
    "--max-polymer-chains",
    type=int,
    default=300,
    help="Maximum number of polymer chains for included structures.",
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
def main(
    metadata_cache_path: Path,
    preprocessed_dir: Path,
    alignment_representatives_fasta: Path,
    output_path: Path,
    dataset_name: str,
    max_release_date: str,
    max_resolution: float,
    max_polymer_chains: int,
    write_no_alignment_repr_entries: bool,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
) -> None:
    """Create a training dataset cache using PDB-weighted-like filtering procedures."""
    # TODO: Improve docstring

    # Set up logger
    logger = logging.getLogger("openfold3")
    logger.setLevel(getattr(logging, log_level))
    logger.addHandler(logging.StreamHandler())

    create_pdb_training_dataset_cache_af3(
        metadata_cache_path=metadata_cache_path,
        preprocessed_dir=preprocessed_dir,
        alignment_representatives_fasta=alignment_representatives_fasta,
        output_path=output_path,
        dataset_name=dataset_name,
        max_release_date=max_release_date,
        max_resolution=max_resolution,
        max_polymer_chains=max_polymer_chains,
        write_no_alignment_repr_entries=write_no_alignment_repr_entries,
    )


if __name__ == "__main__":
    main()
