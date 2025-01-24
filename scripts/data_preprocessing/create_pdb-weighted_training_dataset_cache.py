import logging
from datetime import datetime
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
    default="2021-09-30",
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
    "--allow-missing-alignment",
    is_flag=True,
    help=(
        "If this flag is set, allow entries where not every RNA and protein sequence "
        "matches to an alignment representative in the alignment_representatives_fasta."
        " Otherwise skip these entries."
    ),
)
@click.option(
    "--missing_alignment_log",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "If this is specified, writes all entries without an alignment representative "
        "to the specified log file."
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
) # TODO: Add docstring
def main(
    metadata_cache_path: Path,
    preprocessed_dir: Path,
    alignment_representatives_fasta: Path,
    output_path: Path,
    dataset_name: str,
    max_release_date: str = "2021-09-30",
    max_resolution: float = 9.0,
    max_polymer_chains: int = 300,
    allow_missing_alignment: bool = False,
    missing_alignment_log: Path | None = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
    log_file: Path | None = None,
) -> None:
    """Create a training dataset cache using PDB-weighted filtering procedures.

    This applies basic filtering procedures to create a training dataset cache that can
    be used by the DataLoader from the more general metadata cache created in
    preprocessing. Following AF3, the filters applied are:
        - release date can be no later than max_release_date
        - resolution can be no higher than max_resolution
        - number of polymer chains can be no higher than max_polymer_chains

    This also adds the following additional information:
        Name of the dataset (for use with the DataSet registry)

        Structure data:
            - alignment_representative_id:
                The ID of the alignment of this chain
            - cluster_id:
                The ID of the cluster this chain/interface belongs to
            - cluster_size:
                The size of the cluster this chain/interface belongs to
        Reference molecule data:
            - set_fallback_to_nan:
                Whether to set the fallback conformer of this molecule to NaN. This
                applies to the very special case where the fallback conformer was
                derived from CCD model coordinates coming from a PDB-ID that was
                released outside of the time cutoff (see AF3 SI 2.8)
    """
    max_release_date = datetime.strptime(max_release_date, "%Y-%m-%d").date()

    # Set up logger
    logger = logging.getLogger("openfold3")
    logger.setLevel(getattr(logging, log_level))
    logger.addHandler(logging.StreamHandler())

    # Add file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        logger.addHandler(file_handler)

    create_pdb_training_dataset_cache_af3(
        metadata_cache_path=metadata_cache_path,
        preprocessed_dir=preprocessed_dir,
        alignment_representatives_fasta=alignment_representatives_fasta,
        output_path=output_path,
        dataset_name=dataset_name,
        max_release_date=max_release_date,
        max_resolution=max_resolution,
        max_polymer_chains=max_polymer_chains,
        filter_missing_alignment=not allow_missing_alignment,
        missing_alignment_log=missing_alignment_log,
    )


if __name__ == "__main__":
    main()
