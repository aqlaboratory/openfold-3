import logging
from pathlib import Path
from typing import Literal

import click

from openfold3.core.data.pipelines.preprocessing.structure import preprocess_cif_dir_af3


@click.command()
@click.option(
    "--cif-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to directory containing input mmCIF files.",
)
@click.option(
    "--ccd-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a Chemical Component Dictionary mmCIF file.",
)
@click.option(
    "--out-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to top-level directory that output files should be written to.",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help=(
        "Number of workers to use for parallel processing. Use None for all available "
        "CPUs, and 0 for single-threaded processing."
    ),
)
@click.option(
    "--chunksize",
    type=int,
    default=50,
    help="Number of CIF files to process in each worker task.",
)
@click.option(
    "--write-additional-cifs",
    is_flag=False,
    help=(
        "Write not only binary .cif files but also standard .cif files (useful for "
        "inspecting results)."
    ),
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="WARNING",
    help="Set the logging level.",
)
def main(
    cif_dir: Path,
    ccd_path: Path,
    out_dir: Path,
    num_workers: int | None = None,
    chunksize: int = 50,
    write_additional_cifs: bool = False,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
) -> None:
    logger = logging.getLogger("openfold3")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(logging.StreamHandler())

    preprocess_cif_dir_af3(
        cif_dir, ccd_path, out_dir, num_workers, chunksize, write_additional_cifs
    )


if __name__ == "__main__":
    main()
