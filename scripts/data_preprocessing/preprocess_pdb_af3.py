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
    "--max-polymer-chains",
    type=int,
    default=None,
    help=(
        "The maximum number of polymer chains in the first bioassembly after which a "
        "structure is skipped by the parser."
    ),
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
    "--output-format",
    type=click.Choice(["cif", "bcif", "pkl"]),
    multiple=True,
    required=True,
    help=(
        "What output formats to write the structures to. "
        "Can be 'cif', 'bcif', and 'pkl'."
    ),
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="WARNING",
    help="Set the logging level.",
)
@click.option(
    "--early-stop",
    type=int,
    default=None,
    help="Stop after processing this many CIFs. Only used for debugging.",
)
def main(
    cif_dir: Path,
    ccd_path: Path,
    out_dir: Path,
    output_format: list[Literal["cif", "bcif", "pkl"]],
    max_polymer_chains: int = 300,
    num_workers: int | None = None,
    chunksize: int = 50,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
    early_stop: int | None = None,
) -> None:
    """Preprocesses a directory of mmCIF files for use in AlphaFold3.

    Cleans up mmCIF files following the AlphaFold3 filtering procedure and writes out a
    metadata JSON and individual FASTA files for all structures.
    """
    # TODO: Add better docstring
    logger = logging.getLogger("openfold3")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(logging.StreamHandler())

    preprocess_cif_dir_af3(
        cif_dir=cif_dir,
        ccd_path=ccd_path,
        out_dir=out_dir,
        max_polymer_chains=max_polymer_chains,
        num_workers=num_workers,
        chunksize=chunksize,
        output_formats=output_format,
        early_stop=early_stop,
    )


if __name__ == "__main__":
    main()
