import logging
from pathlib import Path

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
    "--write-additional-cifs",
    is_flag=False,
    help=(
        "Write not only binary .cif files but also standard .cif files (useful for "
        "inspecting results)."
    ),
)
def main(
    cif_dir: Path,
    ccd_path: Path,
    out_dir: Path,
    num_workers: int | None = None,
    write_additional_cifs: bool = False,
) -> None:
    logger = logging.getLogger("openfold3")
    logger.setLevel(logging.WARNING)
    logger.addHandler(logging.StreamHandler())

    preprocess_cif_dir_af3(
        cif_dir, ccd_path, out_dir, num_workers, write_additional_cifs
    )


if __name__ == "__main__":
    main()
