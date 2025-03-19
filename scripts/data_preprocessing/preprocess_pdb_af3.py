import logging
from pathlib import Path
from typing import Literal

import click

from openfold3.core.data.pipelines.preprocessing.structure import preprocess_cif_dir_af3


# TODO: rename to make it more clear this script is for metadata cache creation
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
    "--preprocessed-ccd-path",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    help=(
        "Path to a .bcif CCD that has been preprocessed with biotite's setup_ccd.py "
        "script, for usage with biotite's set_ccd_path. This can be used to make sure "
        "that the CCD that is used in preprocessing perfectly matches a particular CCD "
        "version, for example to match the version that the PDB was downloaded with."
    ),
    default=None,
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
    type=click.Choice(["npz", "cif", "bcif", "pkl"]),
    multiple=True,
    required=True,
    help=(
        "What output formats to write the structures to. "
        "Can be 'npz', 'cif', 'bcif', and 'pkl'."
    ),
)
@click.option(
    "--n-chains-precropping",
    type=int,
    default=20,
    help=(
        "The number of chains to keep in the precropping step. If the structure has "
        "less than N chains, all of them are kept."
    ),
)
@click.option(
    "--disable-rna-precropping",
    is_flag=True,
    help=(
        "Whether to disable the N-chain precropping for structures that contain RNA."
    ),
)
@click.option(
    "--permissive-small-ligand-precropping",
    is_flag=True,
    help=(
        "If this is set to True, small ligands will be ignored in the N-chain counter "
        "for precropping, and included based on proximity to the selected chains."
    ),
)
@click.option(
    "--random-seed",
    type=int,
    default=None,
    help="Seed for reproducibility in large-assembly subsetting.",
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
    preprocessed_ccd_path: Path | None,
    out_dir: Path,
    output_format: list[Literal["npz", "cif", "bcif", "pkl"]],
    max_polymer_chains: int = 300,
    num_workers: int | None = None,
    chunksize: int = 50,
    n_chains_precropping: int = 20,
    disable_rna_precropping: bool = False,
    permissive_small_ligand_precropping: bool = False,
    random_seed: int | None = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
    early_stop: int | None = None,
) -> None:
    """Preprocesses a directory of mmCIF files for use in AlphaFold3.

    Cleans up mmCIF files following the AlphaFold3 filtering procedure and writes out a
    metadata JSON and individual FASTA files for all structures.
    """
    # TODO: Add better docstring
    logger = logging.getLogger("openfold3")
    logger.setLevel(getattr(logging, log_level))
    logger.addHandler(logging.StreamHandler())

    preprocess_cif_dir_af3(
        cif_dir=cif_dir,
        ccd_path=ccd_path,
        preprocessed_ccd_path=preprocessed_ccd_path,
        out_dir=out_dir,
        max_polymer_chains=max_polymer_chains,
        num_workers=num_workers,
        chunksize=chunksize,
        output_formats=output_format,
        n_chains_precropping=n_chains_precropping,
        disable_rna_precropping=disable_rna_precropping,
        permissive_small_ligand_precropping=permissive_small_ligand_precropping,
        random_seed=random_seed,
        early_stop=early_stop,
    )


if __name__ == "__main__":
    main()
