import logging
import multiprocessing
from pathlib import Path
from typing import Literal

import click

from openfold3.core.data.pipelines.preprocessing.structure import preprocess_cif_dir_of3
from openfold3.core.utils.logging_utils import ContextInjectingFilter


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
    help=(
        "Path to a Chemical Component Dictionary mmCIF file. This sets the CCD that is "
        "used by OF3 processing functions."
    ),
)
@click.option(
    "--biotite-ccd-path",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    help=(
        "Path to a .bcif CCD that has been preprocessed with biotite's setup_ccd.py "
        "script, for usage with biotite's set_ccd_path. This will be used to set the "
        "internal CCD of Biotite itself for functions where it is not exposed, such as"
        " `connect_via_residue_names`. If not specified, Biotite's default CCD will be"
        " used which might have slight mismatches with older PDB snapshots."
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
    "--precrop-n-chains",
    type=int,
    default=20,
    help=(
        "The number of chains to keep in the precropping step. If the structure has "
        "less than N chains, all of them are kept."
    ),
)
@click.option(
    "--precrop-ignore-ligands-below",
    type=float,
    default=6,
    help=(
        "Ligand chains with fewer atoms than this value will be ignored in the N-chain "
        "counter for precropping, and included based on proximity to the selected "
        "chains. Set this to inf to ignore all ligands from the chain budget."
    ),
)
@click.option(
    "--precrop-disable-rna",
    is_flag=True,
    help=(
        "Whether to disable the N-chain precropping for structures that contain RNA."
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
    biotite_ccd_path: Path | None,
    out_dir: Path,
    output_format: list[Literal["npz", "cif", "bcif", "pkl"]],
    max_polymer_chains: int = 1000,
    num_workers: int | None = None,
    chunksize: int = 50,
    precrop_n_chains: int = 20,
    precrop_disable_rna: bool = False,
    precrop_ignore_ligands_below: int = 6,
    random_seed: int | None = None,
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING",
    early_stop: int | None = None,
) -> None:
    """Preprocesses a directory of mmCIF files for use in AlphaFold3.

    Cleans up mmCIF files following the AlphaFold3 filtering procedure and writes out a
    metadata JSON and individual FASTA files for all structures.
    """
    # TODO: Add better docstring

    # Casting this to int if it is one is cleaner
    if precrop_ignore_ligands_below.is_integer():
        precrop_ignore_ligands_below = int(precrop_ignore_ligands_below)

    log_level = getattr(logging, log_level.upper())
    log_file = out_dir / "preprocess_pdb_af3.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure console handlers and formatters (only for main process)
    stream_formatter = logging.Formatter("%(message)s")  # minimalist format for console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(log_level)

    # Configure file handlers and formatters (all processes)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_formatter = logging.Formatter(
        "[%(levelname)s - %(module)s:%(funcName)s - pid:%(process)d - PDB:%(pdb_id)s]:"
        " %(message)s"
    )  # detailed format for log file
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    # This sets a default PDB-ID value "-" if it's not yet provided in the context
    file_handler.addFilter(
        ContextInjectingFilter("openfold3", expected_keys=["pdb_id"])
    )

    # Multiprocessing queue for logs
    log_queue = multiprocessing.Queue(-1)

    # Set up Queue listener for multi-worker logging
    listener = logging.handlers.QueueListener(
        log_queue,
        file_handler,
    )
    listener.start()

    # Set up main logger
    main_logger = logging.getLogger("openfold3")
    main_logger.addHandler(stream_handler)
    main_logger.addHandler(file_handler)

    try:
        main_logger.info("Starting preprocessing...")
        main_logger.info(f"Writing logs to {log_file}")
        preprocess_cif_dir_of3(
            cif_dir=cif_dir,
            ccd_path=ccd_path,
            biotite_ccd_path=biotite_ccd_path,
            out_dir=out_dir,
            max_polymer_chains=max_polymer_chains,
            num_workers=num_workers,
            chunksize=chunksize,
            output_formats=output_format,
            precrop_n_chains=precrop_n_chains,
            precrop_disable_rna=precrop_disable_rna,
            precrop_ignore_ligands_below=precrop_ignore_ligands_below,
            random_seed=random_seed,
            log_queue=log_queue,
            log_level=log_level,
            early_stop=early_stop,
        )
    finally:
        # Stop the listener
        listener.stop()
        main_logger.info("Preprocessing completed.")


if __name__ == "__main__":
    main()
