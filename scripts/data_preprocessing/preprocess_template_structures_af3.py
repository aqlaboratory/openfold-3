from pathlib import Path

import click

from openfold3.core.data.pipelines.preprocessing.template import (
    preprocess_template_structures,
)


@click.command()
@click.option(
    "--template_structures_directory",
    required=True,
    help="Directory containing cif or pdb files for templates.",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--template_file_format",
    required=True,
    help="File format of the template structures.",
    type=click.Choice(["cif", "pdb"], case_sensitive=True),
)
@click.option(
    "--template_structure_array_directory",
    required=True,
    help=(
        "Output directory to where the pre-parsed, processed per-chain atom arrays "
        "of the template structures are saved."
    ),
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--ccd_file",
    required=True,
    help="Chemical component dictionary file path.",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--num_workers",
    required=True,
    type=int,
    help=(
        "Number of workers to parallelize the template cache computation and filtering"
        " over."
    ),
)
@click.option(
    "--log_level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Set the logging level",
)
@click.option(
    "--log_to_file", default=True, type=bool, help="Enable logging to a file."
)
@click.option(
    "--log_to_console", default=False, type=bool, help="Enable logging to the console."
)
def main(
    template_structures_directory: Path,
    template_file_format: str,
    template_structure_array_directory: Path,
    ccd_file: Path,
    num_workers: int,
    log_level: str,
    log_to_file: bool,
    log_to_console: bool,
) -> None:
    if not template_structure_array_directory.exists():
        template_structure_array_directory.mkdir(parents=True, exist_ok=True)
    try:
        # Preprocess the template structures
        log_dir = (
            template_structure_array_directory.parent
            / "template_structure_preprocessing_logs"
        )
        preprocess_template_structures(
            template_structures_directory=template_structures_directory,
            template_file_format=template_file_format,
            template_structure_array_directory=template_structure_array_directory,
            ccd_file=ccd_file,
            num_workers=num_workers,
            log_level=log_level,
            log_to_file=log_to_file,
            log_to_console=log_to_console,
            log_dir=log_dir,
        )
    except Exception as e:
        raise e
    finally:
        # Collate logs
        log_files = sorted(
            log_dir.glob("process_*.log"), key=lambda p: int(p.stem.split("_")[1])
        )
        combined_log = template_structure_array_directory.parent / "combined.log"
        with combined_log.open("w") as out_file:
            for log_file in log_files:
                out_file.write(log_file.read_text())
                log_file.unlink()
        log_dir.rmdir()


if __name__ == "__main__":
    main()
