"""Template preprocessing script for AF3 dataset."""

import logging
from pathlib import Path

import click

from openfold3.core.data.io.s3 import parse_s3_config
from openfold3.core.data.pipelines.preprocessing.template import (
    create_template_cache_af3,
    create_template_seq_cache_af3,
    filter_template_cache_af3,
)


@click.command()
@click.option(
    "--template_alignment_directory",
    required=True,
    help=(
        "Directory containing per-chain folders with template alignments. If the "
        "directory lives in an S3 bucket, the path should be 's3:/<bucket>/<prefix>'."
    ),
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--template_alignment_filename",
    required=True,
    help="Filename for the template alignments.",
    type=str,
)
@click.option(
    "--template_structures_directory",
    required=True,
    help="Directory containing the template structures.",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--template_cache_directory",
    required=True,
    help="Filepath to where the template cache should be saved.",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--query_structures_directory",
    required=True,
    help="Directory containing query structures used for training or inference.",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--query_structures_filename",
    required=False,
    default="None",
    help=(
        "Filename for the query structures. If 'None', uses the per-entry dir names "
        "as filenames."
    ),
    type=str,
)
@click.option(
    "--query_file_format",
    required=True,
    help="File format for the query structures.",
    type=str,
)
@click.option(
    "--template_file_format",
    required=True,
    help="File format for the template structures.",
    type=str,
)
@click.option(
    "--query_seq_load_logic",
    required=True,
    help=(
        "Whether to load the query sequences associated with structures from fasta "
        "or structure files."
    ),
    type=click.Choice(["fasta", "structure"], case_sensitive=True),
)
@click.option(
    "--single_moltype",
    required=False,
    default=None,
    help=(
        "Constant molecule type to use for datasets that have one molecule type "
        "across all entries and whose dataset cache is missing the per-chain "
        "molecule type field."
    ),
    type=click.Choice(["PROTEIN", "RNA", "DNA", "LIGAND"], case_sensitive=True),
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
    "--dataset_cache_file",
    required=True,
    help="Filepath to the dataset cache.",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--updated_dataset_cache_file",
    required=True,
    help="Filepath to where the updated dataset cache should be saved.",
    type=click.Path(
        exists=False,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--max_templates_construct",
    required=True,
    type=int,
    help="Maximum number of templates to keep per query chain.",
)
@click.option(
    "--max_templates_filter",
    required=True,
    type=int,
    help="Maximum number of templates to keep per query chain.",
)
@click.option(
    "--is_core_train",
    type=bool,
    help=(
        "Flag to specify the dataset cache is for the core training set. False for"
        " distillation and inference sets."
    ),
)
@click.option(
    "--max_release_date",
    required=False,
    help=(
        "Maximum release date for templates in format YYYY-MM-DD. Used for"
        " distillation and inference sets"
    ),
    type=str,
    default=None,
)
@click.option(
    "--min_release_date_diff",
    required=False,
    help=(
        "Minimum number of days required for the template to be released before a"
        " query structure. Used for core training sets."
    ),
    type=int,
    default=None,
)
@click.option(
    "--log_level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=True),
    help="Set the logging level",
)
@click.option(
    "--filter_only",
    default=False,
    type=bool,
    help=(
        "Whether to filter an existing template cache. True if the full template "
        "cache has already been created."
    ),
)
@click.option(
    "--log_to_file", default=True, type=bool, help="Enable logging to a file."
)
@click.option(
    "--log_to_console", default=False, type=bool, help="Enable logging to the console."
)
@click.option(
    "--s3_client_config",
    required=False,
    default=None,
    type=str,
    help="The argument s3_client_config "
    "input as a JSON string with keys 'profile' and 'max_keys'.",
)
def main(
    template_alignment_directory: Path,
    template_alignment_filename: str,
    template_structures_directory: Path,
    template_cache_directory: Path,
    query_structures_directory: Path,
    query_structures_filename: str,
    query_file_format: str,
    template_file_format: str,
    query_seq_load_logic: str,
    single_moltype: str | None,
    num_workers: int,
    dataset_cache_file: Path,
    updated_dataset_cache_file: Path,
    max_templates_construct: int,
    max_templates_filter: int,
    is_core_train: bool,
    max_release_date: str,
    min_release_date_diff: int,
    log_level: str,
    filter_only: bool,
    log_to_file: bool,
    log_to_console: bool,
    s3_client_config: str | None,
) -> None:
    """Preprocesses templates for AF3 datasets.

    Args:
        template_alignment_directory (Path):
            Directory containing per-chain folders with template alignments.
        template_alignment_filename (str):
            Filename for the template alignments. Typically hmm_output.sto.
        template_structures_directory (Path):
            Directory containing the template structures.
        template_cache_directory (Path):
            Filepath to where the template cache should be saved.
        query_structures_directory (Path):
            Directory containing the sanitized query structures used for training or
            inference.
        query_structures_filename (str):
            Name of the query file.
        query_file_format (str):
            File format for the query structures.
        template_file_format (str):
            File format for the template structures.
        query_seq_load_logic (str):
            Whether to load the query sequences associated with structures from fasta or
            structure files.
        single_moltype (str | None):
            Constant molecule type to use if all query structures contain the same
            molecule type. Needed if the input dataset cache is missing the per-chain
            molecule type field.
        num_workers (int):
            Number of workers to parallelize the template cache computation and
            filtering over.
        dataset_cache_file (Path):
            Filepath to the dataset cache.
        updated_dataset_cache_file (Path):
            Filepath to where the updated dataset cache should be saved.
        max_templates_construct (int):
            Max number of template to keep per query chain during template cache
            construction. This includes all valid templates not filtered for release
            dates.
        max_templates_filter (int):
            Maximum number of templates to keep per query chain after filtering.
        is_core_train (bool):
            Flag to specify the dataset cache is for the core training set. False for
            distillation and inference sets.
        save_frequency (int):
            Number of query chains after which to save the dataset cache update with
            valid templates.
        max_release_date (str):
            Maximum release date for templates in format YYYY-MM-DD. Used for
            distillation and inference sets.
        min_release_date_diff (int):
            Minimum number of days required for the template to be released before a
            query structure. Used for core training sets.
        log_level (str):
            Logging level.
        filter_only (bool):
            Whether to only filter the template cache. True if the full template cache
            has already been created.
        log_to_file (bool):
            Enable logging to a file.
        log_to_console (bool):
            Enable logging to the console.
        s3_client_config (str | None):
            The argument s3_client_config input as a JSON string with keys 'profile' and
            'max_keys'.

    Raises:
        ValueError:
            If is_core_train is True and min_release_date_diff is None.
        ValueError:
            If is_core_train is False and max_release_date is None.
    """
    # Parse S3 config
    s3_client_config = parse_s3_config(s3_client_config)

    # Error handling
    if is_core_train & (min_release_date_diff is None):
        raise ValueError(
            "Minimum release date difference for core training must be specified."
        )

    if (not is_core_train) & (max_release_date is None):
        raise ValueError(
            "Max release date difference for distillation and inference sets must be"
            " specified."
        )

    # Run
    if not filter_only:
        logging.info("1/3: Creating the template sequence cache.")
        create_template_seq_cache_af3(
            template_structures_directory=template_structures_directory,
            template_cache_directory=template_cache_directory,
            template_file_format=template_file_format,
            num_workers=num_workers,
            log_level=log_level,
            log_to_file=log_to_file,
            log_to_console=log_to_console,
            log_dir=template_cache_directory.parent / Path("template_seq_logs"),
        )

        logging.info("2/3: Creating the template cache.")
        create_template_cache_af3(
            dataset_cache_file=dataset_cache_file,
            template_alignment_directory=template_alignment_directory,
            template_alignment_filename=template_alignment_filename,
            template_structures_directory=template_structures_directory,
            template_cache_directory=template_cache_directory,
            query_structures_directory=query_structures_directory,
            max_templates_construct=max_templates_construct,
            query_structures_filename=query_structures_filename,
            query_file_format=query_file_format,
            query_seq_load_logic=query_seq_load_logic,
            single_moltype=single_moltype,
            num_workers=num_workers,
            log_level=log_level,
            log_to_file=log_to_file,
            log_to_console=log_to_console,
            log_dir=template_cache_directory.parent / "template_construct_logs",
            s3_client_config=s3_client_config,
        )
    else:
        logging.info(
            "Skipping template cache generation. Using existing cache "
            f"from {template_cache_directory}."
        )

    logging.info("3/3: Filtering the template cache.")
    filter_template_cache_af3(
        dataset_cache_file=dataset_cache_file,
        updated_dataset_cache_file=updated_dataset_cache_file,
        template_cache_directory=template_cache_directory,
        max_templates_filter=max_templates_filter,
        single_moltype=single_moltype,
        is_core_train=is_core_train,
        num_workers=num_workers,
        log_level=log_level,
        log_to_file=log_to_file,
        log_to_console=log_to_console,
        log_dir=template_cache_directory.parent / "template_filter_logs",
        max_release_date=max_release_date,
        min_release_date_diff=min_release_date_diff,
    )


if __name__ == "__main__":
    main()
