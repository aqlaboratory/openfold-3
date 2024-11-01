"""Template preprocessing script for AF3 dataset."""

import logging
from pathlib import Path

import click

from openfold3.core.data.pipelines.preprocessing.template import (
    create_template_cache_af3,
    filter_template_cache_af3,
)


@click.command()
@click.option(
    "--template_alignment_directory",
    required=True,
    help="Directory containing per-chain folders with template alignments.",
    type=click.Path(
        exists=True,
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
        exists=True,
        file_okay=False,
        dir_okay=True,
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
    "--max_templates",
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
    "--save_frequency",
    required=True,
    type=int,
    help=(
        "Number of query chains after which to save the dataset cache update with"
        " valid templates."
    ),
    default=1000,
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
        " query structure. Used for core trianing sets."
    ),
    type=int,
    default=None,
)
@click.option(
    "--log_level",
    default="WARNING",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
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
def main(
    template_alignment_directory: Path,
    template_alignment_filename: str,
    template_structures_directory: Path,
    template_cache_directory: Path,
    query_structures_directory: Path,
    num_workers: int,
    dataset_cache_file: Path,
    updated_dataset_cache_file: Path,
    max_templates: int,
    is_core_train: bool,
    save_frequency: int,
    max_release_date: str,
    min_release_date_diff: int,
    log_level: str,
    filter_only: bool,
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
        num_workers (int):
            Number of workers to parallelize the template cache computation and
            filtering over.
        dataset_cache_file (Path):
            Filepath to the dataset cache.
        updated_dataset_cache_file (Path):
            Filepath to where the updated dataset cache should be saved.
        max_templates (int):
            Maximum number of templates to keep per query chain.
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

    Raises:
        ValueError:
            If is_core_train is True and min_release_date_diff is None.
        ValueError:
            If is_core_train is False and max_release_date is None.
    """

    # Configure the logger
    logger = logging.getLogger("openfold3")
    numeric_level = getattr(logging, log_level.upper())
    logger.setLevel(numeric_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    file_handler = logging.FileHandler(
        updated_dataset_cache_file.parent / Path("preprocess_templates_af3.log")
    )
    file_handler.setLevel(numeric_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    # Error handling
    if is_core_train & (min_release_date_diff is None):
        raise ValueError(
            "Minimum release date difference for core training must be specified."
        )

    if not is_core_train & (max_release_date is None):
        raise ValueError(
            "Max release date difference for distillation and inference sets must be"
            " specified."
        )

    # Run
    if not filter_only:
        logging.info("Creating the template cache.")
        create_template_cache_af3(
            dataset_cache_file=dataset_cache_file,
            template_alignment_directory=template_alignment_directory,
            template_alignment_filename=template_alignment_filename,
            template_structures_directory=template_structures_directory,
            template_cache_directory=template_cache_directory,
            query_structures_directory=query_structures_directory,
            num_workers=num_workers,
        )
    else:
        logging.info(
            "Skipping template cache generation. Using existing cache "
            f"from {template_cache_directory}."
        )

    logging.info("Filtering the template cache.")
    filter_template_cache_af3(
        dataset_cache_file=dataset_cache_file,
        updated_dataset_cache_file=updated_dataset_cache_file,
        template_cache_directory=template_cache_directory,
        max_templates=max_templates,
        is_core_train=is_core_train,
        num_workers=num_workers,
        save_frequency=save_frequency,
        max_release_date=max_release_date,
        min_release_date_diff=min_release_date_diff,
    )


if __name__ == "__main__":
    main()
