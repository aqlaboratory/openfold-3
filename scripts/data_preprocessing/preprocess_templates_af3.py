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
    "--metadata-cache-path",
    required=True,
    help="Filepath to the metadata cache.",
    exists=True,
    file_okay=True,
    dir_okay=False,
    path_type=Path,
)
@click.option(
    "--template-alignment-base-path",
    required=True,
    help="Directory containing per-chain folders with template alignments.",
    exists=True,
    file_okay=False,
    dir_okay=True,
    path_type=Path,
)
@click.option(
    "--template-alignment-filename",
    required=True,
    help="Filename for the template alignments.",
    type=str,
)
@click.option(
    "--template-structures-path",
    required=True,
    help="Directory containing the template structures.",
    exists=True,
    file_okay=False,
    dir_okay=True,
    path_type=Path,
)
@click.option(
    "--template-cache-path",
    required=True,
    multiple=True,
    help="Filepath to where the template cache should be saved.",
    exists=False,
    file_okay=False,
    dir_okay=True,
    path_type=Path,
)
@click.option(
    "--query-structures-path",
    required=True,
    help="Directory containing query structures used for training or inference.",
    exists=True,
    file_okay=False,
    dir_okay=True,
    path_type=Path,
)
@click.option(
    "--num-workers",
    required=True,
    type=int,
    help=(
        "Number of workers to parallelize the template cache computation and filtering"
        " over."
    ),
)
@click.option(
    "--dataset-cache-path",
    required=True,
    help="Filepath to the dataset cache.",
    exists=True,
    file_okay=True,
    dir_okay=False,
    path_type=Path,
)
@click.option(
    "--updated-dataset-cache-path",
    required=True,
    help="Filepath to where the updated dataset cache should be saved.",
    exists=False,
    file_okay=True,
    dir_okay=False,
    path_type=Path,
)
@click.option(
    "--max-templates",
    required=True,
    type=int,
    help="Maximum number of templates to keep per query chain.",
)
@click.option(
    "--is-core-train",
    type=bool,
    help=(
        "Flag to specify the dataset cache is for the core training set. False for"
        " distillation and inference sets."
    ),
)
@click.option(
    "--save-frequency",
    required=True,
    type=int,
    help=(
        "Number of query chains after which to save the dataset cache update with"
        " valid templates."
    ),
    default=1000,
)
@click.option(
    "--max-release-date",
    required=False,
    help=(
        "Maximum release date for templates in format YYYY-MM-DD. Used for"
        " distillation and inference sets"
    ),
    type=str,
    default=None,
)
@click.option(
    "--min-release-date-diff-core-train",
    required=False,
    help=(
        "Minimum number of days required for the template to be released before a"
        " query structure. Used for core trianing sets."
    ),
    type=int,
    default=None,
)
def main(
    metadata_cache_path,
    template_alignment_base_path,
    template_alignment_filename,
    template_structures_path,
    template_cache_path,
    query_structures_path,
    num_workers,
    dataset_cache_path,
    updated_dataset_cache_path,
    max_templates,
    is_core_train,
    save_frequency,
    max_release_date,
    min_release_date_diff_core_train,
) -> None:
    logger = logging.getLogger("openfold3")
    logger.setLevel(logging.WARNING)
    logger.addHandler(logging.StreamHandler())

    if is_core_train & (min_release_date_diff_core_train is None):
        raise ValueError(
            "Minimum release date difference for core training must be specified."
        )

    if not is_core_train & (max_release_date is None):
        raise ValueError(
            "Max release date difference for distillation and inference sets must be"
            " specified."
        )

    create_template_cache_af3(
        metadata_cache_path,
        template_alignment_base_path,
        template_alignment_filename,
        template_structures_path,
        template_cache_path,
        query_structures_path,
        num_workers,
    )

    filter_template_cache_af3(
        dataset_cache_path,
        updated_dataset_cache_path,
        template_cache_path,
        max_templates,
        is_core_train,
        num_workers,
        save_frequency,
        max_release_date,
        min_release_date_diff_core_train,
    )


if __name__ == "__main__":
    main()
