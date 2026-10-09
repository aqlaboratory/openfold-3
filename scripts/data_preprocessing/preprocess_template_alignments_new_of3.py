"""
Script to preprocess template alignments separately from model training or inference.

Train mode sets template_ids for each chain of a dataset cache from ColabFold template
alignments (one <rep_id>/colabfold_template.m8 per alignment representative, with label
chain IDs), and writes the template cache and preparsed template structure arrays that
training reads. Predict mode does the same for an inference query set JSON.

Example (train mode):

    python scripts/data_preprocessing/preprocess_template_alignments_new_of3.py \
        --input_set_path dataset_cache.json \
        --output_directory template_data \
        --input_set_type train \
        --runner_yaml template_preprocessing.yml

with template_preprocessing.yml:

    template_preprocessor_settings:
      structure_directory: template_structures
      template_alignment_directory: template_alignments
      alignment_representatives_fasta: alignment_representatives.fasta
      min_release_date_diff: 60
      fetch_missing_structures: false
      preparse_structures: true

Relative paths resolve against the working directory. Everything is written to
--output_directory:

    template_data/
      dataset_cache.json                    the input set, with template information
      template_preprocessor_settings.json   the settings used
      template_cache/                       template cache entries
      template_structure_arrays/            preparsed template structures

output_directory, cache_directory and log_directory in the YAML are ignored.
precache_directory and structure_array_directory are used if given, e.g. to reuse
template structures preparsed once for the whole PDB.
"""

# TODO: rename to preprocess_template_alignments_of3.py
from pathlib import Path
from typing import Literal, cast

import click

from openfold3.core.config import config_utils
from openfold3.entry_points.template_preprocessing import run_template_preprocessing


@click.command()
@click.option(
    "--input_set_path",
    required=True,
    help=(
        "Input dataset cache JSON for training and validation or inference "
        "query set JSON for inference."
    ),
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--output_directory",
    required=True,
    help=(
        "Directory for all outputs: the input set with updated template "
        "information (same file name as the input), the settings used, the "
        "template cache and the preparsed template structures."
    ),
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
@click.option(
    "--input_set_type",
    required=True,
    help=("Mode of template preprocessing. One of 'train' or 'predict'."),
    type=click.Choice(
        ["train", "predict"],
        case_sensitive=False,
    ),
)
@click.option(
    "--runner_yaml",
    required=True,
    help=(
        "Runner.yml file to be parsed into settings for the template preprocessor "
        "pipeline."
    ),
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
def main(
    input_set_path: Path,
    output_directory: Path,
    input_set_type: str,
    runner_yaml: Path,
):
    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()
    output_set_path = run_template_preprocessing(
        input_set_path=input_set_path,
        input_set_type=cast(Literal["train", "predict"], input_set_type),
        output_directory=output_directory,
        settings_kwargs=runner_args.get("template_preprocessor_settings", {}),
    )
    print(f"Wrote {output_set_path} and template data to {output_directory}.")


if __name__ == "__main__":
    main()
