"""
Script to preprocess template alignments separately from model training or inference.
"""

# TODO: rename to preprocess_template_alignments_of3.py
from pathlib import Path

import click

from openfold3.core.config import config_utils
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessor,
    TemplatePreprocessorSettings,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)


@click.command()
@click.option(
    "--input_set_path",
    required=True,
    help=(
        "Input dataset cache JSON for training and validation or inference"
        "query set JSON for inference."
    ),
    type=click.Path(
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
@click.option(
    "--output_set_path",
    required=True,
    help=(
        "Output path to the JSON cache file with updated template information."
        " Dataset cache for training and validation, inference"
        "query set JSON for inference."
    ),
    type=click.Path(
        file_okay=True,
        dir_okay=False,
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
    input_set_path: Path, output_set_path: Path, input_set_type: str, runner_yaml: Path
):
    # Load input set
    if input_set_type == "train":
        # load into dataset cache
        input_set = None
        raise NotImplementedError(
            "Offline template preprocessing with the new template"
            " pipeline is not yet implemented."
        )
    elif input_set_type == "predict":
        # load into InferenceQuerySet
        input_set = InferenceQuerySet.from_json(input_set_path)

    # Load runner YAML and template_preprocessor_settings
    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()
    template_preprocessor_kwargs = runner_args.get("template_preprocessor_settings", {})

    # Override defaults
    template_preprocessor_settings = TemplatePreprocessorSettings(
        mode=input_set_type, **template_preprocessor_kwargs
    )

    # Run template preprocessing
    template_preprocessor = TemplatePreprocessor(
        input_set=input_set, config=template_preprocessor_settings
    )
    template_preprocessor()

    # Save updated set
    if input_set_type == "train":
        # Save dataset cache with template information
        input_set = None
        raise NotImplementedError(
            "Offline template preprocessing with the new template"
            " pipeline is not yet implemented."
        )
    elif input_set_type == "predict":
        # Save InferenceQuerySet with template information
        with open(output_set_path, "w") as f:
            f.write(input_set.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
