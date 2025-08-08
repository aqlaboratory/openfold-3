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
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
def main(input_set_path: Path, input_set_type: str, runner_yaml: Path):
    # Load input set
    if input_set_type == "train":
        # load into dataset cache
        input_set = None
        raise NotImplementedError(
            "Offline template preprocessing with the new template"
            " pipeline is not yet implemented."
        )
    elif input_set_type == "inference":
        # load into InferenceQuerySet
        input_set = InferenceQuerySet.from_json(input_set_path)

    # Load runner YAML and extract template_preprocessor_settings if present
    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()

    # Extract template_preprocessor_settings from runner YAML or use default
    template_preprocessor_kwargs = runner_args.get("template_preprocessor_settings", {})

    # Create template preprocessor settings with defaults, overriding with YAML values
    template_preprocessor_settings = TemplatePreprocessorSettings(
        mode=input_set_type, **template_preprocessor_kwargs
    )

    template_preprocessor = TemplatePreprocessor(
        input_set=input_set, config=template_preprocessor_settings
    )
    template_preprocessor()


if __name__ == "__main__":
    main()
