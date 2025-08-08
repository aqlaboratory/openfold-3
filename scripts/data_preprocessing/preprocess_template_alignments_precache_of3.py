"""
Script to preprocess template alignments separately from model training or inference.
"""

from pathlib import Path

import click

from openfold3.core.config import config_utils
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePrecachePreprocessor,
    TemplatePreprocessorSettings,
)


@click.command()
@click.option(
    "--runner_yaml",
    required=True,
    help=(
        "Runner.yml file to be parsed into settings for the template precache "
        "preprocessor pipeline."
    ),
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        path_type=Path,
    ),
)
def main(runner_yaml: Path):
    # Load runner YAML and extract template_preprocessor_settings if present
    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()

    # Extract template_preprocessor_settings from runner YAML or use default
    template_preprocessor_kwargs = runner_args.get("template_preprocessor_settings", {})

    # Create template preprocessor settings with defaults, overriding with YAML values
    template_preprocessor_settings = TemplatePreprocessorSettings(
        **template_preprocessor_kwargs
    )

    template_precache_preprocessor = TemplatePrecachePreprocessor(
        config=template_preprocessor_settings
    )
    template_precache_preprocessor()


if __name__ == "__main__":
    main()
