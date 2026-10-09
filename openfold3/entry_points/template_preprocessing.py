# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Entry point for running template preprocessing on its own.

Wraps TemplatePreprocessor for
scripts/data_preprocessing/preprocess_template_alignments_new_of3.py: reads a dataset
cache or inference query set JSON, resolves the settings so that every output goes to
one output directory, runs preprocessing, and writes the updated set and the settings
used next to the template data.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from openfold3.core.data.io.dataset_cache import read_datacache, write_datacache_to_json
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessor,
    TemplatePreprocessorSettings,
)
from openfold3.core.data.primitives.caches.format import DatasetCache
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)

logger = logging.getLogger(__name__)

TEMPLATE_PREPROCESSOR_SETTINGS_FILENAME = "template_preprocessor_settings.json"

# Settings for where template preprocessing writes its own outputs. The precache and
# structure array directories are not among them: they hold parsed template structures
# that runs reuse, so they are taken from the settings when given.
_OUTPUT_PATH_SETTINGS = ("output_directory", "cache_directory", "log_directory")


def resolve_template_preprocessor_settings(
    settings_kwargs: dict[str, Any],
    input_set_type: Literal["train", "predict"],
    output_directory: Path,
) -> TemplatePreprocessorSettings:
    """Builds the settings for a run whose outputs all go to output_directory.

    output_directory, cache_directory and log_directory in settings_kwargs are
    ignored, with a warning; the template cache and logs go inside output_directory.
    The precache and structure array directories are kept when given, and otherwise
    also go inside output_directory.

    Args:
        settings_kwargs (dict[str, Any]):
            TemplatePreprocessorSettings fields from the runner YAML, other than mode.
        input_set_type (Literal["train", "predict"]):
            The template preprocessing mode.
        output_directory (Path):
            Directory for all outputs.

    Returns:
        TemplatePreprocessorSettings: The settings to run with.
    """
    if "mode" in settings_kwargs:
        raise ValueError(
            "Do not set 'mode' in the template preprocessor settings; it follows "
            "the input set type."
        )
    output_directory = Path(output_directory)
    ignored = [key for key in _OUTPUT_PATH_SETTINGS if settings_kwargs.get(key)]
    if ignored:
        logger.warning(
            f"Ignoring template_preprocessor_settings {ignored}: all outputs go to "
            f"{output_directory}."
        )
    return TemplatePreprocessorSettings(
        **{
            key: value
            for key, value in settings_kwargs.items()
            if key not in _OUTPUT_PATH_SETTINGS
        },
        mode=input_set_type,
        output_directory=output_directory,
    )


def run_template_preprocessing(
    input_set_path: Path,
    input_set_type: Literal["train", "predict"],
    output_directory: Path,
    settings_kwargs: dict[str, Any],
) -> Path:
    """Runs template preprocessing on a dataset cache or inference query set JSON.

    Everything is written to output_directory: the updated set, under the input's
    file name; the settings used, as template_preprocessor_settings.json; and the
    template data (see resolve_template_preprocessor_settings).

    Args:
        input_set_path (Path):
            Dataset cache JSON (train) or inference query set JSON (predict).
        input_set_type (Literal["train", "predict"]):
            The template preprocessing mode.
        output_directory (Path):
            Directory for all outputs.
        settings_kwargs (dict[str, Any]):
            TemplatePreprocessorSettings fields from the runner YAML, other than mode.

    Returns:
        Path: The updated dataset cache or inference query set JSON.
    """
    output_directory = Path(output_directory)
    output_set_path = output_directory / input_set_path.name
    if output_set_path.resolve() == input_set_path.resolve():
        raise ValueError(
            f"Writing to {output_directory} would overwrite the input set "
            f"{input_set_path}."
        )
    settings = resolve_template_preprocessor_settings(
        settings_kwargs=settings_kwargs,
        input_set_type=input_set_type,
        output_directory=output_directory,
    )
    (output_directory / TEMPLATE_PREPROCESSOR_SETTINGS_FILENAME).write_text(
        settings.model_dump_json(indent=4)
    )

    input_set: DatasetCache | InferenceQuerySet
    if input_set_type == "train":
        dataset_cache = read_datacache(input_set_path)
        if not isinstance(dataset_cache, DatasetCache):
            raise ValueError(f"{input_set_path} is not a training dataset cache.")
        input_set = dataset_cache
    else:
        input_set = InferenceQuerySet.from_json(input_set_path)

    TemplatePreprocessor(input_set=input_set, config=settings)()

    if isinstance(input_set, DatasetCache):
        write_datacache_to_json(input_set, output_set_path)
    else:
        output_set_path.write_text(input_set.model_dump_json(indent=2))
    return output_set_path
