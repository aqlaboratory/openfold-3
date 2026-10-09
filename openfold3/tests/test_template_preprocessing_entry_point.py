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

"""Tests for the template preprocessing entry point.

The entry point behind scripts/data_preprocessing/preprocess_template_alignments_new_of3.py:
resolving where outputs go, and running preprocessing on a dataset cache or
inference query set JSON. Runs use the 1fdl fixture of the template preprocessing
tests.
"""

import json
import logging
from pathlib import Path

import pytest

from openfold3.core.data.io.dataset_cache import read_datacache, write_datacache_to_json
from openfold3.core.data.io.sequence.fasta import get_chain_id_to_seq_from_fasta
from openfold3.core.data.primitives.caches.format import ClusteredDatasetCache
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.entry_points.template_preprocessing import (
    resolve_template_preprocessor_settings,
    run_template_preprocessing,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Chain,
    InferenceQuerySet,
    Query,
)
from openfold3.tests.core.data.pipelines.preprocessing.test_template_train import (
    TRAIN_EXPECTED_TEMPLATE_IDS,
    TRAIN_FIXTURE_DIR,
    _train_dataset_cache,
    _train_settings_kwargs,
    _train_template_ids,
)

# ---------------------------------------------------------------------------
# Where outputs go
# ---------------------------------------------------------------------------


# The resolved output paths for the same settings, whatever the YAML gave. Relative
# to tmp_path, which the test prefixes to every path.
DEFAULT_OUTPUT_PATHS = {
    "output_directory": Path("out"),
    "cache_directory": Path("out/template_cache"),
    "log_directory": Path("out/template_logs"),
    "precache_directory": None,  # create_precache is off
    "structure_array_directory": Path("out/template_structure_arrays"),
    "structure_directory": Path("structures"),  # an input, as given
}


@pytest.mark.parametrize(
    ("yaml_paths", "expected_paths", "expected_ignored"),
    [
        pytest.param(
            {"structure_directory": "structures"},
            DEFAULT_OUTPUT_PATHS,
            [],
            id="derived-from-output-directory",
        ),
        pytest.param(
            {
                "structure_directory": "structures",
                "output_directory": "from_yaml",
                "cache_directory": "from_yaml/cache",
                "log_directory": "from_yaml/logs",
            },
            DEFAULT_OUTPUT_PATHS,
            ["output_directory", "cache_directory", "log_directory"],
            id="yaml-output-paths-ignored",
        ),
        pytest.param(
            {
                "structure_directory": "structures",
                "precache_directory": "shared/template_precache",
                "structure_array_directory": "shared/template_structure_arrays",
            },
            {
                **DEFAULT_OUTPUT_PATHS,
                "precache_directory": Path("shared/template_precache"),
                "structure_array_directory": Path("shared/template_structure_arrays"),
            },
            [],
            id="yaml-structure-caches-kept",
        ),
    ],
)
def test_resolved_output_paths(
    tmp_path, caplog, yaml_paths, expected_paths, expected_ignored
):
    """Where each output goes, given paths from the YAML.

    --output_directory wins over output_directory, cache_directory and log_directory
    from the YAML, with a warning. Precache and structure array directories hold
    parsed template structures reused across runs, so the YAML's are kept.
    """
    yaml_paths = {key: str(tmp_path / value) for key, value in yaml_paths.items()}
    expected_paths = {
        key: None if value is None else tmp_path / value
        for key, value in expected_paths.items()
    }
    output_directory = tmp_path / "out"
    settings_kwargs = {"preparse_structures": True, "create_logs": True, **yaml_paths}

    with caplog.at_level(logging.WARNING):
        settings = resolve_template_preprocessor_settings(
            settings_kwargs=settings_kwargs,
            input_set_type="train",
            output_directory=output_directory,
        )

    actual = {key: getattr(settings, key) for key in expected_paths}
    assert actual == expected_paths
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    expected_warnings = (
        [
            f"Ignoring template_preprocessor_settings {expected_ignored}: all "
            f"outputs go to {output_directory}."
        ]
        if expected_ignored
        else []
    )
    assert warnings == expected_warnings
    for key in expected_ignored:
        assert not Path(yaml_paths[key]).exists(), key


def test_mode_in_settings_is_rejected(tmp_path):
    """The mode comes from the input set type, not the YAML."""
    with pytest.raises(ValueError, match="mode"):
        resolve_template_preprocessor_settings(
            settings_kwargs={"mode": "predict"},
            input_set_type="train",
            output_directory=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# Running on a dataset cache or inference query set JSON
# ---------------------------------------------------------------------------


def _write_dataset_cache_json(directory: Path) -> Path:
    """Train-mode input: the 1fdl dataset cache."""
    input_path = directory / "dataset_cache.json"
    write_datacache_to_json(_train_dataset_cache(), input_path)
    return input_path


def _write_inference_query_set_json(directory: Path) -> Path:
    """Predict-mode input: lysozyme (1fdl chain C) with its ColabFold template hits."""
    lysozyme = get_chain_id_to_seq_from_fasta(
        TRAIN_FIXTURE_DIR / "representatives.fasta"
    )["1fdl_C"]
    query_set = InferenceQuerySet(
        queries={
            "lysozyme": Query(
                chains=[
                    Chain(
                        molecule_type=MoleculeType.PROTEIN,
                        chain_ids=["A"],
                        sequence=lysozyme,
                        template_alignment_file_path=TRAIN_FIXTURE_DIR
                        / "template_alignments"
                        / "1fdl_C"
                        / "colabfold_template.m8",
                    )
                ]
            )
        }
    )
    input_path = directory / "inference_query_set.json"
    input_path.write_text(query_set.model_dump_json())
    return input_path


def _read_template_assignments(output_path: Path, input_set_type: str) -> dict:
    """Templates assigned in the output JSON.

    train: structure -> chain -> template_ids. predict: query -> template IDs of its
    first chain.
    """
    if input_set_type == "train":
        dataset_cache = read_datacache(output_path)
        assert isinstance(dataset_cache, ClusteredDatasetCache)
        return _train_template_ids(dataset_cache)
    return {
        name: query.chains[0].template_entry_chain_ids
        for name, query in InferenceQuerySet.from_json(output_path).queries.items()
    }


@pytest.mark.parametrize(
    (
        "input_set_type",
        "write_input_set",
        "dropped_settings",
        "expected_files",
        "expected_templates",
    ),
    [
        pytest.param(
            "train",
            _write_dataset_cache_json,
            [],
            [
                "dataset_cache.json",
                "template_cache",
                "template_preprocessor_settings.json",
                "template_structure_arrays",
            ],
            TRAIN_EXPECTED_TEMPLATE_IDS,
            id="train",
        ),
        pytest.param(
            "predict",
            _write_inference_query_set_json,
            # Train-only settings; predict mode has no query release date to compare
            # against
            [
                "template_alignment_directory",
                "alignment_representatives_fasta",
                "min_release_date_diff",
            ],
            [
                "inference_query_set.json",
                "template_cache",
                "template_preprocessor_settings.json",
                "template_structure_arrays",
            ],
            {"lysozyme": ["1ior_A", "7ynv_A", "5lyz_A"]},
            id="predict",
        ),
    ],
)
def test_run_template_preprocessing(
    tmp_path,
    input_set_type,
    write_input_set,
    dropped_settings,
    expected_files,
    expected_templates,
):
    """The updated set, the settings used and the template data go to one directory.

    The updated set keeps the input's file name. Read from JSON, the train-mode dataset
    cache's release dates are strings rather than dates.
    """
    (tmp_path / "input").mkdir()
    input_path = write_input_set(tmp_path / "input")
    settings_kwargs = _train_settings_kwargs(tmp_path)
    for key in ["output_directory", *dropped_settings]:
        del settings_kwargs[key]
    output_directory = tmp_path / "out"

    output_path = run_template_preprocessing(
        input_set_path=input_path,
        input_set_type=input_set_type,
        output_directory=output_directory,
        settings_kwargs=settings_kwargs,
    )

    assert output_path == output_directory / input_path.name
    assert sorted(path.name for path in output_directory.iterdir()) == expected_files
    assert _read_template_assignments(output_path, input_set_type) == (
        expected_templates
    )
    saved = json.loads(
        (output_directory / "template_preprocessor_settings.json").read_text()
    )
    assert saved["cache_directory"] == str(output_directory / "template_cache")


def test_run_template_preprocessing_refuses_to_overwrite_its_input(tmp_path):
    """The updated set is not written over the input set."""
    input_path = _write_dataset_cache_json(tmp_path)
    settings_kwargs = _train_settings_kwargs(tmp_path)
    del settings_kwargs["output_directory"]

    with pytest.raises(ValueError, match="would overwrite the input set"):
        run_template_preprocessing(
            input_set_path=input_path,
            input_set_type="train",
            output_directory=input_path.parent,
            settings_kwargs=settings_kwargs,
        )
