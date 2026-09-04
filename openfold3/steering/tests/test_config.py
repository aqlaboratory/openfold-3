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

"""Validation of the run-level steering settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from openfold3.steering import defaults
from openfold3.steering.config import SteeringSettings, TermSettings


def test_steering_is_off_by_default():
    settings = SteeringSettings()
    assert settings.enabled is False
    assert settings.num_gd_steps == defaults.NUM_GD_STEPS
    assert set(settings.terms) == {"distance_bounds_potential"}


def test_terms_are_named_in_snake_case():
    """A potential's registry key -- the name a user writes in the runner
    yaml -- is snake_case, not the class name. Config keys, batch feature
    prefixes and error messages all use the one string, so it has to be the
    one that reads well in yaml."""
    settings = SteeringSettings.model_validate(
        {"terms": {"distance_bounds_potential": {"weight": 0.02}}}
    )
    assert settings.terms["distance_bounds_potential"].weight == 0.02


def test_the_class_name_is_not_an_accepted_term_key():
    """One name per term. Accepting the CamelCase class name as an alias too
    would put two spellings of every term into circulation."""
    with pytest.raises(ValidationError, match="unknown steering term"):
        SteeringSettings.model_validate({"terms": {"DistanceBoundsPotential": {}}})


def test_default_term_weight_comes_from_the_derived_defaults_table():
    """The derived parameter table lives in defaults.py and nowhere else."""
    assert TermSettings().weight == defaults.DISTANCE_WEIGHT


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        pytest.param({"num_gd_steps": 0}, "num_gd_steps", id="zero_gd_steps"),
        pytest.param({"unknown_key": 1}, "extra_forbidden", id="unknown_setting"),
        pytest.param(
            {"terms": {"distance_bounds_potential": {"weight": -1.0}}},
            "weight",
            id="negative_weight",
        ),
        pytest.param(
            {"terms": {"distance_bounds_potential": {"interval": 0}}},
            "interval",
            id="zero_interval",
        ),
        pytest.param(
            {"terms": {"distance_bounds_potential": {"nope": 1}}},
            "extra_forbidden",
            id="unknown_term_setting",
        ),
        pytest.param(
            {"terms": {"NotAPotential": {}}},
            "unknown steering term",
            id="unregistered_term",
        ),
        pytest.param(
            {"terms": {"distance_bounds_potential": {"weight": float("inf")}}},
            "finite_number",
            id="infinite_weight",
        ),
        pytest.param(
            {"terms": {"distance_bounds_potential": {"weight": float("nan")}}},
            "finite_number",
            id="nan_weight",
        ),
    ],
)
def test_settings_reject_invalid_values(payload: dict, match: str):
    with pytest.raises(ValueError, match=match):
        SteeringSettings.model_validate(payload)


def test_settings_route_from_a_runner_config_to_the_inference_job(tmp_path):
    """The path a runner yaml actually takes: dataset_config_kwargs.steering
    -> InferenceExperimentConfig -> InferenceJobConfig -> the dataset."""
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
        Query,
    )

    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    experiment = InferenceExperimentConfig(
        inference_ckpt_path=checkpoint,
        cache_path=tmp_path,
        dataset_config_kwargs={
            "steering": {"enabled": True, "num_gd_steps": 7},
        },
    )
    assert experiment.dataset_config_kwargs.steering.num_gd_steps == 7

    runner = InferenceExperimentRunner(experiment)
    runner.inference_query_set = InferenceQuerySet(
        queries={
            "ligand": Query.model_validate(
                {
                    "chains": [
                        {
                            "molecule_type": "ligand",
                            "chain_ids": ["L"],
                            "smiles": "CCO",
                        }
                    ]
                }
            )
        }
    )
    inference_job = runner.data_module_config.datasets[0].config

    assert inference_job.steering == experiment.dataset_config_kwargs.steering
    assert inference_job.steering.enabled is True


def test_active_terms_skips_disabled_and_zero_weight_terms():
    settings = SteeringSettings.model_validate(
        {"terms": {"distance_bounds_potential": {"enabled": False}}}
    )
    assert settings.active_terms() == {}

    settings = SteeringSettings.model_validate(
        {"terms": {"distance_bounds_potential": {"weight": 0.0}}}
    )
    assert settings.active_terms() == {}

    settings = SteeringSettings()
    assert set(settings.active_terms()) == {"distance_bounds_potential"}
