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

from unittest.mock import patch

import pytest
import torch

from openfold3.projects.of3_all_atom.project_entry import ModelUpdate, OF3ProjectEntry
from openfold3.projects.of3_all_atom.runner import (
    CONFIDENCE_GROUP,
    MODULE_GROUPS,
    OpenFold3AllAtom,
    logger,
)


def mock_forward_with_name_based_oom(batch):
    """Mock the forward call to OOM based on sample name"""
    query_id = batch["query_id"][0]

    if "oom" in query_id:
        # Sample named "one" - simulate OOM
        raise torch.OutOfMemoryError(
            f"Mock CUDA out of memory error for sample {query_id}"
        )
    else:
        # Other samples - return successful mock outputs
        mock_outputs = {
            "atom_positions_predicted": torch.randn(1, 5, 100, 3),
            # Add other expected outputs as needed
        }
        return batch, mock_outputs


one_oom_in_batch = [
    {
        "query_id": ["oom_one"],
        "seed": torch.tensor([123]),
        "is_repeated_sample": torch.tensor([False]),
        "valid_sample": True,
    },
    {
        "query_id": ["two"],
        "seed": torch.tensor([456]),
        "is_repeated_sample": torch.tensor([False]),
        "valid_sample": torch.tensor([True]),
    },
]
multiple_ooms_in_batch = [
    {
        "query_id": ["oom_one"],
        "seed": torch.tensor([123]),
        "is_repeated_sample": torch.tensor([False]),
        "valid_sample": torch.tensor([True]),
    },
    {
        "query_id": ["two"],
        "seed": torch.tensor([456]),
        "is_repeated_sample": torch.tensor([False]),
        "valid_sample": torch.tensor([True]),
    },
    {
        "query_id": ["oom_three"],
        "seed": torch.tensor([789]),
        "is_repeated_sample": torch.tensor([False]),
        "valid_sample": torch.tensor([True]),
    },
]


@pytest.mark.parametrize(
    "batches",
    [one_oom_in_batch, multiple_ooms_in_batch],
    ids=["one_oom", "multiple_oom"],
)
def test_oom_exception_handling(batches):
    # two queries:
    project_entry = OF3ProjectEntry()
    config = project_entry.get_model_config_with_presets()
    model_runner = OpenFold3AllAtom(model_config=config)
    batches = [
        {
            "query_id": ["oom_one"],
            "seed": torch.tensor([123]),
            "is_repeated_sample": torch.tensor([False]),
            "valid_sample": torch.tensor([True]),
        },
        {
            "query_id": ["two"],
            "seed": torch.tensor([456]),
            "is_repeated_sample": torch.tensor([False]),
            "valid_sample": torch.tensor([True]),
        },
    ]
    results = {}

    with (
        patch.object(
            logger, "exception", return_value=None
        ),  # silence generated exceptions
        patch.object(
            model_runner.model, "forward", side_effect=mock_forward_with_name_based_oom
        ),
        patch.object(
            model_runner, "_compute_confidence_scores", return_value={"plddt": 0.75}
        ),
    ):
        for idx, batch in enumerate(batches):
            outputs = model_runner.predict_step(batch, idx)
            query_name = batch["query_id"][0]
            results[query_name] = outputs

    assert len(results) == len(batches)
    expected_results = [not bool("oom" in query_id) for query_id in results]
    actual_results = [bool(result) for result in results.values()]
    assert expected_results == actual_results


def test_version_registration():
    project_entry = OF3ProjectEntry()
    config = project_entry.get_model_config_with_presets()
    model_runner = OpenFold3AllAtom(model_config=config)

    # Check that the version property returns the expected version string
    expected_version = "1.0.0"
    assert model_runner.version == expected_version


def _is_confidence_param(name: str) -> bool:
    return any(
        name.startswith(f"{prefix}.") for prefix in MODULE_GROUPS[CONFIDENCE_GROUP]
    )


@pytest.mark.parametrize(
    ("freezing_settings", "expect_frozen"),
    [
        ({}, lambda name: False),
        ({"freeze_modules": [CONFIDENCE_GROUP]}, _is_confidence_param),
        (
            {"train_only_modules": [CONFIDENCE_GROUP]},
            lambda name: not _is_confidence_param(name),
        ),
    ],
    ids=["no_freezing", "freeze_confidence", "train_confidence_only"],
)
def test_module_freezing(freezing_settings, expect_frozen):
    """Exactly the requested module groups lose their grads."""
    project_entry = OF3ProjectEntry()
    config = project_entry.get_model_config_with_update(
        ModelUpdate(
            presets=["train"],
            custom={
                "settings": {
                    # The grad manager requires an attached trainer
                    "gradient_clipping": {"per_sample_clipping": False},
                    **freezing_settings,
                }
            },
        )
    )
    model_runner = OpenFold3AllAtom(model_config=config)

    model_runner.setup(stage="fit")

    frozen = {
        name
        for name, param in model_runner.model.named_parameters()
        if not param.requires_grad
    }
    expected = {
        name for name, _ in model_runner.model.named_parameters() if expect_frozen(name)
    }

    assert frozen == expected


@pytest.mark.parametrize(
    ("settings", "error", "match"),
    [
        (
            {
                "train_only_modules": [CONFIDENCE_GROUP],
                "freeze_modules": [CONFIDENCE_GROUP],
            },
            AssertionError,
            "mutually exclusive",
        ),
        ({"freeze_modules": ["diffusion"]}, ValueError, "Unknown module group"),
        ({"train_confidence_only": True}, KeyError, "has been removed"),
    ],
    ids=["mutually_exclusive", "unknown_group", "removed_setting"],
)
def test_module_freezing_config_validation(settings, error, match):
    project_entry = OF3ProjectEntry()

    with pytest.raises(error, match=match):
        project_entry.get_model_config_with_update(
            ModelUpdate(custom={"settings": settings})
        )


@pytest.mark.parametrize(
    ("train_only_modules", "expected"),
    [([], False), ([CONFIDENCE_GROUP], True)],
    ids=["default", "confidence_only"],
)
def test_train_confidence_only_is_derived(train_only_modules, expected):
    """The internal compute-skipping flag follows train_only_modules."""
    project_entry = OF3ProjectEntry()
    config = project_entry.get_model_config_with_update(
        ModelUpdate(custom={"settings": {"train_only_modules": train_only_modules}})
    )

    assert config.settings.train_confidence_only == expected
    # The loss module shares the setting through a field reference
    assert config.architecture.loss_module.train_confidence_only == expected
