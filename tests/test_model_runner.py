from unittest.mock import patch

import pytest
import torch

from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom, logger


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
