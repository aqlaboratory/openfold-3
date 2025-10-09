"""Integration test for inference

Runs two small inference queries without msa or templates.
"""

import logging
import pytest

from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
from openfold3.entry_points.validator import (
    InferenceExperimentConfig,
    InferenceExperimentSettings,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from tests.compare_utils import skip_unless_cuda_available

logger = logging.getLogger(__name__)

protein_only_query = InferenceQuerySet.model_validate(
    {
        "queries": {
            "query1": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A", "B"],
                        "sequence": "XRMKQLEDKVEELLSKNYHLENEVARLKKLVGER",
                    }
                ]
            }
        }
    }
)

protein_and_ligand_query = InferenceQuerySet.model_validate(
    {
        "queries": {
            "query1": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A", "B"],
                        "sequence": "XRMKQLEDKVEELLSKNYHLENEVARLKKLVGER",
                    },
                    {
                        "molecule_type": "ligand",
                        "chain_ids": ["C"],
                        "smiles": "c1ccccc1O",
                    },
                ]
            }
        }
    }
)

@pytest.mark.skip(
    reason="Manually enable this for now, will add flag to run slow tests later."
)
@skip_unless_cuda_available()
@pytest.mark.parametrize("query_set", [protein_only_query, protein_and_ligand_query])
def test_inference_run(tmp_path, query_set):
    experiment_config = InferenceExperimentConfig()
    expt_runner = InferenceExperimentRunner(
        experiment_config, num_diffusion_samples=1, output_dir=tmp_path
    )
    expt_runner.setup()
    expt_runner.run(query_set)
    expt_runner.cleanup()

    err_log_dir = tmp_path / "logs"
    if err_log_dir.exists():
        raise RuntimeError(
            f"Found error logs in  directory {err_log_dir}, "
            "check for errors in inference."
        )

    logging.info(f"Checking output contents at {tmp_path}")
    expected_output_dir = tmp_path / "query1" / "seed_42"
    expected_files = [
        "query1_seed_42_sample_1_confidences.json",
        "query1_seed_42_sample_1_confidences_aggregated.json",
        "query1_seed_42_sample_1_model.cif",
        "timing.json",
    ]
    for f in expected_files:
        assert (expected_output_dir / f).exists()
