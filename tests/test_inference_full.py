# Copyright 2025 AlQuraishi Laboratory
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

"""Integration test for inference

Runs two small inference queries without msa or templates.
"""

import os
import logging

import pytest

from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
from openfold3.entry_points.validator import (
    InferenceExperimentConfig,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from tests.compare_utils import skip_unless_cuda_available
from unittest.mock import patch


pytestmark = pytest.mark.inference_verification

logging.basicConfig(level=logging.WARNING)
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


@skip_unless_cuda_available()
@pytest.mark.parametrize("query_set", [protein_only_query, protein_and_ligand_query])
def test_inference_run(tmp_path, query_set):
    # Trigger validation logic to replace the cache path
    with patch('builtins.input', return_value='no'):
        # your test code that calls _maybe_download_parameters
        experiment_config = InferenceExperimentConfig.model_validate({})
    expt_runner = InferenceExperimentRunner(
        experiment_config, num_diffusion_samples=1, output_dir=tmp_path
    )
    try:
        expt_runner.setup()
    except ValueError as e:
        # If called from setup script, fail the test
        if "is not a valid file or directory" in str(e):
            if os.environ.get('OPENFOLD_SETUP_SCRIPT') == '1':
                pytest.fail(
                    "No checkpoint files found after running setup script. "
                    "Please check that the download completed successfully."
                )
            else:
                logger.warning(
                    "No checkpoint files found, skipping for now. "
                    "Please use scripts/setup_openfold3.sh to download the weights."
                )
                pytest.skip("No checkpoint files available")
        else:
            raise

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
