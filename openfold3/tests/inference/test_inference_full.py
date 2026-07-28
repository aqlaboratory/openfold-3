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

"""Smoke tests for inference: two small queries run end-to-end.

``test_protein_only`` / ``test_protein_and_ligand`` run with MSA server + templates and
check the expected output files are written. See ``test_templates.py`` for the functional
check that a supplied template actually steers the prediction.

Both require a GPU and downloaded model weights; they skip otherwise.

Run with:
    pytest openfold3/tests/inference/test_inference_full.py
"""

import logging

import pytest

from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from openfold3.tests.inference.helpers import run_inference
from openfold3.tests.utils.compare_utils import skip_unless_cuda_available

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


def _assert_inference_writes_outputs(query_set, tmp_path):
    run_inference(
        query_set,
        tmp_path,
        use_msa_server=True,
        use_templates=True,
        num_diffusion_samples=1,
    )
    logger.info("Checking output contents at %s", tmp_path)
    seed_dir = tmp_path / "query1" / "seed_42"
    expected_files = [
        "query1_seed_42_sample_1_confidences.json",
        "query1_seed_42_sample_1_confidences_aggregated.json",
        "query1_seed_42_sample_1_model.cif",
        "timing.json",
    ]
    for name in expected_files:
        assert (seed_dir / name).exists(), (
            f"Expected output file not found: {seed_dir / name}"
        )


@skip_unless_cuda_available()
def test_protein_only(tmp_path):
    _assert_inference_writes_outputs(protein_only_query, tmp_path)


@skip_unless_cuda_available()
def test_protein_and_ligand(tmp_path):
    _assert_inference_writes_outputs(protein_and_ligand_query, tmp_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv"]))
