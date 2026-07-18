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

import random
import unittest

import numpy as np

from openfold3.core.data.framework.single_datasets.inference import (
    InferenceDataset,
    _seeded_feature_creation,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Query

LIGAND_QUERY = Query.model_validate(
    {
        "query_name": "ethanol",
        "chains": [
            {
                "molecule_type": "ligand",
                "chain_ids": "A",
                "smiles": "CCO",
            }
        ],
    }
)


def _reference_conformer_coords(seed: int) -> np.ndarray:
    with _seeded_feature_creation(seed):
        swrm = InferenceDataset.get_structure_with_ref_mols(LIGAND_QUERY)
    return swrm.processed_reference_mols[0].mol.GetConformer().GetPositions()


class TestInferenceFeatureSeeding(unittest.TestCase):
    def test_same_seed_is_independent_of_global_rng(self):
        for _ in range(100):
            random.random()
        first = _reference_conformer_coords(42)
        for _ in range(100):
            random.random()
        second = _reference_conformer_coords(42)
        np.testing.assert_allclose(first, second)

    def test_different_seeds_can_differ(self):
        first = _reference_conformer_coords(1)
        second = _reference_conformer_coords(2)
        self.assertFalse(np.allclose(first, second))

    def test_restores_global_python_rng(self):
        random.seed(7)
        before = random.random()
        with _seeded_feature_creation(999):
            _reference_conformer_coords(999)
        after = random.random()

        random.seed(7)
        expected_before = random.random()
        expected_after = random.random()

        self.assertEqual(before, expected_before)
        self.assertEqual(after, expected_after)
