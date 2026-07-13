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

"""End-to-end featurization check for the inference template-cache gate (PR #306).

Runs the real inference featurization on CPU (no model weights) through
``InferenceDataset.create_template_features``, using a short protein query and a
hand-built preparsed template structure array. It asserts the downstream consequence of
the bug: when ``template_cache_directory`` is ``None`` (pre-#306) the returned template
feature masks are entirely zero, whereas with the preprocessor ``cache_directory`` set
(post-#306) they are populated for the aligned tokens.

This exercises the whole chain sample_templates -> align_template_to_query ->
featurize_template_structures_of3, so reverting the one-line fix at
inference.py:261 turns the "populated" case red.
"""

from pathlib import Path

import numpy as np
import pytest

from openfold3.core.data.framework.single_datasets.inference import InferenceDataset
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.core.data.primitives.structure.tokenization import get_token_count
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    TemplateSettings,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Chain, Query
from openfold3.tests.utils.template_helpers import (
    TEMPLATE_ID,
    make_cache_entry,
    write_cache_npz,
    write_template_structure_array,
)

QUERY_SEQUENCE = "AWKATV"  # 6 residues, no glycine -> every residue has a CB


def _make_dataset(tmp_path: Path) -> InferenceDataset:
    """Bare InferenceDataset with only the attributes create_template_features reads."""
    dataset = object.__new__(InferenceDataset)
    # take_top_k=True -> deterministic top-k selection (inference semantics). The default
    # (False) draws k = randint(0, n_available+1), which with a single template is a coin
    # flip between k=0 (template dropped) and k=1, making the featurized masks flaky.
    dataset.template_settings = TemplateSettings(take_top_k=True)
    dataset.template_preprocessor_settings = TemplatePreprocessorSettings(
        output_directory=tmp_path,
        preparse_structures=True,
        structure_file_format="npz",
    )
    dataset.ccd = None  # unused for the preparsed-structure-array path
    return dataset


# cache_directory is forwarded into sample_templates as the non-None gate; None
# reproduces the pre-#306 bug (templates dropped -> masks all zero), a real dir loads the
# template for every query token (mask sum == n_tokens == len(QUERY_SEQUENCE)).
@pytest.mark.parametrize(
    "make_cache_directory, expected_mask_sum",
    [
        pytest.param(lambda tmp_path: None, 0, id="none_cache_directory_masks_empty"),
        pytest.param(
            lambda tmp_path: tmp_path / "dummy_cache",
            len(QUERY_SEQUENCE),
            id="dummy_cache_directory_masks_populated",
        ),
    ],
)
def test_create_template_features_masks(
    tmp_path, make_cache_directory, expected_mask_sum
):
    n_res = len(QUERY_SEQUENCE)

    # Fixtures: preparsed template structure array + per-chain cache entry.
    dataset = _make_dataset(tmp_path)
    write_template_structure_array(
        dataset.template_preprocessor_settings.structure_array_directory, n_res
    )
    cache_npz = tmp_path / "chainA.npz"
    idx_map = np.stack([np.arange(1, n_res + 1)] * 2, axis=1)
    write_cache_npz(cache_npz, {TEMPLATE_ID: make_cache_entry(idx_map)})

    dataset.template_preprocessor_settings.cache_directory = make_cache_directory(
        tmp_path
    )

    query = Query(
        chains=[
            Chain(
                molecule_type="protein",
                chain_ids=["A"],
                sequence=QUERY_SEQUENCE,
                template_alignment_file_path=cache_npz,
                template_entry_chain_ids=[TEMPLATE_ID],
            )
        ]
    )
    atom_array, _ = InferenceDataset.get_structure_with_ref_mols(query)
    n_tokens = get_token_count(atom_array)

    features = dataset.create_template_features(query, atom_array, n_tokens)

    assert features["template_pseudo_beta_mask"].sum() == expected_mask_sum
    assert features["template_backbone_frame_mask"].sum() == expected_mask_sum
