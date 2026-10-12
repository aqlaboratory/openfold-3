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

"""This module contains preprocessing pipelines for template data."""

import numpy as np
import torch

from openfold3.core.data.primitives.featurization.template import (
    create_template_feature_precursor_of3,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.template import TemplateSliceCollection
from openfold3.core.data.resources.residues import get_with_unknown_3_to_idx


def featurize_templates_dummy_of3(n_templ, n_token):
    """Temporary function to generate dummy template features."""
    return {
        "template_restype": torch.zeros((n_templ, n_token), dtype=torch.int32),
        "template_pseudo_beta_mask": torch.ones((n_templ, n_token)).float(),
        "template_backbone_frame_mask": torch.ones((n_templ, n_token)).float(),
        "template_pseudo_beta_coords": torch.ones((n_templ, n_token, 3)).float(),
        "template_frame_atom_coords": torch.ones((n_templ, n_token, 3, 3)).float(),
    }


@log_runtime_memory(runtime_dict_key="runtime-template-feat")
def featurize_template_structures_of3(
    template_slice_collection: TemplateSliceCollection,
    n_templates: int,
    n_tokens: int,
) -> dict[str, torch.Tensor]:
    """Featurizes template data for AF3.

    Returned features:
        - template_pseudo_beta_mask:
            Mask indicating if pseudo beta atoms (C-alpha atom for GLY, C-beta
            otherwise) are present in the corresponding token.
        - template_backbone_frame_mask:
            Mask indicating if backbone frame atoms (N, CA, C) are present in the
            corresponding token.
        - template_restype:
            Integer indices of template residue types. The model one-hots
            these inside the template embedder.
        - template_pseudo_beta_coords:
            Coordinates of pseudo beta atoms. The pairwise distogram is
            computed on-the-fly inside the template embedder.
        - template_frame_atom_coords:
            Coordinates of the N, CA, C atoms of each token. The model
            constructs the rigid frame and unit vector features from these.

    Args:
        template_slice_collection (TemplateSliceCollection):
            The collection of cropped template atom arrays per chain, per template.
        n_templates (int):
            Number of templates.
        n_tokens (int):
            Number of tokens in the target structure.

    Returns:
        dict[str, torch.Tensor]:
            The featurized template data.
    """
    template_feature_precursor = create_template_feature_precursor_of3(
        template_slice_collection,
        n_templates,
        n_tokens,
    )

    features = {}
    features["template_pseudo_beta_mask"] = torch.tensor(
        ~np.isnan(template_feature_precursor.pseudo_beta_atom_coords).any(axis=-1),
        dtype=torch.float,
    )
    features["template_backbone_frame_mask"] = torch.tensor(
        ~np.isnan(template_feature_precursor.frame_atom_coords).any(axis=(-2, -1)),
        dtype=torch.float,
    )
    # Integer residue indices. Model one-hots inside the template embedder.
    features["template_restype"] = torch.tensor(
        get_with_unknown_3_to_idx(template_feature_precursor.res_names),
        dtype=torch.int32,
    )
    # NaN positions correspond to absent atoms and are zeroed
    features["template_pseudo_beta_coords"] = torch.nan_to_num(
        torch.tensor(
            template_feature_precursor.pseudo_beta_atom_coords, dtype=torch.float32
        ),
        nan=0.0,
    )
    features["template_frame_atom_coords"] = torch.nan_to_num(
        torch.tensor(template_feature_precursor.frame_atom_coords, dtype=torch.float32),
        nan=0.0,
    )
    return features
