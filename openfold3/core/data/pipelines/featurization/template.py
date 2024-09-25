"""This module contains preprocessing pipelines for template data."""

import numpy as np
import torch

from openfold3.core.data.primitives.featurization.template import (
    create_feature_precursor_af3,
    create_template_distogram,
    create_template_restype,
    create_template_unit_vector,
)
from openfold3.core.data.primitives.structure.template import TemplateSliceCollection


def featurize_templates_dummy_af3(batch_size, n_templ, n_token):
    """Temporary function to generate dummy template features."""
    return {
        "template_restype": torch.ones((batch_size, n_templ, n_token, 32)).int(),
        "template_pseudo_beta_mask": torch.ones((batch_size, n_templ, n_token)).float(),
        "template_backbone_frame_mask": torch.ones(
            (batch_size, n_templ, n_token)
        ).float(),
        "template_distogram": torch.ones(
            (batch_size, n_templ, n_token, n_token, 39)
        ).int(),
        "template_unit_vector": torch.ones(
            (batch_size, n_templ, n_token, n_token, 3)
        ).float(),
    }


def featurize_templates_af3(
    template_slice_collection: TemplateSliceCollection,
    n_templates: int,
    token_budget: int,
    min_bin: float,
    max_bin: float,
    n_bins: int,
) -> dict[str, torch.Tensor]:
    """Featurizes template data for AF3.

    Args:
        template_slice_collection (TemplateSliceCollection):
            The collection of cropped template atom arrays per chain, per template.
        n_templates (int):
            Number of templates.
        token_budget (int):
            Crop size.
        min_bin (float):
            The minimum distance for the distogram bins.
        max_bin (float):
            The maximum distance for the distogram bins.
        n_bins (int):
            The number of bins in the distogram.

    Returns:
        dict[str, torch.Tensor]:
            The featurized template data.
    """
    template_feature_precursor = create_feature_precursor_af3(
        template_slice_collection,
        n_templates,
        token_budget,
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
    features["template_restype"] = create_template_restype(
        template_feature_precursor.res_names,
        features["template_pseudo_beta_mask"],
    )
    features["template_distogram"] = create_template_distogram(
        template_feature_precursor.pseudo_beta_atom_coords,
        features["template_pseudo_beta_mask"],
        min_bin,
        max_bin,
        n_bins,
    )
    features["template_unit_vector"] = create_template_unit_vector(
        template_feature_precursor.frame_atom_coords,
        features["template_backbone_frame_mask"],
    )
    return features
