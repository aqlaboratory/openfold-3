"""This module contains preprocessing pipelines for template data."""

import torch


def featurize_templates_dummy_af3(n_templ, n_token):
    """Temporary function to generate dummy template features."""
    return {
        "template_restype": torch.ones((n_templ, n_token, 32)).int(),
        "template_pseudo_beta_mask": torch.ones((n_templ, n_token)).float(),
        "template_backbone_frame_mask": torch.ones((n_templ, n_token)).float(),
        "template_distogram": torch.ones((n_templ, n_token, n_token, 39)).int(),
        "template_unit_vector": torch.ones((n_templ, n_token, n_token, 3)).float(),
    }
