"""This module contains building blocks for template feature generation."""

import dataclasses

import numpy as np

from openfold3.core.data.primitives.structure.template import TemplateSliceCollection


@dataclasses.dataclass(frozen=False)
class AF3TemplateFeaturePrecursor:
    """Dataclass for storing information for AF3 template feature generation.

    Attributes:
        query_res_names (np.ndarray[str]):
            The names of the residues in the template structures. Has shape
            [n_templates, n_tokens].
        pseudo_beta_atom_coords (np.ndarray[float]):
            The coordinates of the pseudo beta atoms. Has shape
            [n_templates, n_tokens, 3].
        n_ca_c_atom_coords (np.ndarray[float]):
            The coordinates of the N, CA, and C atoms in this order. Has shape
            [n_templates, n_tokens, 3, 3].
    """

    res_names: np.ndarray[str]
    pseudo_beta_atom_coords: np.ndarray[float]
    n_ca_c_atom_coords: np.ndarray[float]


def create_feature_precursor_af3(template_slice_collection: TemplateSliceCollection):
    return
