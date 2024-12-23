"""This module contains building blocks for template feature generation."""

import dataclasses

import biotite.structure as struc
import numpy as np
import torch

from openfold3.core.data.primitives.featurization.structure import encode_one_hot
from openfold3.core.data.primitives.structure.template import TemplateSliceCollection
from openfold3.core.data.resources.residues import (
    STANDARD_RESIDUES_WITH_GAP_3,
    get_with_unknown_3_to_idx,
)
from openfold3.core.utils.all_atom_multimer import make_transform_from_reference
from openfold3.core.utils.geometry.vector import Vec3Array


@dataclasses.dataclass(frozen=False)
class AF3TemplateFeaturePrecursor:
    """Dataclass for storing information for AF3 template feature generation.

    Attributes:
        res_names (np.ndarray[str]):
            The names of the residues in the template structures. Has shape
            [n_templates, n_tokens].
        pseudo_beta_atom_coords (np.ndarray[float]):
            The coordinates of the pseudo beta atoms. Has shape
            [n_templates, n_tokens, 3].
        frame_atom_coords (np.ndarray[float]):
            The coordinates of the N, CA, and C atoms in this order along dim -2. Has
            shape [n_templates, n_tokens, 3, 3].
    """

    res_names: np.ndarray[str]
    pseudo_beta_atom_coords: np.ndarray[float]
    frame_atom_coords: np.ndarray[float]


def create_template_feature_precursor_af3(
    template_slice_collection: TemplateSliceCollection,
    n_templates: int,
    n_tokens: int,
) -> AF3TemplateFeaturePrecursor:
    """Generates set of precursor features for AF3 template feature generation.

    Args:
        template_slice_collection (TemplateSliceCollection):
            The collection of cropped template atom arrays per chain, per template.
        n_templates (int):
            Number of templates.
        token_budget (int):
            Number of tokens in the target structure.

    Returns:
        AF3TemplateFeaturePrecursor:
            The precursor features for AF3 template feature generation. Includes
            residue names, pseudo beta atom coordinates, and N, CA, C atom coordinates.
    """
    res_names = np.full((n_templates, n_tokens), "GAP", dtype=np.dtype("U3"))
    pseudo_beta_atom_coords = np.full((n_templates, n_tokens, 3), np.nan, dtype=float)
    frame_atom_coords = np.full((n_templates, n_tokens, 3, 3), np.nan, dtype=float)

    # Iterate over chains then templates per chain
    for _, template_slices in template_slice_collection.template_slices.items():
        for template_idx, template_slice in enumerate(template_slices):
            # Unpack template slice
            template_atom_array = template_slice.atom_array
            template_residue_repeats = template_slice.template_residue_repeats
            query_token_positions = template_slice.query_token_positions
            residue_starts = struc.get_residue_starts(template_atom_array)

            # Residue names and unmask corresponding tokens/template positions
            res_names[template_idx, query_token_positions] = np.repeat(
                template_atom_array[residue_starts].res_name, template_residue_repeats
            )

            # Pseudo beta atom coordinates
            is_gly = template_atom_array.res_name == "GLY"
            is_ca = template_atom_array.atom_name == "CA"
            is_cb = template_atom_array.atom_name == "CB"
            is_pseudo_beta_atom = (is_gly & is_ca) | (~is_gly & is_cb)
            pseudo_beta_atom_coords[template_idx, query_token_positions, :] = np.repeat(
                template_atom_array[is_pseudo_beta_atom].coord,
                template_residue_repeats,
                axis=0,
            )

            # Frame (N, CA, C) atom coordinates
            is_n = template_atom_array.atom_name == "N"
            is_ca = template_atom_array.atom_name == "CA"
            is_c = template_atom_array.atom_name == "C"
            frame_atom_coords[template_idx, query_token_positions, 0, :] = np.repeat(
                template_atom_array[is_n].coord, template_residue_repeats, axis=0
            )
            frame_atom_coords[template_idx, query_token_positions, 1, :] = np.repeat(
                template_atom_array[is_ca].coord, template_residue_repeats, axis=0
            )
            frame_atom_coords[template_idx, query_token_positions, 2, :] = np.repeat(
                template_atom_array[is_c].coord, template_residue_repeats, axis=0
            )

    return AF3TemplateFeaturePrecursor(
        res_names=res_names,
        pseudo_beta_atom_coords=pseudo_beta_atom_coords,
        frame_atom_coords=frame_atom_coords,
    )


def create_template_restype(
    res_names: np.ndarray[str],
    template_pseudo_beta_mask: np.ndarray[float],
) -> torch.Tensor:
    """Creates the restype template feature for AF3.

    Args:
        res_names (np.ndarray[str]):
            The precursor features for AF3 template feature generation.
        template_pseudo_beta_mask (np.ndarray[float]):
            The mask for pseudo beta atoms. Has shape [n_templates, n_tokens].

    Returns:
        torch.Tensor:
            The restype template feature.
    """
    restype_index = torch.tensor(
        get_with_unknown_3_to_idx(res_names), dtype=torch.int64
    )
    template_restype = encode_one_hot(restype_index, len(STANDARD_RESIDUES_WITH_GAP_3))
    return (template_restype * template_pseudo_beta_mask.unsqueeze(-1)).to(torch.int32)


def create_template_distogram(
    pseudo_beta_atom_coords: np.ndarray[float],
    pseudo_beta_mask: np.ndarray[float],
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    n_bins: int = 39,
    inf_value: float = 1e8,
) -> torch.Tensor:
    """Creates the distogram template feature for AF3.

    Note: the pseudo_beta_mask is applied to the distogram to zero out masked
    tokens.

    Args:
        pseudo_beta_atom_coords (np.ndarray[float]):
            The coordinates of the pseudo beta atoms. Has shape
            [n_templates, n_tokens, 3].
        min_bin (float, optional):
            Bin lower bound. Defaults to 3.25.
        max_bin (float, optional):
            Bin upper bound. Defaults to 50.75.
        n_bins (int, optional):
            Number of bins between lower and upper bounds. Defaults to 39.
        inf_value (float, optional):
            Value to set last bin entries to. Defaults to 1e8.

    Returns:
        torch.Tensor:
            The distogram template feature [N_templates, N_token, N_token, N_bins].
    """
    distogram = np.sum(
        (
            pseudo_beta_atom_coords[..., None, :]
            - pseudo_beta_atom_coords[..., None, :, :]
        )
        ** 2,
        axis=-1,
        keepdims=True,
    )

    # Generate squared bin edges
    lower = np.linspace(min_bin, max_bin, n_bins) ** 2
    upper = np.concatenate(
        [lower[1:], np.array([inf_value], dtype=lower.dtype)], axis=-1
    )

    # Bin the distogram
    template_distogram = torch.tensor(
        ((distogram > lower) * (distogram < upper)).astype(distogram.dtype),
        dtype=torch.float,
    )

    return (
        template_distogram
        * (pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :])[..., None]
    )


def create_template_unit_vector(
    frame_atom_coords: np.ndarray[float],
    backbone_frame_mask: np.ndarray[float],
) -> torch.Tensor:
    """Creates the unit vector template feature for AF3.

    Args:
        frame_atom_coords (np.ndarray[float]):
            The coordinates of the N, CA, and C atoms in this order along dim -2. Has
            shape [n_templates, n_tokens, 3, 3].
        backbone_frame_mask (np.ndarray[float]):
            The mask indicating for each token in each template if all backbone frame
            atoms are available. Has shape [n_templates, n_tokens].

    Returns:
        torch.Tensor:
            The unit vector template feature [N_templates, N_token, N_token, 3].
    """

    # Convert nans to 0s and tensors to Vec3Arrays
    frame_atom_coords = torch.nan_to_num(
        torch.tensor(frame_atom_coords, dtype=torch.float32), nan=0.0
    )
    n_vec3array = Vec3Array.from_array(frame_atom_coords[:, :, 0, :])
    ca_vec3array = Vec3Array.from_array(frame_atom_coords[:, :, 1, :])
    c_vec3array = Vec3Array.from_array(frame_atom_coords[:, :, 2, :])

    # Create rigid frame and unit vector
    rigid = make_transform_from_reference(
        a_xyz=n_vec3array,
        b_xyz=ca_vec3array,
        c_xyz=c_vec3array,
    )
    unit_vector = (
        rigid[..., None]
        .inverse()
        .apply_to_point(rigid.translation[..., None, :])
        .normalized()
    )

    # Cast back to tensor and apply backbone frame mask
    masked_unit_vector = []
    for coord in unit_vector:
        # Create a 2D backbone frame mask
        backbone_frame_2d = (
            backbone_frame_mask[..., None] * backbone_frame_mask[..., None, :]
        )
        # Apply mask
        masked_coord = (coord * backbone_frame_2d)[..., None]
        masked_unit_vector.append(masked_coord)

    return torch.cat(masked_unit_vector, dim=-1)
