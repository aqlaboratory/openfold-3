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

"""This module contains building blocks for template feature generation."""

import dataclasses
import logging
import traceback

import biotite.structure as struc
import numpy as np

from openfold3.core.data.primitives.structure.template import TemplateSliceCollection

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=False)
class OF3TemplateFeaturePrecursor:
    """Dataclass for storing information for OF3 template feature generation.

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


def create_template_feature_precursor_of3(
    template_slice_collection: TemplateSliceCollection,
    n_templates: int,
    n_tokens: int,
) -> OF3TemplateFeaturePrecursor:
    """Generates set of precursor features for OF3 template feature generation.

    Args:
        template_slice_collection (TemplateSliceCollection):
            The collection of cropped template atom arrays per chain, per template.
        n_templates (int):
            Number of templates.
        n_tokens (int):
            Number of tokens in the target structure.

    Returns:
        OF3TemplateFeaturePrecursor:
            The precursor features for OF3 template feature generation. Includes
            residue names, pseudo beta atom coordinates, and N, CA, C atom coordinates.
    """
    res_names = np.full((n_templates, n_tokens), "GAP", dtype=np.dtype("U3"))
    pseudo_beta_atom_coords = np.full((n_templates, n_tokens, 3), np.nan, dtype=float)
    frame_atom_coords = np.full((n_templates, n_tokens, 3, 3), np.nan, dtype=float)

    # Iterate over chains then templates per chain
    for _, template_slices in template_slice_collection.template_slices.items():
        template_idx = 0
        for template_slice in template_slices:
            # TODO add support for these cases below Skip templates that fail to be
            # featurized these will be templates with composite residues like 4v2r
            # residue 195
            try:
                # Unpack template slice
                template_atom_array = template_slice.atom_array
                template_residue_repeats = template_slice.template_residue_repeats
                query_token_positions = template_slice.query_token_positions
                residue_starts = struc.get_residue_starts(template_atom_array)

                # Residue names and unmask corresponding tokens/template positions
                res_names[template_idx, query_token_positions] = np.repeat(
                    template_atom_array[residue_starts].res_name,
                    template_residue_repeats,
                )

                # Pseudo beta atom coordinates
                is_gly = template_atom_array.res_name == "GLY"
                is_ca = template_atom_array.atom_name == "CA"
                is_cb = template_atom_array.atom_name == "CB"
                is_pseudo_beta_atom = (is_gly & is_ca) | (~is_gly & is_cb)

                # Skip templates that have non-canonical residues with "non-canonical
                # C-beta atoms" or missing C-beta atoms
                # TODO: add handling for these
                # cases to be able to include them or skip them in a way that allows for
                # retaining the correct number of sampled templates
                if sum(is_pseudo_beta_atom) != len(residue_starts):
                    logger.debug(
                        "Skipping template with non-canonical/missing C-beta atoms."
                    )
                    continue

                pseudo_beta_atom_coords[template_idx, query_token_positions, :] = (
                    np.repeat(
                        template_atom_array[is_pseudo_beta_atom].coord,
                        template_residue_repeats,
                        axis=0,
                    )
                )

                # Frame (N, CA, C) atom coordinates
                is_n = template_atom_array.atom_name == "N"
                is_ca = template_atom_array.atom_name == "CA"
                is_c = template_atom_array.atom_name == "C"
                frame_atom_coords[template_idx, query_token_positions, 0, :] = (
                    np.repeat(
                        template_atom_array[is_n].coord,
                        template_residue_repeats,
                        axis=0,
                    )
                )
                frame_atom_coords[template_idx, query_token_positions, 1, :] = (
                    np.repeat(
                        template_atom_array[is_ca].coord,
                        template_residue_repeats,
                        axis=0,
                    )
                )
                frame_atom_coords[template_idx, query_token_positions, 2, :] = (
                    np.repeat(
                        template_atom_array[is_c].coord,
                        template_residue_repeats,
                        axis=0,
                    )
                )

                template_idx += 1
            except Exception as e:
                tb = traceback.format_exc()
                logger.warning(
                    "-" * 40
                    + "\n"
                    + f"Skipping template with exception: {str(e)}\n"
                    + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                    + "-" * 40
                )
                continue

    return OF3TemplateFeaturePrecursor(
        res_names=res_names,
        pseudo_beta_atom_coords=pseudo_beta_atom_coords,
        frame_atom_coords=frame_atom_coords,
    )
