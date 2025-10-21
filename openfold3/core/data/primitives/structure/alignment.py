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

"""Primitives for aligning atom arrays."""

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.primitives.structure.interface import (
    get_query_interface_atom_pair_idxs,
)


# TODO add docstring and comments
def extend_chain_map_via_alignment(
    target_atom_array: AtomArray,
    source_atom_array: AtomArray,
    chain_map: dict[str, str],
    transform_array: np.ndarray,
):
    affine_transformation = struc.AffineTransformation(
        center_translation=np.zeros(3),
        rotation=transform_array[:3, :3],
        target_translation=transform_array[:3, 3],
    )
    unaligned_target_chain_ids = set(np.unique(target_atom_array.chain_id)) - set(
        chain_map.values()
    )
    unaligned_source_chain_ids = set(np.unique(source_atom_array.chain_id)) - set(
        chain_map.keys()
    )
    target_atom_array_aligned = affine_transformation.apply(target_atom_array)

    chain_map_update = {}
    for source_chain_id in unaligned_source_chain_ids:
        source_chain = source_atom_array[source_atom_array.chain_id == source_chain_id]
        source_backbone = source_chain[
            np.isin(source_chain.atom_name, ["N", "CA", "C"])
        ]
        source_backbone_is_resolved = source_backbone.occupancy > 0
        target_chain_id_to_rmsd = {}
        for target_chain_id in unaligned_target_chain_ids.copy():
            target_chain = target_atom_array_aligned[
                target_atom_array_aligned.chain_id == target_chain_id
            ]
            target_backbone = target_chain[
                np.isin(target_chain.atom_name, ["N", "CA", "C"])
            ]
            if len(source_backbone) == len(target_backbone):
                target_chain_id_to_rmsd[target_chain_id] = struc.rmsd(
                    source_backbone[source_backbone_is_resolved],
                    target_backbone[source_backbone_is_resolved],
                )

        if len(target_chain_id_to_rmsd) > 0:
            target_chain_id_match = min(
                target_chain_id_to_rmsd, key=target_chain_id_to_rmsd.get
            )
            chain_map_update[source_chain_id] = target_chain_id_match
            unaligned_target_chain_ids.remove(target_chain_id_match)

    return {**chain_map, **chain_map_update}


# TODO: improve docstring and refactor for generality
def coalign_atom_arrays(
    fixed: AtomArray,
    mobile: AtomArray,
    comobile: AtomArray,
    alignment_mask_atom_names: list[str],
    mobile_distance_atom_names: list[str],
    distance_threshold: float,
) -> AtomArray:
    """Pocket aligns chains of the comobile AtomArray to the fixed AtomArray.

    Alignment is performed separately for each chain in the comobile AtomArray.

    Args:
        fixed (AtomArray):
            AtomArray that serves as the reference for the alignment.
        mobile (AtomArray):
            AtomArray that is to be aligned to the fixed AtomArray.
        comobile (AtomArray):
            AtomArray to which the mobile -> fixed transformation is applied.
        alignment_mask_atom_names (list[str]):
            Atom names to use for the alignment mask. Only atoms with these atom
            names are considered in the mobile -> fixed alignment.
        mobile_distance_atom_names (list[str]):
            Atom names to use for distance calculations in the mobile AtomArray
            to all atoms in the comobile AtomArray.
        distance_threshold (float):
            Distance threshold within which to include residues from the fixed and
            mobile AtomArrays in the alignment.

    Returns:
        AtomArray:
            The concatenated AtomArray containing the aligned chains from the
            comobile AtomArray.
    """
    comobile_aligned = []
    for comobile_chain_id in np.unique(comobile.chain_id):
        comobile_chain = comobile[comobile.chain_id == comobile_chain_id]

        # Subset to permitted mobile atoms
        mobile_subset = mobile[np.isin(mobile.atom_name, mobile_distance_atom_names)]

        # Find atoms within the distance threshold - these are atoms in the residues
        # that form the pocket in the mobile AtomArray
        atom_pair_idxs = get_query_interface_atom_pair_idxs(
            query_atom_array=comobile_chain,
            target_atom_array=mobile_subset,
            distance_threshold=distance_threshold,
        )

        # Find associated mobile and fixed pocket residues
        mobile_pocket_atoms = mobile_subset[np.unique(atom_pair_idxs[:, 1])]
        mobile_pocket_residue_start_atoms = mobile_pocket_atoms[
            struc.get_residue_starts(mobile_pocket_atoms)
        ]
        mobile_pocket_residues = mobile[
            np.isin(mobile.res_id, mobile_pocket_residue_start_atoms.res_id)
            & np.isin(mobile.chain_id, mobile_pocket_residue_start_atoms.chain_id)
        ]
        fixed_pocket_residues = fixed[
            np.isin(fixed.res_id, mobile_pocket_residue_start_atoms.res_id)
            & np.isin(fixed.chain_id, mobile_pocket_residue_start_atoms.chain_id)
        ]

        # Get subset of mobile and fixed atoms in the pocket to be aligned
        mobile_pocket_align_subset = mobile_pocket_residues[
            np.isin(mobile_pocket_residues.atom_name, alignment_mask_atom_names)
        ]
        fixed_pocket_align_subset = fixed_pocket_residues[
            np.isin(fixed_pocket_residues.atom_name, alignment_mask_atom_names)
        ]

        # Get resolved mask
        resolved_mask = mobile_pocket_align_subset.occupancy == 1.0

        # Align
        _, transformation = struc.superimpose(
            fixed=fixed_pocket_align_subset,
            mobile=mobile_pocket_align_subset,
            atom_mask=resolved_mask,
        )

        # Apply transformation to the comobile chain
        comobile_aligned.append(transformation.apply(comobile_chain))

    return struc.concatenate(comobile_aligned)


def calculate_distance_clash_map(
    atom_array: AtomArray,
    distance_thresholds: list[float],
) -> dict[float, bool]:
    """Finds if an atom array has any chain pair within a list of distances.

    Only checks for cross-chain clashes.

    Args:
        atom_array (AtomArray):
            Atom array to check for clashes.
        distance_thresholds (list[float]):
            List of distance thresholds.

    Returns:
        dict[float, bool]:
            Dictionary mapping distance thresholds to whether the two AtomArrays
            have any atoms within the corresponding distance.
    """
    chain_ids = np.unique(atom_array.chain_id)
    distance_clash_map = {}
    for d in distance_thresholds:
        # Monomers
        if len(chain_ids) == 1:
            distance_clash_map[d] = False

        # Multimers
        else:
            for chain_id in chain_ids:
                atom_pair_idxs = get_query_interface_atom_pair_idxs(
                    query_atom_array=atom_array[atom_array.chain_id == chain_id],
                    target_atom_array=atom_array[atom_array.chain_id != chain_id],
                    distance_threshold=d,
                )
                # Check if any atom pairs were found at d
                if atom_pair_idxs is None:
                    is_clashing = False
                else:
                    is_clashing = atom_pair_idxs.shape[0] != 0
                distance_clash_map[d] = is_clashing
                if is_clashing:
                    break
                else:
                    continue

    return distance_clash_map
