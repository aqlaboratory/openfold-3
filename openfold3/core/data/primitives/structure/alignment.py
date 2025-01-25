"""Primitives for aligning atom arrays."""

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.primitives.structure.interface import (
    get_query_interface_atom_pair_idxs,
)


# TODO: improve docstring
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
    query_atom_array: AtomArray,
    target_atom_array: AtomArray,
    distance_thresholds: list[float],
) -> dict[float, bool]:
    """Returns whether two AtomArrays have atoms within given distance thresholds.

    Args:
        query_atom_array (AtomArray):
            First AtomArray to compare.
        target_atom_array (AtomArray):
            Second AtomArray to compare.
        distance_thresholds (list[float]):
            List of distance thresholds.

    Returns:
        dict[float, bool]:
            Dictionary mapping distance thresholds to whether the two AtomArrays
            have any atoms within the corresponding distance.
    """

    distance_clash_map = {}
    for d in distance_thresholds:
        atom_pair_idxs = get_query_interface_atom_pair_idxs(
            query_atom_array=query_atom_array,
            target_atom_array=target_atom_array,
            distance_threshold=d,
        )
        distance_clash_map[d] = atom_pair_idxs.shape[0] != 0

    return distance_clash_map
