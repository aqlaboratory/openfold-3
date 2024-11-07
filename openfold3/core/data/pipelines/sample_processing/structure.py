"""This module contains pipelines for processing structural features on-the-fly."""

from pathlib import Path
from typing import Literal

from biotite.structure import AtomArray

from openfold3.core.data.io.structure.cif import parse_target_structure
from openfold3.core.data.primitives.permutation.mol_labels import (
    assign_mol_permutation_ids_and_subset,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.cropping import set_crop_mask
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array


@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc")
def process_target_structure_af3(
    target_structures_directory: Path,
    pdb_id: str,
    crop_weights: dict[str, float],
    token_budget: int,
    preferred_chain_or_interface: str,
    structure_format: Literal["cif", "bcif", "pkl"],
    return_full_atom_array=False,
) -> tuple[AtomArray, AtomArray]:
    """AF3 pipeline for processing target structure into AtomArrays.

    Args:
        target_structures_directory (Path):
            Path to the directory containing the directories of target structure files.
        pdb_id (str):
            PDB ID of the target structure.
        crop_weights (dict[str, float]):
            Dataset-specific weights for each cropping strategy.
        token_budget (int):
            Crop size.
        preferred_chain_or_interface (str):
            Sampled preferred chain or interface to sample the crop around.
        structure_format (Literal["cif", "bcif", "pkl"]):
            File extension of the target structure. One of "cif", "bcif", or "pkl".
        return_full_atom_array (bool):
            Whether to return the full, uncropped atom array.

    Returns:
        tuple[AtomArray, AtomArray]:
            Tuple of two or three atom arrays:
            - Atoms inside the crop.
            - Ground truth atoms expanded for chain permutation alignment.
            (- Full atom array - optional.)
    """
    # Parse target structure
    atom_array = parse_target_structure(
        target_structures_directory, pdb_id, structure_format
    )

    # Tokenize
    tokenize_atom_array(atom_array=atom_array)

    # Set the crop_mask attribute of the atom_array
    set_crop_mask(
        atom_array,
        token_budget,
        preferred_chain_or_interface,
        crop_weights,
    )

    # Make copy before doing any subsetting
    if return_full_atom_array:
        atom_array_full = atom_array.copy()

    # Add labels to identify symmetric mols in permutation alignment and subset GT to
    # only contain molecules symmetric to cropped molecules
    atom_array_gt_w_perm_subset = assign_mol_permutation_ids_and_subset(atom_array)

    # Crop the annotated ground-truth atom array to carry over the permutation labels
    atom_array_w_perm_cropped = atom_array_gt_w_perm_subset[
        atom_array_gt_w_perm_subset.crop_mask
    ]

    if return_full_atom_array:
        return atom_array_w_perm_cropped, atom_array_gt_w_perm_subset, atom_array_full
    else:
        return atom_array_w_perm_cropped, atom_array_gt_w_perm_subset
