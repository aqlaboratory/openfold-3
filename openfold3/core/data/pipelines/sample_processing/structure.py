"""This module contains pipelines for processing structural features on-the-fly."""

import logging
from pathlib import Path
from typing import Literal, NamedTuple

from biotite.structure import AtomArray

from openfold3.core.data.io.structure.cif import parse_target_structure
from openfold3.core.data.primitives.permutation.mol_labels import (
    assign_mol_permutation_ids,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.cleanup import filter_bonds
from openfold3.core.data.primitives.structure.cropping import sample_crop_and_set_mask
from openfold3.core.data.primitives.structure.labels import (
    assign_component_ids_from_metadata,
    assign_uniquified_atom_names,
)
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array

logger = logging.getLogger(__name__)


class ProcessedTargetStructure(NamedTuple):
    atom_array_gt: AtomArray
    crop_strategy: str


# TODO: Update docstring
@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc")
def process_target_structure_af3(
    target_structures_directory: Path,
    pdb_id: str,
    crop_weights: dict[str, float],
    token_budget: int,
    preferred_chain_or_interface: str,
    structure_format: Literal["cif", "bcif", "pkl"],
    per_chain_metadata: dict[str, dict[str, str]],
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
        structure_format (Literal["pkl", "npz"]):
            File extension of the target structure. Only "pkl" and "npz" are currently
            supported.
        per_chain_metadata (dict[str, dict[str, str]]):
            Metadata for each chain in the target structure, obtained from the dataset
            cache.

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

    # Mark individual components (which get unique conformers)
    assign_component_ids_from_metadata(atom_array, per_chain_metadata)

    # Remove bonds not following AF3 criteria, but keep intra-residue bonds and
    # consecutive inter-residue bonds for now (necessary for molecule detection in
    # permutation IDs)
    filter_bonds(
        atom_array=atom_array,
        keep_consecutive=True,
        keep_polymer_ligand=True,
        keep_ligand_ligand=True,
        remove_larger_than=2.4,
        mask_intra_component=True,
    )

    # Tokenize
    tokenize_atom_array(atom_array=atom_array)

    # Set the crop_mask attribute of the atom_array, marking which atoms are in the crop
    # and which aren't, and get the crop strategy that was used
    crop_strategy = sample_crop_and_set_mask(
        atom_array,
        token_budget,
        preferred_chain_or_interface,
        crop_weights,
    )

    # Add labels to identify symmetric mols in permutation alignment
    atom_array = assign_mol_permutation_ids(atom_array)

    # NOTE: could move this to conformer processing
    # TODO: make this logic more robust (potentially by reverting treating multi-residue
    # ligands as single compnents)
    # Necessary for multi-residue ligands (which can have duplicated atom names) to
    # identify which atom names ended up in the crop.
    atom_array = assign_uniquified_atom_names(atom_array)

    return ProcessedTargetStructure(
        atom_array_gt=atom_array,
        crop_strategy=crop_strategy,
    )
