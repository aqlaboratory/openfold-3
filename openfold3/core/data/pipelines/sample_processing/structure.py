"""This module contains pipelines for processing structural features on-the-fly."""

import logging
from pathlib import Path
from typing import Literal, NamedTuple

from biotite.structure import AtomArray

from openfold3.core.data.io.structure.cif import parse_target_structure
from openfold3.core.data.primitives.caches.format import PreprocessingChainData
from openfold3.core.data.primitives.permutation.mol_labels import (
    assign_mol_permutation_ids,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.cleanup import filter_bonds
from openfold3.core.data.primitives.structure.component import (
    assign_component_ids_from_metadata,
)
from openfold3.core.data.primitives.structure.cropping import sample_crop_and_set_mask
from openfold3.core.data.primitives.structure.labels import (
    assign_uniquified_atom_names,
)
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array

logger = logging.getLogger(__name__)


class ProcessedTargetStructure(NamedTuple):
    atom_array_gt: AtomArray
    crop_strategy: str
    n_tokens: int


# TODO: Update docstring
@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc")
def process_target_structure_af3(
    target_structures_directory: Path,
    pdb_id: str,
    apply_crop: bool,
    crop_config: dict,
    preferred_chain_or_interface: str | list[str, str] | None,
    structure_format: Literal["pkl", "npz"],
    per_chain_metadata: dict[str, PreprocessingChainData],
) -> ProcessedTargetStructure:
    """AF3 pipeline for processing target structure into AtomArrays.

    Args:
        target_structures_directory (Path):
            Path to the directory containing the directories of target structure files.
        pdb_id (str):
            PDB ID of the target structure.
        apply_crop (bool):
            Whether to apply cropping.
        crop_config (dict):
            Crop configuration dictionary, containing the following keys:
            - token_budget (int): Number of tokens to sample.
            - crop_weights (dict): Weights of different crop strategies.
        preferred_chain_or_interface (str | list[str, str] | None):
            Sampled preferred chain or interface to sample the crop around.
        structure_format (Literal["pkl", "npz"]):
            File extension of the target structure. Only "pkl" and "npz" are currently
            supported.
        per_chain_metadata (dict[str, PreprocessingChainData]):
            Metadata for each chain in the target structure, obtained from the dataset
            cache.

    Returns:
        ProcessedTargetStructure:
            A NamedTuple containing the full AtomArray, the crop strategy, and the
            number of tokens in the full AtomArray if apply_crop is False or the slice
            of the AtomArray with crop_mask=True if apply_crop is True.
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
        remove_metal_coordination=True,
        mask_intra_component=True,
    )

    # Tokenize
    tokenize_atom_array(atom_array=atom_array)

    # Set the crop_mask attribute of the atom_array, marking which atoms are in the crop
    # and which aren't, and get the crop strategy that was used
    crop_strategy = sample_crop_and_set_mask(
        atom_array=atom_array,
        apply_crop=apply_crop,
        crop_config=crop_config,
        preferred_chain_or_interface=preferred_chain_or_interface,
    )

    # Add labels to identify symmetric mols in permutation alignment
    atom_array = assign_mol_permutation_ids(atom_array, retokenize=True)

    # NOTE: could move this to conformer processing
    # TODO: make this logic more robust (potentially by reverting treating multi-residue
    # ligands as single compnents)
    # Necessary for multi-residue ligands (which can have duplicated atom names) to
    # identify which atom names ended up in the crop.
    atom_array = assign_uniquified_atom_names(atom_array)

    # The number of tokens is used in downstream parts of the data pipeline
    # if cropping was done, we want to set the number of tokens to the token budget
    if apply_crop:
        n_tokens = crop_config["token_budget"]
    # otherwise set it to the number of tokens in the target structure
    else:
        n_tokens = len(set(atom_array.token_id))

    return ProcessedTargetStructure(
        atom_array_gt=atom_array,
        crop_strategy=crop_strategy,
        n_tokens=n_tokens,
    )
