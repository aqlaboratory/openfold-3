"""This module contains pipelines for processing structural features on-the-fly."""

from pathlib import Path
from typing import Literal

from biotite.structure import AtomArray

from openfold3.core.data.io.structure.cif import parse_target_structure
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.cropping import apply_crop
from openfold3.core.data.primitives.structure.duplicate_expansion import (
    expand_duplicate_chains,
)
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array


@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc")
def process_target_structure_af3(
    target_structures_directory: Path,
    pdb_id: str,
    crop_config: dict,
    preferred_chain_or_interface: str | list[str, str] | None,
    structure_format: Literal["cif", "bcif", "pkl"],
    return_full_atom_array: bool,
) -> tuple[dict[str, AtomArray]]:
    """AF3 pipeline for processing target structure into AtomArrays.

    Args:
        target_structures_directory (Path):
            Path to the directory containing the directories of target structure files.
        pdb_id (str):
            PDB ID of the target structure.
        crop_config (dict):
            Crop configuration dictionary, containing the following keys:
            - apply_crop (bool): Whether to apply cropping.
            - token_budget (int): Number of tokens to sample.
            - crop_weights (dict): Weights of different crop strategies.
        preferred_chain_or_interface (str | list[str, str] | None):
            Sampled preferred chain or interface to sample the crop around.
        structure_format (Literal["cif", "bcif", "pkl"]):
            File extension of the target structure. One of "cif", "bcif", or "pkl".
        return_full_atom_array (bool):
            Whether to return the full, uncropped atom array.

    Returns:
        Tuple containing a dict of AtomArrays, including:
            - Atoms inside the crop.
            - Ground truth atoms expanded for chain permutation alignment.
            - Full atom array - optional.
        and the number of tokens in the target structure AtomArray.
    """
    target_structure_data = {}

    # Parse target structure
    atom_array = parse_target_structure(
        target_structures_directory, pdb_id, structure_format
    )

    # Tokenize
    tokenize_atom_array(atom_array=atom_array)

    # Create crop mask
    atom_array_cropped = apply_crop(
        atom_array=atom_array,
        crop_config=crop_config,
        preferred_chain_or_interface=preferred_chain_or_interface,
    )

    # Expand duplicate chains
    atom_array_gt = expand_duplicate_chains(atom_array)

    target_structure_data["atom_array_cropped"] = atom_array_cropped
    target_structure_data["atom_array_gt"] = atom_array_gt

    if return_full_atom_array:
        target_structure_data["atom_array"] = atom_array

    # The number of tokens is used in downstream parts of the data pipeline
    # if cropping was done, we want to set the number of tokens to the token budget
    if crop_config["apply_crop"]:
        n_tokens = crop_config["token_budget"]
    # otherwise set it to the number of tokens in the target structure
    else:
        n_tokens = len(set(atom_array.token_id))

    return target_structure_data, n_tokens
