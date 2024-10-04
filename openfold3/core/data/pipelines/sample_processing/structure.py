"""This module contains pipelines for processing structural features on-the-fly."""

import pickle
from pathlib import Path
from typing import Literal

from biotite.structure import AtomArray

from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure.cropping import apply_crop
from openfold3.core.data.primitives.structure.duplicate_expansion import (
    expand_duplicate_chains,
)
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array


def process_target_structure_af3(
    target_structures_directory: Path,
    pdb_id: str,
    crop_weights: dict[str, float],
    token_budget: int,
    preferred_chain_or_interface: str,
    structure_format: Literal["cif", "bcif", "pkl"],
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

    Returns:
        tuple[AtomArray, AtomArray]:
            Tuple of two atom arrays:
            - Atoms inside the crop.
            - Ground truth atoms expanded for chain permutation alignment.
    """
    # Parse target structure
    target_file = target_structures_directory / pdb_id / f"{pdb_id}.{structure_format}"

    if structure_format == "pkl":
        with open(target_file, "rb") as f:
            atom_array = pickle.load(f)
    elif structure_format in ["cif", "bcif"]:
        _, atom_array = parse_mmcif(
            file_path=target_file,
            expand_bioassembly=False,
            include_bonds=True,
            renumber_chain_ids=False,
        )
    else:
        raise ValueError(f"Invalid structure format: {structure_format}")

    # Tokenize
    tokenize_atom_array(atom_array=atom_array)

    # Crop and pad
    apply_crop(atom_array, token_budget, preferred_chain_or_interface, crop_weights)

    # Expand duplicate chains
    return expand_duplicate_chains(atom_array)
