"""This module contains pipelines for processing structural features on-the-fly."""

from pathlib import Path

from biotite.structure import AtomArray

from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure.cropping import apply_crop
from openfold3.core.data.primitives.structure.duplicate_expansion import (
    expand_duplicate_chains,
)
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array


def process_target_structure_af3(
    target_path: Path,
    pdb_id: str,
    crop_weights: dict[str, float],
    token_budget: int,
    preferred_chain_or_interface: str,
) -> tuple[AtomArray, AtomArray]:
    """AF3 pipeline for processing target structure into AtomArrays.

    Args:
        target_path (Path):
            Path to the directory containing the directories of target structure files.
        pdb_id (str):
            PDB ID of the target structure.
        crop_weights (dict[str, float]):
            Dataset-specific weights for each cropping strategy.
        token_budget (int):
            Crop size.
        preferred_chain_or_interface (str):
            Sampled preferred chain or interface to sample the crop around.

    Returns:
        tuple[AtomArray, AtomArray]:
            Tuple of two atom arrays:
            - Atoms inside the crop.
            - Ground truth atoms expanded for chain permutation alignment.
    """
    # Parse target structure
    structure = parse_mmcif(
        file_path=target_path + pdb_id + pdb_id + ".bcif",
        expand_bioassembly=True,
        include_bonds=True,
    )
    atom_array = structure.atom_array

    # Tokenize
    tokenize_atom_array(atom_array=atom_array)

    # Crop and pad
    apply_crop(atom_array, token_budget, preferred_chain_or_interface, crop_weights)

    # Expand duplicate chains
    return expand_duplicate_chains(atom_array)
