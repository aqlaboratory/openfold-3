"""This module contains pipelines for processing structural features on-the-fly."""

from pathlib import Path
from typing import NamedTuple

from biotite.structure import AtomArray
from rdkit.Chem import Mol

from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure.cropping import apply_crop
from openfold3.core.data.primitives.structure.duplicate_expansion import (
    expand_duplicate_chains,
)
from openfold3.core.data.primitives.structure.ligand import assign_reference_molecules
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array


class AF3ProcessedTargetStructure(NamedTuple):
    """Processed target structure containing coordinates and RDKit conformers."""

    atom_array_cropped: AtomArray # cropped and tokenized atom array
    atom_array_expanded: AtomArray # expanded atom array for chain permutation alignment
    mol_objects: list[Mol] # RDKit Mol objects for reference conformer features


# TODO: make usage of mol_objects list more clear
def process_target_structure_af3(
    target_path: Path,
    pdb_id: str,
    ccd_sdfs_path: Path,
    crop_weights: dict[str, float],
    token_budget: int,
    preferred_chain_or_interface: str,
) -> tuple[AtomArray, AtomArray, list[Mol]]:
    """AF3 pipeline for processing target structure into AtomArrays.

    Args:
        target_path (Path):
            Path to the directory containing the directories of target structure files.
        pdb_id (str):
            PDB ID of the target structure.
        ccd_sdfs_path (Path):
            Path to the directory containing SDF files for all standard components in
            the chemical_component_dictionary.
        crop_weights (dict[str, float]):
            Dataset-specific weights for each cropping strategy.
        token_budget (int):
            Crop size.
        preferred_chain_or_interface (str):
            Sampled preferred chain or interface to sample the crop around.

    Returns:
        tuple[AtomArray, AtomArray, list[Mol]]:
            Tuple of the following objects:
            - Atoms inside the crop.
            - Ground truth atoms expanded for chain permutation alignment.
            - flat list of RDKit Mol objects for reference conformer features.
            
    """
    target_dir = target_path / pdb_id

    # Parse target structure
    structure = parse_mmcif(
        file_path=target_dir / pdb_id.with_suffix(".bcif"),
        expand_bioassembly=False,
        include_bonds=True,
    )
    atom_array = structure.atom_array

    # Tokenize
    tokenize_atom_array(atom_array=atom_array)

    # Crop and pad
    apply_crop(atom_array, token_budget, preferred_chain_or_interface, crop_weights)

    # Compute reference conformers
    mol_list, atom_array = assign_reference_molecules(
        atom_array, ccd_sdfs_path, target_dir / "special_ligand_sdfs"
    )

    # Expand duplicate chains
    atom_array, atom_array_expanded = expand_duplicate_chains(atom_array)
    
    return atom_array, atom_array_expanded, mol_list
