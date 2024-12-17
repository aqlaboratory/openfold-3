"""This module contains pipelines for processing structural features on-the-fly."""

import logging
import time
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
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
from openfold3.core.data.resources.residues import (
    STANDARD_NUCLEIC_ACID_RESIDUES,
    STANDARD_PROTEIN_RESIDUES_3,
    MoleculeType,
)

logger = logging.getLogger(__name__)


class ProcessedTargetStructure(NamedTuple):
    atom_array_gt: AtomArray
    crop_strategy: str


# Imported from new unresolved fix
def set_residue_hetero_values(atom_array: AtomArray) -> None:
    """Sets the "hetero" annotation in the AtomArray based on the residue names.

    This function sets the "hetero" annotation in the AtomArray based on the residue
    names. If the residue name is in the list of standard residues for the respective
    molecule type, the "hetero" annotation is set to False, otherwise it is set to True.

    Args:
        atom_array:
            AtomArray containing the structure to set the "hetero" annotation for.

    Returns:
        None, the "hetero" annotation is modified in-place.
    """
    protein_mask = atom_array.molecule_type_id == MoleculeType.PROTEIN
    if protein_mask.any():
        in_standard_protein_residues = np.isin(
            atom_array.res_name, STANDARD_PROTEIN_RESIDUES_3
        )
    else:
        in_standard_protein_residues = np.zeros(len(atom_array), dtype=bool)

    rna_mask = atom_array.molecule_type_id == MoleculeType.RNA
    if rna_mask.any():
        in_standard_rna_residues = np.isin(
            atom_array.res_name, STANDARD_NUCLEIC_ACID_RESIDUES
        )
    else:
        in_standard_rna_residues = np.zeros(len(atom_array), dtype=bool)

    dna_mask = atom_array.molecule_type_id == MoleculeType.DNA
    if dna_mask.any():
        in_standard_dna_residues = np.isin(
            atom_array.res_name, STANDARD_NUCLEIC_ACID_RESIDUES
        )
    else:
        in_standard_dna_residues = np.zeros(len(atom_array), dtype=bool)

    atom_array.hetero[:] = True
    atom_array.hetero[
        (protein_mask & in_standard_protein_residues)
        | (rna_mask & in_standard_rna_residues)
        | (dna_mask & in_standard_dna_residues)
    ] = False


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

    assert (atom_array.occupancy != 0).any(), f"No non-zero occupancy atoms in {pdb_id}"

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

    # TODO: put add_token_positions here

    start = time.perf_counter()
    # Add labels to identify symmetric mols in permutation alignment
    atom_array = assign_mol_permutation_ids(atom_array)
    end = time.perf_counter()
    logger.debug(f"Time to assign mol permutation ids: {end - start:.2f}s")

    # NOTE: could move this to conformer processing
    # TODO: make this logic more robust (potentially by reverting treating multi-residue
    # ligands as single compnents)
    # Necessary for multi-residue ligands (which can have duplicated atom names) to
    # identify which atom names ended up in the crop.
    atom_array = assign_uniquified_atom_names(atom_array)

    assert (
        atom_array.occupancy[atom_array.crop_mask] != 0
    ).any(), f"No non-zero occupancy atoms in crop of {pdb_id}"

    return ProcessedTargetStructure(
        atom_array_gt=atom_array,
        crop_strategy=crop_strategy,
    )
