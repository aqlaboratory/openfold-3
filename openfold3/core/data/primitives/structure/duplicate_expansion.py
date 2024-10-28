"""
Primitives for expanding duplicates chains in an assembly for chain permutation
alignment.
"""

import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.primitives.quality_control.logging_utils import log_runtime
from openfold3.core.data.primitives.structure.labels import (
    assign_atom_indices,
    remove_atom_indices,
)
from openfold3.core.data.resources.residues import MoleculeType


@log_runtime(name="runtime-target-structure-proc-expand")
def expand_duplicate_chains(
    atom_array: AtomArray,
) -> AtomArray:
    """Finds subset of atoms in the assembly needed for permutation alignment.

    Need to be called after tokenization and cropping.

    Args:
        atom_array (AtomArray):
            Atom array of the assembly.

    Returns:
        AtomArray:
            Ground truth atoms expanded for chain permutation alignment.
    """
    # Assign atom indices and duplicate chains mask
    assign_atom_indices(atom_array)
    duplicate_chains_mask = np.repeat(False, len(atom_array))

    # Get subset of atom array that falls into the crop - exclude ligands
    # TODO add logic for ligands and covalently modified chains
    atom_array_in_crop = atom_array[
        (atom_array.crop_mask) & (atom_array.molecule_type_id != MoleculeType.LIGAND)
    ]
    # Get subset of atom array that includes all atoms of all chains that have at least
    # one atom in the crop
    # TODO determine if this is necessary
    # atom_array_in_crop_full_chains = atom_array[
    #     np.isin(
    #         atom_array.chain_id_renumbered,
    #         list(set(atom_array_in_crop.chain_id_renumbered)),
    #     )
    # ]

    # Get all entity IDs in the crop
    entity_ids_in_crop = set(atom_array_in_crop.entity_id)

    # Create duplicate chain mask
    for entity_id in entity_ids_in_crop:
        # Get atoms in chains with given entity ID
        atoms_in_crop_in_entity = atom_array_in_crop[
            atom_array_in_crop.entity_id == entity_id
        ]

        # Get residue IDs for residues
        # in chains with given entity ID
        resids_in_crop_in_entity = list(set(atoms_in_crop_in_entity.res_id))

        # From the atoms of chains with any atom in the crop, get atoms
        # in residues with same residue ID as residues
        # in chains with given entity ID
        atoms_in_assembly_in_entity = atom_array[atom_array.entity_id == entity_id]
        atomids_in_assembly_in_crop_in_entity = atoms_in_assembly_in_entity[
            np.isin(atoms_in_assembly_in_entity.res_id, resids_in_crop_in_entity)
        ]._atom_idx

        # Unmask corresponding atoms in the whole assembly
        duplicate_chains_mask[
            np.isin(atom_array._atom_idx, atomids_in_assembly_in_crop_in_entity)
        ] = True

    remove_atom_indices(atom_array)

    return atom_array[duplicate_chains_mask]
