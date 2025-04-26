# TODO add license

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.primitives.featurization.structure import get_token_starts
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.labels import (
    assign_atom_indices,
    assign_residue_indices,
    remove_atom_indices,
    remove_residue_indices,
)
from openfold3.core.data.resources.residues import (
    PEPTIDE_BOND_ATOMS,
    PHOSPHODIESTER_BOND_ATOMS,
    STANDARD_RESIDUES_3,
    TOKEN_CENTER_ATOMS,
    MoleculeType,
)


@log_runtime_memory(runtime_dict_key="runtime-add-token-pos")
def add_token_positions(atom_array: AtomArray) -> None:
    """Adds token_position annotation to the input atom array.

    Args:
        atom_array (AtomArray):
            AtomArray of the input assembly.
    """
    # Create token ID to token position mapping
    token_starts = get_token_starts(atom_array)
    token_positions_map = {
        token: position
        for position, token in enumerate(atom_array[token_starts].token_id)
    }

    # Map token ID to token position for all atoms and add annotation
    token_positions = np.vectorize(token_positions_map.get)(atom_array.token_id)
    atom_array.set_annotation("token_position", token_positions)


def find_canonical_residue_in_polymer_start_ids(
    atom_array: AtomArray, return_mask: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Finds atom IDs canonical residues in polymers.

    Args:
        atom_array (AtomArray):
            AtomArray in which to find the starting atom IDs of canonical residues in
            polymers.
        return_mask (bool, optional):
            Whether to return a mask indicating which atom belongs to a canonical
            residue in a polymer. Defaults to True.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]:
            If return_mask is True, returns a tuple containing the starting atom IDs of
            canonical residues in polymers and a mask indicating which atoms belong to
            those residues. Otherwise, returns only the starting atom IDs.
    """
    is_l_atom = atom_array.molecule_type_id == MoleculeType.LIGAND
    is_cr_atom = np.isin(atom_array.res_name, STANDARD_RESIDUES_3)
    is_crp_atom = ~is_l_atom & is_cr_atom
    crp_atom_ids = atom_array._atom_idx[is_crp_atom]
    crp_token_start_ids = np.unique(
        struc.get_residue_starts_for(atom_array, crp_atom_ids)
    )
    if return_mask:
        return crp_token_start_ids, is_crp_atom
    else:
        return crp_token_start_ids


def find_modified_residue_atom_ids(atom_array: AtomArray) -> np.ndarray:
    """Finds the IDs of all atoms in modified residues.

    A residue is considered modified if it is connected to any other residue via a
    non-peptide bond or a non-phospho-diester bond.

    Args:
        atom_array (AtomArray):
            AtomArray in which to find the atom IDs of modified residues.

    Returns:
        np.ndarray:
            IDs of all atoms in modified residues.
    """
    bonds = atom_array.bonds.as_array()[:, :2]
    has_different_chain_id = (
        atom_array.chain_id[bonds[:, 0]] != atom_array.chain_id[bonds[:, 1]]
    )
    has_different_res_id = (
        atom_array.res_id[bonds[:, 0]] != atom_array.res_id[bonds[:, 1]]
    )
    not_peptide_bond = ~(
        np.isin(atom_array.atom_name[bonds[:, 0]], PEPTIDE_BOND_ATOMS)
        & np.isin(atom_array.atom_name[bonds[:, 1]], PEPTIDE_BOND_ATOMS)
    )
    not_phospho_diester_bond = ~(
        np.isin(atom_array.atom_name[bonds[:, 0]], PHOSPHODIESTER_BOND_ATOMS)
        & np.isin(atom_array.atom_name[bonds[:, 1]], PHOSPHODIESTER_BOND_ATOMS)
    )
    mod_crp_bonds = bonds[
        (has_different_chain_id | has_different_res_id)
        & (not_peptide_bond & not_phospho_diester_bond)
    ]

    # Get corresponding canonical residue atoms
    mod_crp_atom_ids = np.unique(mod_crp_bonds[:, :2].flatten())
    mod_crp_atoms = atom_array[np.isin(atom_array._atom_idx, mod_crp_atom_ids)]

    # Get corresponding canonical residue token starts
    atomized_crp_token_ids = atom_array[
        np.isin(
            atom_array._residue_idx,
            mod_crp_atoms._residue_idx,
        )
    ]._atom_idx

    return atomized_crp_token_ids


@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc-token")
def tokenize_atom_array(atom_array: AtomArray):
    """Creates token id, token center atom, and is_atomized annotations for atom array.

    Tokenizes the input atom array according to section 2.6. in the AF3 SI. The
    tokenization is added to the input atom array as a 'token_id' annotation alongside
    'token_center_atom' and 'is_atomized' annotations.

    High-level logic of the tokenizer:
        1. Get atoms in canonical residues in polymers
        2. Get atoms in small molecule ligands, non-canonical residues in polymers and
        amino acid or nucleotide small molecule ligands
        3. Get atoms in canonical residues in polymer that are modified
            We consider a residue to be modified if it is connected to any other residue
            via a non-peptide bond for proteins or a non-phospho-diester bond for
            nucleic acids.
        4. Tokenize residues with any atoms from
            - set 2. per atom
            - set 3. per atom
            - the difference of sets 1.-3. per residue

    Args:
        atom_array (AtomArray):
            biotite atom array of the first bioassembly of a PDB entry

    Returns:
        None
    """
    assign_atom_indices(atom_array)
    assign_residue_indices(atom_array)

    # 1. Find token start IDs of canonical residues in-polymer (CRP), excluding amino
    #    acid and nucleotide small molecule ligands
    crp_token_start_ids, is_crp_atom = find_canonical_residue_in_polymer_start_ids(
        atom_array
    )

    # 2. Find atom-token start ids: includes small molecule ligands, non-canonical
    #    residues and amino acid or nucleotide small molecule ligands
    atom_token_ids = atom_array._atom_idx[~is_crp_atom]

    # 3. Find canonical residues in-polymer (CRP) bonded to other chemical species via
    #    non-canonical bonds, i.e. via bonds that are not peptide or phospho-diester
    atomized_crp_token_ids = find_modified_residue_atom_ids(atom_array)

    # Remove the corresponding residue token start ids
    mod_crp_token_start_ids = np.unique(
        struc.get_residue_starts_for(atom_array, atomized_crp_token_ids)
    )
    crp_token_start_ids = crp_token_start_ids[
        ~np.isin(crp_token_start_ids, mod_crp_token_start_ids)
    ]

    # Combine all token start ids
    all_token_start_ids = np.unique(
        np.concatenate(
            [
                crp_token_start_ids,
                atom_token_ids,
                atomized_crp_token_ids,
            ]
        )
    )

    # Add is_atomized annotation
    n_atoms = len(atom_array)
    is_atomized = np.repeat(False, n_atoms)
    is_atomized[np.concatenate([atom_token_ids, atomized_crp_token_ids])] = True
    atom_array.set_annotation("is_atomized", is_atomized)

    # Create token index
    token_id_repeats = np.diff(np.append(all_token_start_ids, n_atoms))
    token_ids_per_atom = np.repeat(np.arange(len(token_id_repeats)), token_id_repeats)
    atom_array.set_annotation("token_id", token_ids_per_atom)

    # Create token center atom annotation
    token_center_atoms = np.repeat(True, n_atoms)
    token_center_atoms[is_crp_atom] = np.isin(
        atom_array[is_crp_atom].atom_name, TOKEN_CENTER_ATOMS
    )
    # Edit token center atoms for covalently modified residues
    token_center_atoms[atomized_crp_token_ids] = True
    atom_array.set_annotation("token_center_atom", token_center_atoms)

    # Remove temporary atom & residue indices
    remove_atom_indices(atom_array)
    remove_residue_indices(atom_array)

    # Add token_position annotation
    add_token_positions(atom_array)


def get_token_count(atom_array: AtomArray) -> int:
    """Get the number of tokens in the input atom array.

    If the input atom array is not yet tokenized, the function will tokenize it.

    Args:
        atom_array (AtomArray):
            AtomArray of the input assembly.

    Returns:
        int: Number of tokens in the input atom array.
    """
    if "token_id" not in atom_array.get_annotation_categories():
        tokenize_atom_array(atom_array)

    return len(np.unique(atom_array.token_id))
