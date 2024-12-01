# TODO add license

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray

from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.labels import (
    assign_atom_indices,
    remove_atom_indices,
)
from openfold3.core.data.resources.residues import (
    NUCLEIC_ACID_MAIN_CHAIN_ATOMS,
    PROTEIN_MAIN_CHAIN_ATOMS,
    STANDARD_RESIDUES_3,
    TOKEN_CENTER_ATOMS,
    MoleculeType,
)


@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc-token")
def tokenize_atom_array(atom_array: AtomArray):
    """Creates token id, token center atom, and is_atomized annotations for atom array.

    Tokenizes the input atom array according to section 2.6. in the AF3 SI. The
    tokenization is added to the input atom array as a 'token_id' annotation alongside
    'token_center_atom' and 'is_atomized' annotations.

    High-level logic of the tokenizer:
        1. Get set of standard polymer residues
        2. Create list of token start indices for all standard residues
        3. Find bonds (atom pair) where at least one atom is coming a standard residue
           (st-(n)st)
        4. Subset bonds
            a. from 3. -> covalent bonds with at least one heteroatom
            b. from 3. -> covalent bonds between residues in different chains
            c. from 3. -> covalent bonds between side chains of residues in the same
               chain
        5. Get list of non-heteroatoms in any bond from step 4.a.-4.c.
        6. Get the start indices of residues containing any non-heteroatom from step 5
        7. Remove start indices of step 6 from the list of start indices of step 2
        8. Get the set of atoms that are not part of standard residues (includes both
            "ligands" and non-standard residues) 9. Combine indices for the following:
            a. residue start indices of standard residues that are not atomized (step 7)
            b. atom indices that are not part of standard residues (step 8) c. atom
            indices of standard residues that are atomized (step 6)
        10. Create is_atomized annotation from steps 9.b. and 9.c.
        11. Create token_id annotation per-residue from step 9.a. and per atom from
            steps 9.b. and 9.c. 12. Create token_center_atom annotation per-residue from
            step 9.a. and per atom from steps 9.b. and 9.c.

    High-level logic of the tokenizer:
        1. Get set of standard polymer residues
        2. Create list of token start indices for all standard residues
        3. Find bonds (atom pair) where at least one atom is coming a standard residue
           (st-(n)st)
        4. Subset bonds
            a. from 3. -> covalent bonds with at least one heteroatom
            b. from 3. -> covalent bonds between residues in different chains
            c. from 3. -> covalent bonds between side chains of residues in the same
               chain
        5. Get list of non-heteroatoms in any bond from step 4.a.-4.c.
        6. Get the start indices of residues containing any non-heteroatom from step 5
        7. Remove start indices of step 6 from the list of start indices of step 2
        8. Get the set of atoms that are not part of standard residues (includes both
            "ligands" and non-standard residues) 9. Combine indices for the following:
            a. residue start indices of standard residues that are not atomized (step 7)
            b. atom indices that are not part of standard residues (step 8) c. atom
            indices of standard residues that are atomized (step 6)
        10. Create is_atomized annotation from steps 9.b. and 9.c.
        11. Create token_id annotation per-residue from step 9.a. and per atom from
            steps 9.b. and 9.c. 12. Create token_center_atom annotation per-residue from
            step 9.a. and per atom from steps 9.b. and 9.c.

    Args:
        atom_array (AtomArray):
            biotite atom array of the first bioassembly of a PDB entry

    Returns:
        None
    """
    # Create temporary atom indices
    assign_atom_indices(atom_array)

    # Create auxiliary residue id annotation
    # The auxiliary residue id is used to tokenize covalently modified residues
    # per atom and is removed afterwards
    atom_array.set_annotation(
        "aux_residue_id",
        struc.spread_residue_wise(
            atom_array, np.arange(struc.get_residue_count(atom_array))
        ),
    )

    # Get standard residues
    n_atoms = len(atom_array)

    # Find standard residues, excluding amino acid and nucleotide ligands
    ligand_mask = atom_array.molecule_type_id == MoleculeType.LIGAND
    std_residue_name_mask = np.isin(atom_array.res_name, STANDARD_RESIDUES_3)
    is_standard_residue_atom = ~ligand_mask & std_residue_name_mask

    # Get standard residue atom IDs & token starts
    standard_residue_atom_ids = atom_array._atom_idx[is_standard_residue_atom]
    residue_token_start_ids = np.unique(
        struc.get_residue_starts_for(atom_array, standard_residue_atom_ids)
    )

    # Tokenize modified residues per atom
    # Get bonds
    bondlist = atom_array.bonds.as_array()

    # Find bonds with AT LEAST ONE standard residue atom
    bondlist_1_std = bondlist[
        np.isin(bondlist[:, 0], standard_residue_atom_ids)
        | np.isin(bondlist[:, 1], standard_residue_atom_ids)
    ]
    # # Find bonds which contain EXACLTY TWO standard residue atoms
    # bondlist_2_std = bondlist_1_std[
    #     np.isin(bondlist_1_std[:, 0], standard_residue_atom_ids)
    #     & np.isin(bondlist_1_std[:, 1], standard_residue_atom_ids)
    # ]

    # Find bonds which contain
    # - exactly one heteroatom
    # -- these are standard or non-standard residues with covalent ligands
    is_heteroatom = atom_array.hetero
    is_one_heteroatom = (
        is_heteroatom[bondlist_1_std[:, 0]] ^ is_heteroatom[bondlist_1_std[:, 1]]
    )
    # - two atoms from two residues in different chains
    # -- these bonds are coming from residue pairs where AT LEAST ONE residue is
    #    a standard residue so should include std-std and std-nonstd pairs
    chain_ids = atom_array.chain_id
    is_different_chain = (
        chain_ids[bondlist_1_std[:, 0]] != chain_ids[bondlist_1_std[:, 1]]
    )
    # - two non-heteroatoms in the same chain but side chains of different residues
    #   (standard residues covalently linking non-consecutive residues in the same
    #   chain)
    atom_names = atom_array.atom_name
    molecule_types = atom_array.molecule_type_id
    # Find atoms connecting residues in the same chain via side chains
    is_side_chain = (
        ~np.isin(atom_names, NUCLEIC_ACID_MAIN_CHAIN_ATOMS)
        & np.isin(molecule_types, [MoleculeType.RNA, MoleculeType.DNA])
    ) | (
        ~np.isin(atom_names, PROTEIN_MAIN_CHAIN_ATOMS)
        & (molecule_types == MoleculeType.PROTEIN)
    )
    is_same_chain = chain_ids[bondlist_1_std[:, 0]] == chain_ids[bondlist_1_std[:, 1]]
    is_both_side_chain = (
        is_side_chain[bondlist_1_std[:, 0]] & is_side_chain[bondlist_1_std[:, 1]]
    )
    is_different_residue = (
        atom_array.aux_residue_id[bondlist_1_std[:, 0]]
        != atom_array.aux_residue_id[bondlist_1_std[:, 1]]
    )
    is_same_chain_diff_sidechain = (
        is_same_chain & is_both_side_chain & is_different_residue
    )

    # Combine
    bondlist_covalent_modification = np.concatenate(
        (
            bondlist_1_std[
                is_one_heteroatom | is_different_chain | is_same_chain_diff_sidechain
            ],
            # bondlist_2_std[is_same_chain_diff_sidechain],
        ),
        axis=0,
    )

    # Get corresponding non-heteroatoms
    atom_ids_covalent_modification = np.unique(
        bondlist_covalent_modification[:, :2].flatten()
    )
    nonhetero_atoms_in_covalent_modification = atom_array[
        atom_ids_covalent_modification
    ][~atom_array[atom_ids_covalent_modification].hetero]

    # Get the start indices of residues with any atom in
    # that need to be atomized according to the above criteria
    atomized_residue_token_start_ids = atom_array[
        np.isin(
            atom_array.aux_residue_id,
            nonhetero_atoms_in_covalent_modification.aux_residue_id,
        )
    ]._atom_idx

    # Remove the corresponding residue token start ids
    modified_residue_token_start_ids = np.unique(
        struc.get_residue_starts_for(atom_array, atomized_residue_token_start_ids)
    )
    residue_token_start_ids = residue_token_start_ids[
        ~np.isin(residue_token_start_ids, modified_residue_token_start_ids)
    ]

    # Get atom-token ids
    atom_token_start_ids = atom_array._atom_idx[~is_standard_residue_atom]

    # Combine all token start ids
    all_token_start_ids = np.sort(
        np.concatenate(
            [
                residue_token_start_ids,
                atom_token_start_ids,
                atomized_residue_token_start_ids,
            ]
        )
    )

    # Add is_atomized annotation
    is_atomized = np.repeat(False, n_atoms)
    is_atomized[
        np.concatenate([atom_token_start_ids, atomized_residue_token_start_ids])
    ] = True
    atom_array.set_annotation("is_atomized", is_atomized)

    # Create token index
    token_id_repeats = np.diff(np.append(all_token_start_ids, n_atoms))
    token_ids_per_atom = np.repeat(np.arange(len(token_id_repeats)), token_id_repeats)
    atom_array.set_annotation("token_id", token_ids_per_atom)

    # Create token center atom annotation
    token_center_atoms = np.repeat(True, n_atoms)
    token_center_atoms[is_standard_residue_atom] = np.isin(
        atom_array[is_standard_residue_atom].atom_name, TOKEN_CENTER_ATOMS
    )
    # Edit token center atoms for covalently modified residues
    token_center_atoms[atomized_residue_token_start_ids] = True
    atom_array.set_annotation("token_center_atom", token_center_atoms)

    # Remove temporary atom & residue indices
    remove_atom_indices(atom_array)
    atom_array.del_annotation("aux_residue_id")

    return None
