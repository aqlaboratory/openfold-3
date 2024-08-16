"""This module contains featurization pipelines for structural data."""

from typing import Union

import numpy as np
import torch
from biotite.structure import AtomArray
from rdkit.Chem import Mol
from torch.nn.functional import one_hot

from openfold3.core.data.primitives.featurization.structure import (
    create_sym_id,
    create_token_bonds,
    encode_one_hot,
    extract_starts_entities,
    get_with_unknown,
)
from openfold3.core.data.primitives.structure.ligand import PERIODIC_TABLE
from openfold3.core.data.resources.tables import (
    MoleculeType,
)
from openfold3.core.np.token_atom_constants import TOKEN_TYPES_WITH_GAP


def featurize_structure_af3(
    atom_array: AtomArray, is_gt: bool
) -> dict[str, torch.Tensor]:
    """AF3 pipeline for creating target and gt structure features.

    Expects the cropped or duplicate-expanded AtomArray as input.

    Args:
        atom_array (AtomArray):
            AtomArray of the target or ground truth structure.
        is_gt (bool):
            Whether the input AtomArray is from the duplicate-expanded ground truth
            structure.

    Returns:
        dict[str, torch.Tensor]:
            Target or ground truth features.
    """
    token_starts_with_stop, entity_ids = extract_starts_entities(atom_array)
    token_starts = token_starts_with_stop[:-1]

    features = {}
    # Indexing
    features["residue_index"] = torch.tensor(
        atom_array.res_id[token_starts], dtype=torch.int
    )
    features["token_index"] = torch.tensor(
        atom_array.token_id[token_starts], dtype=torch.int
    )
    features["asym_id"] = torch.tensor(
        atom_array.chain_id_renumbered[token_starts], dtype=torch.int
    )
    features["entity_id"] = torch.tensor(
        atom_array.entity_id[token_starts], dtype=torch.int
    )
    features["sym_id"] = torch.tensor(create_sym_id(entity_ids), dtype=torch.int)
    restype_index = torch.tensor(
        get_with_unknown(atom_array.res_name[token_starts]), dtype=torch.int64
    )
    features["restype"] = encode_one_hot(restype_index, len(TOKEN_TYPES_WITH_GAP))
    features["is_protein"] = torch.tensor(
        atom_array.molecule_type_id[token_starts] == MoleculeType.PROTEIN,
        dtype=torch.bool,
    )
    features["is_rna"] = torch.tensor(
        atom_array.molecule_type_id[token_starts] == MoleculeType.RNA,
        dtype=torch.bool,
    )
    features["is_dna"] = torch.tensor(
        atom_array.molecule_type_id[token_starts] == MoleculeType.DNA,
        dtype=torch.bool,
    )
    features["is_ligand"] = torch.tensor(
        atom_array.molecule_type_id[token_starts] == MoleculeType.LIGAND,
        dtype=torch.bool,
    )

    # Bonds
    features["token_bonds"] = create_token_bonds(
        atom_array, features["token_index"].numpy()
    )

    # Masks

    # Atomization
    features["num_atoms_per_token"] = torch.tensor(
        np.diff(token_starts_with_stop),
        dtype=torch.int,
    )
    features["start_atom_index"] = torch.tensor(
        token_starts,
        dtype=torch.int,
    )
    features["is_atomized"] = torch.tensor(atom_array.is_atomized, dtype=torch.bool)

    # Ground-truth-specific features
    # TODO reorganize GT feature logic
    if is_gt:
        features["atom_positions"] = torch.tensor(atom_array.coord, dtype=torch.float)
        features["atom_resolved_mask"] = torch.tensor(
            atom_array.occupancy, dtype=torch.bool
        )

    return features


def featurize_target_gt_structure_af3(
    atom_array_cropped: AtomArray,
    atom_array_gt: AtomArray,
) -> dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]]:
    """AF3 pipeline for creating target and gt structure features.

    Expects the cropped and duplicate-expanded AtomArray as input. The target structure
    features are a flat dictionary, while the ground truth features are nested in a
    subdictionary under the 'ground_truth' key.

    Args:
        atom_array_cropped (AtomArray):
            AtomArray of the target structure.
        atom_array_gt (AtomArray):
            AtomArray of the duplicate-expanded ground truth structure.

    Returns:
        dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]]:
            Target and ground truth features. Ground truth features are nested
            in a subdictionary under the 'ground_truth' key.
    """
    features_target = featurize_structure_af3(atom_array_cropped, is_gt=False)
    features_gt = featurize_structure_af3(atom_array_gt, is_gt=True)
    features_target["ground_truth"] = features_gt
    return features_target


# TODO: better docstring
def featurize_ref_conformers_af3(
    mol_list: list[Mol]
):
    """AF3 pipeline for creating reference conformer features."""
    ref_pos = []
    ref_mask = []
    ref_element = []
    ref_charge = []
    ref_atom_name_chars = []
    ref_space_uid = [] # deviation from SI, TODO: explain
    
    for mol_idx, mol in enumerate(mol_list):
        conf = mol.GetConformer()
        
        for atom in mol.GetAtoms():
            ref_pos.append(conf.GetAtomPosition(atom.GetIdx()))
            ref_mask.append(int(atom.GetProp("used_mask")))
            
            element_symbol = atom.GetSymbol()
            ref_element.append(PERIODIC_TABLE.GetAtomicNumber(element_symbol))

            ref_charge.append(atom.GetFormalCharge())
            ref_space_uid.append(mol_idx)

            atom_name_padded = atom.GetProp("name").ljust(4)
            chars = []
            for char in atom_name_padded:
                chars.append(ord(char) - 32)
            ref_atom_name_chars.append(chars)
    
    ref_pos = torch.tensor(ref_pos, dtype=torch.float32)
    ref_mask = torch.tensor(ref_mask, dtype=torch.int32)
    ref_element = one_hot(torch.tensor(ref_element), 128).to(torch.int32)
    ref_charge = torch.tensor(ref_charge, dtype=torch.float32)
    ref_atom_name_chars = one_hot(torch.tensor(ref_atom_name_chars), 64).to(torch.int32)
    ref_space_uid = torch.tensor(ref_space_uid, dtype=torch.int32)
    
    return {
        "ref_pos": ref_pos,
        "ref_mask": ref_mask,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
    }
