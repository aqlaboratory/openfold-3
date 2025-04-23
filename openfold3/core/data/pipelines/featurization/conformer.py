"""Conformer featurization pipeline."""

import logging

import torch
import torch.nn.functional as F

from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.component import PERIODIC_TABLE
from openfold3.core.model.structure.diffusion_module import centre_random_augmentation

logger = logging.getLogger(__name__)


@log_runtime_memory(runtime_dict_key="runtime-ref-conf-feat")
def featurize_reference_conformers_af3(
    processed_ref_mol_list: list[ProcessedReferenceMolecule],
) -> dict[str, torch.Tensor]:
    """AF3 pipeline for creating reference conformer features.

    This function creates all conformer features as outlined in Table 5 of the
    AlphaFold3 SI under the "ref_" prefix.

    NOTE: We implement the "ref_space_uid" feature slightly differently by simply making
    it a unique identifier for each conformer instance, which we believe is the purpose
    of this feature, but making no explicit attempt to base this on (chain_id,
    residue_id) pairs. The output between those strategies should be equivalent except
    for cases like the monomers of glycans, which may have different residue IDs if not
    accounted for, but will still get one reference conformer for the entire linked
    glycan and a single corresponding ref_space_uid in our implementation.

    Args:
        processed_ref_mol_list (list[ProcessedReferenceMolecule]):
            List of RDKit Mol objects corresponding to each reference conformer
            instance.

    Returns:
        dict[str, torch.Tensor]:
            Dictionary of reference conformer features:
                - ref_pos:
                    Reference atomic positions (torch.float32)
                - ref_mask:
                    Mask for used atoms (torch.int32)
                - ref_element:
                    One-hot encoded atomic numbers (torch.int32)
                - ref_charge:
                    Atomic charges (torch.float32)
                - ref_atom_name_chars:
                    One-hot encoded atom names (torch.int32)
                - ref_space_uid:
                    Unique identifier for each conformer instance (torch.int32)
    """
    ref_pos = []
    ref_mask = []
    ref_element = []
    ref_charge = []
    ref_atom_name_chars = []
    ref_space_uid = []  # deviation from SI! see docstring
    ref_space_uid_to_perm = {}

    for mol_idx, processed_mol in enumerate(processed_ref_mol_list):
        mol = processed_mol.mol
        in_crop_mask = processed_mol.in_crop_mask
        permutations = processed_mol.permutations

        conf = mol.GetConformer()

        # Intermediate for this conformer's coordinates so that we can jointly apply a
        # random translation & rotation after collecting the coordinates
        mol_ref_pos = []
        mol_ref_mask = []

        # Featurize the parts of the molecule that ended up in the selected crop
        for atom, mask in zip(mol.GetAtoms(), in_crop_mask):
            # Skip atom not in crop
            if mask == 0:
                continue

            # Intermediate reference coordinates (without random rotation & translation)
            coords = conf.GetAtomPosition(atom.GetIdx())
            mol_ref_mask.append(int(atom.GetBoolProp("annot_used_atom_mask")))
            mol_ref_pos.append(coords)

            # Atom elements (0-indexed)
            element_symbol = atom.GetSymbol()

            # Unknown atom type, assign to last bin (118)
            if element_symbol == "R":
                ref_element.append(118)
            else:
                ref_element.append(PERIODIC_TABLE.GetAtomicNumber(element_symbol) - 1)

            # TODO: check whether pdbeccdutils sets this correctly for charged amino
            # acids
            # Charges
            ref_charge.append(atom.GetFormalCharge())

            # ID for each unique conformer instance
            ref_space_uid.append(mol_idx)

            # Encoding of atom names
            atom_name_padded = atom.GetProp("annot_atom_name").ljust(4)
            chars = []
            for char in atom_name_padded:
                chars.append(ord(char) - 32)
            ref_atom_name_chars.append(chars)

        mol_ref_mask = torch.tensor(mol_ref_mask, dtype=torch.int32)
        mol_ref_pos = torch.tensor(mol_ref_pos, dtype=torch.float32)

        if torch.any(mol_ref_mask):
            # Apply random translation & rotation to reference coordinates
            final_ref_pos = centre_random_augmentation(mol_ref_pos, mol_ref_mask)

            ref_pos.append(final_ref_pos)
            ref_mask.append(mol_ref_mask)
        else:
            # If all-NaN conformer, skip random augmentation
            ref_pos.append(mol_ref_pos)
            ref_mask.append(mol_ref_mask)

        # Append mapping of each unique conformer instance to its permutations
        ref_space_uid_to_perm[mol_idx] = torch.tensor(permutations, dtype=torch.int32)

    ref_pos = torch.concat(ref_pos)
    ref_mask = torch.concat(ref_mask)

    # DISCREPANCY TO SI: We set the maximum atomic number to 118, not 128, which is the
    # largest known atomic number, and add an additional bin for unknown elements
    ref_element = F.one_hot(torch.tensor(ref_element), 119).to(torch.int32)

    ref_charge = torch.tensor(ref_charge, dtype=torch.float32)
    ref_atom_name_chars = F.one_hot(torch.tensor(ref_atom_name_chars), 64).to(
        torch.int32
    )
    ref_space_uid = torch.tensor(ref_space_uid, dtype=torch.int32)

    return {
        "ref_pos": ref_pos,
        "ref_mask": ref_mask,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "ref_space_uid_to_perm": ref_space_uid_to_perm,
    }
