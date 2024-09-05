"""Conformer featurization pipeline."""

import torch
import torch.nn.functional as F
from rdkit.Chem import Mol

from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.structure.component import PERIODIC_TABLE
# from openfold3.core.model.structure.diffusion_module import centre_random_augmentation


# TODO: REMOVE
import torch
from openfold3.core.utils.rigid_utils import quat_to_rot


def sample_rotations(shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Sample random quaternions"""
    q = torch.randn(*shape, 4, dtype=dtype, device=device)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)

    rots = quat_to_rot(q)

    return rots


def centre_random_augmentation(
    xl: torch.Tensor, atom_mask: torch.Tensor, scale_trans: float = 1.0
) -> torch.Tensor:
    """
    Implements AF3 Algorithm 19.

    Args:
        xl:
            [*, N_atom, 3] Atom positions
        atom_mask:
            [*, N_atom] Atom mask
        scale_trans:
            Translation scaling factor
    Returns:
        Updated atom position with random global rotation and translation
    """
    rots = sample_rotations(shape=xl.shape[:-2], dtype=xl.dtype, device=xl.device)

    trans = scale_trans * torch.randn(
        (*xl.shape[:-2], 3), dtype=xl.dtype, device=xl.device
    )

    mean_xl = torch.sum(
        xl * atom_mask[..., None],
        dim=-2,
        keepdim=True,
    ) / torch.sum(atom_mask[..., None], dim=-2, keepdim=True)

    # center coordinates
    pos_centered = xl - mean_xl
    return pos_centered @ rots.transpose(-1, -2) + trans[..., None, :]


def featurize_ref_conformers_af3(
    processed_mol_list: list[ProcessedReferenceMolecule],
) -> dict[str, torch.Tensor]:
    """AF3 pipeline for creating reference conformer features.

    This function creates all conformer features as outlined in Table 5 of the
    AlphaFold3 SI under the "ref_" prefix.

    NOTE: We implement the "ref_space_uid" feature slightly differently by simply making
    it a unique identifier for each conformer instance, which we believe is the purpose
    of this feature, but making no explicit attempt to basing this on (chain_id,
    residue_id) pairs. The output between those strategies should be equivalent except
    for cases like the monomers of glycans, which may have different residue IDs if not
    accounted for, but will still get one reference conformer for the entire linked
    glycan and a single corresponding ref_space_uid in our implementation.

    Args:
        mol_list (list[Mol]):
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

    for mol_idx, processed_mol in enumerate(processed_mol_list):
        mol_id, mol, in_array_mask = processed_mol
        conf = mol.GetConformer()

        # Intermediate for this conformer's coordinates so that we can jointly apply a
        # random translation & rotation after collecting the coordinates
        intermediate_ref_pos = []

        # Featurize the parts of the molecule that ended up in the selected crop
        for atom, mask in zip(mol.GetAtoms(), in_array_mask):
            # Skip atom not in crop
            if mask == 0:
                continue

            # Intermediate reference coordinates (without random rotation & translation)
            coords = conf.GetAtomPosition(atom.GetIdx())
            intermediate_ref_pos.append(coords)
            ref_mask.append(int(atom.GetProp("used_mask")))

            # Atom elements
            element_symbol = atom.GetSymbol()
            ref_element.append(PERIODIC_TABLE.GetAtomicNumber(element_symbol))

            # Charges
            ref_charge.append(atom.GetFormalCharge())

            # ID for each unique conformer instance
            ref_space_uid.append(mol_idx)

            # Encoding of atom names
            atom_name_padded = atom.GetProp("name").ljust(4)
            chars = []
            for char in atom_name_padded:
                chars.append(ord(char) - 32)
            ref_atom_name_chars.append(chars)

        # Apply random translation & rotation to reference coordinates
        final_ref_pos = centre_random_augmentation(intermediate_ref_pos)
        ref_pos.extend(final_ref_pos)

    ref_pos = torch.tensor(ref_pos, dtype=torch.float32)
    ref_mask = torch.tensor(ref_mask, dtype=torch.int32)
    ref_element = F.one_hot(torch.tensor(ref_element), 128).to(torch.int32)
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
    }
