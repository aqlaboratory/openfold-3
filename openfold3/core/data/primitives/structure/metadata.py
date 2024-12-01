from collections import defaultdict
from datetime import datetime
from typing import Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import BondType
from biotite.structure.info.bonds import BOND_TYPES
from biotite.structure.io.pdbx import CIFBlock, CIFFile

from openfold3.core.data.primitives.structure.labels import (
    get_chain_to_entity_dict,
)


def get_pdb_id(cif_file: CIFFile, format: Literal["upper", "lower"] = "lower") -> str:
    """Get the PDB ID of the structure.

    Args:
        cif_file:
            Parsed mmCIF file containing the structure.
        format:
            The case of the PDB ID to return. Options are "upper" and "lower". Defaults
            to "lower".

    Returns:
        The PDB ID of the structure.
    """
    (pdb_id,) = cif_file.keys()

    if format == "upper":
        return pdb_id.upper()
    elif format == "lower":
        return pdb_id.lower()
    else:
        raise ValueError(f"Invalid format: {format}")


def get_release_date(cif_data: CIFBlock) -> datetime:
    """Get the release date of the structure.

    Release date is defined as the earliest revision date of the structure.

    Args:
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        The release date of the structure.
    """
    release_dates = cif_data["pdbx_audit_revision_history"]["revision_date"].as_array()
    release_dates = [datetime.strptime(date, "%Y-%m-%d") for date in release_dates]

    return min(release_dates)


def get_resolution(cif_data: CIFBlock) -> float:
    """Get the resolution of the structure.

    The resolution is obtained by sequentially checking the following data items:
    - refine.ls_d_res_high
    - em_3d_reconstruction.resolution
    - reflns.d_resolution_high

    and returning the first one that is found. If none of the above data items are
    found, the function returns NaN.

    Args:
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        The resolution of the structure.
    """
    keys_to_check = [
        ("refine", "ls_d_res_high"),
        ("em_3d_reconstruction", "resolution"),
        ("reflns", "d_resolution_high"),
    ]

    for key in keys_to_check:
        try:
            resolution = cif_data[key[0]][key[1]].as_item()

            # Try next if not specified
            if resolution in ("?", "."):
                continue

            # If successful, convert to float and return
            resolution = float(resolution)
            break

        # Try next if key not found
        except KeyError:
            continue
    else:
        resolution = float("nan")

    # dev-only: TODO remove
    assert isinstance(resolution, float), "Resolution is not a float"

    return resolution


def get_experimental_method(cif_data: CIFBlock) -> str:
    """Get the experimental method used to determine the structure.

    Args:
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        The experimental method used to determine the structure.
    """
    method = cif_data["exptl"]["method"].as_array()[0].item()

    return method


def get_cif_block(cif_file: CIFFile) -> CIFBlock:
    """Get the CIF block of the structure.

    Args:
        cif_file:
            Parsed mmCIF file containing the structure.

    Returns:
        The CIF block of the structure.
    """
    (pdb_id,) = cif_file.keys()
    cif_block = cif_file[pdb_id]

    return cif_block


def get_entity_to_canonical_seq_dict(cif_data: CIFBlock) -> dict[int, str]:
    """Get a dictionary mapping entity IDs to their canonical sequences.

    Args:
        The CIF data block containing the entity_poly table.

    Returns:
        A dictionary mapping entity IDs to their sequences.
    """
    polymer_entities = cif_data["entity_poly"]["entity_id"].as_array(dtype=int)
    polymer_canonical_seqs = cif_data["entity_poly"][
        "pdbx_seq_one_letter_code_can"
    ].as_array()
    polymer_canonical_seqs = np.char.replace(polymer_canonical_seqs, "\n", "")

    return dict(zip(polymer_entities.tolist(), polymer_canonical_seqs.tolist()))


def get_chain_to_canonical_seq_dict(
    atom_array: struc.AtomArray, cif_data: CIFBlock
) -> dict[int, str]:
    """Get a dictionary mapping chain IDs to their canonical sequences.

    Args:
        atom_array:
            AtomArray containing the chain IDs and entity IDs.
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)
    """
    entity_to_seq_dict = get_entity_to_canonical_seq_dict(cif_data)
    chain_to_entity_dict = get_chain_to_entity_dict(atom_array)

    chain_to_seq_dict = {
        chain: entity_to_seq_dict[entity]
        for chain, entity in chain_to_entity_dict.items()
        if entity in entity_to_seq_dict
    }

    return chain_to_seq_dict


def get_entity_to_three_letter_codes_dict(cif_data: CIFBlock) -> dict[int, list[str]]:
    """Get a dictionary mapping entity IDs to their three-letter-code sequences.

    Note that in the special case of multiple amino acids being set to the same residue
    ID, this will currently default to taking the first one and make no special attempt
    to take occupancy into account.

    Args:
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        A dictionary mapping entity IDs to their three-letter-code sequences.
    """
    # Flat list of residue-wise entity IDs for all polymeric sequences
    entity_ids_flat = cif_data["entity_poly_seq"]["entity_id"].as_array(dtype=int)

    # Deduplicated entity IDs
    entity_ids = np.unique(entity_ids_flat)

    # Get full (3-letter code) residue sequence for every polymeric entity
    entity_monomers = cif_data["entity_poly_seq"]["mon_id"].as_array()

    entity_residue_ids = cif_data["entity_poly_seq"]["num"].as_array()

    # Get map of residue IDs to monomers sharing that residue ID for every entity
    res_id_to_monomers = defaultdict(lambda: defaultdict(list))
    for entity_id, res_id, ccd_id in zip(
        entity_ids_flat.tolist(), entity_residue_ids.tolist(), entity_monomers.tolist()
    ):
        res_id_to_monomers[entity_id][res_id].append(ccd_id)

    # In case where multiple monomers are set to the same residue ID, take the first one
    # (TODO: this should ideally take occupancy into account)
    entity_id_to_3l_codes = {
        entity_id: [monomers[0] for monomers in res_id_to_monomers[entity_id].values()]
        for entity_id in entity_ids
    }

    return entity_id_to_3l_codes


def get_chain_to_three_letter_codes_dict(
    atom_array: struc.AtomArray, cif_data: CIFBlock
) -> dict[int, list[str]]:
    """Get dictionary mapping chain IDs to their three-letter-code sequences.

    Args:
        atom_array:
            AtomArray containing the chain IDs and entity IDs.
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        A dictionary mapping chain IDs to their three-letter-code sequences.
    """
    entity_ids_to_3l_codes = get_entity_to_three_letter_codes_dict(cif_data)
    chain_to_entity_dict = get_chain_to_entity_dict(atom_array)

    chain_to_3l_codes_dict = {
        chain: entity_ids_to_3l_codes[entity]
        for chain, entity in chain_to_entity_dict.items()
        if entity in entity_ids_to_3l_codes
    }

    return chain_to_3l_codes_dict


def get_ccd_atom_pair_to_bond_dict(ccd_entry: CIFBlock) -> dict[(str, str), BondType]:
    """Gets the list of bonds from a CCD entry.

    Args:
        ccd_entry:
            CIFBlock containing the CCD entry.

    Returns:
        Dictionary mapping each pair of atom names to the respective Biotite bond type.
    """

    chem_comp_bonds = ccd_entry.get("chem_comp_bond")

    if chem_comp_bonds is None:
        return {}

    atom_pair_to_bond = {}

    for atom_1, atom_2, ccd_bond_type, aromatic_flag in zip(
        chem_comp_bonds["atom_id_1"].as_array(),
        chem_comp_bonds["atom_id_2"].as_array(),
        chem_comp_bonds["value_order"].as_array(),
        chem_comp_bonds["pdbx_aromatic_flag"].as_array(),
    ):
        bond_type = BOND_TYPES[ccd_bond_type, aromatic_flag]
        atom_pair_to_bond[(atom_1.item(), atom_2.item())] = bond_type

    return atom_pair_to_bond


def get_ccd_atom_id_to_element_dict(ccd_entry: CIFBlock) -> dict[str, str]:
    """Gets the dictionary mapping atom IDs to element symbols from a CCD entry.

    Args:
        ccd_entry:
            CIFBlock containing the CCD entry.

    Returns:
        Dictionary mapping atom IDs to element symbols.
    """

    atom_id_to_element = {
        atom_id.item(): element.item()
        for atom_id, element in zip(
            ccd_entry["chem_comp_atom"]["atom_id"].as_array(),
            ccd_entry["chem_comp_atom"]["type_symbol"].as_array(),
        )
    }

    return atom_id_to_element


def get_ccd_atom_id_to_charge_dict(ccd_entry: CIFBlock) -> dict[str, float]:
    """Gets the dictionary mapping atom IDs to charges from a CCD entry.

    Args:
        ccd_entry:
            CIFBlock containing the CCD entry.

    Returns:
        Dictionary mapping atom IDs to charges.
    """

    atom_id_to_charge = {
        atom_id.item(): charge.item()
        for atom_id, charge in zip(
            ccd_entry["chem_comp_atom"]["atom_id"].as_array(),
            ccd_entry["chem_comp_atom"]["charge"].as_array().astype(int),
        )
    }

    return atom_id_to_charge


def get_first_bioassembly_polymer_count(cif_data: CIFBlock) -> int:
    """Returns the number of polymer chains in the first bioassembly."""
    return (
        cif_data["pdbx_struct_assembly"]["oligomeric_count"]
        .as_array(dtype=int)[0]
        .item()
    )
