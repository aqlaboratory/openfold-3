import biotite.structure as struc
import numpy as np
from biotite.structure.io.pdbx import CIFBlock, CIFFile


def get_release_date(cif_data: CIFBlock) -> str:
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
    try:
        resolution = cif_data["refine"]["ls_d_res_high"].as_item()
    except KeyError:
        try:
            resolution = cif_data["em_3d_reconstruction"]["resolution"].as_item()
        except KeyError:
            try:
                resolution = cif_data["reflns"]["d_resolution_high"].as_item()
            except KeyError:
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
    method = cif_data["exptl"]["method"].as_item()
    
    return method


def get_cif_block(cif_file: CIFFile) -> CIFBlock:
    """Get the CIF block of the structure.

    Args:
        cif_file:
            Parsed mmCIF file containing the structure.

    Returns:
        The CIF block of the structure.
    """
    (cif_block,) = cif_file.keys()

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

    return dict(zip(polymer_entities, polymer_canonical_seqs))


def get_chain_to_entity_dict(atom_array: struc.AtomArray) -> dict[int, int]:
    """Get a dictionary mapping renumbered chain IDs to their entity IDs.

    Args:
        atom_array:
            AtomArray containing the chain IDs and entity IDs.

    Returns:
        A dictionary mapping renumbered chain IDs to their entity IDs.
    """
    return dict(zip(atom_array.chain_id_renumbered, atom_array.entity_id))


def get_chain_to_canonical_seq_dict(
    atom_array: struc.AtomArray, cif_data: CIFBlock
) -> dict[int, str]:
    """Get a dictionary mapping renumbered chain IDs to their canonical sequences.

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

    Args:
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        A dictionary mapping entity IDs to their three-letter-code sequences.
    """
    # Flat list of residue-wise entity IDs for all polymeric sequences
    entity_ids_flat = cif_data["entity_poly_seq"]["entity_id"].as_array(dtype=int)

    # Get sequence lenfths and entity starts for each polymeric entity
    entity_ids, new_entity_starts, seq_lengths = np.unique(
        entity_ids_flat, return_index=True, return_counts=True
    )

    # Get full (3-letter code) residue sequence for every polymeric entity
    entity_monomers = cif_data["entity_poly_seq"]["mon_id"].as_array()
    entity_id_to_3l_codes = {
        entity_id: entity_monomers[start : start + length]
        for entity_id, start, length in zip(entity_ids, new_entity_starts, seq_lengths)
    }

    return entity_id_to_3l_codes


def get_chain_to_three_letter_codes_dict(
    atom_array: struc.AtomArray, cif_data: CIFBlock
) -> dict[int, list[str]]:
    """Get a dictionary mapping renumbered chain IDs to their three-letter-code sequences.

    Args:
        atom_array:
            AtomArray containing the chain IDs and entity IDs.
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        A dictionary mapping renumbered chain IDs to their three-letter-code sequences.
    """
    entity_ids_to_3l_codes = get_entity_to_three_letter_codes_dict(cif_data)
    chain_to_entity_dict = get_chain_to_entity_dict(atom_array)

    chain_to_3l_codes_dict = {
        chain: entity_ids_to_3l_codes[entity]
        for chain, entity in chain_to_entity_dict.items()
        if entity in entity_ids_to_3l_codes
    }

    return chain_to_3l_codes_dict
