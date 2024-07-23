import logging
from pathlib import Path
from typing import NamedTuple

from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from .structure_primitives import (
    assign_atom_indices,
    assign_entity_ids,
    assign_molecule_type_ids,
    assign_renumbered_chain_ids,
)


class ParsedStructure(NamedTuple):
    cif_file: pdbx.CIFFile
    atom_array: AtomArray


def parse_mmcif_bioassembly(
    file_path: Path | str,
    use_author_fields: bool = False,
    include_bonds: bool = True,
    extra_fields: list | None = None,
) -> ParsedStructure:
    """Convenience wrapper around biotite's bioassembly CIF parsing

    Parses the mmCIF file, expands the first bioassembly, and creates an AtomArray from
    it. This includes only the first model, resolves alternative locations by taking the
    one with the highest occupancy, defaults to inferring bond information, and defaults
    to using the PDB-automated chain/residue annotation instead of author annotations.

    This function also creates the following additional annotations in the AtomArray:
        - occupancy: inferred from atom_site.occupancy
        - entity_id: inferred from atom_site.label_entity_id
        - atom_idx: numerical atom indices starting from 0
        - molecule_type_id: numerical code for the molecule type (see tables.py)
        - chain_id_renumbered: numerical chain IDs starting from 0 to circumvent
          duplicate chain IDs after bioassembly expansion. This is used in place of the
          original chain ID in all of the cleanup and preprocessing functions.

    Args:
        file_path:
            Path to the mmCIF file.
        use_author_fields:
            Whether to use author fields. Defaults to False.
        include_bonds:
            Whether to infer bond information. Defaults to True.
        extra_fields:
            Extra fields to include in the AtomArray. Defaults to None.

    Returns:
        A NamedTuple containing the parsed CIF file and the AtomArray.
    """
    cif_file = pdbx.CIFFile.read(file_path)

    (pdb_id,) = cif_file.keys()  # Single-element unpacking

    extra_fields_preset = ["label_entity_id", "occupancy"]

    if extra_fields:
        extra_fields_preset.extend(extra_fields)

    # Shared args between get_assembly and get_structure
    parser_args = {
        "pdbx_file": cif_file,
        "model": 1,
        "altloc": "occupancy",
        "use_author_fields": use_author_fields,
        "include_bonds": include_bonds,
        "extra_fields": extra_fields_preset,
    }

    # Check if the CIF file contains bioassembly information
    if "pdbx_struct_assembly_gen" in cif_file[pdb_id]:
        atom_array = pdbx.get_assembly(
            **parser_args,
            assembly_id="1",
        )
    else:
        logging.warning(
            "No bioassembly information found in the CIF file, "
            "falling back to parsing the asymmetric unit."
        )
        atom_array = pdbx.get_structure(
            **parser_args,
        )

    # Add entity IDs
    assign_entity_ids(atom_array)

    # Add atom indices for convenience
    assign_atom_indices(atom_array)

    # Add molecule types for convenience
    assign_molecule_type_ids(atom_array)

    # Renumber chain IDs from 0 to avoid duplicate chain labels after bioassembly
    # expansion
    assign_renumbered_chain_ids(atom_array)

    return ParsedStructure(cif_file, atom_array)
