"""This module contains IO functions for reading and writing mmCIF files."""

import logging
from pathlib import Path
from typing import Literal, NamedTuple

from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from openfold3.core.data.primitives.structure.labels import (
    assign_entity_ids,
    assign_molecule_type_ids,
    assign_renumbered_chain_ids,
    update_author_to_pdb_labels,
)


class ParsedStructure(NamedTuple):
    cif_file: pdbx.CIFFile
    atom_array: AtomArray


def parse_mmcif(
    file_path: Path | str,
    expand_bioassembly: bool = False,
    include_bonds: bool = True,
    extra_fields: list | None = None,
) -> ParsedStructure:
    """Convenience wrapper around biotite's CIF parsing

    Parses the mmCIF file and creates an AtomArray from it while optionally expanding
    the first bioassembly. This includes only the first model, resolves alternative
    locations by taking the one with the highest occupancy, defaults to inferring bond
    information, and defaults to using the PDB-automated chain/residue annotation
    instead of author annotations.

    This function also creates the following additional annotations in the AtomArray:
        - occupancy: inferred from atom_site.occupancy
        - entity_id: inferred from atom_site.label_entity_id
        - molecule_type_id: numerical code for the molecule type (see tables.py)
        - chain_id_renumbered: numerical chain IDs starting from 0 to circumvent
          duplicate chain IDs after bioassembly expansion. This is used in place of the
          original chain ID in all of the cleanup and preprocessing functions.

    Args:
        file_path:
            Path to the mmCIF (or binary mmCIF) file.
        expand_bioassembly:
            Whether to expand the first bioassembly. Defaults to False.
        include_bonds:
            Whether to infer bond information. Defaults to True.
        extra_fields:
            Extra fields to include in the AtomArray. Defaults to None. Fields
            "entity_id" and "occupancy" are always included.

    Returns:
        A NamedTuple containing the parsed CIF file and the AtomArray.
    """
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

    if file_path.suffix == ".cif":
        cif_class = pdbx.CIFFile
    elif file_path.suffix == ".bcif":
        cif_class = pdbx.BinaryCIFFile
    else:
        raise ValueError("File must be in mmCIF or binary mmCIF format")

    cif_file = cif_class.read(file_path)

    (pdb_id,) = cif_file.keys()  # Single-element unpacking

    # Always include these fields
    label_fields = [
        "label_entity_id",
        "label_atom_id",
        "label_comp_id",
        "label_asym_id",
        "label_seq_id",
    ]
    extra_fields_preset = [
        "occupancy",
    ] + label_fields

    if extra_fields:
        extra_fields = extra_fields_preset + extra_fields
    else:
        extra_fields = extra_fields_preset

    # Shared args between get_assembly and get_structure
    parser_args = {
        "pdbx_file": cif_file,
        "model": 1,
        "altloc": "occupancy",
        "use_author_fields": True,
        "include_bonds": include_bonds,
        "extra_fields": extra_fields,
    }

    # Check if the CIF file contains bioassembly information
    if expand_bioassembly & ("pdbx_struct_assembly_gen" not in cif_file[pdb_id]):
        logging.warning(
            "No bioassembly information found in the CIF file, "
            "falling back to parsing the asymmetric unit."
        )
        expand_bioassembly = False

    if expand_bioassembly:
        atom_array = pdbx.get_assembly(
            **parser_args,
            assembly_id="1",
        )
    else:
        atom_array = pdbx.get_structure(
            **parser_args,
        )

    # Replace author-assigned IDs with PDB-assigned IDs
    update_author_to_pdb_labels(atom_array)

    # Add entity IDs
    assign_entity_ids(atom_array)

    # Add molecule types for convenience
    assign_molecule_type_ids(atom_array)

    # Renumber chain IDs from 0 to avoid duplicate chain labels after bioassembly
    # expansion
    assign_renumbered_chain_ids(atom_array)

    return ParsedStructure(cif_file, atom_array)


def write_minimal_cif(
    atom_array: AtomArray,
    output_path: Path,
    format: Literal["cif", "bcif"] = "cif",
    include_bonds: bool = True,
) -> None:
    """Write a minimal CIF file

    The resulting CIF file will only contain the atom_site records and bond information
    by default, not any other mmCIF metadata.

    Args:
        atom_array:
            AtomArray to write to the CIF file.
        output_path:
            Path to write the CIF file to.
        format:
            Format of the CIF file. Defaults to "cif".
        include_bonds:
            Whether to include bond information in the CIF file. Defaults to True.
    """
    cif_file = pdbx.CIFFile() if format == "cif" else pdbx.BinaryCIFFile()
    pdbx.set_structure(cif_file, atom_array, include_bonds=include_bonds)

    cif_file.write(output_path)
