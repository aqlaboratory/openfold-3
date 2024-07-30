import logging
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from .structure_primitives import (
    assign_entity_ids,
    assign_molecule_type_ids,
    assign_renumbered_chain_ids,
)


class ParsedStructure(NamedTuple):
    cif_file: pdbx.CIFFile
    atom_array: AtomArray


def parse_mmcif(
    file_path: Path | str,
    expand_bioassembly: bool = False,
    include_bonds: bool = True,
    extra_fields: list | None = None,
    use_author_fields: bool = False,
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
        use_author_fields:
            Whether to use author fields. Note that the structure preprocessing and
            cleanup functions in this module were only tested with the PDB-auto-assigned
            labels (i.e. this argument set to False), so be careful when enabling this
            setting. Defaults to False.

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
    extra_fields_preset = ["label_entity_id", "occupancy"]

    if extra_fields:
        extra_fields = extra_fields_preset + extra_fields
    else:
        extra_fields = extra_fields_preset

    # Shared args between get_assembly and get_structure
    parser_args = {
        "pdbx_file": cif_file,
        "model": 1,
        "altloc": "occupancy",
        "use_author_fields": use_author_fields,
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


class ChainMetadataDict(TypedDict):
    molecule_type: str  # "protein", "RNA", "DNA", "ligand"
    chain_id_renumbered: int  # renumbered chain ID assigned by parse_mmcif_bioassembly
    chain_id_pdb: str  # PDB-assigned chain ID "label_asym_id"
    chain_id_author: str  # Author-assigned chain ID "auth_asym_id"


def write_annotated_chains_fasta(
    output_path: Path,
    chain_to_sequence: dict,
    pdb_id: str,
    chain_metadata_list: list[ChainMetadataDict],
) -> Path:
    """Writes a FASTA file annotating chain IDs and molecule types

    Will write all chains in the chain_metadata_list to a single FASTA file. The header
    will follow the format
    `>{pdb_id}_{chain_id_pdb}_{chain_id_author}_{chain_id_renum}_{molecule_type}`.

    Args:
        output_path:
            Path to write the FASTA file to.
        chain_to_sequence:
            Dictionary mapping chain IDs to sequences.
        pdb_id:
            PDB ID of the structure.
        chain_metadata_list:
            List of dictionaries containing chain metadata. Each dictionary should
            contain the following keys:
                - molecule_type: "protein", "RNA", "DNA", "ligand"
                - chain_id_renumbered: renumbered chain ID assigned by
                  parse_mmcif_bioassembly
                - chain_id_pdb: PDB-assigned chain ID "label_asym_id"
                - chain_id_author: Author-assigned chain ID "auth_asym_id"

    Returns:
        Path to the written FASTA file.
    """
    fasta_lines = []

    # Go through list of chains and extract sequences for all polymers
    for chain_data in chain_metadata_list:
        mol_type = chain_data["molecule_type"]

        # Skip ligands
        if mol_type == "ligand":
            continue

        chain_id_renum = chain_data["chain_id_renumbered"]
        chain_id_pdb = chain_data["chain_id_pdb"]
        chain_id_author = chain_data["chain_id_author"]

        # Header stores multiple chain IDs and molecule type
        chain_header = (
            f">{pdb_id}_{chain_id_pdb}_{chain_id_author}_{chain_id_renum}_"
            f"{mol_type}\n"
        )
        seq = chain_to_sequence[chain_id_renum] + "\n"

        fasta_lines.extend([chain_header, seq])

    with open(output_path, "w") as f:
        f.writelines(fasta_lines)

    return output_path


def encode_numpy_types(obj: object):
    """An encoding function for NumPy types (useful for JSON serialisation)"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
