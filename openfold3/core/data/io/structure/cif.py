"""This module contains IO functions for reading and writing mmCIF files."""

import logging
import pickle
from pathlib import Path
from typing import NamedTuple

from biotite.structure import AtomArray
from biotite.structure.io import pdbx, pdb

from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.labels import (
    assign_entity_ids,
    assign_molecule_type_ids,
    assign_renumbered_chain_ids,
    update_author_to_pdb_labels,
)
from openfold3.core.data.primitives.structure.metadata import (
    get_cif_block,
    get_first_bioassembly_polymer_count,
)
import  numpy as np 
from openfold3.core.data.resources.residues import MoleculeType



logger = logging.getLogger(__name__)


class ParsedStructure(NamedTuple):
    cif_file: pdbx.CIFFile
    atom_array: AtomArray | None


class SkippedStructure(NamedTuple):
    cif_file: pdbx.CIFFile
    n_polymer_chains: int


def _load_ciffile(file_path: Path | str) -> pdbx.CIFFile:
    """Load a CIF file from a given path.

    Args:
        file_path (Path):
            Path to the CIF file.

    Returns:
        pdbx.CIFFile:
            CIF file object.
    """
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

    if file_path.suffix == ".cif":
        cif_class = pdbx.CIFFile
    elif file_path.suffix == ".bcif":
        cif_class = pdbx.BinaryCIFFile
    else:
        raise ValueError("File must be in mmCIF or binary mmCIF format")

    return cif_class.read(file_path)


# TODO: update docstring with new residue ID handling and preset fields
def parse_mmcif(
    file_path: Path | str,
    expand_bioassembly: bool = False,
    include_bonds: bool = True,
    renumber_chain_ids: bool = False,
    extra_fields: list | None = None,
    max_polymer_chains: int | None = None,
) -> ParsedStructure | SkippedStructure:
    """Convenience wrapper around biotite's CIF parsing

    Parses the mmCIF file and creates an AtomArray from it while optionally expanding
    the first bioassembly. This includes only the first model, resolves alternative
    locations by taking the one with the highest occupancy, defaults to inferring bond
    information, and defaults to using the PDB-automated chain/residue annotation
    instead of author annotations, except for ligand residue IDs which are kept as
    author-assigned IDs because they would otherwise be None.

    This function also creates the following additional annotations in the AtomArray:
        - occupancy: inferred from atom_site.occupancy
        - charge: charge of the atom
        - entity_id: inferred from atom_site.label_entity_id
        - molecule_type_id: numerical code for the molecule type (see tables.py)
        - label_asym_id: original PDB-assigned chain ID
        - label_seq_id: original PDB-assigned residue ID
        - label_comp_id: original PDB-assigned residue name
        - label_atom_id: original PDB-assigned atom name
        - auth_asym_id: author-assigned chain ID
        - auth_seq_id: author-assigned residue ID
        - auth_comp_id: author-assigned residue name
        - auth_atom_id: author-assigned atom name

    Args:
        file_path:
            Path to the mmCIF (or binary mmCIF) file.
        expand_bioassembly:
            Whether to expand the first bioassembly. Defaults to False.
        include_bonds:
            Whether to infer bond information. Defaults to True.
        renumber_chain_ids:
            Whether to renumber chain IDs from 1 to avoid duplicate chain labels after
            bioassembly expansion. Defaults to False.
        extra_fields:
            Extra fields to include in the AtomArray. Defaults to None. Fields
            "entity_id" and "occupancy" are always included.
        max_polymer_chains:
            Maximum number of polymer chains in the first bioassembly after which a
            structure is skipped by the get_structure() parser. Defaults to None.

    Returns:
        A ParsedStructure NamedTuple containing the parsed CIF file and the AtomArray,
        or a SkippedStructure NamedTuple containing the CIF file and the number of
        polymer chains in the first bioassembly.
    """

    cif_file = _load_ciffile(file_path)
    cif_data = get_cif_block(cif_file)

    if max_polymer_chains is not None:
        # Polymers in first bioassembly
        n_polymers = get_first_bioassembly_polymer_count(cif_data)

        if n_polymers > max_polymer_chains:
            return SkippedStructure(cif_file, n_polymers)

    cif_data = get_cif_block(cif_file)

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
        "charge",
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
    if expand_bioassembly & ("pdbx_struct_assembly_gen" not in cif_data):
        logger.warning(
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
    assign_molecule_type_ids(atom_array, cif_file)

    # Renumber chain IDs from 1 to avoid duplicate chain labels after bioassembly
    # expansion
    if renumber_chain_ids:
        assign_renumbered_chain_ids(atom_array)

    return ParsedStructure(cif_file, atom_array)

def parse_pdb(file_path: Path | str, include_bonds: bool = True, extra_fields: list | None = None):
    """_summary_

    Args:
        file_path (Path | str): _description_
        include_bonds (bool, optional): _description_. Defaults to True.
        extra_fields (list | None, optional): _description_. Defaults to None.

    Returns:
        ParsedStructure : _description_
    """
    
    ## no label fields in pdb files
    pdb_file = pdb.PDBFile.read(file_path)
    extra_fields_preset = [
        "occupancy",
        "charge",
    ] 

    if extra_fields:
        extra_fields = extra_fields_preset + extra_fields
    else:
        extra_fields = extra_fields_preset
    
    parser_args = {
        "pdb_file": pdb_file,
        "model": 1,
        "altloc": "occupancy",
        "include_bonds": include_bonds,
        "extra_fields": extra_fields,
    }
    atom_array = pdb.get_structure(
                **parser_args,
            )
    
    ## manually assign th entity and molecule type ids; 
    ## monomers are all "single chain", so should have the same entity id, 
    ## everything is a single asym, and sym id should be 1(identity)
    chain_ids = np.array([1] * len(atom_array), dtype=int)
    molecule_type_ids = np.array([MoleculeType.PROTEIN] * len(atom_array), dtype=int)
    entity_ids = np.array([1] * len(atom_array), dtype=int)
    asym_ids = np.array([1] * len(atom_array), dtype=int)
    sym_ids = np.array([1] * len(atom_array), dtype=int)
    
    atom_array.set_annotation("chain_id", chain_ids)
    atom_array.set_annotation("molecule_type_id", molecule_type_ids)
    atom_array.set_annotation("entity_id", entity_ids)
    atom_array.set_annotation("asym_id", asym_ids)
    atom_array.set_annotation("sym_id", sym_ids)
    
    return ParsedStructure(pdb_file, atom_array)

def write_structure(
    atom_array: AtomArray,
    output_path: Path,
    data_block: str = None,
    include_bonds: bool = True,
) -> None:
    """Write a structure file from an AtomArray

    The resulting CIF file will only contain the atom_site records and bond information
    by default, not any other mmCIF metadata.

    Args:
        atom_array:
            AtomArray to write to an output file.
        output_path:
            Path to write the output file to. The output format is inferred from the
            file suffix. Allowed values are .cif, .bcif, and .pkl.
        data_block:
            Name of the data block in the CIF/BCIF file. Defaults to None. Ignored if
            the format is pkl.
        include_bonds:
            Whether to include bond information. Defaults to True. Ignored if the format
            is pkl in which the entire BondList is written to the file.
    """
    suffix = output_path.suffix
    if suffix == ".pkl":
        with open(output_path, "wb") as f:
            pickle.dump(atom_array, f)
        return
    elif suffix == ".cif":
        cif_file = pdbx.CIFFile()
    elif suffix == ".bcif":
        cif_file = pdbx.BinaryCIFFile()
    else:
        raise NotImplementedError("Only .cif, .bcif, and .pkl formats are supported")

    pdbx.set_structure(
        cif_file, atom_array, data_block=data_block, include_bonds=include_bonds
    )

    cif_file.write(output_path)


@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc-parse")
def parse_target_structure(
    target_structures_directory: Path, pdb_id: str, structure_format: str
) -> AtomArray:
    """Parses a target structure from a pickle file.

    Args:
        target_structures_directory (Path):
            Directory containing target structure folders.
        pdb_id (str):
            PDB ID of the target structure.
        structure_format (str):
            File extension of the target structure. Only "pkl" is supported.

    Raises:
        ValueError:
            If the structure format is not "pkl".

    Returns:
        AtomArray:
            AtomArray of the target structure.
    """
    target_file = target_structures_directory / pdb_id / f"{pdb_id}.{structure_format}"

    if structure_format == "pkl":
        with open(target_file, "rb") as f:
            atom_array = pickle.load(f)
    else:
        raise ValueError(
            f"Invalid structure format: {structure_format}. Only pickle "
            "format is supported in a torch dataset __getitem__."
        )

    return atom_array
