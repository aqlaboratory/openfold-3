from pathlib import Path

import numpy as np
from biotite.structure.io import pdb

from openfold3.core.data.io.s3 import open_local_or_s3
from openfold3.core.data.io.structure.cif import ParsedStructure
from openfold3.core.data.primitives.structure.cleanup import (
    fix_arginine_naming,
    remove_hydrogens,
    remove_std_residue_terminal_atoms,
)
from openfold3.core.data.resources.residues import MoleculeType


# TODO: refactor PDB file reading logic as it currently only supports monomers
def parse_protein_monomer_pdb_tmp(
    file_path: Path | str,
    include_bonds: bool = True,
    extra_fields: list | None = None,
    s3_profile: str | None = None,
):
    """Temporary function to parse a protein monomer from a PDB file.

    Args:
        file_path (Path | str): _description_
        include_bonds (bool, optional): _description_. Defaults to True.
        extra_fields (list | None, optional): _description_. Defaults to None.

    Returns:
        ParsedStructure : _description_
    """

    ## no label fields in pdb files
    with open_local_or_s3(file_path, profile=s3_profile) as f:
        pdb_file = pdb.PDBFile.read(f)
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

    atom_array.set_annotation("chain_id", chain_ids)
    atom_array.set_annotation("molecule_type_id", molecule_type_ids)
    atom_array.set_annotation("entity_id", entity_ids)

    # Clean up structure
    fix_arginine_naming(atom_array)
    atom_array = remove_hydrogens(atom_array)
    atom_array = remove_std_residue_terminal_atoms(atom_array)

    return ParsedStructure(pdb_file, atom_array)
