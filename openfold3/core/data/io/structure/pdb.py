# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
from biotite.structure.io import pdb

from openfold3.core.data.io.s3 import open_local_or_s3
from openfold3.core.data.io.structure.cif import (
    ParsedStructure,
    SkippedStructure,
)
from openfold3.core.data.primitives.structure.cleanup import (
    fix_arginine_naming,
    remove_hydrogens,
    remove_std_residue_terminal_atoms,
)
from openfold3.core.data.primitives.structure.labels import (
    assign_entity_ids,
    assign_renumbered_chain_ids,
    update_author_to_pdb_labels,
)
from openfold3.core.data.resources.residues import MoleculeType


def _load_pdbfile(file_path: Path | str) -> pdb.PDBFile:
    """Load a PDB file from a given path.

    Args:
        file_path (Path):
            Path to the PDB file.

    Returns:
        pdb.PDBFile:
            PDB file object.
    """
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path

    if file_path.suffix == ".pdb":
        return pdb.PDBFile.read(file_path)
    else:
        raise ValueError("File must be in pdb format")


# TODO: refactor with PDB reader below as it currently only supports monomers
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


# TODO: check if extending existing primitives used below to support AF2 is ok
def parse_pdb_af2(
    file_path: Path | str,
    include_bonds: bool = True,
    renumber_chain_ids: bool = False,
) -> ParsedStructure | SkippedStructure:
    """Parses AF2 predictions from a PDB file."""

    pdb_file = _load_pdbfile(file_path)

    # Shared args between get_assembly and get_structure
    parser_args = {
        "model": 1,
        "altloc": "occupancy",
        "include_bonds": include_bonds,
        "extra_fields": [
            "atom_id",
            "occupancy",
            "charge",
        ],
    }

    atom_array = pdb_file.get_structure(
        **parser_args,
    )

    # Replace author-assigned IDs with PDB-assigned IDs
    update_author_to_pdb_labels(atom_array, atom_array_source_format="pdb_af2")

    # Add entity IDs
    assign_entity_ids(atom_array, atom_array_source_format="pdb_af2")

    # Add protein molecule type
    atom_array.set_annotation(
        "molecule_type_id",
        np.array([MoleculeType.PROTEIN] * len(atom_array), dtype=int),
    )

    # Renumber chain IDs to match CIF chain ID renumbering convention
    if renumber_chain_ids:
        assign_renumbered_chain_ids(atom_array)

    return ParsedStructure(pdb_file, atom_array)
