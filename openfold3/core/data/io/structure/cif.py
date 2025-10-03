"""This module contains IO functions for reading and writing mmCIF files."""

import logging
import pickle
from pathlib import Path
from typing import Literal, NamedTuple

from biotite.structure import AtomArray, get_chain_count
from biotite.structure.io import pdb, pdbx, save_structure

from openfold3.core.data.io.structure.atom_array import (
    read_atomarray_from_npz,
    write_atomarray_to_npz,
)
from openfold3.core.data.primitives.quality_control.logging_utils import (
    log_runtime_memory,
)
from openfold3.core.data.primitives.structure.cleanup import (
    convert_intra_residue_dative_to_single,
    get_polymer_mask,
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

logger = logging.getLogger(__name__)


class ParsedStructure(NamedTuple):
    structure_file: pdbx.CIFFile | pdb.PDBFile
    atom_array: AtomArray | None


class SkippedStructure(NamedTuple):
    structure_file: pdbx.CIFFile | pdb.PDBFile
    reason: str


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
@log_runtime_memory(runtime_dict_key="runtime-parse-mmcif", multicall=True)
def parse_mmcif(
    file_path: Path | str,
    expand_bioassembly: bool = False,
    include_bonds: bool = True,
    renumber_chain_ids: bool = False,
    extra_fields: list | None = None,
    max_polymer_chains: int | None = None,
    skip_all_zero_occ: bool = True,
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
        skip_all_zero_occ:
            Whether to skip structures where all atoms have zero occupancy. Defaults to
            True.

    Returns:
        A ParsedStructure NamedTuple containing the parsed CIF file and the AtomArray,
        or a SkippedStructure NamedTuple containing the CIF file and the reason for why
        the structure was skipped.
    """

    cif_file = _load_ciffile(file_path)
    cif_data = get_cif_block(cif_file)

    # Try predetermining from the CIF metadata if the structure has too many chains
    if max_polymer_chains is not None:
        n_polymers = get_first_bioassembly_polymer_count(cif_data)

        if n_polymers > max_polymer_chains:
            return SkippedStructure(cif_file, f"Too many polymer chains: {n_polymers}")

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
        try:
            atom_array = pdbx.get_structure(
                **parser_args,
            )
        except Exception as e:
            raise ValueError(f"Failed to parse {file_path}: ") from e

    # Skip structures where all atoms have zero occupancy
    if skip_all_zero_occ and atom_array.occupancy.sum() == 0:
        return SkippedStructure(cif_file, "All atoms have zero occupancy.")

    # Replace author-assigned IDs with PDB-assigned IDs, but transfer over author
    # residue IDs where necessary (see function documentation)
    update_author_to_pdb_labels(atom_array, use_author_res_id_if_missing=True)

    # Add entity IDs
    assign_entity_ids(atom_array)

    # Renumber chain IDs from 1 to avoid duplicate chain labels after bioassembly
    # expansion
    if renumber_chain_ids:
        assign_renumbered_chain_ids(atom_array)

    # Add IDs for major molecular types (PROTEIN, DNA, RNA, LIGAND)
    assign_molecule_type_ids(atom_array, cif_file)

    # Check again if the structure has too many chains based on the actual structure to
    # not only rely on earlier metadata annotation, which may not always be complete.
    if max_polymer_chains is not None:
        n_polymers = get_chain_count(
            atom_array[get_polymer_mask(atom_array, use_molecule_type_id=True)]
        )

        if n_polymers > max_polymer_chains:
            return SkippedStructure(cif_file, f"Too many polymer chains: {n_polymers}")

    return ParsedStructure(cif_file, atom_array)


def _create_cif_file(
    suffix: str, atom_array: AtomArray, data_block: str, include_bonds: bool
):
    """Helper function to create and populate CIF or BCIF files."""
    if suffix == ".cif":
        cif_file = pdbx.CIFFile()
    elif suffix == ".bcif":
        cif_file = pdbx.BinaryCIFFile()
    else:
        raise ValueError("Suffix must be either .cif or .bcif")

    try:
        pdbx.set_structure(
            cif_file, atom_array, data_block=data_block, include_bonds=include_bonds
        )
    # This error sometimes happens in the PDB preprocessing
    except KeyError:
        logger.warning(
            "KeyError while writing structure to CIF file. Retrying with "
            "intra-residue COORDINATION bonds set to SINGLE."
        )
        atom_array = convert_intra_residue_dative_to_single(atom_array)
        pdbx.set_structure(
            cif_file, atom_array, data_block=data_block, include_bonds=include_bonds
        )

    return cif_file


def write_structure(
    atom_array: AtomArray,
    output_path: Path | str,
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
            file suffix. Allowed values are .npz, .cif, .bcif, and .pkl.
        data_block:
            Name of the data block in the CIF/BCIF file. Defaults to None. Ignored if
            the format is not cif or bcif.
        include_bonds:
            Whether to include bond information. Defaults to True. Ignored if the format
            is pkl in which the entire BondList is written to the file.
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)

    suffix = output_path.suffix

    match suffix:
        case ".npz":
            write_atomarray_to_npz(atom_array, output_path)

        case ".pkl":
            with open(output_path, "wb") as f:
                pickle.dump(atom_array, f)

        case ".cif" | ".bcif":
            file_obj = _create_cif_file(
                suffix=suffix,
                atom_array=atom_array,
                data_block=data_block,
                include_bonds=include_bonds,
            )
            file_obj.write(output_path)

        case ".pdb":
            # Ensure that residue names are 3 letters max
            if any(len(name) > 3 for name in atom_array.res_name):
                atom_array = atom_array.copy()
                atom_array.res_name = atom_array.res_name.astype("<U3")

            save_structure(output_path, atom_array)

        case _:
            raise NotImplementedError(
                "Only .cif, .bcif, and .pkl formats are supported"
            )


@log_runtime_memory(runtime_dict_key="runtime-target-structure-proc-parse")
def parse_target_structure(
    target_structures_directory: Path,
    pdb_id: str,
    structure_format: Literal["pkl", "npz"],
) -> AtomArray:
    """Parses a preprocessed structure from a pickle or numpy array.

    Args:
        target_structures_directory (Path):
            Directory containing target structure folders.
        pdb_id (str):
            PDB ID of the target structure.
        structure_format (str):
            File extension of the target structure. Only "pkl" and "npz" are currently
            supported.

    Raises:
        ValueError:
            If the structure format is not "pkl" or "npz".

    Returns:
        AtomArray:
            AtomArray of the target structure.
    """
    target_file = target_structures_directory / pdb_id / f"{pdb_id}.{structure_format}"

    if structure_format == "pkl":
        with open(target_file, "rb") as f:
            atom_array = pickle.load(f)
    elif structure_format == "npz":
        atom_array = read_atomarray_from_npz(target_file)
    else:
        raise ValueError(
            f"Invalid structure format: {structure_format}. Only pickle "
            "or npz formats are supported in a torch dataset __getitem__."
        )

    return atom_array
