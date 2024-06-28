from pathlib import Path
from typing import NamedTuple

import biotite.structure as struc
from biotite.structure.io import pdbx
from structure_primitives import assign_renumbered_chain_ids


class ParsedStructure(NamedTuple):
    cif_file: pdbx.CIFFile
    atom_array: struc.AtomArray


def parse_mmcif_bioassembly(
    file_path: Path | str,
    use_author_fields=False,
    include_bonds=True,
    add_renumbered_chains=True,
) -> ParsedStructure:
    """Convenience wrapper around biotite's bioassembly CIF parsing

    Parses the mmCIF file, expands the first bioassembly, and creates an
    AtomArray from it. This includes only the first model, resolves alternative
    locations by taking the one with the highest occupancy, defaults to
    inferring bond information, and defaults to using the PDB-automated
    chain/residue annotation instead of author annotations.
    
    This function also creates a "chain_ids_renumbered" field in the AtomArray
    with numerical chain IDs starting from 0, which can be useful when
    bioassembly expansion results in multiple chains with the same chain ID.

    Args:
        file_path:
            Path to the mmCIF file.
        use_author_fields:
            Whether to use author fields. Defaults to False.
        include_bonds:
            Whether to infer bond information. Defaults to True.

    Returns:
        A NamedTuple containing the parsed CIF file and the AtomArray.
    """
    cif_file = pdbx.CIFFile.read(file_path)
    atom_array = pdbx.get_assembly(
        cif_file,
        assembly_id="1",
        model=1,
        altloc="occupancy",
        use_author_fields=use_author_fields,
        include_bonds=include_bonds,
    )

    atom_array = assign_renumbered_chain_ids(atom_array)

    return ParsedStructure(cif_file, atom_array)
