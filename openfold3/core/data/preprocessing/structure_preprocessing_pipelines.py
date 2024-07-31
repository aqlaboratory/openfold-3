"""
Centralized module for pre-assembled workflows corresponding to structure cleanup
procedures of different models.
"""

# TODO: give this file a different name

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFBlock, CIFFile

from openfold3.core.data.preprocessing.metadata_extraction import (
    get_experimental_method,
)
from openfold3.core.data.preprocessing.structure_cleanup import (
    convert_MSE_to_MET,
    fix_arginine_naming,
    remove_chains_with_CA_gaps,
    remove_clashing_chains,
    remove_crystallization_aids,
    remove_fully_unknown_polymers,
    remove_hydrogens,
    remove_non_CCD_atoms,
    remove_small_polymers,
    remove_waters,
    subset_large_structure,
)
from openfold3.core.data.preprocessing.tokenization import tokenize_atom_array


def cleanup_structure_af3(
    atom_array: AtomArray, cif_data: CIFBlock, ccd: CIFFile
) -> AtomArray:
    """Cleans up a structure following the AlphaFold3 SI

    This function applies all cleaning steps outlined in the AlphaFold3 SI 2.5.4. The
    only non-applied filters are the release date and resolution filters as those are
    deferred to the training cache generation script for easier adjustment in the
    future.

    Args:
        atom_array:
            AtomArray containing the structure to clean up
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see
            `metadata_extraction.get_cif_block`)
        ccd:
            CIFFile containing the parsed CCD (components.cif)

    Returns:
        AtomArray with all cleaning steps applied
    """
    atom_array = atom_array.copy()

    convert_MSE_to_MET(atom_array)
    fix_arginine_naming(atom_array)
    atom_array = remove_waters(atom_array)

    if get_experimental_method(cif_data) == "X-RAY DIFFRACTION":
        atom_array = remove_crystallization_aids(atom_array)

    atom_array = remove_hydrogens(atom_array)
    atom_array = remove_small_polymers(atom_array)
    atom_array = remove_fully_unknown_polymers(atom_array)
    atom_array = remove_clashing_chains(atom_array)
    atom_array = remove_non_CCD_atoms(atom_array, ccd)
    atom_array = remove_chains_with_CA_gaps(atom_array)

    return atom_array
