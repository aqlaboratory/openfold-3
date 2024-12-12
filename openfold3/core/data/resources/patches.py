"""Util file for patching bugs in used packages."""

import re

import biotite.structure as struc
import numpy as np


def construct_atom_array(atoms: list[struc.Atom]) -> struc.AtomArray:
    """Patches the Biotite structure.array function.

    Biotite's function infers the dtype of annotations from a type() call on the first
    atom which is then used to initialize the annotation array in the AtomArray. This is
    problematic, because if a new array is created with np.str_ dtype, it will default
    to dtype '<U1' which will truncate longer strings to a single character. This
    function patches this by creating and assigning numpy arrays for every annotation
    considering all atoms at once, which will infer the correct dtype with numpy
    automatically.
    """
    # CODE COPIED FROM https://github.com/biotite-dev/biotite/blob/main/src/biotite/structure/atoms.py#L1176

    # Check if all atoms have the same annotation names
    # Equality check requires sorting
    names = sorted(atoms[0]._annot.keys())
    for i, atom in enumerate(atoms):
        if sorted(atom._annot.keys()) != names:
            raise ValueError(
                f"The atom at index {i} does not share the same "
                f"annotation categories as the atom at index 0"
            )
    array = struc.AtomArray(len(atoms))
    # Add all (also optional) annotation categories
    ##### PATCH START #####
    for name in names:
        # This will infer the correct dtype as well
        annot_array = np.array([atom._annot[name] for atom in atoms])
        array.set_annotation(name, annot_array)

    array._coord = np.array([atom.coord for atom in atoms])
    ##### PATCH END #####

    return array


def correct_cif_string(cif_str: str, ccd_id: str):
    """Temporary fix for a current bug in Biotite CIFBlock.serialize()

    Essentially adds back erroneously missing line-breaks between comments and data
    blocks. Also adds the data block name as a header.

    Args:
        cif_str:
            CIF string to fix.
        ccd_id:
            CCD ID of the component to extract.

    Returns:
        Fixed CIF string.
    """
    # Matches `#` or `#  #` followed by a character
    pattern = r"(^#\s*#?\s*)(\S)"

    # Puts a newline between the two matched groups
    fixed_str = re.sub(pattern, r"\1\n\2", cif_str, flags=re.MULTILINE)

    return f"data_{ccd_id}\n{fixed_str}"
