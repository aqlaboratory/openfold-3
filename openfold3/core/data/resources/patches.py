"""Util file for patching bugs in used packages."""

import re

import biotite.structure as struc


def construct_atom_array(atoms: list[struc.Atom]) -> struc.AtomArray:
    """Patches the Biotite structure.array function.

    Biotite's function infers the dtype of annotations from a type() call on the first
    element which is then used to initialize the annotation array in the AtomArray. This
    is problematic, because if a new array is created with np.str_ dtype, it will
    default to dtype '<U1' which will truncate longer strings to a single character.
    This function patches this by calling .dtype or type() on the first element
    depending on whether it's a numpy type or not, which will result in numpy types
    getting a more accurate dtype. This is not the most general patch but works for our
    purposes because the first element's dtype will be consistent with the other atoms
    in the list.
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
    for name in names:
        first_atom_val = atoms[0]._annot[name]

        ##### PATCH START #####
        try:
            dtype = first_atom_val.dtype
        except AttributeError:
            dtype = type(first_atom_val)

        array.add_annotation(name, dtype=dtype)
        ##### PATCH END #####

    # Add all atoms to AtomArray
    for i in range(len(atoms)):
        for name in names:
            array._annot[name][i] = atoms[i]._annot[name]
        array._coord[i] = atoms[i].coord
    return array

