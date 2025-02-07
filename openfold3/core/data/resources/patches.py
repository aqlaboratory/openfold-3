"""Util file for patching bugs in used packages."""

from collections.abc import Generator

import biotite.structure as struc
import networkx as nx
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


def get_molecule_indices(atom_array: struc.AtomArray) -> list[np.ndarray]:
    """Alternative implementation of Biotite's get_molecule_indices.

    We are getting segfault errors on rare occasions when using Biotite's
    get_molecule_indices function. This is a temporary alternative implementation that
    should work the same way but is more robust.
    """
    # Currently only works with AtomArrays
    if isinstance(atom_array, struc.AtomArray):
        if atom_array.bonds is None:
            raise ValueError("An associated BondList is required")
        bonds = atom_array.bonds
    else:
        raise TypeError(f"Expected an 'AtomArray', not '{type(atom_array).__name__}'")

    g = bonds.as_graph()

    # Add any atoms that are not in the BondList as single-atom components
    all_atoms = np.arange(len(atom_array))
    atoms_in_graph = np.unique(list(g.nodes))
    singleton_components = np.setdiff1d(all_atoms, atoms_in_graph)
    singleton_components_formatted = [np.array([atom]) for atom in singleton_components]

    # Add connected components and sort each by internal atom index
    connected_components = nx.connected_components(g)
    connected_components_formatted = [np.sort(list(c)) for c in connected_components]

    # Combine indices and do outer sort on first atom index
    all_components = singleton_components_formatted + connected_components_formatted
    all_components_sorted = sorted(all_components, key=lambda x: x[0])

    return all_components_sorted


def molecule_iter(
    atom_array: struc.AtomArray,
) -> Generator[struc.AtomArray, None, None]:
    """Alternative implementation of Biotite's molecule_iter.

    We are getting segfault errors on rare occasions when using Biotite's molecule_iter
    function. This is a temporary alternative implementation that should work the same
    way but is more robust.
    """
    molecule_indices = get_molecule_indices(atom_array)

    for indices in molecule_indices:
        yield atom_array[indices]
