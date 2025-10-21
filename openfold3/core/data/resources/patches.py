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

"""Util file for patching bugs in used packages."""

from collections.abc import Generator

import biotite.structure as struc
import networkx as nx
import numpy as np
from biotite.structure.info import link_type


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


_PEPTIDE_LINKS = ["PEPTIDE LINKING", "L-PEPTIDE LINKING", "D-PEPTIDE LINKING"]
_NUCLEIC_LINKS = ["RNA LINKING", "DNA LINKING"]


def connect_via_residue_names(atoms, inter_residue=True, custom_bond_dict=None):
    """
    IMPORTANT: This is a copy of Biotite's original connect_via_residue_names function,
    ported from Cython to Python. It uses the patched _connect_inter_residue function,
    which previously had a case-sensitivity issue with extracted link types that could
    cause disconnected chains.

    ================================================================================
    ORIGINAL DOCSTRING BELOW:
    ================================================================================


    connect_via_residue_names(atoms, atom_mask=None, inter_residue=True)

    Create a :class:`BondList` for a given atom array (stack), based on the deposited
    bonds for each residue in the RCSB ``components.cif`` dataset.

    Bonds between two adjacent residues are created for the atoms expected to connect
    these residues, i.e. ``'C'`` and ``'N'`` for peptides and ``"O3'"`` and ``'P'`` for
    nucleotides.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The structure to create the :class:`BondList` for.
    inter_residue : bool, optional
        If true, connections between consecutive amino acids and nucleotides are also
        added.
    custom_bond_dict : dict (str -> dict ((str, str) -> int)), optional
        A dictionary of dictionaries: The outer dictionary maps residue names to inner
        dictionaries. The inner dictionary maps tuples of two atom names to their
        respective :class:`BondType` (represented as integer). If given, these bonds are
        used instead of the bonds read from ``components.cif``.

    Returns
    -------
    BondList
        The created bond list. No bonds are added for residues that are not found in
        ``components.cif``.

    See also
    --------
    connect_via_distances

    Notes
    -----
    This method can only find bonds for residues in the RCSB *Chemical Component
    Dictionary*, unless `custom_bond_dict` is set. Although this includes most molecules
    one encounters, this will fail for exotic molecules, e.g. specialized inhibitors.

    .. currentmodule:: biotite.structure.info

    To supplement `custom_bond_dict` with bonds for residues from the *Chemical
    Component Dictionary*  you can use :meth:`bonds_in_residue()`.

    >>> import pprint
    >>> custom_bond_dict = {
    ...     "XYZ": {
    ...         ("A", "B"): BondType.SINGLE,
    ...         ("B", "C"): BondType.SINGLE
    ...     }
    ... }
    >>> # Supplement with bonds for common residues
    >>> custom_bond_dict["ALA"] = bonds_in_residue("ALA")
    >>> pp = pprint.PrettyPrinter(width=40)
    >>> pp.pprint(custom_bond_dict)
    {'ALA': {('C', 'O'): <BondType.DOUBLE: 2>,
             ('C', 'OXT'): <BondType.SINGLE: 1>,
             ('CA', 'C'): <BondType.SINGLE: 1>,
             ('CA', 'CB'): <BondType.SINGLE: 1>,
             ('CA', 'HA'): <BondType.SINGLE: 1>,
             ('CB', 'HB1'): <BondType.SINGLE: 1>,
             ('CB', 'HB2'): <BondType.SINGLE: 1>,
             ('CB', 'HB3'): <BondType.SINGLE: 1>,
             ('N', 'CA'): <BondType.SINGLE: 1>,
             ('N', 'H'): <BondType.SINGLE: 1>,
             ('N', 'H2'): <BondType.SINGLE: 1>,
             ('OXT', 'HXT'): <BondType.SINGLE: 1>},
     'XYZ': {('A', 'B'): <BondType.SINGLE: 1>,
             ('B', 'C'): <BondType.SINGLE: 1>}}

    """
    bonds = []
    atom_names = atoms.atom_name
    res_names = atoms.res_name

    residue_starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    # Omit exclusive stop in 'residue_starts'
    for res_i in range(len(residue_starts) - 1):
        curr_start_i = residue_starts[res_i]
        next_start_i = residue_starts[res_i + 1]

        if custom_bond_dict is None:
            bond_dict_for_res = struc.info.bonds_in_residue(res_names[curr_start_i])
        else:
            bond_dict_for_res = custom_bond_dict.get(res_names[curr_start_i], {})

        atom_names_in_res = atom_names[curr_start_i:next_start_i]
        for (atom_name1, atom_name2), bond_type in bond_dict_for_res.items():
            atom_indices1 = np.where(atom_names_in_res == atom_name1)[0].astype(
                np.int64, copy=False
            )
            atom_indices2 = np.where(atom_names_in_res == atom_name2)[0].astype(
                np.int64, copy=False
            )
            # In rare cases the same atom name may appear multiple times
            # (e.g. in altlocs)
            # -> create all possible bond combinations
            for i in range(atom_indices1.shape[0]):
                for j in range(atom_indices2.shape[0]):
                    bonds.append(
                        (
                            curr_start_i + atom_indices1[i],
                            curr_start_i + atom_indices2[j],
                            bond_type,
                        )
                    )

    bond_list = struc.bonds.BondList(atoms.array_length(), np.array(bonds))

    if inter_residue:
        inter_bonds = _connect_inter_residue(atoms, residue_starts)
        return bond_list.merge(inter_bonds)
    else:
        return bond_list


def _connect_inter_residue(atoms, residue_starts):
    """
    IMPORTANT: This is a copy of Biotite's original _connect_inter_residue function
    ported from Cython to Python, and patches an issue introduced by case-sensitivity of
    extracted link types which can cause disconnected chains.

    ================================================================================
    ORIGINAL DOCSTRING BELOW:
    ================================================================================

    Create a :class:`BondList` containing the bonds between adjacent
    amino acid or nucleotide residues.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The structure to create the :class:`BondList` for.
    residue_starts : ndarray, dtype=int
        Return value of
        ``get_residue_starts(atoms, add_exclusive_stop=True)``.

    Returns
    -------
    BondList
        A bond list containing all inter residue bonds.
    """
    bonds = []
    atom_names = atoms.atom_name
    res_names = atoms.res_name
    res_ids = atoms.res_id
    chain_ids = atoms.chain_id

    # Iterate over all starts excluding:
    #   - the last residue and
    #   - exclusive end index of 'atoms'
    for i in range(len(residue_starts) - 2):
        curr_start_i = residue_starts[i]
        next_start_i = residue_starts[i + 1]
        after_next_start_i = residue_starts[i + 2]

        # Check if the current and next residue is in the same chain
        if chain_ids[next_start_i] != chain_ids[curr_start_i]:
            continue
        # Check if the current and next residue
        # have consecutive residue IDs
        # (Same residue ID is also possible if insertion code is used)
        if res_ids[next_start_i] - res_ids[curr_start_i] > 1:
            continue

        # Get link type for this residue from RCSB components.cif
        curr_link = link_type(res_names[curr_start_i]).upper()  # FIX IS HERE
        next_link = link_type(res_names[next_start_i]).upper()  # FIX IS HERE

        if curr_link in _PEPTIDE_LINKS and next_link in _PEPTIDE_LINKS:
            curr_connect_atom_name = "C"
            next_connect_atom_name = "N"
        elif curr_link in _NUCLEIC_LINKS and next_link in _NUCLEIC_LINKS:
            curr_connect_atom_name = "O3'"
            next_connect_atom_name = "P"
        else:
            # Create no bond if the connection types of consecutive
            # residues are not compatible
            continue

        # Index in atom array for atom name in current residue
        # Addition of 'curr_start_i' is necessary, as only a slice of
        # 'atom_names' is taken, beginning at 'curr_start_i'
        curr_connect_indices = (
            curr_start_i
            + np.where(atom_names[curr_start_i:next_start_i] == curr_connect_atom_name)[
                0
            ]
        )
        # Index in atom array for atom name in next residue
        next_connect_indices = (
            next_start_i
            + np.where(
                atom_names[next_start_i:after_next_start_i] == next_connect_atom_name
            )[0]
        )
        if len(curr_connect_indices) == 0 or len(next_connect_indices) == 0:
            # The connector atoms are not found in the adjacent residues
            # -> skip this bond
            continue

        bonds.append(
            (
                curr_connect_indices[0],
                next_connect_indices[0],
                struc.bonds.BondType.SINGLE,
            )
        )

    return struc.bonds.BondList(atoms.array_length(), np.array(bonds, dtype=np.uint32))
