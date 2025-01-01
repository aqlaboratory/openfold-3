"""Module for any added custom assert functions."""

import numpy as np
from biotite.structure import AtomArray


def assert_atomarray_equal(atom_array_1: AtomArray, atom_array_2: AtomArray):
    """Checks if two AtomArrays are fully equivalent.

    Args:
        atom_array_1 (AtomArray):
            First AtomArray to compare
        atom_array_2 (AtomArray):
            Second AtomArray to compare

    Raises:
        AssertionError
    """
    annotations = atom_array_1.get_annotation_categories()

    assert (
        annotations == atom_array_2.get_annotation_categories()
    ), "AtomArrays have different annotations."

    annotations.append("coord")

    for annotation in annotations:
        values_1 = getattr(atom_array_1, annotation)
        values_2 = getattr(atom_array_2, annotation)

        if annotation == "coord":
            equal_nan = True
        else:
            equal_nan = False

        assert np.array_equal(
            values_1, values_2, equal_nan=equal_nan
        ), f"AtomArrays have different values for: {annotation}."

    bonds_1 = atom_array_1.bonds
    bonds_2 = atom_array_2.bonds

    # If no BondList for both, skip check
    if bonds_1 is None and bonds_2 is None:
        return

    assert (
        bonds_1 is not None and bonds_2 is not None
    ), "Only one of the AtomArrays has an undefined BondList."

    bondlist_1 = atom_array_1.bonds.as_array()
    bondlist_2 = atom_array_2.bonds.as_array()

    assert np.array_equal(
        bondlist_1, bondlist_2
    ), "AtomArrays have different BondLists."
