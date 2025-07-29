"""Module for any added custom assert functions."""

import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem

from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)


def assert_atomarray_equal(
    atom_array_1: AtomArray, atom_array_2: AtomArray, strict_annot_order: bool = True
):
    """Checks if two AtomArrays are fully equivalent.

    Args:
        atom_array_1 (AtomArray):
            First AtomArray to compare
        atom_array_2 (AtomArray):
            Second AtomArray to compare
        strict_order (bool):
            If True, checks that annotations are in the same order. Otherwise only
            checks that annotation categories are identical.

    Raises:
        AssertionError
    """
    annotations_1 = atom_array_1.get_annotation_categories()
    annotations_2 = atom_array_2.get_annotation_categories()

    if not strict_annot_order:
        # If strict order is not required, just check if categories are the same
        annotations_1 = sorted(annotations_1)
        annotations_2 = sorted(annotations_2)

    assert annotations_1 == annotations_2, "AtomArrays have different annotations."

    annotations_1.append("coord")

    for annotation in annotations_1:
        values_1 = getattr(atom_array_1, annotation)
        values_2 = getattr(atom_array_2, annotation)

        if annotation == "coord":
            equal_nan = True
        else:
            equal_nan = False

        assert np.array_equal(values_1, values_2, equal_nan=equal_nan), (
            f"AtomArrays have different values for: {annotation}."
        )

    bonds_1 = atom_array_1.bonds
    bonds_2 = atom_array_2.bonds

    # If no BondList for both, skip check
    if bonds_1 is None and bonds_2 is None:
        return

    assert bonds_1 is not None and bonds_2 is not None, (
        "Only one of the AtomArrays has an undefined BondList."
    )

    bondlist_1 = atom_array_1.bonds.as_array()
    bondlist_2 = atom_array_2.bonds.as_array()

    assert np.array_equal(bondlist_1, bondlist_2), (
        "AtomArrays have different BondLists."
    )


def assert_ref_mols_equal(
    ref_mol_1: ProcessedReferenceMolecule, ref_mol_2: ProcessedReferenceMolecule
):
    """Checks if two ProcessedReferenceMolecules are equivalent.

    Verifies that both molecules yield the same SMILES in the same order, and checks
    that all internal annotations are the same.

    Args:
        ref_mol_1 (ProcessedReferenceMolecule):
            First ProcessedReferenceMolecule to compare
        ref_mol_2 (ProcessedReferenceMolecule):
            Second ProcessedReferenceMolecule to compare

    Raises:
        AssertionError
    """
    # Check that molecules are equivalent (order-sensitive)
    smiles_1 = Chem.MolToSmiles(ref_mol_1.mol, canonical=False)
    smiles_2 = Chem.MolToSmiles(ref_mol_2.mol, canonical=False)

    assert smiles_1 == smiles_2, (
        "Reference molecules have different SMILES representations."
    )

    # Check internal molecule properties
    for atom_1, atom_2 in zip(ref_mol_1.mol.GetAtoms(), ref_mol_2.mol.GetAtoms()):
        if atom_1.HasProp("annot_atom_name") and atom_2.HasProp("annot_atom_name"):
            assert atom_1.GetProp("annot_atom_name") == atom_2.GetProp(
                "annot_atom_name"
            ), "Reference molecules have different annot_atom_name annotation."

        if atom_1.HasProp("annot_used_atom_mask") and atom_2.HasProp(
            "annot_used_atom_mask"
        ):
            assert atom_1.GetProp("annot_used_atom_mask") == atom_2.GetProp(
                "annot_used_atom_mask"
            ), "Reference molecules have different annot_used_atom_mask annotation."

    # Check remaining attributes
    assert np.array_equal(ref_mol_1.in_crop_mask, ref_mol_2.in_crop_mask), (
        "Reference molecules have different in_crop_mask."
    )
    assert ref_mol_1.component_id == ref_mol_2.component_id, (
        "Reference molecules have different component_id."
    )
    if (ref_mol_1.permutations is not None) and (ref_mol_2.permutations is not None):
        for perm_1, perm_2 in zip(ref_mol_1.permutations, ref_mol_2.permutations):
            assert np.array_equal(perm_1, perm_2), (
                "Reference molecules have different permutations."
            )
