"""This module contains IO functions for reading and writing MOL files."""

from collections import defaultdict
from os import PathLike
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Mol

from openfold3.core.data.io import utils
from openfold3.core.data.primitives.structure.component import AnnotatedMol


def read_single_sdf(path: PathLike) -> Mol:
    """Reads an SDF file and returns the RDKit Mol object of the first entry.

    Convenience method to avoid boilerplate code when reading SDF files that only
    contain one molecule.

    Args:
        path:
            Path to the SDF file.

    Returns:
        The RDKit Mol object.
    """
    reader = Chem.SDMolSupplier(str(path))
    mol = next(reader)

    return mol


def write_annotated_sdf(mol: AnnotatedMol, out: PathLike | str) -> Path:
    """Writes an SDF file from a mol with atom-wise annotations.

    Some molecule objects in data preprocessing contain additional atom-wise
    annotations, like original atom IDs and a mask for used conformer atoms, which SDF
    files don't natively support. Therefore in addition to writing the SDF file, this
    function searches for all atom-wise annotations with an "annot_" prefix and writes
    them as an "atom_annot_{...}" molecule property by joining the values together with
    a space.

    Args:
        mol:
            The molecule to write.
        out:
            Path to the output file.

    Returns:
        The path to the written file.
    """
    mol_annotations = defaultdict(list)

    # Convert atom-wise annotations to global annotation which .sdf can handle
    for atom in mol.GetAtoms():
        atom_annotations = atom.GetPropsAsDict()

        for key, value in atom_annotations.items():
            if key.startswith("annot_"):
                mol_annotations[f"atom_annot_{key[6:]}"].append(str(value))

    # Write the global molecule-level annotations
    for key, value in mol_annotations.items():
        mol.SetProp(key, " ".join(value))

    with Chem.SDWriter(str(out)) as writer:
        writer.write(mol)


def read_single_annotated_sdf(path: PathLike) -> AnnotatedMol:
    """Reads an SDF file with special atom annotations and returns an RDKit Mol.

    This function reads an SDF file, extracts the first molecule, and looks for global
    properties formatted like "atom_annot_{...}" which are then converted to atom-wise
    annotations in the Mol object. The original global properties are deleted.

    Args:
        path:
            Path to the SDF file.

    Returns:
        The RDKit Mol object with atom-wise annotation properties prefixed with
        "annot_". Each property can be accessed with `atom.GetProp("annot_{...}")`.
    """

    mol = read_single_sdf(path)

    mol_annotations = mol.GetPropsAsDict()

    for key, value in mol_annotations.items():
        if key.startswith("atom_annot_"):
            # Remove atom_ prefix
            key = key[5:]

            # Set to atom-wise annotations with proper type
            for atom, annot in zip(mol.GetAtoms(), value.split()):
                if annot.lower() == "true":
                    atom.SetBoolProp(key, True)
                elif annot.lower() == "false":
                    atom.SetBoolProp(key, False)
                elif utils.is_intlike_string(annot):
                    atom.SetIntProp(key, int(annot))
                else:
                    atom.SetProp(key, annot)

            mol.ClearProp(key)  # delete the global property

    return mol
