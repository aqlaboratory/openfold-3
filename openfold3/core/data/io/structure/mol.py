"""This module contains IO functions for reading and writing MOL files."""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Mol


def read_single_sdf(path: Path | str) -> Mol:
    """Reads an SDF file and returns the RDKit Mol object.

    Args:
        path:
            Path to the SDF file.

    Returns:
        The RDKit Mol object.
    """
    if not isinstance(path, Path):
        path = Path
    
    reader = Chem.SDMolSupplier(str(path))
    mol = next(reader)

    return mol


# TODO: improve docstring (explain what used_mask is)
def read_single_annotated_sdf(path: Path | str) -> Mol:
    """Reads an annotated SDF file and returns the RDKit Mol object."""

    mol = read_single_sdf(path)
    
    # Set the annotations to atom-wise properties
    used_atom_mask = mol.GetProp("used_atom_mask").split()
    atom_names = mol.GetProp("atom_names").split()
    
    for atom, used, name in zip(mol.GetAtoms(), used_atom_mask, atom_names):
        atom.SetProp("name", name)
        atom.SetProp("used_mask", used)
    
    # delete old properties
    mol.ClearProp("used_atom_mask")
    mol.ClearProp("atom_names")
    
    return mol