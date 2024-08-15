import contextlib
import logging

import numpy as np
from biotite.structure import AtomArray, BondType
from pdbeccdutils.core.ccd_reader import Component
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from rdkit.Geometry import Point3D

logger = logging.getLogger(__name__)

PERIODIC_TABLE = Chem.GetPeriodicTable()

# Biotite -> RDKit bond conversion NOTE: ANY is converted to SINGLE because Biotite
# relies on _struct_conn.pdbx_value_order for inter-residue bond orders, however this
# category is not present in the vast majority of CIF files (see
# https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Items/_struct_conn.pdbx_value_order.html)
bondtype_conversion = {
    BondType.ANY: Chem.BondType.SINGLE,
    BondType.SINGLE: Chem.BondType.SINGLE,
    BondType.DOUBLE: Chem.BondType.DOUBLE,
    BondType.TRIPLE: Chem.BondType.TRIPLE,
    BondType.QUADRUPLE: Chem.BondType.QUADRUPLE,
    BondType.AROMATIC_SINGLE: Chem.BondType.AROMATIC,
    BondType.AROMATIC_DOUBLE: Chem.BondType.AROMATIC,
    BondType.AROMATIC_TRIPLE: Chem.BondType.AROMATIC,
}


class ConformerGenerationError(ValueError):
    """An error that is raised when the generation of a conformer fails."""

    pass


# TODO: could improve warning handling of this to send less UFFTYPER warnings
def compute_conformer(mol: Mol) -> Mol:
    """Computes 3D coordinates for a molecule using RDKit.

    Args:
        mol:
            The molecule for which the 3D coordinates should be computed.
    """
    try:
        mol = Chem.AddHs(mol)
    except Exception as e:
        logger.warning(f"Failed to add hydrogens before conformer generation: {e}")

    strategy = AllChem.ETKDGv3()
    try:
        return_code = AllChem.EmbedMolecule(mol, strategy)
    except Exception as e:
        logger.warning(f"Exception when calling EmbedMolecule: {e}")
        return_code = -1

    if return_code == -1:
        # TODO: remove
        print("Failed to generate 3D coordinates, trying random coordinates")

        strategy.useRandomCoords = True

        try:
            return_code = AllChem.EmbedMolecule(mol, strategy)
        except Exception as e:
            logger.warning(
                f"Exception when calling EmbedMolecule with random coordinates: {e}"
            )
            return_code = -1

        if return_code == -1:
            with contextlib.suppress(Exception):
                mol = Chem.RemoveAllHs(mol)
            # TODO: remove
            print("Failed to generate 3D coordinates with random coordinates")
            raise ConformerGenerationError("Failed to generate 3D coordinates")

    conf = mol.GetConformer(mol.GetNumConformers() - 1)
    conf.SetProp("name", "Computed")  # following pdbeccdutils ConformerType

    mol = Chem.RemoveAllHs(mol)

    return mol


def set_fallback_conformer(mol: Mol) -> Chem.Mol:
    # Try conformer generation
    try:
        # If it works, no need to supply a conformer because dataloading can handle it
        # on the fly
        mol = compute_conformer(mol)
        mol.SetProp("use_conformer", "False")
    except ConformerGenerationError:
        conf_names = [conf.GetProp("name") for conf in mol.GetConformers()]

        # Look for Ideal and Model coordinates in that order and use the first one that
        # is present as fallback conformer
        if "Ideal" in conf_names:
            fallback_conf = mol.GetConformer(conf_names.index("Ideal"))
            mol.ClearProp("model_pdb_id")
            mol.ClearProp("model_used_atom_mask")

        elif "Model" in conf_names and mol.GetProp("model_pdb_id") != "?":
            logger.debug("Ideal not found, using Model coordinates as fallback")

            fallback_conf = mol.GetConformer(conf_names.index("Model"))
            mol.ClearProp("ideal_used_atom_mask")

            # assert that not all coordinates are 0, dev-only TODO: remove
            assert not all(
                all(coord == 0 for coord in fallback_conf.GetAtomPosition(atom_id))
                for atom_id in range(mol.GetNumAtoms())
            )
        # If no valid conformer is found, use a fallback conformer with all atoms at the
        # origin
        else:
            # TODO: make this a primitive because featurization will need it again
            fallback_conf = Chem.Conformer(mol.GetNumAtoms())
            for atom_id in range(mol.GetNumAtoms()):
                fallback_conf.SetAtomPosition(atom_id, Point3D(0, 0, 0))

        # Clean up everything but the fallback conformer
        mol = Chem.Mol(mol)  # make a copy, see rdkit issue #3817
        mol.RemoveAllConformers()
        mol.AddConformer(fallback_conf, assignId=True)

        mol.SetProp("use_conformer", "True")

    return mol


def mol_from_atomarray(
    atom_array: AtomArray, assign_fallback_conformer: bool = True
) -> Mol:
    mol = AllChem.RWMol()

    mol.BeginBatchEdit()
    for atom in atom_array:
        element = atom.element.capitalize()
        atomic_number = PERIODIC_TABLE.GetAtomicNumber(element)

        isotope = None

        if element == "X":
            element = "*"
        elif element == "D":
            element = "H"
            isotope = 2

        new_atom = Chem.Atom(atomic_number)

        if isotope is not None:
            new_atom.SetIsotope(isotope)

        new_atom.SetFormalCharge(int(atom.charge.item()))

        mol.AddAtom(Chem.Atom(new_atom))

    for atom_1, atom_2, bond_type_id in atom_array.bonds.as_array():
        mol.AddBond(
            int(atom_1), int(atom_2), bondtype_conversion[BondType(bond_type_id)]
        )
    mol.CommitBatchEdit()

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            logger.warning("Failed to sanitize molecule, trying to remove charges")
            for atom in mol.GetAtoms():
                atom.SetFormalCharge(0)
            Chem.SanitizeMol(mol)
        except Exception as e:
            logger.warning(f"Failed to sanitize molecule: {e}")

    Chem.AssignStereochemistryFrom3D(mol)

    # Add original atom IDs as properties
    mol.SetProp(
        "atom_names", " ".join(str(atom_id) for atom_id in atom_array.atom_name)
    )

    if assign_fallback_conformer:
        mol = set_fallback_conformer(mol)

    return mol


# TODO: explain in docstring that this follows the pdbeccdutils Component Mol
def mol_from_parsed_component(
    component: Component, assign_fallback_conformer: bool = True
) -> Mol:
    # Get mol
    mol = component.mol

    cif_block = component.ccd_cif_block

    # Atom elements in original CCD entry (including Hs)
    reference_atom_elements = list(cif_block.find_values("_chem_comp_atom.type_symbol"))

    # TODO: remove
    assert len(reference_atom_elements) == mol.GetNumAtoms()

    # Remove hydrogens
    try:
        mol = Chem.RemoveAllHs(mol)
    except Exception:
        mol = Chem.RemoveAllHs(mol, sanitize=False)

    # Set atom names as property
    atom_names = [
        name
        for name, element in zip(component.atoms_ids, reference_atom_elements)
        if element not in ("H", "D")
    ]
    mol.SetProp("atom_names", " ".join(atom_names))

    # TODO: remove
    assert len(atom_names) == mol.GetNumAtoms()

    ## Get masks for "unused" atoms in the individual conformers
    conf_types = ["Ideal", "Model"]
    coords_suffixes = ["x", "y", "z"]
    conf_masks = {
        conf_type: np.ones(mol.GetNumAtoms(), dtype=bool) for conf_type in conf_types
    }
    for conf_type in conf_types:
        coord_layout = (
            "_chem_comp_atom.model_Cartn_{}"
            if conf_type == "Model"
            else "_chem_comp_atom.pdbx_model_Cartn_{}_ideal"
        )

        for suffix in coords_suffixes:
            coords = list(cif_block.find_values(coord_layout.format(suffix)))

            # TODO: remove
            assert len(coords) == len(reference_atom_elements)

            # Subset to non-hydrogen coords
            coords = [
                coord
                for coord, element in zip(coords, reference_atom_elements)
                if element not in ("H", "D")
            ]

            # TODO: remove
            assert len(coords) == mol.GetNumAtoms()

            for i, value in enumerate(coords):
                if value in [".", "?"]:
                    conf_masks[conf_type][i] = False

        mol.SetProp(
            f"{conf_type.lower()}_used_atom_mask",
            " ".join(str(int(x)) for x in conf_masks[conf_type]),
        )

    # Get PDB ID of the structure that model coordinates are taken from
    model_pdb_id = cif_block.find_value("pdbx_model_coordinates_db_code")
    if model_pdb_id is None:
        model_pdb_id = "?"
    mol.SetProp("model_pdb_id", model_pdb_id)

    if assign_fallback_conformer:
        mol = set_fallback_conformer(mol)

    return mol
