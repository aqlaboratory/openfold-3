import numpy as np
from biotite.structure import AtomArray, BondType
from pdbeccdutils.core.ccd_reader import Component
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from rdkit.Geometry import Point3D

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


def compute_conformer(mol: Mol, cleanup_hs: bool = True) -> Mol:
    """Computes 3D coordinates for a molecule using RDKit.

    Args:
        mol:
            The molecule for which the 3D coordinates should be computed.
        cleanup_hs:
            If set to True, hydrogens are removed after the generation of 3D
            coordinates. Defaults to False.

    """
    Chem.AddHs(mol)

    strategy = AllChem.ETKDGv3()
    return_code = AllChem.EmbedMolecule(mol, strategy)

    if return_code == -1:
        # TODO: remove
        print("Failed to generate 3D coordinates, trying random coordinates")
        strategy.useRandomCoords = True

        return_code = AllChem.EmbedMolecule(mol, strategy)

        if return_code == -1:
            # TODO: remove
            print("Failed to generate 3D coordinates with random coordinates")
            raise ConformerGenerationError("Failed to generate 3D coordinates")

    conf = mol.GetConformer(mol.GetNumConformers() - 1)
    conf.SetProp("name", "Computed")  # following pdbeccdutils ConformerType

    if cleanup_hs:
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

            # assert that not all coordinates are 0, dev-only TODO: remove
            assert not all(
                all(coord == 0 for coord in fallback_conf.GetAtomPosition(atom_id))
                for atom_id in range(mol.GetNumAtoms())
            )
        elif "Model" in conf_names and mol.GetProp("model_pdb_id") != "?":
            fallback_conf = mol.GetConformer(conf_names.index("Model"))
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

        mol.AddAtom(Chem.Atom(new_atom))

    for atom_1, atom_2, bond_type_id in atom_array.bonds.as_array():
        mol.AddBond(
            int(atom_1), int(atom_2), bondtype_conversion[BondType(bond_type_id)]
        )
    mol.CommitBatchEdit()

    mol = Chem.RemoveAllHs(mol)
    Chem.SanitizeMol(mol)
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
    conf_types = ["Ideal", "Model"]
    coords_suffixes = ["x", "y", "z"]

    # Get masks for "unused" atoms
    conf_masks = {
        conf_type: np.ones(len(mol.GetAtoms()), dtype=bool) for conf_type in conf_types
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
            assert len(coords) == len(mol.GetAtoms())

            for i, value in enumerate(coords):
                if value in [".", "?"]:
                    conf_masks[conf_type][i] = False

        mol.SetProp(
            f"{conf_type.lower()}_used_atom_mask",
            "".join(str(int(x)) for x in conf_masks[conf_type]),
        )

    # Get PDB ID of the structure that model coordinates are taken from
    model_pdb_id = cif_block.find_value("pdbx_model_coordinates_db_code")
    if model_pdb_id is None:
        model_pdb_id = "?"
    mol.SetProp("model_pdb_id", model_pdb_id)

    if assign_fallback_conformer:
        mol = set_fallback_conformer(mol)

    return mol
