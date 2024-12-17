# TODO: note in module level docstrings that nothing here supports hydrogens
import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import NamedTuple, TypeAlias

import biotite.structure as struc
import gemmi
import numpy as np
from biotite.structure import AtomArray, BondType
from biotite.structure.io.pdbx import CIFFile
from pdbeccdutils.core import ccd_reader
from pdbeccdutils.core.ccd_reader import Component
from rdkit import Chem
from rdkit.Chem import AllChem, Mol

from openfold3.core.data.resources.patches import correct_cif_string
from openfold3.core.data.resources.residues import MoleculeType

logger = logging.getLogger(__name__)

AnnotatedMol: TypeAlias = Mol
"""An RDKit mol object containing additional atom-wise annotations.

The custom atom-wise annotations are stored as atom properties in the Mol object,
following the schema "{property_name}_annot"
"""

PERIODIC_TABLE = Chem.GetPeriodicTable()

# Biotite -> RDKit bond conversion
# --------------------------------
# NOTE: ANY is converted to SINGLE because Biotite relies on
# _struct_conn.pdbx_value_order for inter-residue bond orders, however this category is
# not present in the vast majority of CIF files (see
# https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Items/_struct_conn.pdbx_value_order.html)
# and we therefore have to assume that inter-residue bonds are single if not explicitly
# stated otherwise
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


class PDBComponents(NamedTuple):
    """Named tuple grouping all the molecular components of a PDB structure.

    residue_components:
        List of all 3-letter codes of residues that are part of a polymer chain.
    standard_ligands:
        Dictionary mapping each unique ligand 3-letter code to the list of respective
        chain IDs
    non_standard_ligands:
        Dictionary mapping each unique non-standard ligand entity ID to the list of
        respective chain IDs. A non-standard ligand can generally be any ligand not
        directly mapping to a CCD code, which in this case are usually covalently
        connected multi-component ligands like glycans or certain BIRDs.
    """

    residue_components: list[str]
    standard_ligands: dict[str, list[str]]
    non_standard_ligands: dict[int, list[str]]


def set_atomwise_annotation(
    mol: Mol, property_name: str, annotations: Iterable
) -> AnnotatedMol:
    """Sets atom-wise annotations in an RDKit molecule object.

    This function takes an iterable as argument and assigns the values to atom-wise
    properties in the RDKit molecule in-place, following the naming scheme
    "annot_{property_name}". The prefix is needed for the annotations to be recognized
    by io-functions like `write_single_annotated_sdf`.

    Args:
        mol:
            RDKit molecule object to set the annotations in.
        property_name:
            Name of the property. The full annotation name will be
            "annot_{property_name}".
        annotations:
            Iterable containing the values to set as annotations. The length of the
            iterable must match the number of atoms in the molecule.

    Returns:
        An RDKit molecule object with the atom-wise annotations set as properties under
        "annot_{property_name}".
    """
    for atom, annotation in zip(mol.GetAtoms(), annotations):
        if isinstance(annotation, bool):
            atom.SetBoolProp(f"annot_{property_name}", annotation)
        elif isinstance(annotation, int):
            atom.SetIntProp(f"annot_{property_name}", annotation)
        else:
            atom.SetProp(f"annot_{property_name}", str(annotation))

    return mol


def get_components(atom_array: AtomArray) -> PDBComponents:
    """Extracts all unique components from an AtomArray.

    Standard residue and ligand components correspond to molecular building blocks of
    the structure which are described in the Chemical Component Dictionary (CCD), such
    as polymeric amino acid and nucleotide residues as well as single ligands.

    The "non_standard_ligand" components on the other hand represent ligand molecules
    that have no direct CCD representative, such as glycans or certain BIRDs which
    consist of multiple covalently-linked CCD entries but should be treated as a single
    molecule by the data pipeline's conformer generation.

    Args:
        atom_array:
            AtomArray containing the structure to extract components from.

    Returns:
        A PDBComponents named tuple containing categorized components of the PDB
        structure. See PDBComponents for more information.
    """
    residue_components = set()
    standard_ligands_to_chain = defaultdict(list)
    non_standard_ligands_to_chain = defaultdict(list)

    ligand_filter = atom_array.molecule_type_id == MoleculeType.LIGAND

    # Get residue components
    for resname in np.unique(atom_array[~ligand_filter].res_name):
        residue_components.add(resname.item())

    # Get ligand components
    ligand_atom_array = atom_array[ligand_filter]
    if ligand_atom_array.array_length() > 0:
        for ligand_chain in struc.chain_iter(ligand_atom_array):
            chain_id = ligand_chain.chain_id[0].item()

            # Append standard single-residue ligand
            if struc.get_residue_count(ligand_chain) == 1:
                ccd_id = ligand_chain.res_name[0].item()
                standard_ligands_to_chain[ccd_id].append(chain_id)
            # Append non-standard multi-residue ligand
            else:
                entity_id = ligand_chain.entity_id[0].item()
                non_standard_ligands_to_chain[entity_id].append(chain_id)

    # TODO: remove later
    # Check that all ligands of the same entity have the same atoms
    for entity_id in non_standard_ligands_to_chain:
        entity_atom_array = atom_array[atom_array.entity_id == entity_id]
        chain_atom_names = set()
        for chain in struc.chain_iter(entity_atom_array):
            atom_names = tuple(chain.atom_name.tolist())
            chain_atom_names.add(atom_names)

        # Asserts that all chains of the same entity have the exact same atoms in the
        # exact same order
        # TODO: improve atom expansion for non-standard ligands
        assert len(chain_atom_names) == 1

    return PDBComponents(
        residue_components=list(residue_components),
        standard_ligands=dict(standard_ligands_to_chain),
        non_standard_ligands=dict(non_standard_ligands_to_chain),
    )


def pdbeccdutils_component_from_ccd(ccd_id: str, ccd: CIFFile) -> Component:
    """Creates a pdbeccdutils Component object from a CCD entry in a CIFFile.

    Internally uses Biotite's serialize() function to convert the CIFBlock to a string
    which can be read by gemmi, which pdbeccdutils is based on. This avoids parsing the
    CCD twice.

    Args:
        ccd_id:
            CCD ID of the component to extract.
        ccd:
            CIFFile containing the CCD entry.

    Returns:
        pdbeccdutils Component object representing the CCD entry.
    """
    cif_block = ccd[ccd_id]
    cif_str = cif_block.serialize()
    cif_str = correct_cif_string(cif_str, ccd_id)

    # Manually recreate ccd_reader.read_pdb_cif_file but using a string instead of
    # file-path input
    doc = gemmi.cif.read_string(cif_str)
    block = doc.sole_block()
    ccd_reader_result = ccd_reader._parse_pdb_mmcif(block)

    return ccd_reader_result.component


def remove_hydrogen_values(values: Iterable, atom_elements: Iterable) -> list:
    """Convenience method to remove values corresponding to hydrogens.

    Takes a list of values, and a list of atom elements, and will only return the values
    where the corresponding atom is not a hydrogen.

    Args:
        values:
            List of values to filter.
        atom_elements:
            List of atom elements corresponding to the values.

    Returns:
        List of values where the corresponding atom is not a hydrogen.
    """
    return [x for x, element in zip(values, atom_elements) if element not in ("H", "D")]


def safe_remove_all_hs(mol: Mol) -> Mol:
    """Safely removes all hydrogens from an RDKit molecule.

    Removes all hydrogens from the molecule. In case the built-in sanitization fails,
    reruns hydrogen removal without sanitization.

    Args:
        mol:
            RDKit molecule object to remove hydrogens from.

    Returns:
        The RDKit molecule object with all hydrogens removed.
    """
    try:
        mol = Chem.RemoveAllHs(mol)
    except Exception:
        mol = Chem.RemoveAllHs(mol, sanitize=False)

    return mol


def mol_from_pdbeccdutils_component(
    component: Component,
) -> AnnotatedMol:
    """Extracts a cleaned-up RDKit Mol object from a pdbeccdutils Component object.

    Extracts the Mol object from the Component, and applies the following cleanup steps:
        - Remove hydrogens from the Mol object
        - Change missing coordinates in the stored conformers to NaN
        - Remove the "Ideal" and/or "Model" conformers if their coordinates are all NaN

    Args:
        component:
            pdbeccdutils Component object to extract the Mol object from.

    Returns:
        An RDKit Mol object with the specified cleanup steps applied. The returned Mol
        will have the following properties:
            - "atom_name_annot": Original atom names from the CCD entry
            - "model_pdb_id": PDB ID of the structure that the "Model" coordinates are
                taken from
            - the "Ideal" conformer under confID=0 from the original CCD entry (if not
                removed in cleanup)
            - the "Model" conformer under confID=1 from the original CCD entry (if not
                removed in cleanup)
    """
    # Get mol
    mol = component.mol

    # Get original CCD CIF information
    cif_block = component.ccd_cif_block

    # Atom elements in original CCD entry (including Hs)
    reference_atom_elements = list(cif_block.find_values("_chem_comp_atom.type_symbol"))

    # TODO: remove
    assert len(reference_atom_elements) == mol.GetNumAtoms()

    # Remove hydrogens from the Mol object itself
    try:
        mol = Chem.RemoveAllHs(mol)
    except Exception:
        mol = Chem.RemoveAllHs(mol, sanitize=False)

    # Set (non-hydrogen) atom names as property
    atom_names = remove_hydrogen_values(component.atoms_ids, reference_atom_elements)
    mol = set_atomwise_annotation(mol, "atom_name", atom_names)

    # TODO: remove
    assert len(atom_names) == mol.GetNumAtoms()

    # If any "Ideal" coordinates are missing, all the coordinates should be missing and
    # we should remove the corresponding conformer
    if cif_block.find_value("_chem_comp.pdbx_ideal_coordinates_missing_flag") == "Y":
        ideal_conf = mol.GetConformer(0)
        assert ideal_conf.GetProp("name") == "Ideal"
        mol.RemoveConformer(0)

    ## "Model" coordinates can be partially missing -> set to NaN for cleaner handling
    model_conf = mol.GetConformer(1)
    cif_coord_section = "_chem_comp_atom.model_Cartn_{}"

    all_nan = True
    for coord_axis in ["x", "y", "z"]:
        axis_coords = list(cif_block.find_values(cif_coord_section.format(coord_axis)))
        axis_coords = remove_hydrogen_values(axis_coords, reference_atom_elements)

        for i, value in enumerate(axis_coords):
            if value in [".", "?"]:
                model_conf.SetAtomPosition(i, [float("nan")] * 3)
            else:
                all_nan = False

    # If all coordinates are missing, also remove the model conformer
    if all_nan:
        mol.RemoveConformer(1)
    else:
        # Get PDB ID of the structure that model coordinates are taken from
        model_pdb_id = cif_block.find_value("pdbx_model_coordinates_db_code")
        if model_pdb_id is None:
            model_pdb_id = "?"
        mol.SetProp("model_pdb_id", model_pdb_id)

    return mol


def mol_from_ccd_entry(ccd_id: str, ccd: CIFFile) -> AnnotatedMol:
    """Generates an RDKit Mol object from a CCD entry in a CIFFile.

    Convenience wrapper around `pdbeccdutils_component_from_ccd` and
    `mol_from_pdbeccdutils_component` which extracts the CCD entry from the CIFFile,
    converts it to a pdbeccdutils Component object, and generates a cleaned-up RDKit Mol
    object from it.

    Args:
        ccd_id:
            CCD ID of the component to extract.
        ccd:
            CIFFile containing the CCD entry.

    Returns:
        An RDKit Mol object representing the CCD entry. The returned Mol will have the
        following properties:
            - "atom_name_annot": Original atom names from the CCD entry
            - "model_pdb_id": PDB ID of the structure that the "Model" coordinates are
                taken from
            - the "Ideal" conformer under confID=0 from the original CCD entry (if not
                removed in cleanup)
            - the "Model" conformer under confID=1 from the original CCD entry (if not
                removed in cleanup)
    """
    component = pdbeccdutils_component_from_ccd(ccd_id, ccd)
    mol = mol_from_pdbeccdutils_component(component)

    return mol


def mol_from_atomarray(atom_array: AtomArray) -> AnnotatedMol:
    """Generates an RDKit Mol object from an AtomArray.

    Tries to naively build an RDKit molecule from the AtomArray by adding the specified
    atoms and inferring bonds from the BondList.

    Args:
        atom_array:
            AtomArray to convert to an RDKit molecule.

    Returns:
        An RDKit molecule object representing the AtomArray. The original names of each
        atom will be stored under the atom-wise property "atom_name_annot".
    """
    mol = AllChem.RWMol()

    mol.BeginBatchEdit()

    # Add all atoms from the AtomArray
    for atom in atom_array:
        element = atom.element.capitalize()
        atomic_number = PERIODIC_TABLE.GetAtomicNumber(element)

        if element == "X":
            element = "*"

        new_atom = Chem.Atom(atomic_number)
        new_atom.SetFormalCharge(int(atom.charge.item()))

        mol.AddAtom(Chem.Atom(new_atom))

    # Form bonds based on the parsed BondList
    for atom_1, atom_2, bond_type_id in atom_array.bonds.as_array():
        mol.AddBond(
            int(atom_1), int(atom_2), bondtype_conversion[BondType(bond_type_id)]
        )
    mol.CommitBatchEdit()

    # Sanitize and assign stereochemistry
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            # Sometimes charges are misassigned by the authors, so try to remove them
            logger.warning("Failed to sanitize molecule, trying to remove charges.")
            for atom in mol.GetAtoms():
                atom.SetFormalCharge(0)
            Chem.SanitizeMol(mol)
            logger.warning("Sanitize successful after removing charges.")
        except Exception as e:
            logger.warning(f"Failed to sanitize molecule: {e}")

    Chem.AssignStereochemistryFrom3D(mol)

    # Add original atom IDs as properties
    mol = set_atomwise_annotation(mol, "atom_name", atom_array.atom_name)

    return mol


def component_iter_from_metadata(atom_array: AtomArray, per_chain_metadata: dict):
    for chain_array in struc.chain_iter(atom_array):
        chain_id = chain_array.chain_id[0]

        ref_mol_id = per_chain_metadata[chain_id].get("reference_mol_id", None)

        # Entire chain corresponds to a single reference molecule (e.g. a ligand chain)
        if ref_mol_id is not None:
            yield chain_array
        # Decompose the chain into individual residues and their reference molecules
        else:
            yield from struc.residue_iter(chain_array)
