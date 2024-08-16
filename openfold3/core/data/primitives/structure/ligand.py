import contextlib
import logging
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, BondType
from pdbeccdutils.core.ccd_reader import Component
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from rdkit.Geometry import Point3D

from openfold3.core.data.io.structure.mol import (
    read_single_annotated_sdf,
    read_single_sdf,
)
from openfold3.core.data.primitives.structure.labels import assign_atom_indices

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


def get_allzero_conformer(mol: Mol) -> Chem.Conformer:
    """Returns a conformer with all atoms at the origin.

    Args:
        mol:
            The molecule for which the conformer should be generated.

    Returns:
        An RDKit conformer object with all atoms at the origin.
    """
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom_id in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(atom_id, Point3D(0, 0, 0))

    return conf


def set_fallback_conformer(mol: Mol) -> Chem.Mol:
    # Try conformer generation
    try:
        # If it works, no need to supply a conformer because dataloading can handle it
        # on the fly
        mol = compute_conformer(mol)
        mol.SetProp("use_conformer", "False")
<<<<<<< Updated upstream

        mol.ClearProp("model_pdb_id")
        mol.ClearProp("model_used_atom_mask")
        mol.ClearProp("ideal_used_atom_mask")

        used_atom_mask = [1] * mol.GetNumAtoms()
        mol.SetProp("used_atom_mask", " ".join(str(x) for x in used_atom_mask))

=======
>>>>>>> Stashed changes
    except ConformerGenerationError:
        conf_names = [conf.GetProp("name") for conf in mol.GetConformers()]

        # Look for Ideal and Model coordinates in that order and use the first one that
        # is present as fallback conformer
        if "Ideal" in conf_names:
            fallback_conf = mol.GetConformer(conf_names.index("Ideal"))
            mol.ClearProp("model_pdb_id")
            mol.ClearProp("model_used_atom_mask")

            mol.SetProp("used_atom_mask", mol.GetProp("ideal_used_atom_mask"))

        elif "Model" in conf_names and mol.GetProp("model_pdb_id") != "?":
            logger.debug("Ideal not found, using Model coordinates as fallback")

            fallback_conf = mol.GetConformer(conf_names.index("Model"))
            mol.ClearProp("ideal_used_atom_mask")

            mol.SetProp("used_atom_mask", mol.GetProp("model_used_atom_mask"))

            # assert that not all coordinates are 0, dev-only TODO: remove
            assert not all(
                all(coord == 0 for coord in fallback_conf.GetAtomPosition(atom_id))
                for atom_id in range(mol.GetNumAtoms())
            )
        # If no valid conformer is found, use a fallback conformer with all atoms at the
        # origin
        else:
            fallback_conf = get_allzero_conformer(mol)

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


# TODO: improve docstring
def resolve_conformer(mol: Mol) -> Mol:
    """Generates a conformer or uses a fallback conformer depending on the annotation.

    This function is meant to be used within dataloading after the preprocessing has
    already set the "use_conformer" property in the RDKit Mol object.

    Args:
        mol:
            The molecule for which the conformer should be generated.

    Returns:
        The input molecule with a single conformer.
    """
    # Check if mol has fallback conformer, if yes use it
    use_conformer = mol.GetProp("use_conformer")
    use_conformer = use_conformer == "True"

    if use_conformer == "True":
        return mol
    # If no, generate conformer
    else:
        # TODO: make function out of this conformer copy-clear operation
        mol = Chem.Mol(mol)  # make a copy, see rdkit issue #3817
        mol.RemoveAllConformers()
        try:
            mol = compute_conformer(mol)
        # This should not happen as the preprocessing should have already confirmed that
        # on-the-fly conformer generation is possible, but as a fallback set all
        # coordinates to 0
        except ConformerGenerationError:
            logging.warning(
                "Failed to generate conformer. This should have been dealt with in "
                "preprocessing. Using all-zero conformer."
            )

            mol.AddConformer(get_allzero_conformer(mol), assignId=True)


def subset_mol_by_atom_names(mol: Mol, atom_names: set):
    """Removes atoms from a molecule that are not in a set of atom names."""

    mol = Chem.RWMol(mol)

    atoms_to_remove = []

    for atom in mol.GetAtoms():
        if atom.GetProp("name") not in atom_names:
            atoms_to_remove.append(atom.GetIdx())

    # Remove in safe order
    for atom_idx in reversed(atoms_to_remove):
        mol.RemoveAtom(atom_idx)

    return Chem.Mol(mol)


def assign_reference_molecules(
    atom_array: AtomArray, ccd_sdfs_path: Path, special_ligand_sdfs_path: Path
) -> tuple[AtomArray, list[Mol]]:
    """Generates a list of reference molecules for all components in the AtomArray."""
    # TODO: improve docstring with conformer information
    """Gets RDKit molecules for all components in the AtomArray."""

    # Get temporary helper indices
    assign_atom_indices(atom_array)

    # Get the set of non-standard entities (like covalently connected components) that
    # require special treatment
    special_entities = set()
    for file in special_ligand_sdfs_path.iterdir():
        entity_id = int(file.stem.replace("entity_", ""))
        special_entities.add(entity_id)

    # Maps CCD codes (for standard components) or entity IDs (for special ligands) to
    # cached RDKit Mol objects to avoid reading the same file many times
    identifier_to_mol = {}

    # Numerical ID for each instance of a component
    ref_mol_ids = []
    current_ref_mol_id = 0

    # Flat list of components
    flat_components = []

    residue_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    residue_start_end_iter = zip(residue_starts[:-1], residue_starts[1:])

    def look_ahead_next_chain_id(res_end):
        next_residue_start = atom_array[res_end]._atom_idx + 1
        if next_residue_start == len(atom_array):
            return -1
        else:
            return atom_array[next_residue_start].chain_id

    # Standard residues get a conformer ID for each residue, while special entities
    # enter a nested loop until the particular chain ends
    for res_start, res_end in residue_start_end_iter:
        first_res_atom = atom_array[res_start]

        chain_id = first_res_atom.chain_id
        ccd_code = first_res_atom.res_name
        entity_id = first_res_atom.entity_id

        n_res_atoms = res_end - res_start

        if entity_id in special_entities:
            special_entity = True
            identifier = entity_id
            mol_path = special_ligand_sdfs_path / f"{entity_id}_entity.sdf"
        else:
            special_entity = False
            identifier = ccd_code
            mol_path = ccd_sdfs_path / f"{ccd_code}.sdf"

        # Get cached mol or read from file
        if identifier not in identifier_to_mol:
            mol = read_single_annotated_sdf(mol_path)
            identifier_to_mol[identifier] = mol
        else:
            mol = identifier_to_mol[identifier]

        # Add atom-wise conformer IDs
        ref_mol_ids.extend([current_ref_mol_id] * n_res_atoms)

        # Enter special loop for non-standard multi-residue entities till the particular
        # current chain ends
        if special_entity:
            # Look-ahead if next residue is within same instance of the entity
            next_residue_chain = look_ahead_next_chain_id(res_end)
            if next_residue_chain == -1:
                break

            # Set residues to same mol ID as long as they're still in the same chain
            while next_residue_chain == chain_id:
                res_start, res_end = next(residue_start_end_iter)
                n_res_atoms = res_end - res_start

                ref_mol_ids.extend([current_ref_mol_id] * n_res_atoms)

                next_residue_chain = look_ahead_next_chain_id(res_end)

        else:
            if ccd_code not in identifier_to_mol:
                mol = read_single_sdf(ccd_sdfs_path / f"{ccd_code}.sdf")
                identifier_to_mol[ccd_code] = mol
            else:
                mol = identifier_to_mol[ccd_code]

            # Add atom-wise conformer IDs
            ref_mol_ids.extend([current_ref_mol_id] * n_res_atoms)

        # Set the final conformer of the molecule
        mol = resolve_conformer(mol)
        
        # Subset to only the present atoms (not required for special ligands which are
        # built without missing atoms) TODO: improve this explanation/hacky logic
        if not special_entity:
            atom_names = set(atom_array.atom_name[res_start:res_end])
            mol = subset_mol_by_atom_names(mol, atom_names)

        flat_components.append(mol)
        current_ref_mol_id += 1  # noqa: SIM113

    # Assign the conformer IDs in the atom array
    atom_array.set_annotation("ref_conf_id", np.array(ref_mol_ids, dtype=int))

    return atom_array, flat_components
