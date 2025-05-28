"""
Contains code related to parsing Query objects into AtomArrays and processed reference
molecules.
"""

import logging
from collections.abc import Container
from typing import NamedTuple

import biotite.structure as struc
import numpy as np
from biotite.interface.rdkit import from_mol, to_mol
from biotite.structure import AtomArray
from rdkit import Chem

from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.structure.cleanup import remove_hydrogens
from openfold3.core.data.primitives.structure.component import set_atomwise_annotation
from openfold3.core.data.primitives.structure.conformer import (
    multistrategy_compute_conformer,
)
from openfold3.core.data.resources.residues import (
    DNA_RESTYPE_1TO3,
    MOLECULE_TYPE_TO_LEAVING_ATOMS,
    MOLECULE_TYPE_TO_UKNOWN_RESIDUES_3,
    PROTEIN_RESTYPE_1TO3,
    RNA_RESTYPE_1TO3,
    MoleculeType,
)
from openfold3.projects.af3_all_atom.config.inference_query_format import Query

logger = logging.getLogger(__name__)


class StructureWithReferenceMolecules(NamedTuple):
    """Central object required for structure feature-creation in inference.

    Attributes:
        atom_array (struc.AtomArray):
            AtomArray parsed from the input Query for which coordinates will be
            predicted.
        processed_reference_mols (list[ProcessedReferenceMolecule]):
            List of processed reference molecules (RDKit mol objects with atom names and
            computed conformers) that are required for feature construction.
    """

    atom_array: struc.AtomArray
    processed_reference_mols: list[ProcessedReferenceMolecule]


def atom_array_from_ccd_code(
    ccd_code: str,
    chain_id: str,
    res_id: int = 1,
    molecule_type: MoleculeType | None = None,
) -> AtomArray:
    res_array = struc.info.residue(ccd_code)
    res_array = remove_hydrogens(res_array)

    res_array.res_id[:] = res_id
    res_array.chain_id[:] = chain_id
    res_array.set_annotation("res_name", np.repeat(ccd_code, len(res_array)))

    if molecule_type is not None:
        res_array.set_annotation(
            "molecule_type_id", np.repeat(molecule_type, len(res_array))
        )

    return res_array


def atom_array_from_mol(
    mol: Chem.Mol, chain_id: str, res_id: int = 1, res_name: str = "LIG"
) -> AtomArray:
    atom_array = from_mol(mol, conformer_id=0, add_hydrogen=False)

    # Set global annotations
    atom_array.chain_id[:] = chain_id
    atom_array.hetero[:] = True
    atom_array.res_id[:] = res_id
    atom_array.set_annotation(
        "molecule_type_id", np.repeat(MoleculeType.LIGAND, len(atom_array))
    )

    # Set specific annotations
    atom_array.set_annotation("res_name", np.repeat(res_name, len(atom_array)))
    atom_array.set_annotation(
        "atom_name", [atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()]
    )

    return atom_array


def processed_reference_molecule_from_atom_array(
    atom_array: struc.AtomArray,
    mask_atoms: Container[str] = None,
) -> ProcessedReferenceMolecule:
    # Mask atoms to exclude
    if mask_atoms is not None:
        atom_mask = ~np.isin(atom_array.atom_name, mask_atoms)
    else:
        atom_mask = np.ones(len(atom_array), dtype=bool)

    # Convert to RDKit mol
    mol = to_mol(atom_array, kekulize=True)
    Chem.SanitizeMol(mol)
    mol.RemoveConformer(0)

    # All coordinates of the conformer are valid
    mol = set_atomwise_annotation(mol, "used_atom_mask", [True] * mol.GetNumAtoms())
    mol = set_atomwise_annotation(mol, "atom_name", atom_array.atom_name)

    return processed_reference_molecule_from_mol(
        mol=mol,
        atom_mask=atom_mask,
        mol_id=atom_array.res_name[0],
    )


def processed_reference_molecule_from_mol(
    mol: Chem.Mol,
    atom_mask: np.ndarray | None = None,
    mol_id: str = "LIG",
) -> ProcessedReferenceMolecule:
    # Assume all atoms are in the structure if no special mask is given
    if atom_mask is None:
        atom_mask = np.ones(mol.GetNumAtoms(), dtype=bool)

    # Compute conformer
    mol, conf_id, _ = multistrategy_compute_conformer(mol, remove_hs=True)
    assert conf_id == 0

    return ProcessedReferenceMolecule(
        mol_id=mol_id,
        mol=mol,
        in_crop_mask=atom_mask,
        permutations=None,
    )


def structure_with_ref_mols_from_sequence(
    sequence: str, poly_type: MoleculeType, chain_id: str
) -> StructureWithReferenceMolecules:
    # Figure out 3-letter code mapping and unknown residue identifier
    if poly_type == MoleculeType.PROTEIN:
        resname_1_to_3 = PROTEIN_RESTYPE_1TO3
        unk_res = MOLECULE_TYPE_TO_UKNOWN_RESIDUES_3[MoleculeType.PROTEIN]
    elif poly_type == MoleculeType.DNA:
        resname_1_to_3 = DNA_RESTYPE_1TO3
        unk_res = MOLECULE_TYPE_TO_UKNOWN_RESIDUES_3[MoleculeType.DNA]
    elif poly_type == MoleculeType.RNA:
        resname_1_to_3 = RNA_RESTYPE_1TO3
        unk_res = MOLECULE_TYPE_TO_UKNOWN_RESIDUES_3[MoleculeType.RNA]
    else:
        raise ValueError(f"Unsupported molecule type: {poly_type}")

    leaving_atoms = MOLECULE_TYPE_TO_LEAVING_ATOMS[poly_type]

    atom_array = None
    processed_reference_mols = []

    from tqdm import tqdm

    for res_id, resname_1 in enumerate(tqdm(sequence), start=1):
        # Get 3-letter code of the residue
        if resname_1 not in resname_1_to_3:
            logger.warning(
                f"Unknown residue {resname_1} at position {res_id} in sequence. "
                f"Using placeholder residue {unk_res}."
            )
            resname_3 = unk_res
        else:
            resname_3 = resname_1_to_3[resname_1]

        # Construct atom array for the residue
        res_array = atom_array_from_ccd_code(
            resname_3,
            chain_id=chain_id,
            res_id=res_id,
            molecule_type=poly_type,
        )

        # Parse into RDKit mol and compute conformer
        processed_ref_mol = processed_reference_molecule_from_atom_array(
            res_array, mask_atoms=leaving_atoms
        )
        processed_reference_mols.append(processed_ref_mol)

        # Remove the leaving atoms from the atom array
        res_array = res_array[~np.isin(res_array.atom_name, leaving_atoms)]

        # Initialize atom array
        if atom_array is None:
            atom_array = res_array

        # Append to atom array
        else:
            atom_array += res_array

    # Auto-connect bonds
    atom_array.bonds = struc.connect_via_residue_names(atom_array)

    # Force coordinates to 0 for consistency
    atom_array.coord[:] = 0.0

    return StructureWithReferenceMolecules(
        atom_array=atom_array,
        processed_reference_mols=processed_reference_mols,
    )


def structure_with_ref_mol_from_mol(
    mol: Chem.Mol,
    chain_id: str,
    atom_mask: np.ndarray | None = None,
    mol_id: str = "LIG",
) -> StructureWithReferenceMolecules:
    # Build the ligand molecule
    proc_ref_mol = processed_reference_molecule_from_mol(
        mol, atom_mask=atom_mask, mol_id=mol_id
    )

    # Get the processed mol that now will have a computed conformer
    mol = proc_ref_mol.mol

    # Convert to AtomArray
    atom_array = atom_array_from_mol(mol, chain_id=chain_id, res_name=mol_id)

    # Force coordinates to 0 for consistency
    atom_array.coord[:] = 0.0

    return StructureWithReferenceMolecules(
        atom_array=atom_array, processed_reference_mols=[proc_ref_mol]
    )


def structure_with_ref_mol_from_ccd_code(
    ccd_code: str,
    chain_id: str,
) -> StructureWithReferenceMolecules:
    # Build ligand AtomArray
    atom_array = atom_array_from_ccd_code(
        ccd_code,
        chain_id=chain_id,
        res_id=1,
        molecule_type=MoleculeType.LIGAND,
    )

    # Get processed reference molecule
    proc_ref_mol = processed_reference_molecule_from_atom_array(atom_array)

    # Force coordinates to 0 for consistency
    atom_array.coord[:] = 0.0

    return StructureWithReferenceMolecules(
        atom_array=atom_array, processed_reference_mols=[proc_ref_mol]
    )


def structure_with_ref_mol_from_smiles(
    smiles: str,
    chain_id: str,
    mol_id: str = "LIG",
) -> StructureWithReferenceMolecules:
    """Creates a single AtomArray and processed ref molecule from a SMILES string.

    Args:
        smiles (str):
            The SMILES string of the molecule to create.
        chain_id (str):
            The chain ID to assign to the created AtomArray.
        mol_id (str):
            The ID of the molecule, used for naming in the processed reference molecule.
    """
    mol = Chem.MolFromSmiles(smiles)

    mol = set_atomwise_annotation(mol, "used_atom_mask", [True] * mol.GetNumAtoms())

    # Set simple atom names like C1, C2, N1, N2, ...
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_names = struc.create_atom_names(elements)

    mol = set_atomwise_annotation(mol, "atom_name", atom_names)

    return structure_with_ref_mol_from_mol(
        mol,
        chain_id=chain_id,
        atom_mask=None,
        mol_id=mol_id,
    )


def structure_with_ref_mols_from_query(query: Query) -> StructureWithReferenceMolecules:
    """Builds an AtomArray and processed reference molecules from a Query object.

    Parses the Query object into a full AtomArray and processed reference molecules
    (RDKit mol objects with atom names and computed conformers).

    The returned AtomArray follows the chain IDs given in the Query object. If a chain
    specifies multiple chain IDs, repeated identical chains with those IDs will be
    constructed and given the same entity ID.

    Residue names will be inferred from the sequence or CCD codes. If a ligand is
    specified through a SMILES string, it will be named as "LIG-X", where X starts at 1
    and is incremented for each unnamed ligand entity found in the Query.

    Args:
        query (Query):
            The Query object containing the chains to construct the structure from.

    Returns:
        StructureWithReferenceMolecules:
            A named tuple containing the AtomArray and a list of processed reference
            molecules.
    """
    # Initialize eventually returned objects
    atom_array = None
    processed_reference_mols: list[ProcessedReferenceMolecule] = []

    # Counter of chains added
    unnamed_lig_entity_count = 1

    # Current entity ID
    entity_id = 1

    # Build the structure segment-wise from all chains in the query.
    for chain in query.chains:
        chain_is_unnamed_lig_entity = False

        for chain_id in chain.chain_ids:
            match chain.molecule_type:
                # Build polymeric part
                case MoleculeType.PROTEIN | MoleculeType.DNA | MoleculeType.RNA:
                    # Create atom array from sequence
                    segment_atom_array, segment_ref_mols = (
                        structure_with_ref_mols_from_sequence(
                            sequence=chain.sequence,
                            poly_type=chain.molecule_type,
                            chain_id=chain_id,
                        )
                    )

                case MoleculeType.LIGAND:
                    # Build ligand from SMILES
                    if chain.smiles is not None:
                        # Mark that this is an unnamed ligand (important for tracking
                        # number)
                        chain_is_unnamed_lig_entity = True

                        segment_atom_array, segment_ref_mols = (
                            structure_with_ref_mol_from_smiles(
                                smiles=chain.smiles,
                                chain_id=chain_id,
                                mol_id=f"LIG-{unnamed_lig_entity_count}",
                            )
                        )

                    # Build ligand from CCD code
                    elif chain.ccd_codes is not None:
                        # TODO: add multi-residue ligand support
                        if len(chain.ccd_codes) > 1:
                            raise NotImplementedError(
                                "Multiple CCD codes for a single chain are not yet "
                                "supported."
                            )

                        segment_atom_array, segment_ref_mols = (
                            structure_with_ref_mol_from_ccd_code(
                                ccd_code=chain.ccd_codes[0],
                                chain_id=chain_id,
                            )
                        )

                    # Build ligand from SDF file
                    elif chain.sdf_file_path is not None:
                        # TODO: add SDF support
                        raise NotImplementedError(
                            "SDF format for ligands is not yet supported."
                        )

                    else:
                        raise ValueError("No valid molecule specification found.")

            # Add processed reference molecules
            processed_reference_mols.extend(segment_ref_mols)

            segment_atom_array.set_annotation(
                "entity_id", np.repeat(entity_id, len(segment_atom_array))
            )

            # Append atom array to end
            if atom_array is None:
                atom_array = segment_atom_array
            else:
                atom_array += segment_atom_array

        # Increment count of unnamed ligand entities if applicable
        if chain_is_unnamed_lig_entity:
            unnamed_lig_entity_count += 1

        # Assume that each new set of chains represents a different entity
        entity_id += 1

    # Force coordinates to 0
    atom_array.coord[:] = 0.0

    return StructureWithReferenceMolecules(
        atom_array=atom_array, processed_reference_mols=processed_reference_mols
    )
