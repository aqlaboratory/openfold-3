"""
Contains code related to parsing Query objects into AtomArrays and processed reference
molecules.
"""

import logging
from collections.abc import Container, Iterable
from functools import lru_cache
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


get_residue_cached = lru_cache(maxsize=500)(struc.info.residue)
"""Cached residue information retrieval from Biotite to speed up preprocessing."""


def atom_array_from_ccd_code(
    ccd_code: str,
    chain_id: str,
    res_id: int = 1,
    molecule_type: MoleculeType | None = None,
) -> AtomArray:
    res_array = get_residue_cached(ccd_code)
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
    atoms_to_mask: Container[str] = None,
) -> ProcessedReferenceMolecule:
    """Creates a processed reference molecule from an AtomArray.

    atom_array
    """
    # Mask certain atoms that should not be present in the final structure
    if atoms_to_mask is not None:
        atom_mask = ~np.isin(atom_array.atom_name, atoms_to_mask)
    else:
        atom_mask = np.ones(len(atom_array), dtype=bool)

    # Convert to RDKit mol
    mol = to_mol(atom_array, kekulize=True)
    Chem.SanitizeMol(mol)
    mol.RemoveConformer(0)

    return processed_reference_molecule_from_mol(
        mol=mol,
        atom_names=atom_array.atom_name,
        atom_mask=atom_mask,
    )


def processed_reference_molecule_from_mol(
    mol: Chem.Mol,
    atom_names: Iterable[str] | None = None,
    atom_mask: np.ndarray | None = None,
) -> ProcessedReferenceMolecule:
    """Creates a processed reference molecule from an RDKit mol object.

    Args:
        mol (Chem.Mol):
            The RDKit molecule to create the processed reference molecule from.
        atom_names (Container[str] | None):
            Optional atom names to set for the atoms in the RDKit mol. If None, the atom
            names will be set to a simple pattern like C1, C2, N1, N2, etc.
        atom_mask (np.ndarray | None):
            Optional mask for atoms in the processed reference molecule that should not
            be included in the feature creation, e.g. leaving atoms. Those atoms will
            still be part of the rdkit.Mol object to retain chemical validity of the
            molecule and generate the correct conformer. If None, which is the default,
            no atoms will be masked.

    Returns:
        ProcessedReferenceMolecule:
            A processed reference molecule containing the RDKit mol with a computed
            conformer and the atom mask.
    """
    # Assume all atoms are in the structure if no special mask is given
    if atom_mask is None:
        atom_mask = np.ones(mol.GetNumAtoms(), dtype=bool)

    # Set atom names if provided, otherwise renumber to C1, C2, N1, N2, etc.
    if atom_names is not None:
        mol = set_atomwise_annotation(mol, "atom_name", atom_names)
    else:
        elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atom_names = struc.create_atom_names(elements)
        mol = set_atomwise_annotation(mol, "atom_name", atom_names)

    # This is a different mask only required for fallback conformers in the training
    # script where some coordinates are not defined
    mol = set_atomwise_annotation(mol, "used_atom_mask", [True] * mol.GetNumAtoms())

    # Compute conformer
    mol, conf_id, _ = multistrategy_compute_conformer(mol, remove_hs=True)
    assert conf_id == 0

    return ProcessedReferenceMolecule(
        mol=mol,
        in_crop_mask=atom_mask,
        permutations=None,
    )


def structure_with_ref_mols_from_sequence(
    sequence: str, poly_type: MoleculeType, chain_id: str
) -> StructureWithReferenceMolecules:
    """Builds an AtomArray and processed reference molecules from a sequence.

    Will read the entire sequence into an AtomArray and create reference molecule
    objects with separate conformers for each residue. Currently only supports standard
    residues, any non-canonical residue will be treated as an unknown residue.

    Args:
        sequence (str):
            The sequence of the polymeric molecule as a string of 1-letter residue
            codes.
        poly_type (MoleculeType):
            The MoleculeType of the polymeric molecule. Should be one of
            MoleculeType.PROTEIN, MoleculeType.DNA, or MoleculeType.RNA.
        chain_id (str):
            The chain ID to assign to the created AtomArray.

    Returns:
        StructureWithReferenceMolecules:
            A named tuple containing the AtomArray and a list of processed reference
            molecules, each corresponding to a residue in the sequence.
    """
    # Figure out 3-letter code mapping
    match poly_type:
        case MoleculeType.PROTEIN:
            resname_1_to_3 = PROTEIN_RESTYPE_1TO3
        case MoleculeType.DNA:
            resname_1_to_3 = DNA_RESTYPE_1TO3
        case MoleculeType.RNA:
            resname_1_to_3 = RNA_RESTYPE_1TO3
        case _:
            raise ValueError(f"Unsupported molecule type: {poly_type}")

    # Figure out the unknown residue 3-letter identifier and leaving atom names
    unk_res = MOLECULE_TYPE_TO_UKNOWN_RESIDUES_3[poly_type]
    leaving_atoms = MOLECULE_TYPE_TO_LEAVING_ATOMS[poly_type]

    atom_array = None
    processed_reference_mols = []

    # TODO: Remove this
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
            res_array, atoms_to_mask=leaving_atoms
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
    res_name: str = "LIG",
) -> StructureWithReferenceMolecules:
    """Creates a single AtomArray and processed reference molecule from an RDKit mol.

    Args:
        mol (Chem.Mol):
            The RDKit molecule to create the AtomArray and processed reference molecule
            from.
        chain_id (str):
            The chain ID to assign to the created AtomArray.
        atom_mask (np.ndarray | None):
            Optional mask for atoms to include in the processed reference molecule. If
            None, all atoms will be included.
        res_name (str):
            The residue name to assign to the created AtomArray. Defaults to "LIG".
    Returns:
        StructureWithReferenceMolecules:
            A named tuple containing the AtomArray and a list with a single processed
            reference molecule. The residue ID will be set to 1.
    """

    # Build the ligand molecule
    proc_ref_mol = processed_reference_molecule_from_mol(mol, atom_mask=atom_mask)

    # Get the processed mol that now will have a computed conformer
    mol = proc_ref_mol.mol

    # Convert to AtomArray
    atom_array = atom_array_from_mol(mol, chain_id=chain_id, res_name=res_name)

    # Force coordinates to 0 for consistency
    atom_array.coord[:] = 0.0

    return StructureWithReferenceMolecules(
        atom_array=atom_array, processed_reference_mols=[proc_ref_mol]
    )


def structure_with_ref_mol_from_ccd_code(
    ccd_code: str,
    chain_id: str,
) -> StructureWithReferenceMolecules:
    """Creates a single AtomArray and processed reference molecule from a CCD code.

    Args:
        ccd_code (str):
            The CCD code of the molecule to create.
        chain_id (str):
            The chain ID to assign to the created AtomArray.

    Returns:
        StructureWithReferenceMolecules:
            A named tuple containing the AtomArray and a list with a single processed
            reference molecule. The residue ID will be set to 1.
    """

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
    res_name: str = "LIG",
) -> StructureWithReferenceMolecules:
    """Creates a single AtomArray and processed ref molecule from a SMILES string.

    Args:
        smiles (str):
            The SMILES string of the molecule to create.
        chain_id (str):
            The chain ID to assign to the created AtomArray.
        res_name (str):
            The residue name to assign to the created AtomArray. Defaults to "LIG".

    Returns:
        StructureWithReferenceMolecules:
            A named tuple containing the AtomArray and a list with a single processed
            reference molecule. The residue ID will be set to 1. Atom names of the
            molecule will be set to follow the pattern C1, C2, N1, N2, etc.
    """
    mol = Chem.MolFromSmiles(smiles)

    return structure_with_ref_mol_from_mol(
        mol,
        chain_id=chain_id,
        res_name=res_name,
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
                # Build polymeric segment
                case MoleculeType.PROTEIN | MoleculeType.DNA | MoleculeType.RNA:
                    segment_atom_array, segment_ref_mols = (
                        structure_with_ref_mols_from_sequence(
                            sequence=chain.sequence,
                            poly_type=chain.molecule_type,
                            chain_id=chain_id,
                        )
                    )

                # Build ligand molecule
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
                                res_name=f"LIG-{unnamed_lig_entity_count}",
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

    # Force coordinates to 0 for consistency
    atom_array.coord[:] = 0.0

    return StructureWithReferenceMolecules(
        atom_array=atom_array, processed_reference_mols=processed_reference_mols
    )
