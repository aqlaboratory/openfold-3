# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contains code related to parsing Query objects into AtomArrays and processed reference
molecules.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import biotite.structure as struc
import numpy as np
from biotite.interface.rdkit import from_mol, to_mol
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from rdkit import Chem

from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.structure.cleanup import remove_hydrogens
from openfold3.core.data.primitives.structure.component import (
    BiotiteCCDWrapper,
    set_atomwise_annotation,
)
from openfold3.core.data.primitives.structure.conformer import (
    multistrategy_compute_conformer,
)
from openfold3.core.data.resources.residues import (
    DNA_RESTYPE_1TO3,
    MOLECULE_TYPE_TO_LEAVING_ATOMS,
    MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_3,
    PROTEIN_RESTYPE_1TO3,
    RNA_RESTYPE_1TO3,
    STANDARD_RESIDUES_3,
    MoleculeType,
)

if TYPE_CHECKING:
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        Atom,
        Query,
    )

logger = logging.getLogger(__name__)

_FORCE_ATOMIZATION_ANNOTATION = "force_atomization"


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


class CovalentQueryError(ValueError):
    """A query-scoped validation error caused by a requested structure edit."""


def get_leaving_atoms(ccd_code: str) -> np.ndarray:
    """Returns the leaving atoms for a given CCD code.

    Args:
        ccd_code (str):
            The CCD code of the residue to get the leaving atoms for.

    Returns:
        np.ndarray:
            An array of leaving atom names for the given CCD code.
    """
    leaving_atom_flag = struc.info.get_from_ccd(
        category_name="chem_comp_atom",
        comp_id=ccd_code,
        column_name="pdbx_leaving_atom_flag",
    ).as_array()
    atom_names = struc.info.get_from_ccd(
        category_name="chem_comp_atom",
        comp_id=ccd_code,
        column_name="atom_id",
    ).as_array()

    leaving_atoms = atom_names[leaving_atom_flag == "Y"]

    return leaving_atoms


def _query_error(query: Query, message: str) -> CovalentQueryError:
    """Create an error whose message identifies the affected inference query."""
    query_name = query.query_name if query.query_name is not None else "<unnamed>"
    return CovalentQueryError(f"Invalid inference query {query_name!r}: {message}")


def _atom_selector_tuple(atom: Atom) -> tuple[str, int, str]:
    """Convert a public atom selector into a hashable lookup key."""
    return atom.chain_id, atom.residue_id, atom.atom_name


def _atom_selector_text(atom: Atom) -> str:
    """Format a public atom selector for a user-facing diagnostic."""
    return repr([atom.chain_id, atom.residue_id, atom.atom_name])


def _ccd_code_for_endpoint(query: Query, endpoint: Atom) -> tuple[str, set[str]] | None:
    """Return an endpoint's CCD code and atoms already removed by construction.

    ``None`` denotes a non-CCD molecule, currently a SMILES ligand. Structural
    existence and uniqueness are validated again against the final AtomArray.
    """
    matching_chains = [
        chain for chain in query.chains if endpoint.chain_id in chain.chain_ids
    ]
    if len(matching_chains) == 0:
        raise _query_error(
            query,
            f"covalent endpoint {_atom_selector_text(endpoint)} references unknown "
            f"chain {endpoint.chain_id!r}",
        )
    if len(matching_chains) > 1:
        raise _query_error(
            query,
            f"covalent endpoint {_atom_selector_text(endpoint)} references chain "
            f"{endpoint.chain_id!r}, which is declared more than once",
        )

    chain = matching_chains[0]
    if chain.smiles is not None:
        return None

    if chain.molecule_type == MoleculeType.LIGAND:
        if chain.ccd_codes is None:
            return None
        if endpoint.residue_id != 1:
            raise _query_error(
                query,
                f"covalent endpoint {_atom_selector_text(endpoint)} references "
                "residue ID other than 1 in a single-component ligand chain",
            )
        if len(chain.ccd_codes) != 1:
            # Multi-residue CCD ligands are rejected by structure construction too.
            return None
        return chain.ccd_codes[0], set()

    if chain.sequence is None or endpoint.residue_id > len(chain.sequence):
        raise _query_error(
            query,
            f"covalent endpoint {_atom_selector_text(endpoint)} references an unknown "
            f"residue in chain {endpoint.chain_id!r}",
        )

    non_canonical_residues = chain.non_canonical_residues or {}
    if endpoint.residue_id in non_canonical_residues:
        # CCD leaving flags for noncanonical polymer residues are already consumed by
        # structure_with_ref_mols_from_sequence().  This set is populated by the
        # caller from the same selected CCD source.
        return non_canonical_residues[endpoint.residue_id], set()

    residue_letter = chain.sequence[endpoint.residue_id - 1]
    match chain.molecule_type:
        case MoleculeType.PROTEIN:
            ccd_code = PROTEIN_RESTYPE_1TO3.get(
                residue_letter,
                MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_3[MoleculeType.PROTEIN],
            )
        case MoleculeType.DNA:
            ccd_code = DNA_RESTYPE_1TO3.get(
                residue_letter,
                MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_3[MoleculeType.DNA],
            )
        case MoleculeType.RNA:
            ccd_code = RNA_RESTYPE_1TO3.get(
                residue_letter,
                MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_3[MoleculeType.RNA],
            )
        case _:
            return None

    return ccd_code, set(MOLECULE_TYPE_TO_LEAVING_ATOMS[chain.molecule_type])


def _read_ccd_leaving_graph(
    ccd: Any,
    ccd_code: str,
    query: Query,
) -> tuple[list[str], set[str], dict[str, set[str]]]:
    """Read atom order, heavy leaving atoms, and adjacency from one CCD block."""
    try:
        ccd_entry = ccd[ccd_code]
        atom_category = ccd_entry["chem_comp_atom"]
        atom_names = atom_category["atom_id"].as_array().astype(str).tolist()
        atom_elements = atom_category["type_symbol"].as_array().astype(str)
        leaving_flags = atom_category["pdbx_leaving_atom_flag"].as_array().astype(str)
    except (KeyError, TypeError, AttributeError) as exc:
        raise _query_error(
            query,
            f"could not read atom/leaving-atom metadata for CCD component "
            f"{ccd_code!r}: {exc}",
        ) from exc

    heavy_leaving_atoms = {
        atom_name
        for atom_name, element, leaving_flag in zip(
            atom_names, atom_elements, leaving_flags, strict=True
        )
        if leaving_flag == "Y" and element.upper() not in {"H", "D"}
    }
    adjacency = {atom_name: set() for atom_name in atom_names}

    try:
        bond_category = ccd_entry.get("chem_comp_bond")
        if bond_category is None:
            return atom_names, heavy_leaving_atoms, adjacency
        atom_names_1 = bond_category["atom_id_1"].as_array().astype(str)
        atom_names_2 = bond_category["atom_id_2"].as_array().astype(str)
    except (KeyError, TypeError, AttributeError) as exc:
        raise _query_error(
            query,
            f"could not read bond metadata for CCD component {ccd_code!r}: {exc}",
        ) from exc

    for atom_name_1, atom_name_2 in zip(atom_names_1, atom_names_2, strict=True):
        # Invalid CCD references should not make the inference algorithm silently
        # invent an incomplete graph.
        if atom_name_1 not in adjacency or atom_name_2 not in adjacency:
            raise _query_error(
                query,
                f"CCD component {ccd_code!r} has a bond referencing an unknown atom "
                f"({atom_name_1!r}, {atom_name_2!r})",
            )
        adjacency[atom_name_1].add(atom_name_2)
        adjacency[atom_name_2].add(atom_name_1)

    return atom_names, heavy_leaving_atoms, adjacency


def _endpoint_local_leaving_groups(
    atom_names: list[str],
    heavy_leaving_atoms: set[str],
    adjacency: dict[str, set[str]],
    endpoint_atom_name: str,
) -> list[list[str]]:
    """Find connected CCD-flagged heavy groups directly adjacent to an endpoint."""
    if atom_names.count(endpoint_atom_name) != 1:
        return []

    atom_order = {atom_name: index for index, atom_name in enumerate(atom_names)}
    direct_seeds = adjacency[endpoint_atom_name] & heavy_leaving_atoms
    groups: list[list[str]] = []
    visited: set[str] = set()
    for seed in sorted(direct_seeds, key=atom_order.__getitem__):
        if seed in visited:
            continue
        stack = [seed]
        group: set[str] = set()
        while stack:
            atom_name = stack.pop()
            if atom_name in group:
                continue
            group.add(atom_name)
            stack.extend((adjacency[atom_name] & heavy_leaving_atoms) - group)
        visited.update(group)
        groups.append(sorted(group, key=atom_order.__getitem__))
    return groups


def normalize_manual_leaving_atoms(query: Query) -> Query:
    """Return a copy with duplicate manual leaving-atom selectors removed.

    Normalization is independent of automatic CCD inference so the effective query log
    is stable whether or not leaving-atom inference is enabled.
    """
    effective_query = query.model_copy(deep=True)
    normalized_atoms = []
    seen_selectors: set[tuple[str, int, str]] = set()
    for atom in effective_query.leaving_atoms or []:
        selector = _atom_selector_tuple(atom)
        if selector not in seen_selectors:
            normalized_atoms.append(atom)
            seen_selectors.add(selector)
    effective_query.leaving_atoms = normalized_atoms or None
    return effective_query


def infer_ccd_leaving_atoms(
    query: Query,
    ccd: Any | str | Path | None = None,
) -> Query:
    """Conservatively infer endpoint-local heavy leaving atoms for a query.

    The returned query is a deep copy; the input query is never mutated.  ``ccd`` may
    be a parsed CIF file, a :class:`BiotiteCCDWrapper`, or a path to an uncompressed
    CCD CIF file. If it is omitted, Biotite's configured global CCD is used.

    Existing manual/effective ``leaving_atoms`` entries take precedence at an endpoint.
    This makes the operation idempotent when a logged effective query is submitted
    again and lets a user explicitly resolve an otherwise ambiguous CCD endpoint.
    """
    effective_query = normalize_manual_leaving_atoms(query)
    if not effective_query.covalent_bonds:
        return effective_query

    if ccd is None:
        ccd = BiotiteCCDWrapper()
    elif isinstance(ccd, (str, Path)):
        ccd = pdbx.CIFFile.read(ccd)

    effective_leaving_atoms = []
    existing_selectors: set[tuple[str, int, str]] = set()
    for atom in effective_query.leaving_atoms or []:
        selector = _atom_selector_tuple(atom)
        if selector not in existing_selectors:
            effective_leaving_atoms.append(atom)
            existing_selectors.add(selector)

    for bond_index, bond in enumerate(effective_query.covalent_bonds):
        for endpoint in bond:
            ccd_info = _ccd_code_for_endpoint(effective_query, endpoint)
            if ccd_info is None:
                continue
            ccd_code, atoms_removed_by_construction = ccd_info
            atom_names, heavy_leaving_atoms, adjacency = _read_ccd_leaving_graph(
                ccd, ccd_code, effective_query
            )

            # Noncanonical polymer construction removes every CCD-flagged atom.  Read
            # that set from the selected CCD rather than Biotite's global CCD.
            chain = next(
                chain
                for chain in effective_query.chains
                if endpoint.chain_id in chain.chain_ids
            )
            if chain.molecule_type != MoleculeType.LIGAND and endpoint.residue_id in (
                chain.non_canonical_residues or {}
            ):
                atoms_removed_by_construction = set(heavy_leaving_atoms)

            heavy_leaving_atoms -= atoms_removed_by_construction
            groups = _endpoint_local_leaving_groups(
                atom_names,
                heavy_leaving_atoms,
                adjacency,
                endpoint.atom_name,
            )
            if not groups:
                continue

            component_manual_names = {
                atom_name
                for chain_id, residue_id, atom_name in existing_selectors
                if chain_id == endpoint.chain_id and residue_id == endpoint.residue_id
            }
            # A manual entry in any candidate group is treated as an explicit choice.
            # Do not second-guess or expand it automatically.
            if any(component_manual_names & set(group) for group in groups):
                continue

            if len(groups) > 1:
                formatted_groups = ", ".join(repr(group) for group in groups)
                raise _query_error(
                    effective_query,
                    f"covalent bond {bond_index} endpoint "
                    f"{_atom_selector_text(endpoint)} has multiple endpoint-local "
                    f"CCD leaving groups in {ccd_code!r}: {formatted_groups}; "
                    "specify leaving_atoms explicitly",
                )

            for atom_name in groups[0]:
                inferred_atom = type(endpoint)(
                    endpoint.chain_id, endpoint.residue_id, atom_name
                )
                selector = _atom_selector_tuple(inferred_atom)
                if selector not in existing_selectors:
                    effective_leaving_atoms.append(inferred_atom)
                    existing_selectors.add(selector)

    effective_query.leaving_atoms = effective_leaving_atoms or None
    return effective_query


def atom_array_from_ccd_code(
    ccd_code: str,
    chain_id: str,
    res_id: int = 1,
    molecule_type: MoleculeType | None = None,
) -> AtomArray:
    """Creates an AtomArray from a CCD code.

    Fetches the residue information from Biotite and constructs an AtomArray with the
    specified chain ID, residue ID, and molecule type.

    Args:
        ccd_code (str):
            The CCD code of the residue to create the AtomArray from.
        chain_id (str):
            The chain ID to assign to the created AtomArray.
        res_id (int):
            The residue ID to assign to the created AtomArray. Defaults to 1.
        molecule_type (MoleculeType | None):
            The MoleculeType of the molecule. If None, no molecule type annotation will
            be set. Defaults to None.
    """
    res_array = get_residue_cached(ccd_code)
    res_array = remove_hydrogens(res_array)

    res_array.res_id[:] = res_id
    res_array.chain_id[:] = chain_id

    if molecule_type is not None:
        res_array.set_annotation(
            "molecule_type_id", np.repeat(molecule_type, len(res_array))
        )

    return res_array


def atom_array_from_mol(
    mol: Chem.Mol,
    atom_names: Iterable[str],
    chain_id: str,
    molecule_type: MoleculeType = MoleculeType.LIGAND,
    res_id: int = 1,
    res_name: str = "LIG",
) -> AtomArray:
    """Creates an AtomArray from an RDKit mol object.

    Args:
        mol (Chem.Mol):
            The RDKit molecule to create the AtomArray from.
        atom_names (Iterable[str]):
            Iterable of atom names to set in the AtomArray.
        chain_id (str):
            The chain ID to assign to the created AtomArray.
        molecule_type (MoleculeType):
            The MoleculeType of the molecule. Defaults to MoleculeType.LIGAND.
        res_id (int):
            The residue ID to assign to the created AtomArray. Defaults to 1.
        res_name (str):
            The residue name to assign to the created AtomArray. Defaults to "LIG".

    Returns:
        AtomArray:
            An AtomArray containing the atoms from the RDKit mol with the specified atom
            names, chain ID, residue ID, and residue name, and molecule type. The order
            of atoms in the AtomArray will match the order of atom names provided.
    """
    atom_array = from_mol(mol, conformer_id=0, add_hydrogen=False)

    # Set global annotations
    atom_array.chain_id[:] = chain_id
    atom_array.hetero[:] = True
    atom_array.res_id[:] = res_id
    atom_array.set_annotation(
        "molecule_type_id", np.repeat(molecule_type, len(atom_array))
    )

    # Set specific annotations
    atom_array.set_annotation("res_name", np.repeat(res_name, len(atom_array)))
    atom_array.set_annotation("atom_name", atom_names)

    return atom_array


def processed_reference_molecule_from_atom_array(
    atom_array: struc.AtomArray,
    atoms_to_mask: Iterable[str] = None,
) -> ProcessedReferenceMolecule:
    """Creates a processed reference molecule from an AtomArray.

    Args:
        atom_array (struc.AtomArray):
            The AtomArray to create the processed reference molecule from. Atom names of
            the processed reference molecule will be set to the atom names of the
            AtomArray.
        atoms_to_mask (Iterable[str] | None):
            Optional iterable of atom names to mask in the processed reference molecule.
            If provided, these atoms will not be included in the final structure, but
            will still be part of the RDKit mol object to retain chemical validity and
            generate the correct conformer. If None, which is the default, no atoms will
            be masked.

    Returns:
        ProcessedReferenceMolecule:
            A processed reference molecule containing the RDKit mol with a computed
            conformer and the atom mask.
    """
    # Mask certain atoms that should not be present in the final structure
    if atoms_to_mask is not None:
        atom_mask = ~np.isin(atom_array.atom_name, atoms_to_mask)
    else:
        atom_mask = np.ones(len(atom_array), dtype=bool)

    # Convert to RDKit mol
    mol = to_mol(atom_array, kekulize=True)
    Chem.SanitizeMol(mol)
    if np.all(atom_array.molecule_type_id == MoleculeType.LIGAND):
        Chem.AssignStereochemistryFrom3D(mol)
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
        atom_names (Iterable[str] | None):
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
    # Compute conformer (note that we call this before creating the annotations, as this
    # function will remove all hydrogens in the input mol and can therefore change the
    # mask length)
    result = multistrategy_compute_conformer(
        mol, remove_hs=True, timeouts={"default": 120, "random_init": 120}
    )
    mol, conf_id = result.mol, result.conf_id
    assert conf_id == 0

    # Assume all atoms are in the structure if no special mask is given
    if atom_mask is None:
        atom_mask = np.ones(mol.GetNumAtoms(), dtype=bool)

    # Set atom names if provided, otherwise renumber to C1, C2, N1, N2, etc.
    if atom_names is not None:
        mol = set_atomwise_annotation(mol, "atom_name", atom_names)
    else:
        # `create_atom_names` currently requires an AtomArray, so make one first
        # See https://github.com/biotite-dev/biotite/issues/915 for more context
        temp_struc = from_mol(mol, conformer_id=0, add_hydrogen=False)
        atom_names = struc.create_atom_names(temp_struc)
        mol = set_atomwise_annotation(mol, "atom_name", atom_names)
        del temp_struc

    # This is a different mask only required for fallback conformers in the training
    # script where some coordinates are not defined
    mol = set_atomwise_annotation(mol, "used_atom_mask", [True] * mol.GetNumAtoms())

    return ProcessedReferenceMolecule(
        mol=mol,
        in_crop_mask=atom_mask,
        permutations=None,
    )


def structure_with_ref_mols_from_sequence(
    sequence: str,
    poly_type: MoleculeType,
    chain_id: str,
    non_canonical_residues: dict[int, str] | None = None,
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
        non_canonical_residues (dict[int, str] | None):
            A dictionary mapping residue IDs to non-canonical residue names. Defaults to
            None.

    Returns:
        StructureWithReferenceMolecules:
            A named tuple containing the AtomArray and a list of processed reference
            molecules, each corresponding to a residue in the sequence.
    """
    if non_canonical_residues is None:
        non_canonical_residues = {}

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
    unk_res = MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_3[poly_type]
    base_leaving_atoms = MOLECULE_TYPE_TO_LEAVING_ATOMS[poly_type]

    atom_array = None
    processed_reference_mols = []

    for res_id, resname_1 in enumerate(sequence, start=1):
        # Swap to non-standard residue if necessary
        if res_id in non_canonical_residues:
            resname_3 = non_canonical_residues[res_id]
            leaving_atoms = get_leaving_atoms(resname_3)

            if len(leaving_atoms) == 0:
                logger.warning(
                    f"Non-canonical residue {resname_3} at position {res_id} has no "
                    "leaving atoms defined. Using default leaving atoms."
                )
                leaving_atoms = base_leaving_atoms

        else:
            leaving_atoms = base_leaving_atoms

            # Get 3-letter code of standard residue
            if resname_1 in resname_1_to_3:
                resname_3 = resname_1_to_3[resname_1]

            # Set unknown placeholder residue
            else:
                logger.warning(
                    f"Unknown residue {resname_1} at position {res_id} in sequence. "
                    f"Using placeholder residue {unk_res}."
                )
                resname_3 = unk_res

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
        leaving_atom_mask = np.isin(res_array.atom_name, leaving_atoms)
        if not leaving_atom_mask.any():
            logger.warning(
                f"Residue {resname_3} at position {res_id} has no leaving atoms to "
                "remove. This could cause issues with incorrect polymer linkage."
            )
        res_array = res_array[~leaving_atom_mask]

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

    # Retrieve atom names from special annotation in the reference mol
    atom_names = [atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()]

    # Convert to AtomArray
    atom_array = atom_array_from_mol(
        mol, atom_names=atom_names, chain_id=chain_id, res_name=res_name
    )

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


def _build_smiles_comp_id_mapping(query: Query) -> dict[str, str]:
    """Build the component-ID mapping for SMILES ligands."""
    all_smiles = sorted(
        {chain.smiles for chain in query.chains if chain.smiles is not None}
    )
    smiles_to_comp_id = {smiles: f"LIG{i}" for i, smiles in enumerate(all_smiles)}

    explicit_names: dict[str, str] = {}
    for chain in query.chains:
        if chain.smiles is None or chain.ligand_name is None:
            continue

        previous_name = explicit_names.get(chain.smiles)
        if previous_name is not None and previous_name != chain.ligand_name:
            raise ValueError("The same SMILES string cannot use multiple ligand names")
        explicit_names[chain.smiles] = chain.ligand_name

    reserved_component_ids = {
        component_id.upper()
        for chain in query.chains
        for component_id in [
            *(chain.ccd_codes or []),
            *(chain.non_canonical_residues or {}).values(),
        ]
    }
    collisions = set(explicit_names.values()) & reserved_component_ids
    if collisions:
        raise ValueError(
            "Explicit SMILES ligand names conflict with CCD or non-canonical "
            f"residue names: {sorted(collisions)}"
        )

    smiles_to_comp_id.update(explicit_names)
    component_ids = list(smiles_to_comp_id.values())
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("Distinct SMILES strings must use distinct ligand names")

    return smiles_to_comp_id


def _reference_atom_names(
    processed_reference_mol: ProcessedReferenceMolecule,
) -> list[str]:
    """Return atom names in the exact order used by a reference molecule mask."""
    return [
        atom.GetProp("annot_atom_name")
        for atom in processed_reference_mol.mol.GetAtoms()
    ]


def _apply_manual_leaving_atoms(
    atom_array: AtomArray,
    processed_reference_mols_by_residue: dict[
        tuple[str, int], list[ProcessedReferenceMolecule]
    ],
    query: Query,
) -> AtomArray:
    """Remove query-selected atoms and update matching reference-molecule masks."""
    if not query.leaving_atoms:
        return atom_array

    chain_ids = set(atom_array.chain_id.tolist())
    residues_by_chain = {
        chain_id: sorted(set(atom_array.res_id[atom_array.chain_id == chain_id]))
        for chain_id in chain_ids
    }
    updates: list[tuple[ProcessedReferenceMolecule, int]] = []
    atom_indices_to_remove: set[int] = set()
    seen_selectors: set[tuple[str, int, str]] = set()

    for leaving_atom in query.leaving_atoms:
        selector = _atom_selector_tuple(leaving_atom)
        if selector in seen_selectors:
            continue
        seen_selectors.add(selector)

        chain_id, residue_id, atom_name = selector
        if chain_id not in chain_ids:
            raise _query_error(
                query,
                f"leaving atom {_atom_selector_text(leaving_atom)} references unknown "
                f"chain {chain_id!r}; valid chains are {sorted(chain_ids)!r}",
            )

        reference_matches = processed_reference_mols_by_residue.get(
            (chain_id, residue_id), []
        )
        if len(reference_matches) == 0:
            raise _query_error(
                query,
                f"leaving atom {_atom_selector_text(leaving_atom)} references unknown "
                f"residue {residue_id} in chain {chain_id!r}; valid residue IDs are "
                f"{residues_by_chain[chain_id]!r}",
            )
        if len(reference_matches) > 1:
            raise _query_error(
                query,
                f"leaving atom {_atom_selector_text(leaving_atom)} is ambiguous "
                f"because chain {chain_id!r}, residue {residue_id} occurs more than "
                "once",
            )

        processed_reference_mol = reference_matches[0]
        reference_atom_names = _reference_atom_names(processed_reference_mol)
        reference_atom_indices = [
            index
            for index, reference_atom_name in enumerate(reference_atom_names)
            if reference_atom_name == atom_name
        ]
        if len(reference_atom_indices) == 0:
            raise _query_error(
                query,
                f"leaving atom {_atom_selector_text(leaving_atom)} has unknown atom "
                f"name {atom_name!r}; valid atom names are "
                f"{sorted(reference_atom_names)!r}",
            )
        if len(reference_atom_indices) > 1:
            raise _query_error(
                query,
                f"leaving atom {_atom_selector_text(leaving_atom)} is ambiguous; atom "
                f"name {atom_name!r} occurs {len(reference_atom_indices)} times",
            )

        reference_atom_index = reference_atom_indices[0]
        if len(processed_reference_mol.in_crop_mask) != len(reference_atom_names):
            raise _query_error(
                query,
                f"reference mask for leaving atom {_atom_selector_text(leaving_atom)} "
                "is not aligned with its reference molecule",
            )
        updates.append((processed_reference_mol, reference_atom_index))

        structure_matches = np.where(
            (atom_array.chain_id == chain_id)
            & (atom_array.res_id == residue_id)
            & (atom_array.atom_name == atom_name)
        )[0]
        if len(structure_matches) > 1:
            raise _query_error(
                query,
                f"leaving atom {_atom_selector_text(leaving_atom)} is ambiguous in the "
                f"assembled structure; it resolves to {len(structure_matches)} atoms",
            )
        if len(structure_matches) == 1:
            atom_indices_to_remove.add(int(structure_matches[0]))
        elif processed_reference_mol.in_crop_mask[reference_atom_index]:
            raise _query_error(
                query,
                f"leaving atom {_atom_selector_text(leaving_atom)} is present in the "
                "reference mask but missing from the assembled structure",
            )

    # Apply all edits only after every selector has validated.
    for processed_reference_mol, reference_atom_index in updates:
        processed_reference_mol.in_crop_mask[reference_atom_index] = False

    if atom_indices_to_remove:
        keep_mask = np.ones(len(atom_array), dtype=bool)
        keep_mask[list(atom_indices_to_remove)] = False
        atom_array = atom_array[keep_mask]

    return atom_array


def _resolve_covalent_endpoint(
    atom_array: AtomArray,
    endpoint: Atom,
    query: Query,
    bond_index: int,
    endpoint_index: int,
    selector_to_indices: dict[tuple[str, int, str], list[int]],
) -> int:
    """Resolve one public endpoint with layered, actionable diagnostics."""
    chain_id, residue_id, atom_name = _atom_selector_tuple(endpoint)
    chain_mask = atom_array.chain_id == chain_id
    if not np.any(chain_mask):
        raise _query_error(
            query,
            f"covalent bond {bond_index} endpoint {endpoint_index} "
            f"{_atom_selector_text(endpoint)} references unknown chain {chain_id!r}; "
            f"valid chains are {sorted(set(atom_array.chain_id.tolist()))!r}",
        )

    residue_mask = chain_mask & (atom_array.res_id == residue_id)
    if not np.any(residue_mask):
        valid_residue_ids = sorted(set(atom_array.res_id[chain_mask].tolist()))
        raise _query_error(
            query,
            f"covalent bond {bond_index} endpoint {endpoint_index} "
            f"{_atom_selector_text(endpoint)} references unknown residue {residue_id} "
            f"in chain {chain_id!r}; valid residue IDs are {valid_residue_ids!r}",
        )

    selector = _atom_selector_tuple(endpoint)
    atom_indices = selector_to_indices.get(selector, [])
    if len(atom_indices) == 0:
        valid_atom_names = sorted(set(atom_array.atom_name[residue_mask].tolist()))
        raise _query_error(
            query,
            f"covalent bond {bond_index} endpoint {endpoint_index} "
            f"{_atom_selector_text(endpoint)} references unknown atom name "
            f"{atom_name!r}; valid atom names are {valid_atom_names!r}",
        )
    if len(atom_indices) > 1:
        raise _query_error(
            query,
            f"covalent bond {bond_index} endpoint {endpoint_index} "
            f"{_atom_selector_text(endpoint)} is ambiguous and resolves to "
            f"{len(atom_indices)} atoms",
        )
    return atom_indices[0]


def _materialize_query_covalent_bonds(
    atom_array: AtomArray,
    query: Query,
) -> None:
    """Validate and atomically merge query-defined bonds into an AtomArray."""
    if not query.covalent_bonds:
        return

    selector_to_indices: dict[tuple[str, int, str], list[int]] = {}
    for atom_index, selector in enumerate(
        zip(
            atom_array.chain_id.tolist(),
            atom_array.res_id.tolist(),
            atom_array.atom_name.tolist(),
            strict=True,
        )
    ):
        selector_to_indices.setdefault(selector, []).append(atom_index)

    leaving_atom_selectors = {
        _atom_selector_tuple(atom) for atom in query.leaving_atoms or []
    }
    intrinsic_bonds = (
        atom_array.bonds.as_array()
        if atom_array.bonds is not None
        else np.empty((0, 3), dtype=np.uint32)
    )
    intrinsic_bond_types: dict[tuple[int, int], set[int]] = {}
    for atom_index_1, atom_index_2, bond_type in intrinsic_bonds:
        pair = tuple(sorted((int(atom_index_1), int(atom_index_2))))
        intrinsic_bond_types.setdefault(pair, set()).add(int(bond_type))

    seen_custom_pairs: set[tuple[int, int]] = set()
    resolved_bonds: list[tuple[int, int, int]] = []
    endpoint_indices: set[int] = set()

    for bond_index, bond in enumerate(query.covalent_bonds):
        endpoint_1, endpoint_2 = bond
        for endpoint in (endpoint_1, endpoint_2):
            if _atom_selector_tuple(endpoint) in leaving_atom_selectors:
                raise _query_error(
                    query,
                    f"covalent bond {bond_index} endpoint "
                    f"{_atom_selector_text(endpoint)} is also selected as a leaving "
                    "atom",
                )

        atom_index_1 = _resolve_covalent_endpoint(
            atom_array,
            endpoint_1,
            query,
            bond_index,
            0,
            selector_to_indices,
        )
        atom_index_2 = _resolve_covalent_endpoint(
            atom_array,
            endpoint_2,
            query,
            bond_index,
            1,
            selector_to_indices,
        )

        if endpoint_1.chain_id == endpoint_2.chain_id:
            raise _query_error(
                query,
                f"covalent bond {bond_index} connects endpoints in the same chain "
                f"{endpoint_1.chain_id!r}; v1 supports inter-chain bonds only",
            )
        if atom_index_1 == atom_index_2:
            raise _query_error(query, f"covalent bond {bond_index} is a self-bond")

        pair = tuple(sorted((atom_index_1, atom_index_2)))
        if pair in seen_custom_pairs:
            raise _query_error(
                query,
                f"covalent bond {bond_index} duplicates an earlier custom bond, "
                "including when its endpoint order is reversed",
            )
        seen_custom_pairs.add(pair)
        endpoint_indices.update(pair)

        existing_types = intrinsic_bond_types.get(pair)
        if existing_types is not None:
            if existing_types == {int(struc.BondType.SINGLE)}:
                logger.debug(
                    "Query %r covalent bond %d is already present as an intrinsic "
                    "single bond",
                    query.query_name,
                    bond_index,
                )
                continue
            raise _query_error(
                query,
                f"covalent bond {bond_index} conflicts with intrinsic bond type(s) "
                f"{sorted(existing_types)!r}",
            )

        resolved_bonds.append((atom_index_1, atom_index_2, int(struc.BondType.SINGLE)))
        logger.debug(
            "Query %r covalent bond %d resolved %s and %s to atom indices %d and %d",
            query.query_name,
            bond_index,
            _atom_selector_text(endpoint_1),
            _atom_selector_text(endpoint_2),
            atom_index_1,
            atom_index_2,
        )

    # All declarations are valid: now perform the mutations as one commit.
    if resolved_bonds:
        custom_bonds = np.asarray(resolved_bonds, dtype=np.uint32)
        merged_bonds = np.concatenate((intrinsic_bonds, custom_bonds), axis=0)
        atom_array.bonds = struc.BondList(len(atom_array), merged_bonds)

    force_atomization = np.zeros(len(atom_array), dtype=bool)
    forced_residues: set[tuple[str, int]] = set()
    for endpoint_index in endpoint_indices:
        if (
            atom_array.molecule_type_id[endpoint_index] != MoleculeType.LIGAND
            and atom_array.res_name[endpoint_index] in STANDARD_RESIDUES_3
        ):
            forced_residues.add(
                (
                    str(atom_array.chain_id[endpoint_index]),
                    int(atom_array.res_id[endpoint_index]),
                )
            )
            force_atomization |= (
                atom_array.chain_id == atom_array.chain_id[endpoint_index]
            ) & (atom_array.res_id == atom_array.res_id[endpoint_index])
    atom_array.set_annotation(_FORCE_ATOMIZATION_ANNOTATION, force_atomization)
    logger.debug(
        "Query %r validated %d custom bond(s), inserted %d new edge(s), and "
        "marked %d canonical endpoint residue(s) for atomization",
        query.query_name,
        len(query.covalent_bonds),
        len(resolved_bonds),
        len(forced_residues),
    )


def structure_with_ref_mols_from_query(query: Query) -> StructureWithReferenceMolecules:
    """Builds an AtomArray and processed reference molecules from a Query object.

    Parses the Query object into a full AtomArray and processed reference molecules
    (RDKit mol objects with atom names and computed conformers).

    The returned AtomArray follows the chain IDs given in the Query object. If a chain
    specifies multiple chain IDs, repeated identical chains with those IDs will be
    constructed and given the same entity ID.

    Residue names will be inferred from the sequence or CCD codes. SMILES ligands use
    an explicit ``ligand_name`` when provided and otherwise retain their deterministic
    ``LIG0``, ``LIG1``, ... defaults.

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
    processed_reference_mols_by_residue: dict[
        tuple[str, int], list[ProcessedReferenceMolecule]
    ] = {}

    # Create entity mapping
    all_entities = set()
    for chain in query.chains:
        if chain.sequence is not None:
            all_entities.add(chain.sequence)
        elif chain.ccd_codes is not None:
            for ccd in chain.ccd_codes:
                all_entities.add(ccd)
        elif chain.smiles is not None:
            all_entities.add(chain.smiles)
    all_entities = sorted(all_entities)
    entity_to_id = {e: i + 1 for i, e in enumerate(all_entities)}

    smiles_to_comp_id = _build_smiles_comp_id_mapping(query)

    # Build the structure segment-wise from all chains in the query.
    for chain in query.chains:
        for chain_id in chain.chain_ids:
            match chain.molecule_type:
                # Build polymeric segment
                case MoleculeType.PROTEIN | MoleculeType.DNA | MoleculeType.RNA:
                    segment_atom_array, segment_ref_mols = (
                        structure_with_ref_mols_from_sequence(
                            sequence=chain.sequence,
                            poly_type=chain.molecule_type,
                            chain_id=chain_id,
                            non_canonical_residues=chain.non_canonical_residues,
                        )
                    )
                    representation = chain.sequence

                # Build ligand molecule
                case MoleculeType.LIGAND:
                    # Build ligand from SMILES
                    if chain.smiles is not None:
                        segment_atom_array, segment_ref_mols = (
                            structure_with_ref_mol_from_smiles(
                                smiles=chain.smiles,
                                chain_id=chain_id,
                                res_name=smiles_to_comp_id[chain.smiles],
                            )
                        )
                        representation = chain.smiles

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
                        representation = chain.ccd_codes[0]

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
            for residue_id, processed_reference_mol in enumerate(
                segment_ref_mols, start=1
            ):
                processed_reference_mols_by_residue.setdefault(
                    (chain_id, residue_id), []
                ).append(processed_reference_mol)

            segment_atom_array.set_annotation(
                "entity_id",
                np.repeat(entity_to_id[representation], len(segment_atom_array)),
            )
            segment_atom_array.set_annotation(
                "is_cyclic",
                np.repeat(chain.cyclic, len(segment_atom_array)),
            )

            # Append atom array to end
            if atom_array is None:
                atom_array = segment_atom_array
            else:
                atom_array += segment_atom_array

    atom_array = _apply_manual_leaving_atoms(
        atom_array,
        processed_reference_mols_by_residue,
        query,
    )
    _materialize_query_covalent_bonds(atom_array, query)

    # Force coordinates to 0 for consistency
    atom_array.coord[:] = 0.0

    return StructureWithReferenceMolecules(
        atom_array=atom_array, processed_reference_mols=processed_reference_mols
    )
