# Copyright 2025 AlQuraishi Laboratory
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

from collections import defaultdict
from datetime import datetime
from typing import Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import BondType
from biotite.structure.info.bonds import BOND_TYPES
from biotite.structure.io.pdbx import BinaryCIFFile, CIFBlock, CIFCategory, CIFFile

from openfold3.core.data.primitives.structure.labels import (
    get_chain_to_entity_dict,
)
from openfold3.core.data.resources.residues import (
    CHEM_COMP_TYPE_TO_MOLECULE_TYPE,
    STANDARD_PROTEIN_RESIDUES_1,
    STANDARD_RESIDUES_3,
    STANDARD_RNA_RESIDUES,
    MoleculeType,
)


def get_pdb_id(cif_file: CIFFile, format: Literal["upper", "lower"] = "lower") -> str:
    """Get the PDB ID of the structure.

    Args:
        cif_file:
            Parsed mmCIF file containing the structure.
        format:
            The case of the PDB ID to return. Options are "upper" and "lower". Defaults
            to "lower".

    Returns:
        The PDB ID of the structure.
    """
    (pdb_id,) = cif_file.keys()

    if format == "upper":
        return pdb_id.upper()
    elif format == "lower":
        return pdb_id.lower()
    else:
        raise ValueError(f"Invalid format: {format}")


def get_release_date(cif_data: CIFBlock) -> datetime:
    """Get the release date of the structure.

    Release date is defined as the earliest revision date of the structure.

    Args:
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        The release date of the structure.
    """
    release_dates = cif_data["pdbx_audit_revision_history"]["revision_date"].as_array()
    release_dates = [datetime.strptime(date, "%Y-%m-%d") for date in release_dates]

    return min(release_dates)


def get_resolution(cif_data: CIFBlock) -> float:
    """Get the resolution of the structure.

    The resolution is obtained by sequentially checking the following data items:
    - refine.ls_d_res_high
    - em_3d_reconstruction.resolution
    - reflns.d_resolution_high

    and returning the first one that is found. If none of the above data items are
    found, the function returns NaN.

    Args:
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        The resolution of the structure.
    """
    keys_to_check = [
        ("refine", "ls_d_res_high"),
        ("em_3d_reconstruction", "resolution"),
        ("reflns", "d_resolution_high"),
    ]

    for key in keys_to_check:
        try:
            # as_array() because very rare structures can have multiple resolutions
            # (e.g. 7TX3)
            resolution = cif_data[key[0]][key[1]].as_array()[0].item()

            # Try next if not specified
            if resolution in ("?", "."):
                continue

            # If successful, convert to float and return
            resolution = float(resolution)
            break

        # Try next if key not found
        except KeyError:
            continue
    else:
        resolution = float("nan")

    # dev-only: TODO remove
    assert isinstance(resolution, float), "Resolution is not a float"

    return resolution


def get_experimental_method(cif_data: CIFBlock) -> str:
    """Get the experimental method used to determine the structure.

    Args:
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        The experimental method used to determine the structure.
    """
    # as_array() because very rare structures can have multiple methods (e.g. 7TX3)
    method = cif_data["exptl"]["method"].as_array()[0].item().upper()

    return method


def get_cif_block(cif_file: CIFFile) -> CIFBlock:
    """Get the CIF block of the structure.

    Args:
        cif_file:
            Parsed mmCIF file containing the structure.

    Returns:
        The CIF block of the structure.
    """
    (pdb_id,) = cif_file.keys()
    cif_block = cif_file[pdb_id]

    return cif_block


def get_entity_to_canonical_seq_dict(
    cif_data: CIFBlock, multi_letter_res_to_X: bool = True, ccd: CIFFile | None = None
) -> dict[int, str]:
    """Get a dictionary mapping entity IDs to their canonical sequences.

    Args:
        cif_data (CIFBlock):
            The CIF data block containing the entity_poly table.
        multi_letter_res_to_X (bool):
            Whether to replace residues that map to multiple one-letter codes with 'X',
            which can be necessary to keep a 1:1 correspondence of MSA-residue features
            downstream. An example for this are GFP chromophores. Defaults to True.
        ccd (CIFFile | None):
            The parsed chemical component dictionary (CCD). Only necessary if
            `multi_letter_res_to_X` is True.

    Returns:
        A dictionary mapping entity IDs to their sequences.
    """
    if multi_letter_res_to_X and ccd is None:
        raise ValueError("If multi_letter_res_to_X is True, the CCD must be provided.")

    polymer_entities = cif_data["entity_poly"]["entity_id"].as_array(dtype=int)
    polymer_canonical_seqs = cif_data["entity_poly"][
        "pdbx_seq_one_letter_code_can"
    ].as_array()
    polymer_canonical_seqs = np.char.replace(polymer_canonical_seqs, "\n", "")

    entity_to_canonical_seq_dict = dict(
        zip(polymer_entities.tolist(), polymer_canonical_seqs.tolist(), strict=True)
    )

    if not multi_letter_res_to_X:
        return entity_to_canonical_seq_dict

    # Check if there is any entity that has multi-letter residues
    entity_to_3l_dict = get_entity_to_three_letter_codes_dict(cif_data)
    for entity in polymer_entities.tolist():
        unique_3l = set(entity_to_3l_dict[entity])

        # If there is any multi-letter residue, rebuild the sequence using the
        # CCD-one-letter-codes, setting everything not in the standard one-letter codes
        # to 'X'.
        # NOTE: Technically this could also respect parent ID annotations for
        # non-standard AA to cast less residues to 'X', but we go with a simpler
        # approach for now, given that multi-letter residues are rare.
        if any(
            len(ccd[three_letter]["chem_comp"]["one_letter_code"].as_item()) > 1
            for three_letter in unique_3l
        ):
            new_seq = []

            for three_letter in entity_to_3l_dict[entity]:
                one_letter = ccd[three_letter]["chem_comp"]["one_letter_code"].as_item()

                # "+" Is an allowed character according to
                # https://mmcif.wwpdb.org/dictionaries/mmcif_std.dic/Items/_chem_comp.one_letter_code.html
                one_letter = one_letter.replace("+", "")

                # Find the standard one-letter code set for this molecule type (RNA set
                # is also used for DNA because they share the same one-letter codes)
                chem_comp_type = (
                    ccd[three_letter]["chem_comp"]["type"].as_item().upper()
                )
                molecule_type = CHEM_COMP_TYPE_TO_MOLECULE_TYPE[chem_comp_type]
                ref_one_letter_codes = (
                    STANDARD_PROTEIN_RESIDUES_1
                    if molecule_type == MoleculeType.PROTEIN
                    else STANDARD_RNA_RESIDUES
                )

                # Set to 'X' if not a standard AA
                one_letter = one_letter if one_letter in ref_one_letter_codes else "X"

                new_seq.append(one_letter)

            entity_to_canonical_seq_dict[entity] = "".join(new_seq)

    return entity_to_canonical_seq_dict


def get_chain_to_canonical_seq_dict(
    atom_array: struc.AtomArray,
    cif_data: CIFBlock,
    multi_letter_res_to_X: bool = True,
    ccd: CIFFile | None = None,
) -> dict[int, str]:
    """Get a dictionary mapping chain IDs to their canonical sequences.

    Args:
        atom_array:
            AtomArray containing the chain IDs and entity IDs.
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)
        multi_letter_res_to_X (bool):
            Whether to replace residues that map to multiple one-letter codes with 'X',
            which can be necessary to keep a 1:1 correspondence of sequence-residue
            features downstream. An example for this are GFP chromophores. Defaults to
            True.
        ccd (CIFFile | None):
            The parsed chemical component dictionary (CCD). Only necessary if
            `multi_letter_res_to_X` is True.
    """
    entity_to_seq_dict = get_entity_to_canonical_seq_dict(
        cif_data=cif_data, multi_letter_res_to_X=multi_letter_res_to_X, ccd=ccd
    )
    chain_to_entity_dict = get_chain_to_entity_dict(atom_array)

    chain_to_seq_dict = {
        chain: entity_to_seq_dict[entity]
        for chain, entity in chain_to_entity_dict.items()
        if entity in entity_to_seq_dict
    }

    return chain_to_seq_dict


# TODO: Revisit multi-letter-res arguments
def get_asym_id_to_canonical_seq_dict(
    cif_file: CIFFile | BinaryCIFFile,
    multi_letter_res_to_X: bool = False,
    ccd: CIFFile | None = None,
) -> dict[str, str]:
    """Get a dictionary mapping asym IDs to their canonical sequences.

    Args:
        cif_file (CIFFile | BinaryCIFFile):
            Parsed mmCIF file containing the structure.
        multi_letter_res_to_X (bool):
            Whether to replace residues that map to multiple one-letter codes with 'X',
            which can be necessary to keep a 1:1 correspondence of sequence-residue
            features downstream. An example for this are GFP chromophores. Defaults to
            False.
        ccd (CIFFile | None):
            The parsed chemical component dictionary (CCD). Only necessary if
            `multi_letter_res_to_X` is True.

    Returns:
        dict[str, str]:
            A dictionary mapping asym IDs to their canonical sequences.
    """
    # Create entity_id -> canonical sequence map
    entity_id_to_can_seq = get_entity_to_canonical_seq_dict(
        get_cif_block(cif_file), multi_letter_res_to_X, ccd
    )

    # Create asym_id -> canonical sequence map
    asym_ids = cif_file.block["pdbx_poly_seq_scheme"]["asym_id"].as_array()
    entity_ids = cif_file.block["pdbx_poly_seq_scheme"]["entity_id"].as_array()
    asym_to_entity_array = np.unique(
        np.concatenate([asym_ids[:, np.newaxis], entity_ids[:, np.newaxis]], axis=1),
        axis=0,
    )
    asym_to_entity_dict = {row[0]: row[1] for row in asym_to_entity_array}

    # Create asym_id -> canonical sequence map
    return {
        asym_id: entity_id_to_can_seq[int(entity_id)]
        for asym_id, entity_id in asym_to_entity_dict.items()
    }


def get_entity_to_three_letter_codes_dict(cif_data: CIFBlock) -> dict[int, list[str]]:
    """Get a dictionary mapping entity IDs to their three-letter-code sequences.

    Note that in the special case of multiple amino acids being set to the same residue
    ID, this will currently default to taking the first one and make no special attempt
    to take occupancy into account.

    Args:
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        A dictionary mapping entity IDs to their three-letter-code sequences.
    """
    # Flat list of residue-wise entity IDs for all polymeric sequences
    entity_ids_flat = cif_data["entity_poly_seq"]["entity_id"].as_array(dtype=int)

    # Deduplicated entity IDs
    entity_ids = np.unique(entity_ids_flat)

    # Get full (3-letter code) residue sequence for every polymeric entity
    entity_monomers = cif_data["entity_poly_seq"]["mon_id"].as_array()

    entity_residue_ids = cif_data["entity_poly_seq"]["num"].as_array()

    # Get map of residue IDs to monomers sharing that residue ID for every entity
    res_id_to_monomers = defaultdict(lambda: defaultdict(list))
    for entity_id, res_id, ccd_id in zip(
        entity_ids_flat.tolist(),
        entity_residue_ids.tolist(),
        entity_monomers.tolist(),
        strict=False,
    ):
        res_id_to_monomers[entity_id][res_id].append(ccd_id)

    # In case where multiple monomers are set to the same residue ID, take the first one
    # (TODO: this should ideally take occupancy into account)
    entity_id_to_3l_codes = {
        entity_id: [monomers[0] for monomers in res_id_to_monomers[entity_id].values()]
        for entity_id in entity_ids
    }

    return entity_id_to_3l_codes


def get_chain_to_three_letter_codes_dict(
    atom_array: struc.AtomArray, cif_data: CIFBlock
) -> dict[int, list[str]]:
    """Get dictionary mapping chain IDs to their three-letter-code sequences.

    Args:
        atom_array:
            AtomArray containing the chain IDs and entity IDs.
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see `get_cif_block`)

    Returns:
        A dictionary mapping chain IDs to their three-letter-code sequences.
    """
    entity_ids_to_3l_codes = get_entity_to_three_letter_codes_dict(cif_data)
    chain_to_entity_dict = get_chain_to_entity_dict(atom_array)

    chain_to_3l_codes_dict = {
        chain: entity_ids_to_3l_codes[entity]
        for chain, entity in chain_to_entity_dict.items()
        if entity in entity_ids_to_3l_codes
    }

    return chain_to_3l_codes_dict


def get_ccd_atom_pair_to_bond_dict(ccd_entry: CIFBlock) -> dict[(str, str), BondType]:
    """Gets the list of bonds from a CCD entry.

    Args:
        ccd_entry:
            CIFBlock containing the CCD entry.

    Returns:
        Dictionary mapping each pair of atom names to the respective Biotite bond type.
    """

    chem_comp_bonds = ccd_entry.get("chem_comp_bond")

    if chem_comp_bonds is None:
        return {}

    atom_pair_to_bond = {}

    for atom_1, atom_2, ccd_bond_type, aromatic_flag in zip(
        chem_comp_bonds["atom_id_1"].as_array(),
        chem_comp_bonds["atom_id_2"].as_array(),
        chem_comp_bonds["value_order"].as_array(),
        chem_comp_bonds["pdbx_aromatic_flag"].as_array(),
        strict=False,
    ):
        bond_type = BOND_TYPES[ccd_bond_type, aromatic_flag]
        atom_pair_to_bond[(atom_1.item(), atom_2.item())] = bond_type

    return atom_pair_to_bond


def get_ccd_atom_id_to_element_dict(ccd_entry: CIFBlock) -> dict[str, str]:
    """Gets the dictionary mapping atom IDs to element symbols from a CCD entry.

    Args:
        ccd_entry:
            CIFBlock containing the CCD entry.

    Returns:
        Dictionary mapping atom IDs to element symbols.
    """

    atom_id_to_element = {
        atom_id.item(): element.item()
        for atom_id, element in zip(
            ccd_entry["chem_comp_atom"]["atom_id"].as_array(),
            ccd_entry["chem_comp_atom"]["type_symbol"].as_array(),
            strict=False,
        )
    }

    return atom_id_to_element


def get_ccd_atom_id_to_charge_dict(ccd_entry: CIFBlock) -> dict[str, float]:
    """Gets the dictionary mapping atom IDs to charges from a CCD entry.

    Args:
        ccd_entry:
            CIFBlock containing the CCD entry.

    Returns:
        Dictionary mapping atom IDs to charges.
    """

    atom_id_to_charge = {
        atom_id.item(): (int(charge) if charge != "?" else 0)
        for atom_id, charge in zip(
            ccd_entry["chem_comp_atom"]["atom_id"].as_array(),
            ccd_entry["chem_comp_atom"]["charge"].as_array(),
            strict=False,
        )
    }

    return atom_id_to_charge


def get_first_bioassembly_polymer_count(cif_data: CIFBlock) -> int:
    """Returns the number of polymer chains in the first bioassembly."""
    return (
        cif_data["pdbx_struct_assembly"]["oligomeric_count"]
        .as_array(dtype=int)[0]
        .item()
    )


def writer_update_atom_site(
    atom_array: struc.AtomArray, cif_block: CIFBlock, make_ost_compatible: bool = True
) -> None:
    """Updates the atom_site field to be consistent with regular PDB-RCSB format."""
    masks = {
        mtn: atom_array.molecule_type_id == mt
        for mt, mtn in zip(MoleculeType, MoleculeType._member_names_, strict=True)
    }

    label_seq_id = cif_block["atom_site"]["label_seq_id"].as_array()
    label_seq_id[masks["LIGAND"]] = "."
    cif_block["atom_site"]["label_seq_id"] = label_seq_id

    PDB_ins_code = cif_block["atom_site"]["pdbx_PDB_ins_code"].as_array()
    PDB_ins_code[masks["LIGAND"]] = "?"
    cif_block["atom_site"]["pdbx_PDB_ins_code"] = PDB_ins_code

    if make_ost_compatible:
        formal_charge = cif_block["atom_site"]["pdbx_formal_charge"].as_array()
        for i in range(1, 8):
            fc_mask = formal_charge == f"+{i}"
            if np.any(fc_mask):
                formal_charge[formal_charge == f"+{i}"] = f"{i}"
        cif_block["atom_site"]["pdbx_formal_charge"] = formal_charge


def writer_add_chem_comp(atom_array: struc.AtomArray, cif_block: CIFBlock) -> None:
    """Adds the chem_comp cif field to a CIFBlock."""
    masks = {
        mtn: atom_array.molecule_type_id == mt
        for mt, mtn in zip(MoleculeType, MoleculeType._member_names_, strict=True)
    }
    polymer_types = [MoleculeType.PROTEIN, MoleculeType.RNA, MoleculeType.DNA]

    chem_comp_data = {
        "id": np.array([], dtype=str),
        "type": np.array([], dtype=str),
        "mon_nstd_flag": np.array([], dtype=str),
        "name": np.array([], dtype=str),
    }
    for mt, mtn in zip(MoleculeType, MoleculeType._member_names_, strict=True):
        chem_comp_i = np.array(
            [str(i) for i in sorted(set(atom_array.res_name[masks[mtn]]))]
        )

        if (mt in polymer_types) & (len(chem_comp_i) > 0):
            mon_nstd_flag = np.array(["n" for _ in range(len(chem_comp_i))])
            is_standard = np.isin(chem_comp_i, STANDARD_RESIDUES_3)
            mon_nstd_flag[is_standard] = "y"
            # TODO: add support for more types, see keys in
            # core/data/resources/residues.py CHEM_COMP_TYPE_TO_MOLECULE_TYPE keys
            match mt:
                case MoleculeType.PROTEIN:
                    chem_comp_type = np.array(
                        ["L-peptide linking" for _ in range(len(chem_comp_i))]
                    )
                case MoleculeType.RNA:
                    chem_comp_type = np.array(
                        ["RNA linking" for _ in range(len(chem_comp_i))]
                    )
                case MoleculeType.DNA:
                    chem_comp_type = np.array(
                        ["DNA linking" for _ in range(len(chem_comp_i))]
                    )
        else:
            mon_nstd_flag = np.array(["." for _ in range(len(chem_comp_i))])
            chem_comp_type = np.array(["non-polymer" for _ in range(len(chem_comp_i))])

        chem_comp_data["id"] = np.concatenate(
            (chem_comp_data["id"], chem_comp_i), axis=0
        )
        chem_comp_data["type"] = np.concatenate(
            (chem_comp_data["type"], chem_comp_type), axis=0
        )
        chem_comp_data["mon_nstd_flag"] = np.concatenate(
            (chem_comp_data["mon_nstd_flag"], mon_nstd_flag), axis=0
        )
        # TODO: update name to actual names
        chem_comp_data["name"] = np.concatenate(
            (chem_comp_data["name"], chem_comp_i), axis=0
        )
        # TODO add columns if needed:
        # pdbx_synonyms, formula, formula_weight

    cif_block["chem_comp"] = CIFCategory(chem_comp_data)


def writer_add_entity(atom_array: struc.AtomArray, cif_block: CIFBlock) -> None:
    """Adds the entity cif field to a CIFBlock."""
    entity_data = {
        "id": np.array([], dtype=str),
        "type": np.array([], dtype=str),
        "pdbx_description": np.array([], dtype=str),
        "details": np.array([], dtype=str),
    }
    for entity_id, mt in sorted(
        set(
            [
                (str(i), MoleculeType(int(j)))
                for i, j in zip(
                    atom_array.entity_id, atom_array.molecule_type_id, strict=False
                )
            ]
        )
    ):
        match mt:
            case MoleculeType.PROTEIN:
                entity_type = "polymer"
                description = details = "?"
            case MoleculeType.RNA:
                entity_type = "polymer"
                description = details = "?"
            case MoleculeType.DNA:
                entity_type = "polymer"
                description = details = "?"
            case MoleculeType.LIGAND:
                entity_type = "non-polymer"
                description = details = "?"

        entity_data["id"] = np.concatenate(
            (entity_data["id"], np.array([entity_id], dtype=str)), axis=0
        )
        entity_data["type"] = np.concatenate(
            (entity_data["type"], np.array([entity_type], dtype=str)), axis=0
        )
        entity_data["pdbx_description"] = np.concatenate(
            (entity_data["pdbx_description"], np.array([description], dtype=str)),
            axis=0,
        )
        entity_data["details"] = np.concatenate(
            (entity_data["details"], np.array([details], dtype=str)), axis=0
        )

    cif_block["entity"] = CIFCategory(entity_data)


def writer_add_struct_asym(atom_array: struc.AtomArray, cif_block: CIFBlock) -> None:
    struct_asym_data = {
        "id": np.array([], dtype=str),
        "entity_id": np.array([], dtype=str),
        "details": np.array([], dtype=str),
    }
    for chain_id, entity_id in sorted(
        set(
            [
                (str(i), str(j))
                for i, j in zip(atom_array.chain_id, atom_array.entity_id, strict=True)
            ]
        )
    ):
        struct_asym_data["id"] = np.concatenate(
            (struct_asym_data["id"], np.array([chain_id], dtype=str)), axis=0
        )
        struct_asym_data["entity_id"] = np.concatenate(
            (struct_asym_data["entity_id"], np.array([entity_id], dtype=str)), axis=0
        )
        struct_asym_data["details"] = np.concatenate(
            (struct_asym_data["details"], np.array(["?"], dtype=str)), axis=0
        )

    cif_block["struct_asym"] = CIFCategory(struct_asym_data)


def writer_add_pdbx_nonpoly_scheme(
    atom_array: struc.AtomArray, cif_block: CIFBlock
) -> None:
    masks = {
        mtn: atom_array.molecule_type_id == mt
        for mt, mtn in zip(MoleculeType, MoleculeType._member_names_, strict=True)
    }
    nonpoly_scheme_data = {
        "entity_id": np.array([], dtype=str),  # entity_id
        "asym_id": np.array([], dtype=str),  # chain_id
        "mon_id": np.array([], dtype=str),  # component code
        "pdb_mon_id": np.array([], dtype=str),  # component code
        "auth_mon_id": np.array([], dtype=str),  # component code
        "ndb_seq_num": np.array([], dtype=str),  # res_id
        "pdb_seq_num": np.array([], dtype=str),  # res_id
        "auth_seq_num": np.array([], dtype=str),  # res_id
        "pdb_strand_id": np.array([], dtype=str),  # chain_id
        "pdb_ins_code": np.array([], dtype=str),  # .
    }

    ## if they are not ligands, don't write empty schema
    if len(atom_array[masks["LIGAND"]]) == 0:
        return

    for entity_id, chain_id, res_id, res_name in sorted(
        set(
            [
                (str(i), str(j), str(k), str(l))
                for i, j, k, l in zip(
                    atom_array[masks["LIGAND"]].entity_id,
                    atom_array[masks["LIGAND"]].chain_id,
                    atom_array[masks["LIGAND"]].res_id,
                    atom_array[masks["LIGAND"]].res_name,
                    strict=False,
                )
            ]
        )
    ):
        nonpoly_scheme_data["entity_id"] = np.concatenate(
            (nonpoly_scheme_data["entity_id"], np.array([entity_id], dtype=str)), axis=0
        )
        nonpoly_scheme_data["asym_id"] = np.concatenate(
            (nonpoly_scheme_data["asym_id"], np.array([chain_id], dtype=str)), axis=0
        )
        nonpoly_scheme_data["mon_id"] = np.concatenate(
            (nonpoly_scheme_data["mon_id"], np.array([res_name], dtype=str)), axis=0
        )
        nonpoly_scheme_data["pdb_mon_id"] = np.concatenate(
            (nonpoly_scheme_data["pdb_mon_id"], np.array([res_name], dtype=str)), axis=0
        )
        nonpoly_scheme_data["auth_mon_id"] = np.concatenate(
            (nonpoly_scheme_data["auth_mon_id"], np.array([res_name], dtype=str)),
            axis=0,
        )
        nonpoly_scheme_data["pdb_strand_id"] = np.concatenate(
            (nonpoly_scheme_data["pdb_strand_id"], np.array([chain_id], dtype=str)),
            axis=0,
        )
        nonpoly_scheme_data["ndb_seq_num"] = np.concatenate(
            (nonpoly_scheme_data["ndb_seq_num"], np.array([res_id], dtype=str)), axis=0
        )
        nonpoly_scheme_data["pdb_seq_num"] = np.concatenate(
            (nonpoly_scheme_data["pdb_seq_num"], np.array([res_id], dtype=str)), axis=0
        )
        nonpoly_scheme_data["auth_seq_num"] = np.concatenate(
            (nonpoly_scheme_data["auth_seq_num"], np.array([res_id], dtype=str)), axis=0
        )
        nonpoly_scheme_data["pdb_ins_code"] = np.concatenate(
            (nonpoly_scheme_data["pdb_ins_code"], np.array(["."], dtype=str)), axis=0
        )

    cif_block["pdbx_nonpoly_scheme"] = CIFCategory(nonpoly_scheme_data)
