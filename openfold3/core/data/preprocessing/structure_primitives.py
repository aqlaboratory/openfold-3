import logging
from typing import Literal

import biotite.structure as struc
import numpy as np
from biotite.structure.io.pdbx import CIFBlock, CIFFile
from scipy.spatial.distance import cdist

from .tables import (
    MOLECULE_TYPE_ID_DNA,
    MOLECULE_TYPE_ID_LIGAND,
    MOLECULE_TYPE_ID_PROTEIN,
    MOLECULE_TYPE_ID_RNA,
    NUCLEIC_ACID_PHOSPHATE_ATOMS,
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES,
    STANDARD_RNA_RESIDUES,
)


def assign_renumbered_chain_ids(atom_array: struc.AtomArray) -> None:
    """Adds a renumbered chain index to the AtomArray

    Iterates through all chains in the atom array and assigns unique numerical chain IDs
    starting with 0 to each chain in the "chain_id_renumbered" field. This is useful for
    bioassembly parsing where chain IDs can be duplicated after the assembly is
    expanded.

    Args:
        atom_array:
            AtomArray containing the structure to assign renumbered chain IDs to.
    """
    chain_start_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # Assign numerical chain IDs
    chain_id_n_repeats = np.diff(chain_start_idxs)
    chain_ids_per_atom = np.repeat(
        np.arange(len(chain_id_n_repeats)), chain_id_n_repeats
    )
    atom_array.set_annotation("chain_id_renumbered", chain_ids_per_atom)


def assign_atom_indices(atom_array: struc.AtomArray) -> None:
    """Assigns atom indices to the AtomArray

    Atom indices are a simple range from 0 to the number of atoms in the AtomArray which
    is used as a convenience feature.

    Args:
        atom_array:
            AtomArray containing the structure to assign atom indices to.
    """
    atom_array.set_annotation("atom_idx", range(len(atom_array)))


def assign_entity_ids(atom_array: struc.AtomArray) -> None:
    """Assigns entity IDs to the AtomArray

    Entity IDs are assigned to each chain in the AtomArray based on the
    "label_entity_id" field. The entity ID is stored in the "entity_id" field of the
    AtomArray.

    Args:
        atom_array:
            AtomArray containing the structure to assign entity IDs to.
    """
    # Cast entity IDs from string to int and shorten name
    atom_array.set_annotation("entity_id", atom_array.label_entity_id.astype(int))
    atom_array.del_annotation("label_entity_id")


def assign_molecule_type_ids(atom_array: struc.AtomArray) -> None:
    """Assigns molecule types to the AtomArray

    Assigns molecule type IDs to each chain based on its residue names. Possible
    molecule types are protein, RNA, DNA, and ligand. The molecule type is stored in the
    "molecule_type_id" field of the AtomArray.

    Args:
        atom_array:
            AtomArray containing the structure to assign molecule types to.
    """
    chain_start_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # Create molecule type annotation
    molecule_type_ids = np.zeros(len(atom_array), dtype=int)

    # Zip together chain starts and ends
    for chain_start, chain_end in zip(chain_start_idxs[:-1], chain_start_idxs[1:]):
        residues_in_chain = set(atom_array[chain_start:chain_end].res_name)

        # Assign protein if any standard protein residue is present
        if residues_in_chain & set(STANDARD_PROTEIN_RESIDUES):
            molecule_type_ids[chain_start:chain_end] = MOLECULE_TYPE_ID_PROTEIN

        # Assign RNA if any standard RNA residue is present
        elif residues_in_chain & set(STANDARD_RNA_RESIDUES):
            molecule_type_ids[chain_start:chain_end] = MOLECULE_TYPE_ID_RNA

        # Assign DNA if any standard DNA residue is present
        elif residues_in_chain & set(STANDARD_DNA_RESIDUES):
            molecule_type_ids[chain_start:chain_end] = MOLECULE_TYPE_ID_DNA

        # Assign ligand otherwise
        else:
            molecule_type_ids[chain_start:chain_end] = MOLECULE_TYPE_ID_LIGAND

    atom_array.set_annotation("molecule_type_id", molecule_type_ids)


def get_interface_atoms(
    query_atom_array: struc.AtomArray,
    target_atom_array: struc.AtomArray,
    distance_threshold: float = 15.0,
) -> struc.AtomArray:
    """Returns interface atoms in the query based on the target

    This will find atoms in the query that are within a given distance threshold of any
    atom with a different chain in the target.

    Args:
        query_atom_array:
            AtomArray containing the structure to find interface atoms in.
        target_atom_array:
            AtomArray containing the structure to compare against.
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 15.0.

    Returns:
        AtomArray with interface atoms.
    """
    pairwise_dists = cdist(query_atom_array.coord, target_atom_array.coord)

    # All unique chains in the query
    query_chain_ids = set(query_atom_array.chain_id_renumbered)

    # Set masks to avoid intra-chain comparisons
    for chain in query_chain_ids:
        query_chain_mask = query_atom_array.chain_id_renumbered == chain
        target_chain_mask = target_atom_array.chain_id_renumbered == chain

        query_chain_block_mask = np.ix_(query_chain_mask, target_chain_mask)
        pairwise_dists[query_chain_block_mask] = np.inf

    interface_atom_mask = np.any(pairwise_dists < distance_threshold, axis=1)

    return query_atom_array[interface_atom_mask]


def get_interface_token_center_atoms(
    query_atom_array: struc.AtomArray,
    target_atom_array: struc.AtomArray,
    distance_threshold: float = 15.0,
) -> struc.AtomArray:
    """Gets interface token center atoms in the query based on the target

    This will find token center atoms in the query that are within a given distance
    threshold of any token center atom with a different chain in the target.

    For example used in 2.5.4 of the AlphaFold3 SI (subsetting of large bioassemblies)

    Args:
        query_atom_array:
            AtomArray containing the structure to find interface token center atoms in.
        target_atom_array:
            AtomArray containing the structure to compare against.
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 15.0.

    Returns:
        AtomArray with interface token center atoms.
    """
    if "af3_token_center_atom" not in query_atom_array.get_annotation_categories():
        raise ValueError(
            "Token center atoms not found in query atom array, run "
            "tokenize_atom_array first"
        )
    elif "af3_token_center_atom" not in target_atom_array.get_annotation_categories():
        raise ValueError(
            "Token center atoms not found in target atom array, run "
            "tokenize_atom_array first"
        )

    query_token_centers = query_atom_array[query_atom_array.af3_token_center_atom]
    target_token_centers = target_atom_array[target_atom_array.af3_token_center_atom]

    return get_interface_atoms(
        query_token_centers, target_token_centers, distance_threshold
    )


def connect_residues(
    atom_array: struc.AtomArray,
    res_id_1: int,
    res_id_2: int,
    chain: int,
    bond_type: Literal["peptide", "phosphate"],
) -> None:
    """Connects two residues in the AtomArray

    This function adds an appropriate bond between two residues in the AtomArray based
    on the canonical way of connecting residues in the given polymer type (peptide or
    nucleic acid).

    Args:
        atom_array:
            AtomArray containing the structure to add the bond to.
        res_id_1:
            Residue ID of the first (upstream) residue to connect. (e.g. will contribute
            the carboxylic carbon in a peptide bond)
        res_id_2:
            Residue ID of the second (downstream) residue to connect. (e.g. will
            contribute the amino nitrogen in a peptide bond)
    """
    # General masks
    chain_mask = atom_array.chain_id_renumbered == chain
    res_id_1_mask = (atom_array.res_id == res_id_1) & chain_mask
    res_id_2_mask = (atom_array.res_id == res_id_2) & chain_mask

    if bond_type == "peptide":
        # Get the index of the C atom in the first residue
        (res_id_1_atom_idx,) = np.where(res_id_1_mask & (atom_array.atom_name == "C"))[
            0
        ]

        # Equivalently get the index of the N atom in the second residue
        (res_id_2_atom_idx,) = np.where(res_id_2_mask & (atom_array.atom_name == "N"))[
            0
        ]

    # Analogously select nucleic acid atoms
    elif bond_type == "phosphate":
        (res_id_1_atom_idx,) = np.where(
            res_id_1_mask & (atom_array.atom_name == "O3'")
        )[0]

        (res_id_2_atom_idx,) = np.where(res_id_2_mask & (atom_array.atom_name == "P"))[
            0
        ]

    atom_array.bonds.add_bond(
        res_id_1_atom_idx, res_id_2_atom_idx, struc.BondType.SINGLE
    )


def build_unresolved_polymer_segment(
    residue_codes: list[str],
    ccd: CIFFile,
    polymer_type: Literal["protein", "nucleic_acid"],
    default_annotations: dict,
    terminal_start: bool = False,
    terminal_end: bool = False,
) -> struc.AtomArray:
    """Builds a polymeric segment with unresolved residues

    This function builds a polymeric segment with unresolved residues based on a list of
    residue 3-letter codes that have matching entries in the Chemical Component
    Dictionary (CCD). The segment is built by adding all atoms of the unresolved
    residues to the AtomArray with dummy coordinates. The BondList of the resulting
    AtomArray is filled appropriately to contain both intra-residue and inter-residue
    bonds.

    Args:
        residue_codes:
            List of 3-letter residue codes of the unresolved residues.
        ccd:
            Parsed Chemical Component Dictionary (CCD) containing the residue
            information.
        polymer_type:
            Type of the polymer segment. Can be either "protein" or "nucleic_acid".
        default_annotations:
            Any annotations and custom annotations to be added to the atoms in the
            output AtomArray. All annotations are kept constant for all atoms in the
            segment except for res_id and atom_idx which are incrementally increased
            appropriately (with the first atom/residue index matching the one in the
            default annotations), and occupancy which is set to 0.0.
        terminal_start:
            Whether the segment is the start of the overall sequence. For proteins this
            will have no effect but for nucleic acids the otherwise overhanging 5'
            phosphate is pruned.
        terminal_end:
            Whether the segment is the end of the overall sequence. For proteins this
            will keep the terminal oxygen, for nucleic acids the otherwise overhanging
            3' phosphate is pruned.
    """
    if polymer_type == "nucleic_acid":
        logging.info("Building unresolved nucleic acid segment!")  # dev-only: del later
        
    # Set occupancy to 0.0
    default_annotations["occupancy"] = 0.0

    atom_list = []

    last_residue_idx = len(residue_codes) - 1

    added_atoms = 0

    for added_residues, residue_code in enumerate(residue_codes):
        atom_names = ccd[residue_code]["chem_comp_atom"]["atom_id"].as_array()
        atom_elements = ccd[residue_code]["chem_comp_atom"]["type_symbol"].as_array()

        # Exclude hydrogens
        hydrogen_mask = atom_elements != "H"

        if polymer_type == "nucleic_acid":
            # Prune 5' phosphate if segment is nucleic acid and segment-start is overall
            # sequence start to prevent overhanging phosphates
            if terminal_start and added_residues == 0:
                phosphate_mask = ~np.isin(atom_names, NUCLEIC_ACID_PHOSPHATE_ATOMS)
                atom_mask = hydrogen_mask & phosphate_mask

            # Always prune terminal phosphate-oxygens because they would only need to be
            # kept in overhanging phosphates which we remove above (in-sequence
            # connecting phosphate-oxygens are annotated as O3' and kept)
            else:
                po3_mask = ~np.isin(atom_names, ["OP3", "O3P"])
                atom_mask = hydrogen_mask & po3_mask

        elif polymer_type == "protein":
            # If segment is protein and segment-end is overall sequence end, do not
            # exclude terminal oxygen
            if terminal_end and added_residues == last_residue_idx:
                atom_mask = hydrogen_mask
            # For any other protein residue, exclude terminal oxygen
            else:
                oxt_mask = atom_names != "OXT"
                atom_mask = hydrogen_mask & oxt_mask

        atom_names = atom_names[atom_mask]
        atom_elements = atom_elements[atom_mask]

        # Add atoms for all unresolved residues
        for atom, element in zip(atom_names, atom_elements):
            annotations = default_annotations.copy()
            annotations["atom_name"] = atom
            annotations["element"] = element
            annotations["res_name"] = residue_code

            base_res_id = default_annotations["res_id"]
            base_atom_idx = default_annotations["atom_idx"]
            annotations["res_id"] = base_res_id + added_residues
            annotations["atom_idx"] = base_atom_idx + added_atoms

            # Append unresolved atom explicitly but with dummy coordinates
            atom_list.append(struc.Atom([np.nan, np.nan, np.nan], **annotations))

            added_atoms += 1

    segment_atom_array = struc.array(atom_list)

    # build standard connectivities
    bond_list = struc.connect_via_residue_names(segment_atom_array)
    segment_atom_array.bonds = bond_list

    return segment_atom_array


def shift_up_atom_indices(
    atom_array: struc.AtomArray, shift: int, idx_threshold: int
) -> None:
    """Shifts all atom indices higher than a threshold by a certain amount

    Args:
        atom_array:
            AtomArray containing the structure to shift atom indices in.
        shift:
            Amount by which to shift the atom indices.
        idx_threshold:
            Threshold index above which to shift the atom indices.
    """
    # Update atom indices for all atoms greater than the given atom index
    update_mask = atom_array.atom_idx > idx_threshold
    atom_array.atom_idx[update_mask] += shift


# TODO: code is a bit redundant but not sure if bad
def add_unresolved_polymer_residues(
    atom_array: struc.AtomArray,
    cif_data: CIFBlock,
    ccd: CIFFile,
) -> struc.AtomArray:
    """Adds all missing polymer residues to the AtomArray

    Missing residues are added to the AtomArray explicitly with dummy NaN coordinates
    and the full atom annotations and bonding patterns. This is useful for contiguous
    cropping or inferring the whole sequence of a polymer chain.

    Args:
        atom_array:
            AtomArray containing the structure to add missing residues to.
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, e.g.
            `cif_data=cif_file["4H1W"]`.
        ccd:
            Parsed Chemical Component Dictionary (CCD) containing the residue
            information.
    """
    # Flat list of residue-wise entity IDs for all polymeric sequences
    entity_ids_flat = cif_data["entity_poly_seq"]["entity_id"].as_array(dtype=int)

    # Get sequence lengths for every polymeric entity
    entity_ids, new_entity_starts, seq_lengths = np.unique(
        entity_ids_flat, return_index=True, return_counts=True
    )
    entity_id_to_seqlength = dict(zip(entity_ids, seq_lengths))

    # Get full (3-letter code) residue sequence for every polymeric entity
    entity_monomers = cif_data["entity_poly_seq"]["mon_id"].as_array()
    entity_id_to_seqres = {
        entity_id: entity_monomers[start : start + length]
        for entity_id, start, length in zip(entity_ids, new_entity_starts, seq_lengths)
    }

    # This will be extended with unresolved residues
    extended_atom_array = atom_array.copy()

    # Iterate through all chains and fill missing residues. To not disrupt the bondlist
    # by slicing the atom_array for an insertion operation, this appends all the missing
    # residue atoms at the end of the atom_array but keeps appropriate bookkeeping of
    # the residue indices. That way the entire array can be sorted in a single
    # reindexing at the end without losing any bond information.
    chain_starts = struc.get_chain_starts(extended_atom_array, add_exclusive_stop=True)
    for chain_start, chain_end in zip(chain_starts[:-1], chain_starts[1:] - 1):
        # Infer some chain-wise properties from first atom (could use any atom)
        first_atom = extended_atom_array[chain_start]
        chain_type = first_atom.molecule_type_id
        chain_entity_id = first_atom.entity_id
        chain_id = first_atom.chain_id_renumbered

        # Only interested in polymer chains
        if chain_type == MOLECULE_TYPE_ID_LIGAND:
            continue
        else:
            if chain_type == MOLECULE_TYPE_ID_PROTEIN:
                polymer_type = "protein"
                bond_type = "peptide"
            elif chain_type in (MOLECULE_TYPE_ID_RNA, MOLECULE_TYPE_ID_DNA):
                polymer_type = "nucleic_acid"
                bond_type = "phosphate"
            else:
                raise ValueError(f"Unknown molecule type: {chain_type}")

        # Three-letter residue codes of the full chain
        chain_seqres = entity_id_to_seqres[chain_entity_id]

        # Atom annotations of first atom (as template for unresolved residue
        # annotations while updating res_id and atom_idx)
        template_annotations = first_atom._annot.copy()
        template_annotations["ins_code"] = ""

        # Fill missing residues at chain start
        if first_atom.res_id > 1:
            n_missing_residues = first_atom.res_id - 1

            default_annotations = template_annotations.copy()
            default_annotations["res_id"] = 1
            default_annotations["atom_idx"] = first_atom.atom_idx

            segment = build_unresolved_polymer_segment(
                residue_codes=chain_seqres[:n_missing_residues],
                ccd=ccd,
                polymer_type=polymer_type,
                default_annotations=default_annotations,
                terminal_start=True,
                terminal_end=False,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            shift_up_atom_indices(
                extended_atom_array,
                n_added_atoms,
                idx_threshold=first_atom.atom_idx - 1,
            )
            extended_atom_array += segment

            # Connect residues in BondList
            connect_residues(
                extended_atom_array,
                first_atom.res_id - 1,
                first_atom.res_id,
                chain=chain_id,
                bond_type=bond_type,
            )

        # Fill missing residues at chain end
        last_atom = extended_atom_array[chain_end]

        if last_atom.res_id < entity_id_to_seqlength[chain_entity_id]:
            n_missing_residues = (
                entity_id_to_seqlength[chain_entity_id] - last_atom.res_id
            )

            default_annotations = template_annotations.copy()
            default_annotations["res_id"] = last_atom.res_id + 1
            default_annotations["atom_idx"] = last_atom.atom_idx + 1

            segment = build_unresolved_polymer_segment(
                residue_codes=chain_seqres[-n_missing_residues:],
                ccd=ccd,
                polymer_type=polymer_type,
                default_annotations=default_annotations,
                terminal_start=False,
                terminal_end=True,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            shift_up_atom_indices(
                extended_atom_array, n_added_atoms, idx_threshold=last_atom.atom_idx
            )
            extended_atom_array += segment

            # Connect residues in BondList
            connect_residues(
                extended_atom_array,
                last_atom.res_id,
                last_atom.res_id + 1,
                chain=chain_id,
                bond_type=bond_type,
            )

        # Gaps between consecutive residues
        res_id_gaps = np.diff(extended_atom_array.res_id[chain_start : chain_end + 1])
        chain_break_start_idxs = np.where(res_id_gaps > 1)[0] + chain_start

        # Fill missing residues within the chain
        for chain_break_start_idx in chain_break_start_idxs:
            chain_break_end_idx = chain_break_start_idx + 1
            break_start_atom = extended_atom_array[chain_break_start_idx]
            break_end_atom = extended_atom_array[chain_break_end_idx]

            n_missing_residues = break_end_atom.res_id - break_start_atom.res_id - 1

            default_annotations = template_annotations.copy()
            default_annotations["res_id"] = break_start_atom.res_id + 1
            default_annotations["atom_idx"] = break_start_atom.atom_idx + 1

            # Residue IDs start with 1 so the indices are offset
            segment_residue_codes = chain_seqres[
                break_start_atom.res_id : break_end_atom.res_id - 1
            ]

            segment = build_unresolved_polymer_segment(
                residue_codes=segment_residue_codes,
                ccd=ccd,
                polymer_type=polymer_type,
                default_annotations=default_annotations,
                terminal_start=False,
                terminal_end=False,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            shift_up_atom_indices(
                extended_atom_array,
                n_added_atoms,
                idx_threshold=break_end_atom.atom_idx - 1,
            )
            extended_atom_array += segment

            # Connect residues at start and end of added segment
            connect_residues(
                extended_atom_array,
                break_start_atom.res_id,
                break_start_atom.res_id + 1,
                chain=chain_id,
                bond_type=bond_type,
            )
            connect_residues(
                extended_atom_array,
                break_end_atom.res_id - 1,
                break_end_atom.res_id,
                chain=chain_id,
                bond_type=bond_type,
            )

    # Finally reorder the array so that the atom indices are in order
    extended_atom_array = extended_atom_array[np.argsort(extended_atom_array.atom_idx)]

    # dev-only: TODO remove
    assert np.array_equal(
        extended_atom_array.atom_idx, np.arange(len(extended_atom_array))
    )

    return extended_atom_array
