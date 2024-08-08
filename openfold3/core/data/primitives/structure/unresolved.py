import logging
from typing import Literal

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFBlock, CIFFile

from openfold3.core.data.primitives.structure.labels import (
    assign_atom_indices,
    remove_atom_indices,
)
from openfold3.core.data.primitives.structure.metadata import (
    get_entity_to_three_letter_codes_dict,
)
from openfold3.core.data.resources.tables import (
    NUCLEIC_ACID_PHOSPHATE_ATOMS,
    MoleculeType,
)


def connect_residues(
    atom_array: AtomArray,
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
) -> AtomArray:
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
            segment except for res_id and _atom_idx (if present) which are incrementally
            increased appropriately (with the first atom/residue index matching the one
            in the default annotations), and occupancy which is set to 0.0.
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

    atom_idx_is_present = "_atom_idx" in default_annotations

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
            atom_annotations = default_annotations.copy()
            atom_annotations["atom_name"] = atom
            atom_annotations["element"] = element
            atom_annotations["res_name"] = residue_code

            base_res_id = default_annotations["res_id"]
            atom_annotations["res_id"] = base_res_id + added_residues

            # Avoid error if _atom_idx is not set
            if atom_idx_is_present:
                base_atom_idx = default_annotations["_atom_idx"]
                atom_annotations["_atom_idx"] = base_atom_idx + added_atoms

            # Append unresolved atom explicitly but with dummy coordinates
            atom_list.append(struc.Atom([np.nan, np.nan, np.nan], **atom_annotations))

            added_atoms += 1

    segment_atom_array = struc.array(atom_list)

    # build standard connectivities
    bond_list = struc.connect_via_residue_names(segment_atom_array)
    segment_atom_array.bonds = bond_list

    return segment_atom_array


def _shift_up_atom_indices(
    atom_array: AtomArray, shift: int, idx_threshold: int
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
    update_mask = atom_array._atom_idx > idx_threshold
    atom_array._atom_idx[update_mask] += shift


def add_unresolved_polymer_residues(
    atom_array: AtomArray,
    cif_data: CIFBlock,
    ccd: CIFFile,
) -> AtomArray:
    """Adds all missing polymer residues to the AtomArray

    Missing residues are added to the AtomArray explicitly with dummy NaN coordinates
    and the full atom annotations and bonding patterns. This is useful for contiguous
    cropping or inferring the whole sequence of a polymer chain.

    Args:
        atom_array:
            AtomArray containing the structure to add missing residues to.
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see
            `metadata_extraction.get_cif_block`)
        ccd:
            Parsed Chemical Component Dictionary (CCD) containing the residue
            information.
    """
    # Three-letter residue codes of all monomers in the entire sequence
    entity_id_to_3l_seq = get_entity_to_three_letter_codes_dict(cif_data)

    # This will be extended with unresolved residues
    extended_atom_array = atom_array.copy()

    # Assign temporary atom indices
    assign_atom_indices(extended_atom_array)

    # Iterate through all chains and fill missing residues. To not disrupt the bondlist
    # by slicing the atom_array for an insertion operation, this appends all the missing
    # residue atoms at the end of the atom_array but keeps appropriate bookkeeping of
    # the atom indices. That way the entire array can be sorted in a single
    # reindexing at the end without losing any bond information.
    chain_starts = struc.get_chain_starts(extended_atom_array, add_exclusive_stop=True)
    for chain_start, chain_end in zip(chain_starts[:-1], chain_starts[1:]):
        # Infer some chain-wise properties from first atom (could use any atom)
        first_atom = extended_atom_array[chain_start]
        chain_type = first_atom.molecule_type_id
        chain_entity_id = first_atom.entity_id
        chain_id = first_atom.chain_id_renumbered

        # Only interested in polymer chains
        if chain_type == MoleculeType.LIGAND:
            continue
        else:
            if chain_type == MoleculeType.PROTEIN:
                polymer_type = "protein"
                bond_type = "peptide"
            elif chain_type in (MoleculeType.RNA, MoleculeType.DNA):
                polymer_type = "nucleic_acid"
                bond_type = "phosphate"
            else:
                raise ValueError(f"Unknown molecule type: {chain_type}")

        # Three-letter residue codes of the full chain
        chain_3l_seq = entity_id_to_3l_seq[chain_entity_id]

        # Atom annotations of first atom (as template for unresolved residue
        # annotations while updating res_id and atom_idx)
        template_annotations = first_atom._annot.copy()
        template_annotations["ins_code"] = ""

        ## Fill missing residues at chain start
        if first_atom.res_id > 1:
            n_missing_residues = first_atom.res_id - 1

            default_annotations = template_annotations.copy()
            default_annotations["res_id"] = 1
            default_annotations["_atom_idx"] = first_atom._atom_idx

            segment = build_unresolved_polymer_segment(
                residue_codes=chain_3l_seq[:n_missing_residues],
                ccd=ccd,
                polymer_type=polymer_type,
                default_annotations=default_annotations,
                terminal_start=True,
                terminal_end=False,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            _shift_up_atom_indices(
                extended_atom_array,
                n_added_atoms,
                idx_threshold=first_atom._atom_idx - 1,
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

        ## Fill missing residues at chain end
        last_atom = extended_atom_array[chain_end]
        full_seq_length = len(entity_id_to_3l_seq[chain_entity_id])

        if last_atom.res_id < full_seq_length:
            n_missing_residues = full_seq_length - last_atom.res_id

            default_annotations = template_annotations.copy()
            default_annotations["res_id"] = last_atom.res_id + 1
            default_annotations["_atom_idx"] = last_atom._atom_idx + 1

            segment = build_unresolved_polymer_segment(
                residue_codes=chain_3l_seq[-n_missing_residues:],
                ccd=ccd,
                polymer_type=polymer_type,
                default_annotations=default_annotations,
                terminal_start=False,
                terminal_end=True,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            _shift_up_atom_indices(
                extended_atom_array, n_added_atoms, idx_threshold=last_atom._atom_idx
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

        ## Fill missing residues within the chain
        for chain_break_start_idx in chain_break_start_idxs:
            chain_break_end_idx = chain_break_start_idx + 1
            break_start_atom = extended_atom_array[chain_break_start_idx]
            break_end_atom = extended_atom_array[chain_break_end_idx]

            n_missing_residues = break_end_atom.res_id - break_start_atom.res_id - 1

            default_annotations = template_annotations.copy()
            default_annotations["res_id"] = break_start_atom.res_id + 1
            default_annotations["_atom_idx"] = break_start_atom._atom_idx + 1

            # Residue IDs start with 1 so the indices are offset
            segment_residue_codes = chain_3l_seq[
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
            _shift_up_atom_indices(
                extended_atom_array,
                n_added_atoms,
                idx_threshold=break_end_atom._atom_idx - 1,
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
    extended_atom_array = extended_atom_array[np.argsort(extended_atom_array._atom_idx)]

    # dev-only: TODO remove
    assert np.array_equal(
        extended_atom_array._atom_idx, np.arange(len(extended_atom_array))
    )

    # Remove temporary atom indices
    remove_atom_indices(extended_atom_array)

    return extended_atom_array
