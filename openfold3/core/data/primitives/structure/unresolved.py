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
    get_ccd_atom_id_to_element_dict,
    get_ccd_atom_pair_to_bond_dict,
    get_chain_to_three_letter_codes_dict,
    get_entity_to_three_letter_codes_dict,
)
from openfold3.core.data.resources.tables import (
    STANDARD_NUCLEIC_ACID_RESIDUES,
    STANDARD_PROTEIN_RESIDUES,
    MoleculeType,
)

logger = logging.getLogger(__name__)


# TODO: Check if this can result in hypervalence in some cases
def update_bond_list(atom_array: AtomArray) -> None:
    """Updates the bond list of the AtomArray in-place with any missing bonds.

    Runs biotite's `connect_via_residue_names` on the AtomArray and merges the result
    with the already existing bond list.

    Args:
        atom_array:
            AtomArray containing the structure to update the bond list for.
    """
    bond_list_update = struc.connect_via_residue_names(atom_array)
    atom_array.bonds = atom_array.bonds.merge(bond_list_update)


def build_unresolved_polymer_segment(
    residue_codes: list[str],
    ccd: CIFFile,
    polymer_type: Literal["protein", "nucleic_acid"],
    default_annotations: dict,
    terminal_start: bool = False,
    terminal_end: bool = False,
    add_bonds: bool = True,
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
        add_bonds:
            Whether to add bonds between the residues in the segment. Defaults to True.
    """
    if polymer_type == "nucleic_acid":
        logger.info("Building unresolved nucleic acid segment!")  # dev-only: del later

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
            if terminal_start and added_residues == 0:
                # Keep overhanging phosphate for sequence start
                atom_mask = hydrogen_mask
            else:
                # Prune terminal phosphate-oxygens (which are not part of the backbone)
                po3_mask = ~np.isin(atom_names, ["OP3", "O3P"])
                atom_mask = hydrogen_mask & po3_mask

        elif polymer_type == "protein":
            if terminal_end and added_residues == last_residue_idx:
                # Include OXT for terminal residue
                atom_mask = hydrogen_mask
            else:
                # For any other protein residue, exclude terminal oxygen
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
    if add_bonds:
        bond_list = struc.connect_via_residue_names(segment_atom_array)
        segment_atom_array.bonds = bond_list

    return segment_atom_array


def _shift_up_atom_indices(
    atom_array: AtomArray,
    shift: int,
    greater_than: int,
) -> None:
    """Shifts all atom indices higher than a threshold by a certain amount

    Atom indices are expected to be present in the "_atom_idx" annotation of the
    AtomArray. This function adds the `shift` to all atom indices greater than a given
    threshold.

    Args:
        atom_array:
            AtomArray containing the structure to shift atom indices in.
        shift:
            Amount by which to shift the atom indices.
        greater_than:
            Threshold index above which to shift the atom indices.
    """
    # Update atom indices for all atoms greater than the given atom index
    update_mask = atom_array._atom_idx > greater_than
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
    for chain_start, chain_end in zip(chain_starts[:-1], chain_starts[1:] - 1):
        # Infer some chain-wise properties from first atom (could use any atom)
        first_atom = extended_atom_array[chain_start]
        chain_type = first_atom.molecule_type_id
        chain_entity_id = first_atom.entity_id

        # Only interested in polymer chains
        if chain_type == MoleculeType.LIGAND:
            continue
        else:
            if chain_type == MoleculeType.PROTEIN:
                polymer_type = "protein"
                # bond_type = "peptide"
            elif chain_type in (MoleculeType.RNA, MoleculeType.DNA):
                polymer_type = "nucleic_acid"
                # bond_type = "phosphate"
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
                add_bonds=False,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            _shift_up_atom_indices(
                extended_atom_array,
                n_added_atoms,
                greater_than=first_atom._atom_idx - 1,
            )
            extended_atom_array += segment

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
                add_bonds=False,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            _shift_up_atom_indices(
                extended_atom_array, n_added_atoms, greater_than=last_atom._atom_idx
            )
            extended_atom_array += segment

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
                add_bonds=False,
            )

            n_added_atoms = len(segment)

            # Shift all atom indices up, then insert the unresolved segment
            _shift_up_atom_indices(
                extended_atom_array,
                n_added_atoms,
                greater_than=break_end_atom._atom_idx - 1,
            )
            extended_atom_array += segment

    # Finally reorder the array so that the atom indices are in order
    extended_atom_array = extended_atom_array[np.argsort(extended_atom_array._atom_idx)]

    # TODO: make numerical chain ID handling cleaner
    extended_atom_array._annot["chain_id"] = extended_atom_array.chain_id.astype(int)

    # dev-only: TODO remove
    assert np.array_equal(
        extended_atom_array._atom_idx, np.arange(len(extended_atom_array))
    )

    # Add bonds between and within all the added residues
    update_bond_list(extended_atom_array)

    # Remove temporary atom indices
    remove_atom_indices(extended_atom_array)

    return extended_atom_array


# TODO: add docstring
def add_unresolved_atoms_within_residue(
    atom_array: AtomArray, cif_data: CIFBlock, ccd: CIFFile
) -> AtomArray:
    # Get theoretical lengths of each chain (needed to identify true terminal residues)
    chain_to_monomers = get_chain_to_three_letter_codes_dict(atom_array, cif_data)
    chain_to_seqlen = {chain: len(seq) for chain, seq in chain_to_monomers.items()}

    # Full atom array that will hold the new atoms
    extended_atom_array = atom_array.copy()

    # We need atom indices for bookkeeping of where to insert the missing atoms
    assign_atom_indices(extended_atom_array)

    std_protein_residues = set(STANDARD_PROTEIN_RESIDUES)
    std_na_residues = set(STANDARD_NUCLEIC_ACID_RESIDUES)

    missing_atom_list = []

    for chain in struc.chain_iter(extended_atom_array):
        chain_id = chain.chain_id[0]

        (chain_molecule_type_id,) = np.unique(chain.molecule_type_id)
        chain_molecule_type = MoleculeType(chain_molecule_type_id)

        is_ligand = chain_molecule_type == MoleculeType.LIGAND
        is_protein = chain_molecule_type == MoleculeType.PROTEIN
        is_nucleic_acid = chain_molecule_type in (MoleculeType.RNA, MoleculeType.DNA)

        # Skip covalently connected ligand chains
        if (struc.get_residue_count(chain) > 1) & is_ligand:
            continue

        # Find unresolved atoms for all residues in each chain
        for residue in struc.residue_iter(chain):
            res_name = residue.res_name[0]

            # Atoms in the structure
            resolved_atom_set = set(residue.atom_name.tolist())

            # Atoms that should be present according to the CCD
            all_atoms = ccd[res_name]["chem_comp_atom"]["atom_id"].as_array()
            all_atom_elements = ccd[res_name]["chem_comp_atom"][
                "type_symbol"
            ].as_array()
            all_atoms_no_h = all_atoms[all_atom_elements != "H"].tolist()

            if is_protein or is_nucleic_acid:
                res_id = residue.res_id[0]
                is_first_residue = res_id == 1
                is_last_residue = res_id == chain_to_seqlen[chain_id]

                # Handle terminal atom inclusion or exclusion
                if is_protein and not is_last_residue:
                    if res_name not in std_protein_residues:
                        logger.debug(
                            "Adding unresolved atoms within protein chain for non-"
                            f"standard protein residue: {res_name}"
                        )
                    # If amino acid is mid-sequence, skip terminal oxygen
                    if "OXT" in all_atoms_no_h:
                        all_atoms_no_h.remove("OXT")

                # Find the "leaving oxygen" for the phosphate group which is displaced
                # during phosphodiester bond formation and should not be considered a
                # missing atom
                elif is_nucleic_acid and not is_first_residue:
                    if res_name not in std_na_residues:
                        logger.debug(
                            "Adding unresolved atoms within NA chain for non-standard "
                            f"NA residue: {res_name}"
                        )

                    # Usually, the "leaving oxygen" for the phosphate group is OP3/O3P,
                    # but in some cases it can be another oxygen which we need to
                    # identify
                    op3_present = "OP3" in resolved_atom_set
                    o3p_present = "O3P" in resolved_atom_set
                    if op3_present or o3p_present:
                        atom_pairs_to_bonds = get_ccd_atom_pair_to_bond_dict(
                            ccd[res_name]
                        )
                        atom_ids_to_elements = get_ccd_atom_id_to_element_dict(
                            ccd[res_name]
                        )

                        p_bonded_oxygens = set()
                        elsewhere_bonded_oxygens = set()

                        for bond_pair in atom_pairs_to_bonds:
                            atom_1 = bond_pair[0]
                            atom_2 = bond_pair[1]

                            if atom_ids_to_elements[atom_1] == "O":
                                # The "leaving oxygen" should be absent from the
                                # structure
                                if atom_1 in resolved_atom_set:
                                    continue

                                if atom_2 == "P":
                                    p_bonded_oxygens.add(atom_1)
                                elif atom_ids_to_elements[atom_2] not in ("H", "D"):
                                    elsewhere_bonded_oxygens.add(atom_1)

                            if atom_ids_to_elements[atom_2] == "O":
                                # The "leaving oxygen" should be absent from the
                                # structure
                                if atom_2 in resolved_atom_set:
                                    continue

                                if atom_1 == "P":
                                    p_bonded_oxygens.add(atom_2)
                                elif atom_ids_to_elements[atom_1] not in ("H", "D"):
                                    elsewhere_bonded_oxygens.add(atom_2)

                        # Oxygens connected to the canonical phosphate "P" but no other
                        # atom qualify as the leaving atom (logic is not perfect but
                        # should cover the majority of cases), so pick arbitrary one
                        leaving_oxygen = next(
                            iter(p_bonded_oxygens - elsewhere_bonded_oxygens)
                        )
                    else:
                        # For normal nucleic acids, the leaving oxygen is always OP3
                        leaving_oxygen = "OP3"

                    # Remove the leaving oxygen from the list of unresolved atoms to not
                    # add it back in
                    if leaving_oxygen in all_atoms_no_h:
                        all_atoms_no_h.remove(leaving_oxygen)

            assert resolved_atom_set.issubset(all_atoms_no_h)  # TODO: remove

            unresolved_atom_set = set(all_atoms_no_h) - resolved_atom_set

            # Skip if all atoms are resolved
            if len(unresolved_atom_set) == 0:
                continue
            else:
                logger.debug(
                    "Adding unresolved atoms %s to residue: (%s, %s, %s, %s)",
                    unresolved_atom_set,
                    residue.res_name[0],
                    residue.res_id[0],
                    residue.chain_id[0],
                    residue.auth_seq_id[0],
                )

            # Push up the atom indices of the subsequent atoms to account for the full
            # residue length
            n_missing_atoms = len(unresolved_atom_set)
            _shift_up_atom_indices(
                extended_atom_array,
                n_missing_atoms,
                greater_than=residue._atom_idx[-1],
            )

            # Rewrite atom indices and add missing atoms to end of atom list
            residue_atom_selection_iter = iter(range(len(residue)))
            residue_first_atom_idx = residue._atom_idx[0]

            for atom_idx, atom_name in enumerate(
                all_atoms_no_h, start=residue_first_atom_idx
            ):
                if atom_name in resolved_atom_set:
                    residue._atom_idx[next(residue_atom_selection_iter)] = atom_idx
                else:
                    atom_annotations = residue[0]._annot.copy()
                    atom_annotations["atom_name"] = atom_name
                    atom_annotations["_atom_idx"] = atom_idx
                    atom_annotations["occupancy"] = 0.0

                    # Add missing atom with dummy coordinates
                    missing_atom_list.append(
                        struc.Atom([np.nan, np.nan, np.nan], **atom_annotations)
                    )

    if len(missing_atom_list) == 0:
        return extended_atom_array

    # Add atoms to end of the atom array
    missing_atom_array = struc.array(missing_atom_list)
    extended_atom_array += missing_atom_array

    # Reorder appropriately
    extended_atom_array = extended_atom_array[np.argsort(extended_atom_array._atom_idx)]

    # TODO: make numerical chain ID handling cleaner
    extended_atom_array._annot["chain_id"] = extended_atom_array.chain_id.astype(int)

    # Remove temporary atom indices
    remove_atom_indices(extended_atom_array)

    # Add bonds within all the added atoms
    update_bond_list(extended_atom_array)

    logger.debug("Added unresolved atoms within residues.")  # TODO: remove

    return extended_atom_array


# TODO: add docstring
def add_unresolved_atoms(atom_array: AtomArray, cif_data: CIFBlock, ccd: CIFFile):
    # Add missing atoms within residues
    extended_atom_array = add_unresolved_atoms_within_residue(atom_array, cif_data, ccd)

    # Add missing residues
    extended_atom_array = add_unresolved_polymer_residues(
        extended_atom_array, cif_data, ccd
    )

    return extended_atom_array
