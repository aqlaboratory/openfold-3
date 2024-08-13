import logging

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile
from scipy.spatial.distance import cdist

from openfold3.core.data.primitives.structure.interface import (
    chain_paired_interface_atom_iter,
    get_interface_token_center_atoms,
)
from openfold3.core.data.primitives.structure.labels import (
    assign_atom_indices,
    remove_atom_indices,
)

logger = logging.getLogger(__name__)


def convert_MSE_to_MET(atom_array: AtomArray) -> None:
    """Converts selenomethionine (MSE) residues to methionine (MET) in-place

    Will change the residue names and convert selenium atoms to sulfur atoms in the
    input AtomArray in-place, following 2.1 of the AlphaFold3 SI.

    Args:
        atom_array: AtomArray containing the structure to convert.
    """
    mse_residues = atom_array.res_name == "MSE"

    # Replace MSE with MET
    atom_array.res_name[mse_residues] = "MET"

    # Change selenium to sulfur
    mse_selenium_atoms = mse_residues & (atom_array.element == "SE")
    atom_array.element[mse_selenium_atoms] = "S"
    atom_array.atom_name[mse_selenium_atoms] = "SD"


def fix_single_arginine_naming(arg_atom_array: AtomArray) -> None:
    """Resolves naming ambiguities for a single arginine residue

    This ensures that NH1 is always closer to CD than NH2, following 2.1 of the
    AlphaFold3 SI.

    Args:
        arg_atom_array: AtomArray containing the arginine residue to fix.
    """
    nh1 = arg_atom_array[arg_atom_array.atom_name == "NH1"]
    nh2 = arg_atom_array[arg_atom_array.atom_name == "NH2"]
    cd = arg_atom_array[arg_atom_array.atom_name == "CD"]

    # If NH2 is closer to CD than NH1, swap the names
    if struc.distance(nh2, cd) < struc.distance(nh1, cd):
        nh1.atom_name = ["NH2"]
        nh2.atom_name = ["NH1"]


def fix_arginine_naming(atom_array: AtomArray) -> None:
    """Resolves naming ambiguities for all arginine residues in the AtomArray

    (see fix_single_arginine_naming for more details)

    Args:
        atom_array: AtomArray containing the structure to fix arginine residues in.
    """
    arginines = atom_array.res_name == "ARG"

    for arginine in struc.residue_iter(atom_array[arginines]):
        fix_single_arginine_naming(arginine)


def remove_waters(atom_array: AtomArray) -> AtomArray:
    """Removes water molecules from the AtomArray

    Returns a new AtomArray with all water (or heavy water) molecules removed.

    Args:
        atom_array: AtomArray containing the structure to remove waters from.

    Returns:
        AtomArray with all water molecules removed.
    """
    water_residues = (atom_array.res_name == "HOH") | (atom_array.res_name == "DOD")
    atom_array = atom_array[~water_residues]

    return atom_array


def remove_crystallization_aids(
    atom_array: AtomArray, ccd_codes=CRYSTALLIZATION_AIDS
) -> AtomArray:
    """Removes crystallization aids from the AtomArray

    Will remove all ligands that are classified as crystallization aids following 2.5.4
    of the AlphaFold3 SI.

    Args:
        atom_array:
            AtomArray containing the structure to remove crystallization aids from.
        ccd_codes:
            List of 3-letter codes for crystallization aids to remove.

    Returns:
        AtomArray with crystallization aids removed.
    """
    crystallization_aids = np.isin(atom_array.res_name, ccd_codes)
    atom_array = atom_array[~crystallization_aids]

    return atom_array


def remove_hydrogens(atom_array: AtomArray) -> AtomArray:
    """Removes all hydrogen atoms from the AtomArray

    Args:
        atom_array: AtomArray containing the structure to remove hydrogens from.

    Returns:
        AtomArray with all hydrogen atoms removed.
    """
    atom_array = atom_array[~np.isin(atom_array.element, ("H", "D"))]

    return atom_array


def remove_small_polymers(atom_array: AtomArray, max_residues: int = 3) -> AtomArray:
    """Removes small polymer chains from the AtomArray

    Follows 2.5.4 of the AlphaFold3 SI and removes all polymer chains with up to
    max_residues residues. We consider proteins and nucleic acids as polymers.

    Args:
        atom_array:
            AtomArray containing the structure to remove small polymers from.
        max_residues:
            Maximum number of residues for a polymer chain to be considered as small.

    Returns:
        AtomArray with all polymer chains with fewer than min_residues residues removed.
    """
    # Get polymers of all sizes
    all_nucleotides = struc.filter_polymer(
        atom_array, pol_type="nucleotide", min_size=2
    )
    all_proteins = struc.filter_polymer(atom_array, pol_type="peptide", min_size=2)
    all_polymers = all_nucleotides | all_proteins

    # Get polymers that are not small
    not_small_nucleotides = struc.filter_polymer(
        atom_array, pol_type="nucleotide", min_size=max_residues + 1
    )
    not_small_proteins = struc.filter_polymer(
        atom_array, pol_type="peptide", min_size=max_residues + 1
    )
    not_small_polymers = not_small_nucleotides | not_small_proteins

    # Get small polymers by subtracting the not small polymers from all polymers
    small_polymers = all_polymers & ~not_small_polymers

    # Remove small polymers
    small_polymer_chains = np.unique(atom_array.chain_id_renumbered[small_polymers])

    for chain_id in small_polymer_chains:
        breakpoint()
        atom_array = remove_chain_and_attached_ligands(atom_array, chain_id)

    return atom_array


def remove_fully_unknown_polymers(atom_array: AtomArray) -> AtomArray:
    """Removes polymer chains with all unknown residues from the AtomArray

    Follows 2.5.4 of the AlphaFold3 SI.

    Args:
        atom_array: AtomArray containing the structure to remove unknown polymers from.

    Returns:
        AtomArray with all polymer chains containing only unknown residues removed.
    """
    # Masks for standard polymers
    proteins = struc.filter_polymer(atom_array, pol_type="peptide")
    nucleic_acids = struc.filter_polymer(atom_array, pol_type="nucleotide")

    # Combine to get mask for all polymers
    polymers = proteins | nucleic_acids

    atom_array_filtered = atom_array

    for chain in struc.chain_iter(atom_array[polymers]):
        # Explicit single-element unpacking (will fail if >1 chain_id)
        (chain_id,) = np.unique(chain.chain_id_renumbered)

        # Remove the chain from the AtomArray if all residues are unknown
        if np.all(chain.res_name == "UNK"):
            breakpoint()
            atom_array_filtered = remove_chain_and_attached_ligands(
                atom_array_filtered, chain_id
            )

    return atom_array_filtered


def remove_chain_and_attached_ligands(
    atom_array: AtomArray, chain_id: int
) -> AtomArray:
    """Removes a chain from an AtomArray including all attached covalent ligands

    While not explicitly stated in the AlphaFold3 SI, the intent of this function is to
    remove protein or nucleic acid chains together with their covalent modifications, so
    that no "floating" covalent ligands are left behind. This doesn't apply the other
    way around, so if the input chain itself is a hetero-chain (e.g. a glycan), only the
    input chain is removed.

    Args:
        atom_array:
            AtomArray containing the structure to remove the chain from.
        chain_id:
            chain ID of the chain to remove (uses the "chain_id_renumbered" field)

    Returns:
        AtomArray with the specified chain and all attached covalent ligands removed
    """
    # Assign temporary helper indices
    assign_atom_indices(atom_array)

    chain_mask = atom_array.chain_id_renumbered == chain_id

    # If the chain itself is a hetero-chain, only remove the chain
    if np.all(atom_array.hetero[chain_mask]):
        return atom_array[~chain_mask]

    # Remove everything but the particular chain and all ligands, so that when we search
    # for connected atoms to the chain we will only find the directly connected ligands
    # and not e.g. a covalent ligand in another chain that has a disulfide bond to the
    # specified chain and is therefore indirectly "connected".
    atom_array_subset = atom_array[chain_mask | atom_array.hetero]
    chain_mask_subset = atom_array_subset.chain_id_renumbered == chain_id

    # Get first atom of the chain in the subset array
    first_chain_atom_idx = np.nonzero(chain_mask_subset)[0][0]

    # Identify bonded ligands as connected atoms with hetero flag
    connected_atoms_mask_subset = np.asarray(
        struc.find_connected(
            atom_array_subset.bonds, first_chain_atom_idx, as_mask=True
        ),
        dtype=bool,
    )
    connected_ligand_atoms_mask_subset = (
        connected_atoms_mask_subset & atom_array_subset.hetero
    )

    # Mark the original chain and all connected ligands for deletion
    atom_deletion_mask = chain_mask_subset | connected_ligand_atoms_mask_subset

    # Determine indices of atoms to keep in original array
    atom_deletion_indices = atom_array_subset._atom_idx[atom_deletion_mask]
    atom_retained_indices = np.setdiff1d(
        atom_array._atom_idx, atom_deletion_indices, assume_unique=True
    )

    if np.any(connected_ligand_atoms_mask_subset):
        breakpoint()

    # Remove chain and connected ligands
    atom_array = atom_array[atom_retained_indices]

    # Clean up temporary indices
    remove_atom_indices(atom_array)

    return atom_array


def remove_clashing_chains(
    atom_array: AtomArray,
    clash_distance: float = 1.7,
    clash_percentage: float = 0.3,
) -> AtomArray:
    """Removes chains with a high fraction of clashes

    This follows 2.5.4 of the AlphaFold3 SI. Pairs of chains are considered to clash if
    either one of them has more than the clash_percentage of atoms within the
    clash_distance of the other chain. The chain with the higher fraction of clashing
    atoms is then removed.

    Args:
        atom_array:
            AtomArray containing the structure to remove clashing chains from.
        clash_distance:
            Distance threshold for clashes in Angstrom. Defaults to 1.7.
        clash_percentage:
            Fraction of clashing atoms in a chain for it to be considered as clashing.
            Defaults to 0.3.

    Returns:
        AtomArray with clashing chains removed.
    """
    # Get atom counts of each chain in the total atom array (index of the resulting
    # array corresponds to chain_id, value to atom count)
    chain_atom_counts = np.bincount(atom_array.chain_id_renumbered)

    ## Get the clashing chains to remove
    chain_ids_to_remove = set()

    # Get all the atom pairs corresponding to clashes and their chain IDs
    for pair_chain_ids, pair_atom_idxs in chain_paired_interface_atom_iter(
        atom_array, distance_threshold=clash_distance, ignore_covalent=True
    ):
        chain1_id, chain2_id = pair_chain_ids

        chain1_clashing_atom_idxs = pair_atom_idxs[:, 0]
        chain2_clashing_atom_idxs = pair_atom_idxs[:, 1]

        # Calculate numbers of unique clashing atoms
        chain1_n_clashing_atoms = np.unique(chain1_clashing_atom_idxs).size
        chain2_n_clashing_atoms = np.unique(chain2_clashing_atom_idxs).size

        # Get fractions of clashing atoms respective to each chain's total atom count
        chain1_n_atoms = chain_atom_counts[chain1_id]
        chain2_n_atoms = chain_atom_counts[chain2_id]
        chain1_clash_fraction = chain1_n_clashing_atoms / chain1_n_atoms
        chain2_clash_fraction = chain2_n_clashing_atoms / chain2_n_atoms

        # If clash, remove chain with higher fraction of clashing atoms
        if (
            chain1_clash_fraction > clash_percentage
            or chain2_clash_fraction > clash_percentage
        ):
            if chain1_clash_fraction > chain2_clash_fraction:
                chain_ids_to_remove.add(chain1_id)
            elif chain2_clash_fraction > chain1_clash_fraction:
                chain_ids_to_remove.add(chain2_id)
            # If fractions are tied break tie by chain ID
            else:
                if chain1_id > chain2_id:
                    chain_ids_to_remove.add(chain1_id)
                else:
                    chain_ids_to_remove.add(chain2_id)

    for chain_id in chain_ids_to_remove:
        breakpoint()
        atom_array = remove_chain_and_attached_ligands(atom_array, chain_id)

    return atom_array


def get_res_atoms_in_ccd_mask(res_atom_array: AtomArray, ccd: CIFFile) -> np.ndarray:
    """Returns a mask for atoms in a residue that are present in the CCD

    Args:
        res_atom_array:
            AtomArray containing the atoms of a single residue
        ccd:
            CIFFile containing the parsed Chemical Component Dictionary (components.cif)

    Returns:
        Mask for atoms in the residue that are present in the CCD
    """
    res_name = res_atom_array.res_name[0]

    # This special unknown token doesn't have chem_comp_atom information. Therefore this
    # function returns an all-False mask which will effectively filter out the unknown
    # ligand.
    if res_name == "UNL":
        return np.zeros(res_atom_array.array_length(), dtype=bool)

    allowed_atoms = ccd[res_name]["chem_comp_atom"]["atom_id"].as_array()

    mask = np.isin(res_atom_array.atom_name, allowed_atoms)
    return mask


def remove_non_CCD_atoms(atom_array: AtomArray, ccd: CIFFile) -> AtomArray:
    """Removes atoms that are not present in the CCD residue definition

    Follows 2.5.4 of the AlphaFold3 SI and removes all atoms that do not appear in the
    residue's definition in the Chemical Component Dictionary (CCD).

    Args:
        atom_array:
            AtomArray containing the structure to remove non-CCD atoms from
        ccd:
            CIFFile containing the parsed CCD (components.cif)

    Returns:
        AtomArray with all atoms not present in the CCD removed
    """
    atom_masks_per_res = [
        get_res_atoms_in_ccd_mask(res_atom_array, ccd)
        for res_atom_array in struc.residue_iter(atom_array)
    ]

    # Inclusion mask over all atoms
    atom_mask = np.concatenate(atom_masks_per_res)

    if not atom_mask.all():
        breakpoint()

    return atom_array[atom_mask]


def remove_chains_with_CA_gaps(
    atom_array: AtomArray, distance_threshold: float = 10.0
) -> AtomArray:
    """Removes protein chains where consecutive C-alpha atoms are too far apart

    This follows 2.5.4 of the AlphaFold3 SI and removes protein chains where the
    distance between consecutive C-alpha atoms is larger than a given threshold.

    Args:
        atom_array:
            AtomArray containing the structure to remove chains with CA gaps from
        distance_threshold:
            Distance threshold in Angstrom. Defaults to 10.0.
    """
    protein_chain_ca = atom_array[
        struc.filter_polymer(atom_array, pol_type="peptide")
        & (atom_array.atom_name == "CA")
    ]

    # If there are no protein chains, return the input atom array
    if len(protein_chain_ca) == 0:
        return atom_array

    # Match C-alpha atoms with their next C-alpha atom
    ca_without_last = protein_chain_ca[:-1]
    ca_shifted_left = struc.array(np.roll(protein_chain_ca, -1, axis=0)[:-1])

    # Distances of every C-alpha atom to the next C-alpha atom
    ca_dists = struc.distance(ca_without_last, ca_shifted_left)

    # Create gap mask for atoms that are directly before a new chain start or chain
    # break
    chain_ends = struc.get_chain_starts(protein_chain_ca)[1:] - 1
    discontinuities = struc.check_res_id_continuity(protein_chain_ca) - 1
    gap_mask = np.union1d(chain_ends, discontinuities)

    # Mask out distances at gaps as they are non-consecutive
    ca_dists[gap_mask] = np.nan

    # Find chains where the distance between consecutive C-alpha atoms is too large
    chain_ids_to_remove = np.unique(
        protein_chain_ca[:-1][ca_dists > distance_threshold].chain_id_renumbered
    )

    for chain_id in chain_ids_to_remove:
        breakpoint()
        atom_array = remove_chain_and_attached_ligands(atom_array, chain_id)

    return atom_array


def subset_large_structure(
    atom_array: AtomArray,
    n_chains: int = 20,
    interface_distance_threshold: float = 15.0,
) -> AtomArray:
    """Subsets structures with too many chains to n chains

    Follows 2.5.4 of the AlphaFold3 SI. Will select a random interface token center atom
    and return the closest N chains based on minimum distances between any token center
    atoms.

    Requires the 'af3_token_center_atom' annotation created by the tokenizer function.

    Args:
        atom_array:
            AtomArray containing the structure to subset
        n_chains:
            Number of chains to keep in the subset
        interface_distance_threshold:
            Distance threshold in Å that an interface token center atom must have to any
            token center atom in another chain to be considered an interface token
            center atom

    Returns:
        AtomArray with the closest n_chains based on token center atom distances
    """
    # Select random interface token center atom
    interface_token_center_atoms = get_interface_token_center_atoms(
        atom_array, distance_threshold=interface_distance_threshold
    )
    selected_atom = np.random.choice(interface_token_center_atoms)

    # Get distances of atom to all token center atoms
    all_token_center_atoms = atom_array[atom_array.af3_token_center_atom]
    dists_to_all_token_centers = cdist(
        selected_atom.coord.reshape(1, 3),
        all_token_center_atoms.coord,
    )[0]

    # Sort (atom-wise) chain IDs by distance
    sort_by_dist_idx = np.argsort(dists_to_all_token_centers)
    chain_ids_sorted = all_token_center_atoms.chain_id_renumbered[sort_by_dist_idx]

    # Get unique chain IDs sorted by distance to selected atom
    unique_chain_idxs_sorted = np.sort(
        np.unique(chain_ids_sorted, return_index=True)[1]
    )

    # Select the closest n chains
    closest_n_chain_ids_idxs = unique_chain_idxs_sorted[:n_chains]
    closest_n_chain_ids = chain_ids_sorted[closest_n_chain_ids_idxs]

    # Subset atom array to the closest n chains
    selected_chain_mask = np.isin(atom_array.chain_id_renumbered, closest_n_chain_ids)

    breakpoint()
    return atom_array[selected_chain_mask]
