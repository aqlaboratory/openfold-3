import logging
from functools import wraps

import biotite.structure as struc
import numpy as np
from biotite.structure import (
    AtomArray,
    BondList,
    BondType,
    index_distance,
)
from biotite.structure.io.pdbx import CIFFile
from scipy.spatial.distance import cdist

from openfold3.core.data.primitives.structure.component import find_cross_chain_bonds
from openfold3.core.data.primitives.structure.interface import (
    chain_paired_interface_atom_iter,
    get_interface_token_center_atoms,
    get_query_interface_atom_pair_idxs,
)
from openfold3.core.data.primitives.structure.labels import (
    AtomArrayView,
    assign_atom_indices,
    get_bond_atom_tuples,
    get_differing_chain_ids,
    get_residue_tuples,
    remove_atom_indices,
    residue_view_iter,
)
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array
from openfold3.core.data.resources.lists import (
    CRYSTALLIZATION_AIDS,
)
from openfold3.core.data.resources.patches import construct_atom_array
from openfold3.core.data.resources.residues import (
    STANDARD_NUCLEIC_ACID_RESIDUES,
    STANDARD_PROTEIN_RESIDUES_3,
    MoleculeType,
)

logger = logging.getLogger(__name__)


def return_on_empty_atom_array(func):
    """Decorator to make the cleanup functions immediately return on empty inputs."""

    @wraps(func)
    def wrapper(atom_array: AtomArray, *args, **kwargs):
        if atom_array.array_length() == 0:
            return atom_array

        return func(atom_array, *args, **kwargs)

    return wrapper


@return_on_empty_atom_array
def convert_MSE_to_MET(atom_array: AtomArray) -> None:
    """Converts selenomethionine (MSE) residues to methionine (MET) in-place

    Will change the residue names and convert selenium atoms to sulfur atoms in the
    input AtomArray in-place, following 2.1 of the AlphaFold3 SI.

    Args:
        atom_array: AtomArray containing the structure to convert.
    """
    mse_residues = atom_array.res_name == "MSE"

    # If there are no MSE residues, return
    if not np.any(mse_residues):
        return None

    # Replace MSE with MET
    atom_array.res_name[mse_residues] = "MET"

    # Change selenium to sulfur
    mse_selenium_atoms = mse_residues & (atom_array.element == "SE")
    atom_array.element[mse_selenium_atoms] = "S"
    atom_array.atom_name[mse_selenium_atoms] = "SD"

    # Set hetero to False for new MET residues
    atom_array.hetero[mse_residues] = False

    # Log modified residues
    former_mse_res_tuples = get_residue_tuples(atom_array[mse_residues])

    logger.info(
        f"Changed {len(former_mse_res_tuples)} MSE residues to MET: "
        f"{former_mse_res_tuples}"
    )


@return_on_empty_atom_array
def fix_arginine_naming(atom_array: AtomArray) -> AtomArray:
    """Resolves naming ambiguities for all arginine residues in the AtomArray

    Ensures that NH1 is always closer to CD than NH2, following 2.1 of the
    AlphaFold3 SI.

    Args:
        atom_array: AtomArray containing the structure to fix arginine residues in.

    Returns:
        AtomArray with arginine residue names fixed.
    """
    assign_atom_indices(atom_array, label="_atom_idx_arginine_fix")

    # Will build up the correct order of atom names in the final array
    name_sort_idx = atom_array._atom_idx_arginine_fix.copy()

    arginines = atom_array.res_name == "ARG"

    # Keeps track of changed residues for logging purposes
    changed_residue_tuples = []

    for arginine_view in residue_view_iter(atom_array[arginines]):
        nh1_mask = arginine_view.atom_name == "NH1"
        nh2_mask = arginine_view.atom_name == "NH2"
        cd_mask = arginine_view.atom_name == "CD"

        nh1_coord = arginine_view.coord[nh1_mask]
        nh2_coord = arginine_view.coord[nh2_mask]
        cd_coord = arginine_view.coord[cd_mask]

        # If NH2 is closer to CD than NH1, swap the names
        if struc.distance(nh2_coord, cd_coord) < struc.distance(nh1_coord, cd_coord):
            nh1_idx = arginine_view._atom_idx_arginine_fix[nh1_mask]
            nh2_idx = arginine_view._atom_idx_arginine_fix[nh2_mask]

            name_sort_idx[nh1_idx] = nh2_idx
            name_sort_idx[nh2_idx] = nh1_idx

            # Record that residue was changed
            changed_residue_tuples.append(
                (arginine_view.chain_id[0], arginine_view.res_id[0])
            )

    # Apply the sorting index to the final atom names
    atom_array.atom_name = atom_array.atom_name[name_sort_idx]

    if len(changed_residue_tuples) > 0:
        logger.info(
            f"Fixed NH1/NH2 naming for {len(changed_residue_tuples)} arginines: "
            f"{changed_residue_tuples}."
        )

    atom_array.del_annotation("_atom_idx_arginine_fix")

    return atom_array


@return_on_empty_atom_array
def remove_waters(atom_array: AtomArray) -> AtomArray:
    """Removes water molecules from the AtomArray

    Returns a new AtomArray with all water (or heavy water) molecules removed.

    Args:
        atom_array: AtomArray containing the structure to remove waters from.

    Returns:
        AtomArray with all water molecules removed.
    """
    water_residues = (atom_array.res_name == "HOH") | (atom_array.res_name == "DOD")
    n_water_residues = struc.get_residue_count(atom_array[water_residues])

    # Remove water residues
    atom_array = atom_array[~water_residues]

    if n_water_residues > 0:
        logger.info(f"Removed {n_water_residues} water molecules.")

    return atom_array


@return_on_empty_atom_array
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
            List of 3-letter codes of crystallization aids to remove (e.g. "SO4").

    Returns:
        AtomArray with crystallization aids removed.
    """
    crystallization_aids = np.isin(atom_array.res_name, ccd_codes)

    # Log which crystallization aids are being removed
    crystallization_aid_residue_tuples = get_residue_tuples(
        atom_array[crystallization_aids], include_resname=True
    )

    atom_array = atom_array[~crystallization_aids]

    if len(crystallization_aid_residue_tuples) > 0:
        logger.info(
            f"Removed {len(crystallization_aid_residue_tuples)} crystallization aids: "
            f"{crystallization_aid_residue_tuples}"
        )

    return atom_array


@return_on_empty_atom_array
def remove_hydrogens(atom_array: AtomArray) -> AtomArray:
    """Removes all hydrogen (and deuterium) atoms from the AtomArray

    Args:
        atom_array: AtomArray containing the structure to remove hydrogens from.

    Returns:
        AtomArray with all hydrogen atoms removed.
    """
    atom_array_len_prev = len(atom_array)
    atom_array = atom_array[~np.isin(atom_array.element, ("H", "D"))]
    atom_array_len_new = len(atom_array)

    n_hydrogens_removed = atom_array_len_prev - atom_array_len_new

    if n_hydrogens_removed > 0:
        logger.info(f"Removed {n_hydrogens_removed} hydrogen atoms.")

    return atom_array


def get_polymer_mask(atom_array, use_molecule_type_id=True):
    """Returns a mask of all standard polymer atoms in the AtomArray.

    Polymers here are defined as proteins and nucleic acids (no carbohydrates).

    Args:
        atom_array:
            AtomArray containing the structure to get the polymer mask for.
        use_molecule_type_id:
            Whether to use the molecule_type_id annotation to identify polymers, which
            is faster, but requires the molecule_type_id annotation to be present. If
            False, biotite's `filter_polymer` function is used to identify polymers.

    Returns:
        Mask for all polymer atoms.
    """

    if use_molecule_type_id:
        if "molecule_type_id" not in atom_array.get_annotation_categories():
            raise ValueError(
                "AtomArray does not have molecule_type_id annotation. "
                "Please run the `assign_molecule_type_ids` function first."
            )

        return np.isin(
            atom_array.molecule_type_id,
            [MoleculeType.PROTEIN, MoleculeType.DNA, MoleculeType.RNA],
        )
    else:
        prot_mask = struc.filter_polymer(atom_array, pol_type="peptide")
        nuc_mask = struc.filter_polymer(atom_array, pol_type="nucleotide")

        return prot_mask | nuc_mask


@return_on_empty_atom_array
def remove_small_polymers(atom_array: AtomArray, max_residues: int = 3) -> AtomArray:
    """Removes small polymer chains from the AtomArray

    Follows 2.5.4 of the AlphaFold3 SI and removes all polymer chains with up to
    max_residues resolved residues. We consider proteins and nucleic acids as polymers.

    Args:
        atom_array:
            AtomArray containing the structure to remove small polymers from.
        max_residues:
            Maximum number of residues for a polymer chain to be considered as small.

    Returns:
        AtomArray with all polymer chains with up to max_residues residues removed.
    """
    polymer_mask = get_polymer_mask(atom_array)

    resolved_mask = atom_array.occupancy > 0.0

    # Get only resolved polymer residues
    resolved_poly_array = atom_array[polymer_mask & resolved_mask]

    # Identify too-small polymers
    small_polymer_chains = [
        chain_arr.chain_id[0]
        for chain_arr in struc.chain_iter(resolved_poly_array)
        if struc.get_residue_count(chain_arr) <= max_residues
    ]

    # Create new reference to keep original atom array
    atom_array_filtered = atom_array

    # Remove small polymer chains
    for chain_id in small_polymer_chains:
        atom_array_filtered = remove_chain_and_attached_ligands(
            atom_array_filtered, chain_id
        )

    # Log what chains were removed
    removed_chains = get_differing_chain_ids(atom_array, atom_array_filtered)

    if len(removed_chains) > 0:
        logger.info(
            f"Removed {len(removed_chains)} small polymer chains: {removed_chains}"
        )

    return atom_array_filtered


@return_on_empty_atom_array
def remove_fully_unknown_polymers(atom_array: AtomArray) -> AtomArray:
    """Removes polymer chains with all unknown residues from the AtomArray

    Follows 2.5.4 of the AlphaFold3 SI.

    Args:
        atom_array: AtomArray containing the structure to remove unknown polymers from.

    Returns:
        AtomArray with all polymer chains containing only unknown residues removed.
    """
    # Masks for standard polymers
    polymer_mask = get_polymer_mask(atom_array)

    # Create a new reference to keep original atom array
    atom_array_filtered = atom_array

    # Remove chains with all unknown residues
    for chain in struc.chain_iter(atom_array[polymer_mask]):
        # Explicit single-element unpacking (will fail if >1 chain_id)
        (chain_id,) = np.unique(chain.chain_id)

        # Remove the chain from the AtomArray if all residues are unknown
        if np.all(chain.res_name == "UNK"):
            atom_array_filtered = remove_chain_and_attached_ligands(
                atom_array_filtered, chain_id
            )

    # Log what chains were removed
    removed_chains = get_differing_chain_ids(atom_array, atom_array_filtered)

    if len(removed_chains) > 0:
        logger.info(
            f"Removed {len(removed_chains)} polymer chains with all unknown residues: "
            f"{removed_chains}"
        )

    return atom_array_filtered


@return_on_empty_atom_array
def remove_chain_and_attached_ligands(
    atom_array: AtomArray, chain_id: str
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
            chain ID of the chain to remove

    Returns:
        AtomArray with the specified chain and all attached covalent ligands removed
    """
    chain_mask = atom_array.chain_id == chain_id

    # If the chain itself is a hetero-chain, only remove the chain
    if np.all(atom_array.hetero[chain_mask]):
        return atom_array[~chain_mask]

    # Assign temporary helper indices
    assign_atom_indices(atom_array)

    # Remove everything but the particular chain and all ligands, so that when we search
    # for connected atoms to the chain we will only find the directly connected ligands
    # and not e.g. a covalent ligand in another chain that has a disulfide bond to the
    # specified chain and is therefore indirectly "connected".
    atom_array_subset = atom_array[chain_mask | atom_array.hetero]
    chain_mask_subset = atom_array_subset.chain_id == chain_id

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

    # Remove chain and connected ligands
    atom_array = atom_array[atom_retained_indices]

    # Clean up temporary indices
    remove_atom_indices(atom_array)

    return atom_array


@return_on_empty_atom_array
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
    # Get atom counts of each chain in the total atom array
    unique_chains, counts = np.unique(atom_array.chain_id, return_counts=True)
    chain_to_atom_count = dict(zip(unique_chains, counts, strict=False))

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
        chain1_n_atoms = chain_to_atom_count[chain1_id]
        chain2_n_atoms = chain_to_atom_count[chain2_id]
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

    # Remove clashing chains from the AtomArray
    for chain_id in chain_ids_to_remove:
        atom_array = remove_chain_and_attached_ligands(atom_array, chain_id)

    # Log what chains were removed
    new_chains = struc.get_chains(atom_array)
    removed_chains = np.setdiff1d(unique_chains, new_chains)

    if len(removed_chains) > 0:
        logger.info(f"Removed {len(removed_chains)} clashing chains: {removed_chains}")

    return atom_array


def get_res_atoms_in_ccd_mask(
    res_atom_array: AtomArray | AtomArrayView, ccd: CIFFile
) -> np.ndarray:
    """Returns a mask for atoms in a residue that are present in the CCD

    Args:
        res_atom_array:
            AtomArray or AtomArrayView containing the atoms of a single residue
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
        return np.zeros(len(res_atom_array), dtype=bool)

    allowed_atoms = ccd[res_name]["chem_comp_atom"]["atom_id"].as_array()

    mask = np.isin(res_atom_array.atom_name, allowed_atoms)
    return mask


@return_on_empty_atom_array
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
        get_res_atoms_in_ccd_mask(residue_view, ccd)
        for residue_view in residue_view_iter(atom_array)
    ]

    # Inclusion mask over all atoms
    atom_mask = np.concatenate(atom_masks_per_res)

    # Log what atoms are getting removed
    n_removed_atoms = len(atom_mask) - np.sum(atom_mask)

    if n_removed_atoms > 0:
        # Find the residues where atoms got removed
        edited_residues = get_residue_tuples(
            atom_array[~atom_mask], include_resname=True
        )
        logger.info(
            f"Removed {n_removed_atoms} non-CCD atoms for {len(edited_residues)} "
            f"residues: {edited_residues}."
        )

    # Remove non-CCD atoms
    return atom_array[atom_mask]


@return_on_empty_atom_array
def canonicalize_atom_order(atom_array: AtomArray, ccd: CIFFile) -> AtomArray:
    """Canonicalizes the order of atoms in the AtomArray by the CCD order.

    This function sorts the atoms in the AtomArray based on the order of atoms for the
    corresponding residue in the Chemical Component Dictionary (CCD). This is necessary
    for downstream functionalities that expect atoms across equivalent residues to
    perfectly match, such as the permutation alignment.

    Args:
        atom_array (AtomArray):
            AtomArray containing the structure to canonicalize the atom order for.
        ccd (CIFFile):
            A parsed CIFFile containing the Chemical Component Dictionary
            (components.cif).

    Returns:
        AtomArray with the atoms sorted in the canonical CCD order.
    """

    def get_ref_atoms(res_name):
        """Convenience function that gets the reference atom names in CCD order."""
        return ccd[res_name]["chem_comp_atom"]["atom_id"].as_array().tolist()

    # Fetches the CCD atoms for each residue name as a list
    res_name_to_ref_atoms = {}

    # Track original order
    assign_atom_indices(atom_array, label="_atom_idx_can_order")

    # Create a resorting index that will be populated to sort the entire AtomArray in a
    # final operation
    resort_idx = np.full(atom_array.array_length(), -1, dtype=int)

    for residue_view in residue_view_iter(atom_array):
        # Get the reference atom order for the residue
        res_name = residue_view.res_name[0]
        if res_name in res_name_to_ref_atoms:
            ref_atoms = res_name_to_ref_atoms[res_name]
        else:
            ref_atoms = get_ref_atoms(res_name)
            res_name_to_ref_atoms[res_name] = ref_atoms

        old_atom_idx = residue_view._atom_idx_can_order

        # Get the sorting index that will sort the residue's atoms to match the
        # reference order
        idx_in_ref = np.array(
            [ref_atoms.index(atom) for atom in residue_view.atom_name]
        )
        reordered_atom_idx = residue_view._atom_idx_can_order[np.argsort(idx_in_ref)]

        # Update the final resorting index with the new order
        resort_idx[old_atom_idx] = reordered_atom_idx

    assert np.all(resort_idx != -1), "Not all atoms were reordered"
    assert np.unique(resort_idx).size == len(resort_idx), "Resort index is not unique."

    # Save old index before reordering (needed for logging in the end)
    prev_idx = atom_array._atom_idx_can_order.copy()

    # Apply the sorting index to the entire AtomArray
    atom_array = atom_array[resort_idx]

    # Clean up temporary indices
    atom_array.del_annotation("_atom_idx_can_order")

    # Log which residues were reordered
    reordered_residue_mask = prev_idx != resort_idx

    if np.any(reordered_residue_mask):
        reordered_residue_tuples = get_residue_tuples(
            atom_array[reordered_residue_mask], include_resname=True
        )
        logger.info(
            f"Reordered atoms in {len(reordered_residue_tuples)} residues to follow CCD"
            f" order: {reordered_residue_tuples}."
        )

    return atom_array


@return_on_empty_atom_array
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
        (atom_array.molecule_type_id == MoleculeType.PROTEIN)
        & (atom_array.atom_name == "CA")
    ]

    # If there are no protein chains, return the input atom array
    if len(protein_chain_ca) == 0:
        return atom_array

    # Match C-alpha atoms with their next C-alpha atom
    ca_without_last = protein_chain_ca[:-1]
    ca_shifted_left = construct_atom_array(np.roll(protein_chain_ca, -1, axis=0)[:-1])

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
        protein_chain_ca[:-1][ca_dists > distance_threshold].chain_id
    )

    # Create new reference to keep original atom array
    atom_array_filtered = atom_array

    for chain_id in chain_ids_to_remove:
        atom_array_filtered = remove_chain_and_attached_ligands(
            atom_array_filtered, chain_id
        )

    # Log what chains were removed
    removed_chains = get_differing_chain_ids(atom_array, atom_array_filtered)

    if len(removed_chains) > 0:
        logger.info(
            f"Removed {len(removed_chains)} chains with C-alpha gaps: {removed_chains}"
        )

    return atom_array_filtered


def get_small_ligand_chain_ids(
    atom_array: AtomArray,
    max_atoms: int = 5,
) -> np.ndarray:
    """Returns chain IDs of small ligands

    The "smallness" of a ligand is determined by the number of atoms it consists of.

    Args:
        atom_array:
            AtomArray containing the structure to get the small ligand chain IDs from.
        max_atoms:
            Maximum number of atoms for a ligand to be considered small. Default is 5.

    Returns:
        Array of unique chain IDs of small ligands.
    """
    ligand_array = atom_array[atom_array.molecule_type_id == MoleculeType.LIGAND]

    if len(ligand_array) == 0:
        return np.array([])

    lig_chain_starts = struc.get_chain_starts(ligand_array, add_exclusive_stop=True)
    lig_chain_sizes = np.diff(lig_chain_starts)
    small_lig_chain_idxs = np.where(lig_chain_sizes <= max_atoms)[0]
    small_lig_chain_ids = ligand_array[lig_chain_starts[small_lig_chain_idxs]].chain_id

    # TODO: Dev-only, remove later
    assert np.unique(small_lig_chain_ids).size == small_lig_chain_ids.size, (
        "Small ligand chain IDs are not unique"
    )

    return small_lig_chain_ids


def get_small_ligand_mask(
    atom_array: AtomArray,
    max_atoms: int = 5,
) -> np.ndarray:
    """Returns a mask for atoms in small ligands.

    The "smallness" of a ligand is determined by the number of atoms it consists of.

    Args:
        atom_array:
            AtomArray containing the structure to get the small ligand mask for.
        max_atoms:
            Maximum number of atoms for a ligand to be considered small. Default is 5.
    """
    small_lig_chain_ids = get_small_ligand_chain_ids(atom_array, max_atoms)
    return np.isin(atom_array.chain_id, small_lig_chain_ids)


def maybe_precrop_chains(
    atom_array: AtomArray,
    n_chains: int = 20,
    disable_for_rna: bool = False,
    ignore_ligands_below: int | float = 6,
    random_seed: int | None = None,
) -> AtomArray:
    """Precrops assemblies meeting the criteria to N chains

    Applies the precropping logic to N chains as described in AlphaFold3 SI 2.5.4, with
    additional options to skip precropping in RNA structures or exclude small ligands
    from the N-chain counter.

    Args:
        atom_array:
            AtomArray containing the structure to precrop.
        n_chains:
            Number of chains to keep in the precrop. If the structure has less than N
            chains, all of them are kept. Default is 20.
        disable_for_rna:
            If True and if the structure contains RNA, skip the N-chain precropping.
        ignore_ligands_below:
            Ligand chains with fewer atoms than this value will be ignored in the
            N-chain counter for precropping, and included based on proximity to the
            selected chains. Set this to inf to ignore all ligands from the chain
            budget.
        random_seed:
            Random seed for reproducibility

    Returns:
        AtomArray (precropped to N chains if chain count > N).
    """
    apply_precropping = True

    # Skip if RNA found and disable_for_rna is True
    if disable_for_rna and (atom_array.molecule_type_id == MoleculeType.RNA).any():
        logger.info("Skipping precropping for RNA structure.")
        apply_precropping = False
    else:
        if ignore_ligands_below:
            total_chain_count = struc.get_chain_count(
                atom_array[
                    ~get_small_ligand_mask(
                        atom_array, max_atoms=ignore_ligands_below - 1
                    )
                ]
            )
        else:
            total_chain_count = struc.get_chain_count(atom_array)

    # Precrop assemblies larger than N chains
    if apply_precropping and (total_chain_count > n_chains):
        tokenize_atom_array(atom_array)
        atom_array_precropped = precrop_chains(
            atom_array=atom_array,
            n_chains=n_chains,
            interface_distance_threshold=15.0,
            ignore_ligands_below=ignore_ligands_below,
            random_seed=random_seed,
        )

        removed_chains = get_differing_chain_ids(atom_array, atom_array_precropped)

        logger.info(
            f"Precropping removed {len(removed_chains)} chains: {removed_chains}"
        )

        return atom_array_precropped

    # Don't apply precropping otherwise
    else:
        return atom_array


@return_on_empty_atom_array
def precrop_chains(
    atom_array: AtomArray,
    n_chains: int = 20,
    interface_distance_threshold: float = 15.0,
    ignore_ligands_below: int | float = 6,
    ligand_inclusion_distance: float = 5.0,
    random_seed: int = None,
) -> AtomArray:
    """Subsets structures with too many chains to N chains

    Follows 2.5.4 of the AlphaFold3 SI. Will select a random interface token center atom
    and return the closest N chains based on minimum distances between any token center
    atoms, therefore acting as an initial fixed precropping of too-large assemblies.

    Requires the 'token_center_atom' annotation created by the tokenizer function.

    Args:
        atom_array:
            AtomArray containing the structure to subset
        n_chains:
            Number of chains to keep in the precrop. If the structure has less than N
            chains, all of them are kept. Default is 20.
        interface_distance_threshold:
            Distance threshold in Å that an interface token center atom must have to any
            token center atom in another chain to be considered an interface token
            center atom
        ignore_ligands_below:
            Any ligands with fewer atoms than this number will be excluded from
            the precrop chain count. Instead, they will be included based on
            proximity to the selected chains in the subset. Set to float('inf')
            to never count any ligands toward the chain budget. Default is 6.
        ligand_inclusion_distance:
            Distance threshold in Å when adding ligands excluded by
            `ignore_ligands_below` back into the precrop. Any such ligand with
            a distance ≤ this threshold to any retained chain will be included.
        random_seed:
            Random seed for reproducibility. Default is None.

    Returns:
        AtomArray with the closest n_chains based on token center atom distances,
        plus any nearby small ligands.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Keep pointer to unfiltered AtomArray
    atom_array_orig = atom_array

    # 1) Carve out "small" ligands before selecting chains
    small_ligand_mask = get_small_ligand_mask(
        atom_array_orig, max_atoms=ignore_ligands_below - 1
    )
    atom_array = atom_array_orig[~small_ligand_mask]

    # 2) All token center atoms
    all_token_center_atoms = atom_array[atom_array.token_center_atom]

    # 3) Select random interface token center atom
    interface_token_center_atoms = get_interface_token_center_atoms(
        atom_array, distance_threshold=interface_distance_threshold
    )
    if len(interface_token_center_atoms) == 0:
        logger.warning(
            "No interface token center atoms found for precropping. "
            "Taking a random token center atom instead."
        )
        selected_atom = np.random.choice(all_token_center_atoms)
    else:
        selected_atom = np.random.choice(interface_token_center_atoms)

    # 4) Compute distances to all token centers
    dists = cdist(
        selected_atom.coord.reshape(1, 3),
        all_token_center_atoms.coord,
    )[0]

    # 5) Sort chain IDs by distance

    # Get indices that sort distances low-to-high
    sort_idx = np.argsort(dists)

    # Chain IDs of all token centers, sorted by distance to the selected atom
    chain_ids_sorted = all_token_center_atoms.chain_id[sort_idx]

    # Get indices of the *first* occurrence (i.e., closest distance) for each unique
    # chain ID.
    # Sort these indices to get the distance rank of unique chains.
    unique_chain_idxs_sorted = np.sort(
        np.unique(chain_ids_sorted, return_index=True)[1]
    )

    # Select indices corresponding to the N closest unique chains
    closest_idxs = unique_chain_idxs_sorted[:n_chains]

    # Get the actual IDs of the N closest unique chains
    closest_chain_ids = chain_ids_sorted[closest_idxs]

    # 6) Mask for the closest N chains on the original AtomArray
    n_chain_mask = np.isin(atom_array_orig.chain_id, closest_chain_ids)

    # 7) Build initial subset
    atom_array_subset = atom_array_orig[n_chain_mask]

    # 8) Re-include small ligands by proximity
    atom_array_small_lig = atom_array_orig[small_ligand_mask]
    if len(atom_array_small_lig) == 0:
        return atom_array_subset

    _, proximal_small_lig_chains = get_query_interface_atom_pair_idxs(
        query_atom_array=atom_array_small_lig,
        target_atom_array=atom_array_subset,
        distance_threshold=ligand_inclusion_distance,
        return_chain_pairs=True,
    )

    # 9) If none requalify, just return the core subset
    if proximal_small_lig_chains is None:
        return atom_array_subset

    # 10) Otherwise, add those ligand chains back in
    proximal_ids = np.unique(proximal_small_lig_chains[:, 0])
    proximal_mask = np.isin(atom_array_orig.chain_id, proximal_ids)
    logger.debug(
        f"Adding {len(proximal_ids)} small (< {ignore_ligands_below}) ligand chains to "
        "the precrop by proximity."
    )
    final_mask = n_chain_mask | proximal_mask
    return atom_array_orig[final_mask]


@return_on_empty_atom_array
def remove_std_residue_terminal_atoms(atom_array: AtomArray) -> AtomArray:
    """Removes terminal atoms like OXT and OP3 from standard residues.

    Models like AF3 and AF2 expect all tokens with the same restype to map to the same
    number of atoms. This makes it awkward to represent terminal atoms like OXT/OP3
    which only appear in the small subset of residues at the end/beginning of a
    protein/NA chain. This function therefore removes these atoms from the AtomArray.
    Note that terminal atoms can be kept for any non-standard residues, as they are
    tokenized per-atom and do not require a fixed number of atoms per residue.

    Args:
        atom_array:
            AtomArray containing the structure to remove terminal atoms from.

    Returns:
        AtomArray with terminal atoms removed.
    """
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    terminal_atom_mask = np.zeros(len(atom_array), dtype=bool)

    std_protein_residues = set(STANDARD_PROTEIN_RESIDUES_3)
    std_nucleic_acid_residues = set(STANDARD_NUCLEIC_ACID_RESIDUES)

    # Iterate through all chains
    for chain_start, chain_end in zip(
        chain_starts[:-1], chain_starts[1:], strict=False
    ):
        chain = atom_array[chain_start:chain_end]
        chain_idx = np.arange(chain_start, chain_end)

        # Search for OXT in last residue
        if chain.molecule_type_id[0] == MoleculeType.PROTEIN:
            last_res_name = chain.res_name[-1]

            # Non-standard residues are fully atomized in AF3 and therefore can have
            # terminal atoms
            if last_res_name not in std_protein_residues:
                continue

            last_res_id = chain.res_id[-1]
            last_res_mask = chain.res_id == last_res_id
            oxt_mask = chain.atom_name[last_res_mask] == "OXT"

            # Mark OXT atom for removal
            terminal_atom_mask[chain_idx[last_res_mask]] = oxt_mask

        # Search for OP3 in first residue (5' end)
        elif chain.molecule_type_id[0] in (MoleculeType.DNA, MoleculeType.RNA):
            last_res_name = chain.res_name[0]

            # Non-standard residues are fully atomized in AF3 and therefore can have
            # terminal atoms
            if last_res_name not in std_nucleic_acid_residues:
                continue

            first_res_id = chain.res_id[0]
            first_res_mask = chain.res_id == first_res_id
            op3_mask = chain.atom_name[first_res_mask] == "OP3"

            # Mark OP3 atom for removal
            terminal_atom_mask[chain_idx[first_res_mask]] = op3_mask

        else:
            continue

    # Log which residues got their terminal atoms removed
    edited_residues = get_residue_tuples(
        atom_array[terminal_atom_mask], include_resname=True
    )

    atom_array = atom_array[~terminal_atom_mask]

    if len(edited_residues) > 0:
        logger.info(
            f"Removed terminal atoms from {len(edited_residues)} residues: "
            f"{edited_residues}."
        )

    return atom_array


def convert_intra_residue_dative_to_single(atom_array: AtomArray):
    """Convert intra-residue dative bonds to single bonds.

    This can be required when used with biotite's set_structure function, which does not
    currently support writing out intra-residue COORDINATION bonds.

    Args:
        atom_array:
            AtomArray containing the structure to convert intra-residue dative bonds
            for.

    Returns:
        Copy of the AtomArray with intra-residue dative bonds converted to single bonds.
    """
    # Copy AtomArray
    atom_array = atom_array.copy()

    bondlist_arr = atom_array.bonds.as_array()

    resid_starts_1, resid_starts_2 = (
        struc.get_residue_starts_for(atom_array, bondlist_arr[:, :2].flatten())
        .reshape(-1, 2)
        .T
    )

    # Identify intra-residue and dative bonds
    is_intra_residue = resid_starts_1 == resid_starts_2
    is_dative = bondlist_arr[:, 2] == BondType.COORDINATION
    is_intra_residue_dative = is_intra_residue & is_dative

    # Convert intra-residue dative bonds to single bonds
    bondlist_arr[is_intra_residue_dative, 2] = BondType.SINGLE

    # Set the new BondList and return
    new_bondlist = BondList(atom_count=len(atom_array), bonds=bondlist_arr)
    atom_array.bonds = new_bondlist

    return atom_array


def prefilter_bonds(
    atom_array: AtomArray,
    remove_inter_chain_dative: bool = True,
    remove_inter_chain_poly_links: bool = True,
    remove_intra_chain_poly_links: bool = True,
    remove_longer_than: float | None = 2.4,
) -> AtomArray:
    """Filters the BondList according to different criteria.

    Args:
        atom_array (AtomArray):
            AtomArray to filter bonds for.
        remove_inter_chain_dative (bool):
            Whether to remove dative (biotite BondType.COORDINATION) bonds between
            different chains. Default is True.
        remove_inter_chain_poly_links (bool):
            Whether to remove polymer-polymer bonds between different chains, such as
            cross-chain disulfide bonds. Polymers are defined as the standard polymers
            [protein, DNA, RNA] (not polymeric ligands). Default is True.
        remove_intra_chain_poly_links (bool):
            Whether to remove polymer-polymer bonds between non-consecutive residues in
            the same chain, such as intra-chain disulfide bonds. Polymers are defined as
            the standard polymers [protein, DNA, RNA] (not polymeric ligands). Default
            is True.
        remove_longer_than (float | None):
            If not None, remove all bonds longer than this cutoff distance (in Å).
            Default is 2.4 Å. If None, no bonds are removed by length.

    Returns:
        AtomArray:
            AtomArray with filtered bond list.
    """

    def log_removed_bonds(removal_mask: np.ndarray, bond_name: str):
        """Convenience function to log any removed atoms on each step."""
        if np.any(removal_mask):
            bond_atom_tuples = get_bond_atom_tuples(
                atom_array, bond_partners[removal_mask], include_resname=True
            )
            logger.info(
                f"Removed {len(bond_atom_tuples)} {bond_name} bonds: {bond_atom_tuples}"
            )

    bondlist = atom_array.bonds.as_array()
    bond_partners = bondlist[:, :2]
    valid_bonds_mask = np.ones(len(bond_partners), dtype=bool)

    # Get inter-/intra-chain masks if required
    if (
        remove_inter_chain_dative
        or remove_inter_chain_poly_links
        or remove_intra_chain_poly_links
    ):
        bond_partner_chain_ids = atom_array.chain_id[bond_partners]

        is_inter_chain = bond_partner_chain_ids[:, 0] != bond_partner_chain_ids[:, 1]
        is_intra_chain = ~is_inter_chain

    # Remove dative bonds between different chains
    if remove_inter_chain_dative:
        bond_types = bondlist[:, 2]
        is_dative = bond_types == BondType.COORDINATION

        inter_chain_dative_mask = is_dative & is_inter_chain
        valid_bonds_mask[inter_chain_dative_mask] = False

        # Log removed bonds
        log_removed_bonds(inter_chain_dative_mask, "inter-chain dative")

    # Handle bonds between polymer residues
    if remove_inter_chain_poly_links or remove_intra_chain_poly_links:
        # Find polymer-polymer bonds
        bond_partner_moltypes = atom_array.molecule_type_id[bond_partners]
        is_polymer = np.isin(
            bond_partner_moltypes,
            [MoleculeType.PROTEIN, MoleculeType.DNA, MoleculeType.RNA],
        )
        is_polymer_polymer = is_polymer.all(axis=1)

        # Remove inter-chain polymer-polymer bonds
        if remove_inter_chain_poly_links:
            inter_chain_poly_link_mask = is_polymer_polymer & is_inter_chain
            valid_bonds_mask[inter_chain_poly_link_mask] = False

            # Log removed bonds
            log_removed_bonds(inter_chain_poly_link_mask, "inter-chain polymer-polymer")

        # Remove intra-chain polymer-polymer bonds between non-consecutive residues
        if remove_intra_chain_poly_links:
            bond_partner_res_ids = atom_array.res_id[bond_partners]

            is_nonconsecutive = (
                np.abs(bond_partner_res_ids[:, 0] - bond_partner_res_ids[:, 1]) > 1
            )

            non_consecutive_intra_chain_poly_link_mask = (
                is_polymer_polymer & is_intra_chain & is_nonconsecutive
            )
            valid_bonds_mask[non_consecutive_intra_chain_poly_link_mask] = False

            # Log removed bonds
            log_removed_bonds(
                non_consecutive_intra_chain_poly_link_mask,
                "non-consecutive intra-chain polymer-polymer",
            )

    # Remove bonds longer than the cutoff
    if remove_longer_than is not None:
        distances = index_distance(atom_array, bond_partners)

        # Unresolved atoms will generate NaN distances, so we always keep their bonds as
        # we don't know their bond lengths
        long_bond_mask = (distances > remove_longer_than) & ~np.isnan(distances)
        valid_bonds_mask[long_bond_mask] = False

        # Log removed bonds
        log_removed_bonds(long_bond_mask, f"too-long (>{remove_longer_than} Å)")

    # Exclude masked bonds from new BondList
    new_bondlist = BondList(
        atom_count=len(atom_array),
        bonds=bondlist[valid_bonds_mask],
    )

    # Create a new AtomArray with the filtered bond list
    atom_array_filtered = atom_array.copy()
    atom_array_filtered.bonds = new_bondlist

    return atom_array_filtered


def filter_fully_atomized_bonds(
    atom_array: AtomArray,
) -> AtomArray:
    """Only retains bonds where both bond partners are atomized.

    Requires the `is_atomized` attribute to be set in the AtomArray (see
    `tokenize_atom_array`).
    """
    if "is_atomized" not in atom_array.get_annotation_categories():
        raise ValueError(
            "The input atom_array must have the is_atomized attribute to filter "
            "atomized bonds. See the 'tokenize_atom_array' function."
        )

    # Get the bond list from the atom array
    bondlist = atom_array.bonds.as_array()

    # Get the bond partners
    bond_partners = bondlist[:, :2]

    # Create a mask for bonds where both partners are atomized
    is_both_atomized = atom_array.is_atomized[bond_partners].all(axis=1)

    # Create a new BondList with only the bonds that are both atomized
    new_bondlist = BondList(
        atom_count=len(atom_array),
        bonds=bondlist[is_both_atomized],
    )

    # Create a new AtomArray with the filtered bond list
    atom_array_filtered = atom_array.copy()
    atom_array_filtered.bonds = new_bondlist

    return atom_array_filtered


def remove_covalent_nonprotein_chains(atom_array: AtomArray) -> AtomArray:
    """Removes non-protein chains covalently linked to protein chains.

    Args:
        atom_array (AtomArray):
            AtomArray from which to remove non-protein chains covalently attached to
            protein chains.

    Returns:
        AtomArray:
            AtomArray with non-protein chains covalently attached to protein chains
            removed.
    """

    # Get cross-chain bonds
    cross_chain_bonds = find_cross_chain_bonds(atom_array)

    # Exclude coordinate bonds
    cross_chain_bonds_covalent = cross_chain_bonds[
        (cross_chain_bonds[:, -1] != BondType.COORDINATION), :
    ][:, :-1]

    # Get molecule types of the atoms involved in the cross-chain bonds
    cross_chain_bonds_covalent_mol_type = atom_array.molecule_type_id[
        cross_chain_bonds_covalent
    ]

    # Find bonds that have exactly one protein atom
    cross_chain_bonds_covalent_has_nonprotein = cross_chain_bonds_covalent[
        (cross_chain_bonds_covalent_mol_type[:, 0] == MoleculeType.PROTEIN)
        ^ (cross_chain_bonds_covalent_mol_type[:, 1] == MoleculeType.PROTEIN)
    ]

    # Get non-protein atom indices then covalent non-protein chain IDs
    cross_chain_atom_nonprotein = cross_chain_bonds_covalent_has_nonprotein.ravel()[
        cross_chain_bonds_covalent_mol_type.ravel() != MoleculeType.PROTEIN
    ]
    covalent_nonprotein_chain_ids = atom_array.chain_id[cross_chain_atom_nonprotein]

    return atom_array[~np.isin(atom_array.chain_id, covalent_nonprotein_chain_ids)]
