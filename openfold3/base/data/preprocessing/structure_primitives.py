from itertools import combinations

import biotite.structure as struc
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tables import (
    CRYSTALLIZATION_AIDS,
    STANDARD_NUCLEIC_ACID_RESIDUES,
    STANDARD_PROTEIN_RESIDUES,
    STANDARD_RESIDUES,
    TOKEN_CENTER_ATOMS,
)


def convert_MSE_to_MET(atom_array: struc.AtomArray) -> None:
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


def fix_single_arginine_naming(arg_atom_array: struc.AtomArray) -> None:
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


def fix_arginine_naming(atom_array: struc.AtomArray) -> None:
    """Resolves naming ambiguities for all arginine residues in the AtomArray

    (see fix_single_arginine_naming for more details)

    Args:
        atom_array: AtomArray containing the structure to fix arginine residues in.
    """
    arginines = atom_array.res_name == "ARG"

    for arginine in struc.residue_iter(atom_array[arginines]):
        fix_single_arginine_naming(arginine)


def remove_waters(atom_array: struc.AtomArray) -> struc.AtomArray:
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
    atom_array: struc.AtomArray, ccd_codes=CRYSTALLIZATION_AIDS
) -> struc.AtomArray:
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


def remove_hydrogens(atom_array: struc.AtomArray) -> struc.AtomArray:
    """Removes all hydrogen atoms from the AtomArray

    Args:
        atom_array: AtomArray containing the structure to remove hydrogens from.

    Returns:
        AtomArray with all hydrogen atoms removed.
    """
    atom_array = atom_array[~np.isin(atom_array.element, ("H", "D"))]

    return atom_array


def remove_fully_unknown_polymers(atom_array: struc.AtomArray) -> struc.AtomArray:
    """Removes polymer chains with all unknown residues from the AtomArray

    Follows 2.5.4 of the AlphaFold3 SI.

    Args:
        atom_array: AtomArray containing the structure to remove unknown polymers from.

    Returns:
        AtomArray with all polymer chains containing only unknown residues removed.
    """
    # Masks for standard polymers
    proteins = struc.filter_polymer(atom_array, poly_type="peptide")
    nucleic_acids = struc.filter_polymer(atom_array, poly_type="nucleotide")

    # Combine to get mask for all polymers
    polymers = proteins | nucleic_acids

    atom_array_filtered = atom_array

    for chain in struc.chain_iter(atom_array[polymers]):
        # Explicit single-element unpacking (will fail if >1 chain_id)
        (chain_id,) = np.unique(chain.chain_id_renumbered)

        # Remove the chain from the AtomArray if all residues are unknown
        if np.all(chain.res_name == "UNK"):
            atom_array_filtered = remove_chain_and_attached_ligands(
                atom_array_filtered, chain_id
            )

    return atom_array_filtered


def assign_renumbered_chain_ids(atom_array: struc.AtomArray) -> struc.AtomArray:
    """Creates a renumbered chain index

    Iterates through all chains in the atom array and assigns unique numerical chain IDs
    starting with 0 to each chain in the "chain_id_renumbered" field. This is useful for
    bioassembly parsing where chain IDs can be duplicated after the assembly is
    expanded.

    Args:
        atom_array (AtomArray): biotite atom array

    Returns:
        AtomArray with the "chain_id_renumbered" field added
    """
    chain_start_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # Assign numerical chain IDs
    chain_id_n_repeats = np.diff(chain_start_idxs)
    chain_ids_per_atom = np.repeat(
        np.arange(len(chain_id_n_repeats)), chain_id_n_repeats
    )
    atom_array.set_annotation("chain_id_renumbered", chain_ids_per_atom)

    return atom_array


def remove_chain_and_attached_ligands(
    atom_array: struc.AtomArray, chain_id: int
) -> struc.AtomArray:
    """Removes a chain from an AtomArray including all attached covalent ligands

    While not explicitly stated in the AlphaFold3 SI, the intent of this function is to
    remove protein or nucleic acid chains together with their covalent modifications, so
    that no "floating" covalent ligands are left behind. This doesn't apply the other
    way around, so if the input chain itself is a hetero-chain (e.g. a glycan), only the
    input chain is removed.

    Args:
        atom_array:
            biotite atom array
        chain_id:
            chain ID of the chain to remove (uses the "chain_id_renumbered" field)

    Returns:
        AtomArray with the specified chain and all attached covalent ligands removed
    """
    chain_mask = atom_array.chain_id_renumbered == chain_id

    # If the chain itself is a hetero-chain, only remove the chain
    if np.all(atom_array.hetero[chain_mask]):
        return atom_array[~chain_mask]

    # Get first atom of the chain
    first_chain_atom_idx = np.nonzero(chain_mask)[0][0]

    # Identify bonded ligands as connected atoms with hetero flag
    connected_atoms_mask = struc.find_connected(
        atom_array.bonds, first_chain_atom_idx, as_mask=True
    )
    connected_ligand_atoms_mask = connected_atoms_mask & atom_array.hetero

    # Remove chain and connected ligands
    atom_array = atom_array[~(chain_mask | connected_ligand_atoms_mask)]

    return atom_array


def remove_clashing_chains(
    atom_array: struc.AtomArray,
    clash_distance: float = 1.7,
    clash_percentage: float = 0.3,
) -> struc.AtomArray:
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
    pairwise_dists = squareform(pdist(atom_array.coord, metric="euclidean"))

    # Mask out self-distances
    np.fill_diagonal(pairwise_dists, np.inf)

    # Early break if no clashes are present
    if not np.any(pairwise_dists < clash_distance):
        return atom_array

    # Mask out distances between covalently bonded atoms which should not be
    # considered clashes
    pairwise_dists[atom_array.bonds.adjacency_matrix()] = np.inf

    # All possible pairings of chains (ignoring order and without self-pairings)
    chain_ids = np.unique(atom_array.chain_id_renumbered)
    chain_pairings = list(combinations(chain_ids, 2))

    # Mask to access every chain
    chain_masks = {
        chain_id: atom_array.chain_id_renumbered == chain_id for chain_id in chain_ids
    }

    chain_ids_to_remove = set()

    # Loop through all pairings and collect chains to remove
    for chain1_id, chain2_id in chain_pairings:
        chain1_mask = chain_masks[chain1_id]
        chain2_mask = chain_masks[chain2_id]
        chain1_chain2_dists = pairwise_dists[chain1_mask][:, chain2_mask]

        chain1_n_clashing_atoms = np.any(
            chain1_chain2_dists < clash_distance, axis=1
        ).sum()
        chain2_n_clashing_atoms = np.any(
            chain1_chain2_dists < clash_distance, axis=0
        ).sum()

        chain1_clash_fraction = chain1_n_clashing_atoms / chain_masks[chain1_id].sum()
        chain2_clash_fraction = chain2_n_clashing_atoms / chain_masks[chain2_id].sum()

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
        atom_array = remove_chain_and_attached_ligands(atom_array, chain_id)

    return atom_array

