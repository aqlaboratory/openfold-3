from itertools import combinations

import biotite.structure as struc
import numpy as np
from biotite.structure.io.pdbx import CIFFile
from scipy.spatial.distance import cdist, pdist, squareform

from .tables import (
    CRYSTALLIZATION_AIDS,
    MOLECULE_TYPE_ID_DNA,
    MOLECULE_TYPE_ID_LIGAND,
    MOLECULE_TYPE_ID_PROTEIN,
    MOLECULE_TYPE_ID_RNA,
    NUCLEIC_ACID_MAIN_CHAIN_ATOMS,
    PROTEIN_MAIN_CHAIN_ATOMS,
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES,
    STANDARD_RESIDUES,
    STANDARD_RNA_RESIDUES,
    TOKEN_CENTER_ATOMS,
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


def assign_molecule_types(atom_array: struc.AtomArray) -> None:
    """Assigns molecule types to the AtomArray

    Assigns molecule types to each chain based on its residue names. Possible molecule
    types are protein, RNA, DNA, and ligand. The molecule type is stored in the
    "molecule_type" field of the AtomArray.

    Args:
        atom_array:
            AtomArray containing the structure to assign molecule types to.
    """
    chain_start_idxs = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # Create molecule type annotation
    molecule_types = np.zeros(len(atom_array))

    # Zip together chain starts and ends
    for chain_start, chain_end in zip(chain_start_idxs[:-1], chain_start_idxs[1:]):
        residues_in_chain = set(atom_array[chain_start:chain_end].res_name)

        # Assign protein if any standard protein residue is present
        if residues_in_chain & set(STANDARD_PROTEIN_RESIDUES):
            molecule_types[chain_start:chain_end] = MOLECULE_TYPE_ID_PROTEIN

        # Assign RNA if any standard RNA residue is present
        elif residues_in_chain & set(STANDARD_RNA_RESIDUES):
            molecule_types[chain_start:chain_end] = MOLECULE_TYPE_ID_RNA

        # Assign DNA if any standard DNA residue is present
        elif residues_in_chain & set(STANDARD_DNA_RESIDUES):
            molecule_types[chain_start:chain_end] = MOLECULE_TYPE_ID_DNA

        # Assign ligand otherwise
        else:
            molecule_types[chain_start:chain_end] = MOLECULE_TYPE_ID_LIGAND

    atom_array.set_annotation("molecule_type", molecule_types)


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


def remove_small_polymers(
    atom_array: struc.AtomArray, max_residues: int = 3
) -> struc.AtomArray:
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
        atom_array = remove_chain_and_attached_ligands(atom_array, chain_id)

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
            atom_array_filtered = remove_chain_and_attached_ligands(
                atom_array_filtered, chain_id
            )

    return atom_array_filtered


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
            AtomArray containing the structure to remove the chain from.
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
    query_token_centers = query_atom_array[query_atom_array.af3_token_center_atom]
    target_token_centers = target_atom_array[target_atom_array.af3_token_center_atom]

    return get_interface_atoms(
        query_token_centers, target_token_centers, distance_threshold
    )


def get_res_atoms_in_ccd_mask(
    res_atom_array: struc.AtomArray, ccd: CIFFile
) -> np.ndarray:
    """Returns a mask for atoms in a residue that are present in the CCD

    Args:
        res_atom_array:
            AtomArray containing the atoms of a single residue
        ccd:
            CIFFile containing the parsed CCD (components.cif)

    Returns:
        Mask for atoms in the residue that are present in the CCD
    """
    res_name = res_atom_array.res_name[0]
    allowed_atoms = ccd[res_name]["chem_comp_atom"]["atom_id"].as_array()

    mask = np.isin(res_atom_array.atom_name, allowed_atoms)
    return mask


def remove_non_CCD_atoms(atom_array: struc.AtomArray, ccd: CIFFile) -> struc.AtomArray:
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

    return atom_array[atom_mask]


def remove_chains_with_CA_gaps(
    atom_array: struc.AtomArray, distance_threshold: float = 10.0
) -> struc.AtomArray:
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
        atom_array = remove_chain_and_attached_ligands(atom_array, chain_id)

    return atom_array


def subset_large_structure(
    atom_array: struc.AtomArray, n_chains: int
) -> struc.AtomArray:
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

    Returns:
        AtomArray with the closest n_chains based on token center atom distances
    """

    # Select random interface token center atom
    interface_token_center_atoms = get_interface_token_center_atoms(
        atom_array, atom_array
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

    return atom_array[selected_chain_mask]

