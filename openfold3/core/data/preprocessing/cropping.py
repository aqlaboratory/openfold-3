# TODO add license

# Some biotite examples
# %%
from typing import Union, Optional

import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure import AtomArray
from numpy.random import Generator

from openfold3.core.data.preprocessing.tokenization import (
    assign_chains,
    tokenize_atom_array,
)

# Protein trimer with covalent and non-covalent glycans
pdbx_file_8FAQ = pdbx.CIFFile.read(
    rcsb.fetch("8FAQ", "cif"),
)
# Protein trimer with non-removed ions and non-covalent ligands
pdbx_file_1PCR = pdbx.CIFFile.read(rcsb.fetch("1PCR", "cif"))
# Protein-DNA complex
pdbx_file_1NVP = pdbx.CIFFile.read(rcsb.fetch("1NVP", "cif"))
# Protein with modified residues
pdbx_file_3US4 = pdbx.CIFFile.read(rcsb.fetch("3US4", "cif"))
# Protein RNA complex
pdbx_file_1A9N = pdbx.CIFFile.read(rcsb.fetch("1A9N", "cif"))

biounit_8FAQ = pdbx.get_assembly(
    pdbx_file_8FAQ,
    assembly_id="1",
    model=1,
    altloc="occupancy",
    use_author_fields=False,
    include_bonds=True,
)
biounit_1PCR = pdbx.get_assembly(
    pdbx_file_1PCR,
    assembly_id="1",
    model=1,
    altloc="occupancy",
    use_author_fields=False,
    include_bonds=True,
)
biounit_1NVP = pdbx.get_assembly(
    pdbx_file_1NVP,
    assembly_id="1",
    model=1,
    altloc="occupancy",
    use_author_fields=False,
    include_bonds=True,
)
biounit_3US4 = pdbx.get_assembly(
    pdbx_file_3US4,
    assembly_id="1",
    model=1,
    altloc="occupancy",
    use_author_fields=False,
    include_bonds=True,
)
biounit_1A9N = pdbx.get_assembly(
    pdbx_file_1A9N,
    assembly_id="1",
    model=1,
    altloc="occupancy",
    use_author_fields=False,
    include_bonds=True,
)

# Remove waters
biounit_8FAQ = biounit_8FAQ[biounit_8FAQ.res_name != "HOH"]
biounit_1PCR = biounit_1PCR[biounit_1PCR.res_name != "HOH"]
biounit_1NVP = biounit_1NVP[biounit_1NVP.res_name != "HOH"]
biounit_3US4 = biounit_3US4[biounit_3US4.res_name != "HOH"]
biounit_1A9N = biounit_1A9N[biounit_1A9N.res_name != "HOH"]

# %%
tokenize_atom_array(biounit_8FAQ)
tokenize_atom_array(biounit_1PCR)
tokenize_atom_array(biounit_1NVP)
tokenize_atom_array(biounit_3US4)
tokenize_atom_array(biounit_1A9N)
assign_chains(biounit_8FAQ)
assign_chains(biounit_1PCR)
assign_chains(biounit_1NVP)
assign_chains(biounit_3US4)
assign_chains(biounit_1A9N)

atom_array = biounit_1PCR
generator = np.random.default_rng(2346)
token_budget = 384
chains = np.array(list(set(atom_array.af3_chain_id)), dtype=int)
chains = generator.permutation(chains)
print(chains)
# %%
# print("Number of chains:", struc.get_chain_count(biounit_3US4))
# struc.get_residues(biounit_3US4)
# import biotite.structure.io as strucio
# strucio.save_structure("/mnt/c/Users/nikol/Documents/biotite_tests/1NVP.pdb", biounit_1NVP)


def crop_contiguous(
    atom_array: AtomArray, token_budget: int, generator: Generator
) -> None:
    """Implements Contiguous Cropping from AF3 SI, 2.7.1.

    Uses Algorithm 1 from AF-Multimer section 7.2.1. to update the input biotite
    atom array with added 'af3_crop_mask' annotation in-place.

    Args:
        atom_array (atom_array):
            Biotite atom array of the first bioassembly of a PDB entry.
        token_budget (int):
            Token budget i.e. total crop size.
        generator (Generator):
            A numpy generator set with a specific seed.

    Returns:
        None
    """

    # Get chain ids and permute
    chains = np.array(list(set(atom_array.af3_chain_id)), dtype=int)
    chains = generator.permutation(chains)

    # Create cropping mask annotation
    atom_array.set_annotation("af3_crop_mask", np.repeat(False, len(atom_array)))

    # Cropping loop
    token_budget = crop_size
    tokens_remaining = total number of tokens










    tokens_added = 0
    tokens_remaining = token_budget
    for chain_id in chains:
        # Get chain atom array and type
        atom_array_chain = atom_array[atom_array.af3_chain_id == chain_id]

        # Get chain length
        chain_length = (
            atom_array_chain.af3_token_id[-1] - atom_array_chain.af3_token_id[0] + 1
        )

        # Cannot have negative number of tokens remaining
        tokens_remaining = max(tokens_remaining - chain_length, 0)

        # Calculate crop length
        crop_size_max = np.min([token_budget - tokens_added, chain_length])
        crop_size_min = np.min(
            [chain_length, np.max([0, token_budget - tokens_added - tokens_remaining])]
        )
        crop_size = generator.integers(crop_size_min, crop_size_max + 1, 1).item()

        tokens_added += crop_size

        # Calculate crop start and map to global atom index
        crop_start = generator.integers(0, chain_length - crop_size + 1, 1).item()
        crop_start_global = atom_array_chain[crop_start].af3_atom_id

        # Edit corresponding segment in crop mask
        atom_array.af3_crop_mask[crop_start_global : crop_start_global + crop_size] = (
            True
        )

        # Break if allocated all of the token budget
        if tokens_remaining == 0:
            break

    return None


def crop_spatial(
    atom_array: AtomArray,
    token_budget: int,
    generator: Generator,
    preferred_chain_or_interface: Optional[Union[int, tuple[int, int]]] = None,
) -> None:
    """Implements Spatial Cropping from AF3 SI, 2.7.2.

    Uses Algorithm 2 from AF-Multimer section 7.2.2. to update the input biotite
    atom array with added 'af3_crop_mask' annotation in-place.

    Args:
        atom_array (AtomArray):
            Biotite atom array of the first bioassembly of a PDB entry.
        token_budget (int):
            Total crop size.
        generator (Generator):
            A numpy generator set with a specific seed.
        preferred_chain_or_interface (Optional[Union[int, tuple[int, int]]]):
            Integer or integer 2-tuple indicating the preferred chain or interface,
            respectively, from which reference atoms are selected. Generated by
            eq. 1 in AF3 SI for the weighted PDB dataset.

    Returns:
        None
    """
    # Subset token center atoms to those in the preferred chain/interface if provided
    token_center_atoms = atom_array[atom_array.af3_token_center_atom]
    if preferred_chain_or_interface:
        # If chain provided
        if isinstance(preferred_chain_or_interface, int):
            token_center_atoms = token_center_atoms[
                token_center_atoms.af3_chain_id == preferred_chain_or_interface
            ]
        # If interface provided
        else:
            token_center_atoms = token_center_atoms[
                token_center_atoms.af3_chain_id in preferred_chain_or_interface
            ]

    # Get reference atom
    reference_atom = generator.choice(token_center_atoms, size=1)[0]

    # Get distance from all other token center atoms and break ties
    distances_to_reference_atom = (
        struc.distance(
            reference_atom,
            token_center_atoms[
                token_center_atoms.af3_atom_id != reference_atom.af3_atom_id
            ],
        )
        + np.arange(len(token_center_atoms) - 1) * 1e-3
    )

    # Get token_budget nearest token center atoms
    nearest_token_center_atom_ids = np.argsort(distances_to_reference_atom)[
        :token_budget
    ]

    # Get all atoms for nearest token center atoms
    atom_array.set_annotation(
        "af3_crop_mask",
        np.isin(
            atom_array.af3_token_id,
            token_center_atoms[nearest_token_center_atom_ids].af3_token_id,
        ),
    )
    return None


def crop_spatial_interface(
    atom_array: AtomArray,
    token_budget: int,
    generator: Generator,
    preferred_chain_or_interface: Optional[Union[int, tuple[int, int]]] = None,
) -> None:
    """Implements Spatial Interface Cropping from AF3 SI, 2.7.3.

    Uses Algorithm 2 from AF-Multimer section 7.2.2. to update the input biotite
    atom array with added 'af3_crop_mask' annotation in-place.

    Args:
        atom_array (AtomArray):
            Biotite atom array of the first bioassembly of a PDB entry.
        token_budget (int):
            Total crop size.
        generator (Generator):
            A numpy generator set with a specific seed.
        preferred_chain_or_interface (Optional[Union[int, tuple[int, int]]]):
            Integer or integer 2-tuple indicating the preferred chain or interface,
            respectively, from which reference atoms are selected. Generated by
            eq. 1 in AF3 SI for the weighted PDB dataset.

    Returns:
        None
    """
    # Subset token center atoms to those in the preferred chain/interface if provided
    token_center_atoms = atom_array[atom_array.af3_token_center_atom]
    if preferred_chain_or_interface:
        # If chain provided
        if isinstance(preferred_chain_or_interface, int):
            token_center_atoms = token_center_atoms[
                token_center_atoms.af3_chain_id == preferred_chain_or_interface
            ]
        # If interface provided
        else:
            token_center_atoms = token_center_atoms[
                token_center_atoms.af3_chain_id in preferred_chain_or_interface
            ]

    # Find interface token center atoms (within 15 A of at least one token center atom in another chain)
    n = len(token_center_atoms)
    token_center_atom_ids = np.arange(n)
    token_center_atom_coords = token_center_atoms._coord
    token_center_atom_chains = token_center_atoms.af3_chain_id

    is_different_chains = np.repeat(token_center_atom_chains, n) != np.tile(
        token_center_atom_chains, n
    )
    query_ids = np.repeat(token_center_atom_ids, n)
    target_ids = np.tile(token_center_atom_ids, n)
    query_coords = token_center_atom_coords[query_ids, :][is_different_chains, :]
    target_coords = token_center_atom_coords[target_ids, :][is_different_chains, :]

    is_interface = struc.distance(query_coords, target_coords) < 15

    interface_token_center_atom_ids = np.array(list(
        set(query_ids[is_different_chains][is_interface])
        | set(target_ids[is_different_chains][is_interface])
    ))





    pairwise_array = struc.array(
        np.column_stack(
            (np.repeat(token_center_atoms, n), np.tile(token_center_atoms, n))
        )
    )
    pairwise_array = struc.array([pairwise_array[:, 0], pairwise_array[:, 1]])
    struc.distance(np.repeat(token_center_atoms, n), np.tile(token_center_atoms, n))
    # TODO add preferred_chain_or_interface logic

    return
