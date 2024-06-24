# TODO add license

# Some biotite examples
#%%
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from numpy.random import Generator
from biotite.structure import AtomArray
from openfold3.core.data.preprocessing.tokenization import tokenize_atom_array, chain_assignment


# Protein trimer with covalent and non-covalent glycans
pdbx_file_8FAQ = pdbx.CIFFile.read(rcsb.fetch("8FAQ", "cif"), )
# Protein trimer with non-removed ions and non-covalent ligands 
pdbx_file_1PCR = pdbx.CIFFile.read(rcsb.fetch("1PCR", "cif"))
# Protein-DNA complex
pdbx_file_1NVP = pdbx.CIFFile.read(rcsb.fetch("1NVP", "cif"))
# Protein with modified residues
pdbx_file_3US4 = pdbx.CIFFile.read(rcsb.fetch("3US4", "cif"))
# Protein RNA complex
pdbx_file_1A9N = pdbx.CIFFile.read(rcsb.fetch("1A9N", "cif"))

biounit_8FAQ = pdbx.get_assembly(pdbx_file_8FAQ, assembly_id="1", model=1, altloc='occupancy', use_author_fields=False, include_bonds=True)
biounit_1PCR = pdbx.get_assembly(pdbx_file_1PCR, assembly_id="1", model=1, altloc='occupancy', use_author_fields=False, include_bonds=True)
biounit_1NVP = pdbx.get_assembly(pdbx_file_1NVP, assembly_id="1", model=1, altloc='occupancy', use_author_fields=False, include_bonds=True)
biounit_3US4 = pdbx.get_assembly(pdbx_file_3US4, assembly_id="1", model=1, altloc='occupancy', use_author_fields=False, include_bonds=True)
biounit_1A9N = pdbx.get_assembly(pdbx_file_1A9N, assembly_id="1", model=1, altloc='occupancy', use_author_fields=False, include_bonds=True)

# Remove waters
biounit_8FAQ = biounit_8FAQ[biounit_8FAQ.res_name != "HOH"]
biounit_1PCR = biounit_1PCR[biounit_1PCR.res_name != "HOH"]
biounit_1NVP = biounit_1NVP[biounit_1NVP.res_name != "HOH"]
biounit_3US4 = biounit_3US4[biounit_3US4.res_name != "HOH"]
biounit_1A9N = biounit_1A9N[biounit_1A9N.res_name != "HOH"]

#%%
tokenize_atom_array(biounit_8FAQ)
tokenize_atom_array(biounit_1PCR)
tokenize_atom_array(biounit_1NVP)
tokenize_atom_array(biounit_3US4)
tokenize_atom_array(biounit_1A9N)
chain_assignment(biounit_8FAQ)
chain_assignment(biounit_1PCR)
chain_assignment(biounit_1NVP)
chain_assignment(biounit_3US4)
chain_assignment(biounit_1A9N)

atom_array = biounit_1A9N
generator = np.random.default_rng(2346)
n_res = 384
chains = np.array(list(set(atom_array.af3_chain_id)), dtype=int)
chains = generator.permutation(chains)
print(chains)
#%%
# print("Number of chains:", struc.get_chain_count(biounit_3US4))
# struc.get_residues(biounit_3US4)

# strucio.save_structure("/mnt/c/Users/nikol/Documents/biotite_tests/1NVP.pdb", biounit_1NVP)


def crop_contiguous(atom_array: AtomArray, n_res: int, generator: Generator):
    """Implements Contiguous Cropping, Algorithm 1 from AF-Multimer section 7.2.1.

    Updates the input biotite atom array with added 'af3_crop_mask' annotation in-place.

    Args:
        atom_array (atom_array):
            biotite atom array of the first bioassembly of a PDB entry
        n_res (int):
            residue budget i.e. total crop size
        generator (Generator): 
            a numpy generator set with a specific seed

    Returns:
        None
    """

    # Get chain ids and permute
    chains = np.array(list(set(atom_array.af3_chain_id)), dtype=int)
    chains = generator.permutation(chains)

    # Create cropping mask annotation
    atom_array.set_annotation("af3_crop_mask", np.repeat(False, len(atom_array)))

    # Cropping loop
    n_added = 0
    n_remaining = n_res
    for chain_id in chains:
        # Get chain atom array and type
        atom_array_chain = atom_array[atom_array.af3_chain_id == chain_id]
        molecule_type_chain = set(atom_array_chain.af3_molecule_type) == 0

        # If standard polymer WITHOUT covalent modifications
        if (molecule_type_chain == 0) | (molecule_type_chain == 1):
            n_k = struc.get_residue_count(atom_array_chain)
        # TODO If standard polymer WITH covalent modifications
        # If nonstandard residue in polymer or ligand
        else:
            n_k = len(atom_array_chain)
        n_remaining -= n_k

        # Calculate crop length
        crop_size_max = np.min([n_res - n_added, n_k])
        crop_size_min = np.min([n_k, np.max([0, n_res - n_added - n_remaining])])
        crop_size = generator.integers(crop_size_min, crop_size_max + 1, 1).item()
        n_added += crop_size

        # Calculate crop start and map to global atom index
        crop_start = generator.integers(0, n_k - crop_size + 1, 1).item()
        crop_start_global = atom_array_chain[crop_start].af3_atom_id

        # Edit corresponding segment in crop mask
        atom_array.af3_crop_mask[crop_start_global:crop_start_global + crop_size] = True



    return None

def crop_spatial():
    return


def crop_spatial_interface():
    return