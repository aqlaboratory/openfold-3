# TODO add license

# Some biotite examples
#%%
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from numpy.random import Generator
from biotite.structure import AtomArray
from openfold3.core.data.preprocessing.tokenization import tokenize_atomarray, chain_assignment


# Protein trimer with covalent and non-covalent glycans
pdbx_file_8FAQ = pdbx.BinaryCIFFile.read(rcsb.fetch("8FAQ", "bcif"))
# Protein trimer with non-removed ions and non-covalent ligands 
pdbx_file_1PCR = pdbx.BinaryCIFFile.read(rcsb.fetch("1PCR", "bcif"))
# Protein-DNA complex
pdbx_file_1NVP = pdbx.BinaryCIFFile.read(rcsb.fetch("1NVP", "bcif"))
# Protein with modified residues
pdbx_file_3US4 = pdbx.BinaryCIFFile.read(rcsb.fetch("3US4", "bcif"))
# Protein RNA complex
pdbx_file_1A9N = pdbx.BinaryCIFFile.read(rcsb.fetch("1A9N", "bcif"))

biounit_8FAQ = pdbx.get_assembly(pdbx_file_8FAQ, assembly_id="1", model=1)
biounit_1PCR = pdbx.get_assembly(pdbx_file_1PCR, assembly_id="1", model=1)
biounit_1NVP = pdbx.get_assembly(pdbx_file_1NVP, assembly_id="1", model=1)
biounit_3US4 = pdbx.get_assembly(pdbx_file_3US4, assembly_id="1", model=1)
biounit_1A9N = pdbx.get_assembly(pdbx_file_1A9N, assembly_id="1", model=1)

# Remove waters
biounit_8FAQ = biounit_8FAQ[biounit_8FAQ.res_name != "HOH"]
biounit_1PCR = biounit_1PCR[biounit_1PCR.res_name != "HOH"]
biounit_1NVP = biounit_1NVP[biounit_1NVP.res_name != "HOH"]
biounit_3US4 = biounit_3US4[biounit_3US4.res_name != "HOH"]
biounit_1A9N = biounit_1A9N[biounit_1A9N.res_name != "HOH"]

#%%
biounit_8FAQ = chain_assignment(tokenize_atomarray(biounit_8FAQ))
biounit_1NVP = chain_assignment(tokenize_atomarray(biounit_1NVP))
biounit_3US4 = chain_assignment(tokenize_atomarray(biounit_3US4))
biounit_1A9N = chain_assignment(tokenize_atomarray(biounit_1A9N))

atomarray = biounit_1A9N
generator = np.random.default_rng(2346)
n_res = 384
chains = np.array(list(set(atomarray.af3_chain_id)), dtype=int)
chains = generator.permutation(chains)
print(chains)
#%%
# print("Number of chains:", struc.get_chain_count(biounit_3US4))
# struc.get_residues(biounit_3US4)

# strucio.save_structure("/mnt/c/Users/nikol/Documents/biotite_tests/1NVP.pdb", biounit_1NVP)


def contiguous_crop(atomarray: AtomArray, n_res: int, generator: Generator):
    """Implements Contiguous Cropping, Algorithm 1 from AF-Multimer section 7.2.1.

    Args:
        atomarray (AtomArray): biotite atom array of the first bioassembly of a PDB entry
        n_res (int): residue budget i.e. total crop size
        generator (Generator): a numpy generator set with a specific seed
    """

    # Get chain ids and permute
    chains = np.array(list(set(atomarray.af3_chain_id)), dtype=int)
    chains = generator.permutation(chains)

    # Create cropping mask annotation
    atomarray.set_annotation("af3_contiguous_crop_mask", np.repeat(False, len(atomarray)))

    # Cropping loop
    n_added = 0
    n_remaining = n_res
    for chain_id in chains:

        # TODO need to deal with atomic tokens including:
        # ligands - separate chain from polymer
        # ligands - same chain as poly
        # covalently modified residues

        atomarray_chain = atomarray[atomarray.af3_chain_id == chain_id]
        n_k = struc.get_residue_count(atomarray_chain)
        n_remaining -= n_k

        # Calculate crop length
        crop_size_max = np.min([n_res - n_added, n_k])
        crop_size_min = np.min([n_k, np.max([0, n_res - n_added - n_remaining])])
        crop_size = generator.integers(crop_size_min, crop_size_max + 1, 1).item()
        n_added += crop_size

        # Calculate crop start
        crop_start = generator.integers(0, n_k - crop_size + 1, 1).item()

        # Create mask - could also edit the AtomArray annotation directly but looks much less clean
        m_k = np.zeros(n_k, dtype=bool)
        m_k[crop_start:crop_start + crop_size] = True

        # Get atom indices of residue starting points for residues in crop
        residue_starts = struc.get_residue_starts(atomarray_chain)
        residue_starts_crop = residue_starts[crop_start:crop_start + crop_size]



    return

def spatial_crop():
    return


def spatial_interface_crop():
    return