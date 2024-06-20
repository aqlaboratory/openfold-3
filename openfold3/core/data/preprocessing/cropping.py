# TODO add license

# Some biotite examples

import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure import AtomArray

# Protein trimer with non-covalent ligands
pdbx_file_8FAQ = pdbx.BinaryCIFFile.read(rcsb.fetch("8FAQ", "bcif"))
# Protein-DNA complex
pdbx_file_1NVP = pdbx.BinaryCIFFile.read(rcsb.fetch("1NVP", "bcif"))
# Protein with modified residues
pdbx_file_3US4 = pdbx.BinaryCIFFile.read(rcsb.fetch("3US4", "bcif"))
# Protein RNA complex
pdbx_file_1A9N = pdbx.BinaryCIFFile.read(rcsb.fetch("1A9N", "bcif"))


pdbx.get_sequence(pdbx_file_8FAQ)
pdbx.get_sequence(pdbx_file_1NVP)
pdbx.get_sequence(pdbx_file_3US4)

biounit_8FAQ = pdbx.get_assembly(pdbx_file_8FAQ, assembly_id="1", model=1)
biounit_1NVP = pdbx.get_assembly(pdbx_file_1NVP, assembly_id="1", model=1)
biounit_3US4 = pdbx.get_assembly(pdbx_file_3US4, assembly_id="1", model=1)
biounit_1A9N = pdbx.get_assembly(pdbx_file_1A9N, assembly_id="1", model=1)

# Remove waters
biounit_8FAQ_noHOH = biounit_8FAQ[biounit_8FAQ.res_name != "HOH"]
biounit_1NVP_noHOH = biounit_1NVP[biounit_1NVP.res_name != "HOH"]
biounit_3US4_noHOH = biounit_3US4[biounit_3US4.res_name != "HOH"]
biounit_1A9N_noHOH = biounit_1A9N[biounit_1A9N.res_name != "HOH"]


# print("Number of chains:", struc.get_chain_count(biounit_3US4))
# struc.get_residues(biounit_3US4_noHOH)

# strucio.save_structure("/mnt/c/Users/nikol/Documents/biotite_tests/1NVP_noHOH.pdb", biounit_1NVP_noHOH)


def get_token_start_ids(atomarray: AtomArray):
    return


def contiguous_crop(atomarray: AtomArray, n_res: int, seed: int):
    """Implements Contiguous Cropping Algorithm 1 from AF-Multimer section 7.2.1.

    Args:
        atomarray (AtomArray): biotite atom array of the first bioassembly of a PDB entry
        n_res (int): residue budget i.e. crop size
        seed (int): seed used for chain permutation and crop sampling
    """
    # Create numpy generator
    generator = np.random.default_rng(seed)

    # Get chain starts and lengths
    chain_start_ids = struc.get_chain_starts(atomarray)
    chain_lens = 0

    n_chains = len(chain_start_ids)
    n_added = 0
    n_remaining = n_res
    for chain_start_id_k in chain_start_ids:
        n_remaining -= 0

    return
