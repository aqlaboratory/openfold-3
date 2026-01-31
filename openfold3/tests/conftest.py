import numpy as np
import pytest
from biotite.structure import AtomArray

from openfold3.setup_openfold import setup_biotite_ccd


@pytest.fixture
def dummy_atom_array():
    # Create dummy atom array
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [2.4, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.4, 0.0, 0.0],
        ],
        dtype=float,
    )
    atom_array = AtomArray(len(coords))
    atom_array.coord = coords
    return atom_array


@pytest.fixture
def mse_ala_atom_array():
    """AtomArray with one MSE residue and one ALA residue for testing MSE->MET conversion."""
    # MSE has a selenium atom (SE), ALA is a simple residue for comparison
    # MSE atoms: N, CA, C, O, CB, CG, SE, CE (8 atoms)
    # ALA atoms: N, CA, C, O, CB (5 atoms)
    n_atoms = 13
    atom_array = AtomArray(n_atoms)

    atom_array.coord = np.zeros((n_atoms, 3))

    # MSE residue (res_id=1)
    atom_array.chain_id[:8] = "A"
    atom_array.res_id[:8] = 1
    atom_array.res_name[:8] = "MSE"
    atom_array.atom_name[:8] = ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"]
    atom_array.element[:8] = ["N", "C", "C", "O", "C", "C", "SE", "C"]
    atom_array.hetero[:8] = True

    # ALA residue (res_id=2)
    atom_array.chain_id[8:] = "A"
    atom_array.res_id[8:] = 2
    atom_array.res_name[8:] = "ALA"
    atom_array.atom_name[8:] = ["N", "CA", "C", "O", "CB"]
    atom_array.element[8:] = ["N", "C", "C", "O", "C"]
    atom_array.hetero[8:] = False

    return atom_array


@pytest.fixture(scope="session", autouse=True)
def ensure_biotite_ccd():
    """Download CCD file before any tests run (once per test session)."""
    setup_biotite_ccd(force_download=False)
