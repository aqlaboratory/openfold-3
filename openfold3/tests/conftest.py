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
    atom_array.chain_id = np.array(["A", "A", "B", "B", "B"])
    return atom_array


@pytest.fixture(scope="session", autouse=True)
def ensure_biotite_ccd():
    """Download CCD file before any tests run (once per test session)."""
    setup_biotite_ccd()
