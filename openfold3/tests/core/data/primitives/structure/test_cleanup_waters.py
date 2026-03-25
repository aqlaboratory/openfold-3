import numpy as np
from biotite.structure import (
    AtomArray,
)

from openfold3.core.data.primitives.structure.cleanup import (
    remove_waters,
)


def test_remove_waters(dummy_atom_array: AtomArray):
    dummy_atom_array.res_name = np.array(["ALA", "GLY", "HOH", "DOD", "NA+"])
    output_atom_array = remove_waters(dummy_atom_array)
    assert len(output_atom_array) == 3
    assert output_atom_array.res_name.tolist() == ["ALA", "GLY", "NA+"]
