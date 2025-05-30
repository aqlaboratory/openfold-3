import numpy as np
import pytest  # noqa: F401  - used for pytest tmp fixture
from biotite import structure
from biotite.structure.io import pdbx

from openfold3.core.runners.writer import write_structure_prediction


class TestPredictionWriter:
    def test_written_coordinates(self, tmp_path):
        atom1 = structure.Atom([1, 2, 3], chain_id="A")
        atom2 = structure.Atom([2, 3, 4], chain_id="A")
        atom3 = structure.Atom([3, 4, 5], chain_id="B")

        atom_array = structure.array([atom1, atom2, atom3])

        # add extra dimension for sample
        new_coords = np.array(
            [
                [
                    [2.0, 2.0, 2.0],
                    [3.5, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                ]
            ]
        )

        dummy_pdb_id = "TEST"
        dummy_seed = 42
        write_structure_prediction(
            dummy_seed, atom_array, new_coords, dummy_pdb_id, tmp_path
        )
        expected_dir = tmp_path / dummy_pdb_id / f"model_{dummy_seed}"
        expected_path = list(expected_dir.glob("*.cif"))[0]

        cif_file = pdbx.CIFFile.read(expected_path)
        parsed_structure = pdbx.get_structure(cif_file)
        parsed_coords = parsed_structure.coord[0]
        np.testing.assert_array_equal(parsed_coords, new_coords[0], strict=False)
