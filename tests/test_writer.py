import numpy as np
import pytest  # noqa: F401  - used for pytest tmp fixture
from biotite import structure
from biotite.structure.io import pdb, pdbx

from openfold3.core.runners.writer import OF3OutputWriter


class TestPredictionWriter:
    @pytest.mark.parametrize(
        "structure_format",
        ["pdb", "cif"],
        ids=lambda x: x,
    )
    def test_written_coordinates(self, tmp_path, structure_format):
        atom1 = structure.Atom([1, 2, 3], chain_id="A")
        atom2 = structure.Atom([2, 3, 4], chain_id="A")
        atom3 = structure.Atom([3, 4, 5], chain_id="B")

        atom_array = structure.array([atom1, atom2, atom3])

        # add extra dimension for sample
        new_coords = np.array(
            [
                [2.0, 2.0, 2.0],
                [3.5, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )
        dummy_plddt = np.array([0.9, 0.8, 0.7])

        output_writer = OF3OutputWriter(
            output_dir=tmp_path,
            structure_format=structure_format,
            full_confidence_out_format="json",
        )
        tmp_file = tmp_path / f"TEST.{structure_format}"
        output_writer.write_structure_prediction(
            atom_array, new_coords, dummy_plddt, tmp_file
        )

        match structure_format:
            case "cif":
                read_file = pdbx.CIFFile.read(tmp_file)
                parsed_structure = pdbx.get_structure(read_file)

            case "pdb":
                parsed_structure = pdb.PDBFile.read(tmp_file).get_structure()

        parsed_coords = parsed_structure.coord[0]
        np.testing.assert_array_equal(parsed_coords, new_coords, strict=False)
