from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx import CIFFile

from openfold3.core.data.io.structure.cif import parse_mmcif

MMCIF_DIR = Path(__file__).parents[3] / "test_data" / "mmcifs"


def test_parse_mmcif_normalizes_uppercase_element_symbols(tmp_path):
    cif_file = CIFFile.read(MMCIF_DIR / "1ubq.cif")
    element_symbols = (
        cif_file.block["atom_site"]["type_symbol"].as_array().astype("<U2")
    )
    element_symbols[:3] = ["BR", "CL", "FE"]
    cif_file.block["atom_site"]["type_symbol"] = element_symbols

    input_path = tmp_path / "uppercase_elements.cif"
    cif_file.write(input_path)

    parsed_structure = parse_mmcif(input_path, include_bonds=False)

    assert parsed_structure.atom_array is not None
    np.testing.assert_array_equal(
        parsed_structure.atom_array.element[:3],
        np.array(["Br", "Cl", "Fe"]),
    )
