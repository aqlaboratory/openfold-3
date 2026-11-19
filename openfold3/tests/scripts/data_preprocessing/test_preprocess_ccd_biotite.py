# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from io import StringIO

import pytest
from biotite.structure.io.pdbx import BinaryCIFFile, CIFFile

from openfold3.core.data.resources.residues import STANDARD_RESIDUES_3
from scripts.data_preprocessing.preprocess_ccd_biotite import (
    CCD_CATEGORIES,
    build_minimal_ccd,
    concatenate_ccd,
    extract_ccd_components,
    read_component_ids_file,
)


def _ccd_block(component_id: str) -> str:
    return f"""data_{component_id}
_chem_comp.id {component_id}
_chem_comp.type 'NON-POLYMER'
_chem_comp.pdbx_synonyms
;
;
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
{component_id} C1 C 0
{component_id} O1 O 0
#
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
{component_id} C1 O1 SING N
#
"""


def test_build_minimal_ccd_selects_standard_and_custom_components(tmp_path):
    source_ccd = tmp_path / "components.cif"
    minimal_ccd = tmp_path / "minimal.cif"
    minimal_bcif = tmp_path / "minimal.bcif"
    source_ccd.write_text(
        "".join(_ccd_block(component_id) for component_id in STANDARD_RESIDUES_3)
        + _ccd_block("LIG")
        + _ccd_block("OMIT")
    )

    found_ids = build_minimal_ccd(
        source_ccd_path=source_ccd,
        output_ccd_path=minimal_ccd,
        custom_component_ids=["lig"],
    )

    expected_ids = set(STANDARD_RESIDUES_3) | {"LIG"}
    assert found_ids == expected_ids
    parsed_minimal_ccd = CIFFile.read(StringIO(minimal_ccd.read_text()))
    assert set(parsed_minimal_ccd.keys()) == expected_ids

    compressed_ccd = concatenate_ccd(minimal_ccd, categories=CCD_CATEGORIES)
    compressed_ccd.write(minimal_bcif)
    parsed_bcif = BinaryCIFFile.read(minimal_bcif)
    component_ids = parsed_bcif["components"]["chem_comp"]["id"].as_array()
    assert set(component_ids.tolist()) == expected_ids


def test_extract_ccd_components_preserves_existing_output_when_id_is_missing(
    tmp_path,
):
    source_ccd = tmp_path / "components.cif"
    output_ccd = tmp_path / "minimal.cif"
    source_ccd.write_text(_ccd_block("ALA"))
    output_ccd.write_text("existing output")

    with pytest.raises(
        ValueError,
        match="Requested CCD components were not found: MISSING",
    ):
        extract_ccd_components(source_ccd, output_ccd, ["ALA", "missing"])

    assert output_ccd.read_text() == "existing output"


def test_read_component_ids_file_ignores_blank_lines(tmp_path):
    component_ids_file = tmp_path / "component_ids.txt"
    component_ids_file.write_text("lig\n\nATP\nlig\n")

    assert read_component_ids_file(component_ids_file) == {"LIG", "ATP"}
