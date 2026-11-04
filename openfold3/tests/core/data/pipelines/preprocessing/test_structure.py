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

from pathlib import Path

import numpy as np
import pytest

import openfold3
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.pipelines.preprocessing.structure import (
    _discover_cif_files,
    cleanup_structure_of3,
)
from openfold3.core.data.primitives.structure.metadata import get_cif_block

MMCIFS_DIR = Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs"


def test_discover_cif_files_recursively_in_deterministic_order(tmp_path):
    nested_dir = tmp_path / "nested" / "entry"
    nested_dir.mkdir(parents=True)

    top_level_cif = tmp_path / "2def.cif"
    nested_cif = nested_dir / "1abc.cif"
    ignored_file = nested_dir / "notes.txt"
    top_level_cif.touch()
    nested_cif.touch()
    ignored_file.touch()

    assert _discover_cif_files(tmp_path) == [top_level_cif, nested_cif]


def test_discover_cif_files_rejects_duplicate_stems(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "1abc.cif").touch()
    (second_dir / "1abc.cif").touch()

    with pytest.raises(
        ValueError,
        match=r"unique stems.*1abc: first/1abc\.cif, second/1abc\.cif",
    ):
        _discover_cif_files(tmp_path)


@pytest.fixture(scope="module")
def parsed_7l39():
    """7l39 (T4 lysozyme L99A with toluene, CCD code MBN) parsed as in preprocessing."""
    cif_file, atom_array = parse_mmcif(
        MMCIFS_DIR / "7l39.cif",
        expand_bioassembly=True,
        include_bonds=True,
        renumber_chain_ids=True,
    )
    return get_cif_block(cif_file), atom_array


def test_cleanup_structure_of3_keeps_ligand_that_matches_ccd(
    parsed_7l39, biotite_ccd_wrapper
):
    cif_data, atom_array = parsed_7l39

    cleaned = cleanup_structure_of3(atom_array, cif_data, biotite_ccd_wrapper)

    toluene = cleaned[cleaned.res_name == "MBN"]
    assert toluene.atom_name.tolist() == ["C", "C1", "C2", "C3", "C4", "C5", "C6"]
    assert not np.isnan(toluene.coord).any()


@pytest.mark.parametrize(
    ("res_name", "methyl_carbon_name", "error_match"),
    [
        # TOL is tolrestat in the CCD, which has atoms C1-C6 but no atom named C
        pytest.param(
            "TOL",
            "C",
            r"\('4', 203, 'TOL', \['C'\]\)",
            id="custom_ligand_reuses_ccd_code",
        ),
        pytest.param(
            "MBN",
            "C7",
            r"\('4', 203, 'MBN', \['C7'\]\)",
            id="ligand_atom_name_not_in_ccd_entry",
        ),
        pytest.param(
            "LIG1",
            "C",
            r"Residue LIG1 \(chain 4, residue 203\) is not in the CCD",
            id="ligand_code_not_in_ccd",
        ),
    ],
)
def test_cleanup_structure_of3_rejects_ligand_that_does_not_match_ccd(
    parsed_7l39, biotite_ccd_wrapper, res_name, methyl_carbon_name, error_match
):
    cif_data, atom_array = parsed_7l39
    atom_array = atom_array.copy()
    toluene = atom_array.res_name == "MBN"
    atom_array.atom_name[toluene & (atom_array.atom_name == "C")] = methyl_carbon_name
    atom_array.res_name[toluene] = res_name

    with pytest.raises(ValueError, match=error_match):
        cleanup_structure_of3(atom_array, cif_data, biotite_ccd_wrapper)
