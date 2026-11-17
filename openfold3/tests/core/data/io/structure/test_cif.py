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

"""Tests for the mmCIF parsers in `cif.py`."""

from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest

import openfold3
from openfold3.core.data.io.structure.cif import (
    _load_ciffile,
    parse_RNA_monomer_cif_tmp,
)


def _write_rna_cif_with_author_numbering(path):
    """Writes a GAC trinucleotide numbered 1-3 in label_seq_id and 101-103 in
    auth_seq_id."""
    residues = []
    for res_id, res_name in enumerate(["G", "A", "C"], start=1):
        residue = struc.info.residue(res_name)
        residue.res_id[:] = res_id
        residue.chain_id[:] = "A"
        residues.append(residue)
    rna = struc.concatenate(residues)
    rna.set_annotation("occupancy", np.ones(len(rna)))

    cif_file = pdbx.CIFFile()
    pdbx.set_structure(cif_file, rna, data_block="rna")
    atom_site = cif_file.block["atom_site"]
    atom_site["auth_seq_id"] = atom_site["label_seq_id"].as_array(int) + 100
    cif_file.write(path)


def test_parse_rna_monomer_keeps_author_residue_numbering(tmp_path):
    """Biotite 2 reads label fields by default; the parser must keep reading the
    author fields it has always used."""
    cif_path = tmp_path / "rna.cif"
    _write_rna_cif_with_author_numbering(cif_path)

    atom_array = parse_RNA_monomer_cif_tmp(cif_path).atom_array

    np.testing.assert_array_equal(np.unique(atom_array.res_id), [101, 102, 103])


TEMPLATE_STRUCTURES_DIR = (
    Path(openfold3.__file__).parent
    / "tests"
    / "test_data"
    / "template_preprocessing"
    / "1fdl"
    / "template_structures"
)


@pytest.mark.parametrize("as_string", [False, True])
def test_load_ciffile_reads_gzipped_cif(as_string):
    """Compressed CIF templates can be loaded from paths and strings."""
    compressed_cif = TEMPLATE_STRUCTURES_DIR / "1fdl.cif.gz"
    path = str(compressed_cif) if as_string else compressed_cif

    cif_file = _load_ciffile(path)

    assert isinstance(cif_file, pdbx.CIFFile)
    assert "atom_site" in cif_file.block
    assert "pdbx_poly_seq_scheme" in cif_file.block
