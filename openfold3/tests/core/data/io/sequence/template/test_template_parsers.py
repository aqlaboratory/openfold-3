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
import pandas as pd
import pytest

import openfold3
from openfold3.core.data.io.sequence.template import (
    A3mParser,
    CifDirectParser,
    M8Parser,
    StoParser,
    TemplateData,
)
from openfold3.core.data.io.structure.cif import _load_ciffile
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessor,
    TemplatePreprocessorInputInference,
)
from openfold3.core.data.primitives.structure.metadata import (
    get_asym_id_to_canonical_seq_dict,
)

TEST_DIR = (
    Path(openfold3.__file__).parent / "tests" / "test_data" / "template_alignments"
)

QUERY_SEQUENCE = """
MLNSFKLSLQYILPKLWLTRLAGWGASKRAGWLTKLVIDLFVKYYKVDMKEAQKPDTASYRTFNEFFVRPLRDEVRPIDTDPNVLV
MPADGVISQLGKIEEDKILQAKGHNYSLEALLAGNYLMADLFRNGTFVTTYLSPRDYHRVHMPCNGILREMIYVPGDLFSVNHLTA
QNVPNLFARNERVICLFDTEFGPMAQILVGATIVGSIETVWAGTITPPREGIIKRWTWPAGENDGSVALLKGQEMGRFKLG
""".replace("\n", "")


@pytest.mark.parametrize(
    "file_path, output_path, max_sequences",
    [
        (
            TEST_DIR / "inputs/sto_hmmalign.sto",
            TEST_DIR / "outputs/sto_hmmalign.npz",
            5,
        ),
        (
            TEST_DIR / "inputs/sto_hmmsearch_same_seq.sto",
            TEST_DIR / "outputs/sto_hmmsearch_same_seq.npz",
            5,
        ),
        (
            TEST_DIR / "inputs/sto_hmmsearch_diff_seq.sto",
            TEST_DIR / "outputs/sto_hmmsearch_diff_seq.npz",
            5,
        ),
    ],
)
def test_sto_parser(file_path, output_path, max_sequences):
    with open(file_path) as f:
        sto_string = f.read()
    sto_parser = StoParser(max_sequences=max_sequences)
    templates = sto_parser(sto_string, QUERY_SEQUENCE)
    expected_templates = np.load(output_path, allow_pickle=True)["templates"].item()
    assert len(templates) == len(expected_templates)
    for actual, expected in zip(
        templates.values(), expected_templates.values(), strict=False
    ):
        _compare_template_data(actual, expected)


@pytest.mark.parametrize(
    "file_path, output_path, max_sequences",
    [
        (
            TEST_DIR / "inputs/a3m_no_realign.a3m",
            TEST_DIR / "outputs/a3m_no_realign.npz",
            5,
        ),
        (
            TEST_DIR / "inputs/a3m_realign.a3m",
            TEST_DIR / "outputs/a3m_realign.npz",
            5,
        ),
    ],
)
def test_a3m_parser(file_path, output_path, max_sequences):
    with open(file_path) as f:
        a3m_string = f.read()
    a3m_parser = A3mParser(max_sequences=max_sequences)
    templates = a3m_parser(a3m_string, query_seq_str=QUERY_SEQUENCE)
    expected_templates = np.load(output_path, allow_pickle=True)["templates"].item()
    assert len(templates) == len(expected_templates)
    for actual, expected in zip(
        templates.values(), expected_templates.values(), strict=False
    ):
        _compare_template_data(actual, expected)


def test_m8_parser():
    file_path = TEST_DIR / "inputs/m8_cigar.m8"
    output_path = TEST_DIR / "outputs/m8_cigar.npz"
    max_sequences = 5

    m8_parser = M8Parser(max_sequences=max_sequences)
    m8_cigar = pd.read_csv(file_path, sep="\t", header=None)
    templates = m8_parser(m8_cigar, query_seq_str=QUERY_SEQUENCE)
    expected_templates = np.load(output_path, allow_pickle=True)["templates"].item()
    assert len(templates) == len(expected_templates)
    for actual, expected in zip(
        templates.values(), expected_templates.values(), strict=False
    ):
        _compare_template_data(actual, expected)

    m8_no_cigar = m8_cigar.loc[:, m8_cigar.columns != "cigar"].copy()
    output_path_no_cigar = TEST_DIR / "outputs/m8_no_cigar.npz"
    templates_no_cigar = m8_parser(m8_no_cigar, query_seq_str=QUERY_SEQUENCE)
    expected_templates_no_cigar = np.load(output_path_no_cigar, allow_pickle=True)[
        "templates"
    ].item()
    assert len(templates) == len(expected_templates)
    for actual, expected in zip(
        templates_no_cigar.values(), expected_templates_no_cigar.values(), strict=False
    ):
        _compare_template_data(actual, expected)


def _compare_template_data(actual, expected):
    for key in TemplateData._fields:
        v_actual = getattr(actual, key)
        v_expected = getattr(expected, key)
        if isinstance(v_actual, np.ndarray):
            np.testing.assert_array_equal(v_actual, v_expected)
        else:
            assert v_actual == v_expected


MMCIFS_DIR = Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs"


def _load_chain_id_seq_map(cif_path: Path) -> dict[str, str]:
    return get_asym_id_to_canonical_seq_dict(_load_ciffile(cif_path))


def test_cif_direct_parser_auto_select():
    """Auto-select picks the chain that matches the query sequence."""
    cif_path = MMCIFS_DIR / "2q2k.cif"
    chain_id_seq_map = _load_chain_id_seq_map(cif_path)
    # Chain B and C are identical 70-residue proteins; A is RNA. Query is chain B.
    query_seq_str = chain_id_seq_map["B"]

    parser = CifDirectParser(max_sequences=None, min_score_threshold=0.1)
    result = parser(
        cif_file_path=cif_path,
        query_seq_str=query_seq_str,
        chain_id_seq_map=chain_id_seq_map,
        entry_id="2q2k",
    )

    assert len(result) == 1
    template = result[0]
    assert template.entry_id == "2q2k"
    # Either B or C should win since they're identical; both score 1.0.
    assert template.chain_id in {"B", "C"}
    assert template.seq_id == pytest.approx(1.0)
    assert template.q_cov == pytest.approx(1.0)
    assert template.seq == query_seq_str


def test_cif_direct_parser_specified_chain():
    """specified_chain_id restricts parsing to one chain."""
    cif_path = MMCIFS_DIR / "2q2k.cif"
    chain_id_seq_map = _load_chain_id_seq_map(cif_path)
    query_seq_str = chain_id_seq_map["B"]

    parser = CifDirectParser(max_sequences=None, min_score_threshold=0.1)
    result = parser(
        cif_file_path=cif_path,
        query_seq_str=query_seq_str,
        chain_id_seq_map=chain_id_seq_map,
        entry_id="2q2k",
        specified_chain_id="C",
    )

    assert len(result) == 1
    assert result[0].chain_id == "C"
    assert result[0].seq_id == pytest.approx(1.0)


def test_cif_direct_parser_below_threshold_returns_empty():
    """No chain passes the threshold -> empty dict."""
    cif_path = MMCIFS_DIR / "2q2k.cif"
    chain_id_seq_map = _load_chain_id_seq_map(cif_path)
    # Unrelated sequence well below threshold against any chain in the CIF.
    query_seq_str = "WWWWWWWWWWWWWWWWWWWW"

    parser = CifDirectParser(max_sequences=None, min_score_threshold=0.9)
    result = parser(
        cif_file_path=cif_path,
        query_seq_str=query_seq_str,
        chain_id_seq_map=chain_id_seq_map,
        entry_id="2q2k",
    )

    assert result == {}


def _make_bare_preprocessor(cif_direct_min_score: float = 0.1) -> TemplatePreprocessor:
    """Build a TemplatePreprocessor with only the fields _parse_templates_from_cif_files reads."""
    pre = object.__new__(TemplatePreprocessor)
    pre.create_logs = False
    pre.cif_direct_min_score = cif_direct_min_score
    pre.precache_directory = None
    return pre


def test_parse_templates_from_cif_files_auto_select():
    """_parse_templates_from_cif_files returns one TemplateData per CIF, indexed and tagged with cif_path."""
    cif_path = MMCIFS_DIR / "2q2k.cif"
    query_seq_str = _load_chain_id_seq_map(cif_path)["B"]

    pre = _make_bare_preprocessor(cif_direct_min_score=0.1)
    input_data = TemplatePreprocessorInputInference(
        query_seq_str=query_seq_str,
        template_cif_paths=[cif_path],
    )

    templates = pre._parse_templates_from_cif_files(input_data)

    assert len(templates) == 1
    template = templates[0]
    assert template.index == 0
    assert template.entry_id == "2q2k"
    assert template.chain_id in {"B", "C"}
    assert template.cif_path == cif_path
    assert template.seq_id == pytest.approx(1.0)
    assert template.q_cov == pytest.approx(1.0)


def test_parse_templates_from_cif_files_specified_chain_ids():
    """When template_cif_chain_ids is provided, only the specified chain is used."""
    cif_path = MMCIFS_DIR / "2q2k.cif"
    query_seq_str = _load_chain_id_seq_map(cif_path)["B"]

    pre = _make_bare_preprocessor(cif_direct_min_score=0.1)
    input_data = TemplatePreprocessorInputInference(
        query_seq_str=query_seq_str,
        template_cif_paths=[cif_path],
        template_cif_chain_ids=["C"],
    )

    templates = pre._parse_templates_from_cif_files(input_data)

    assert len(templates) == 1
    assert templates[0].chain_id == "C"


def test_parse_templates_from_cif_files_missing_cif_skipped():
    """Missing CIF paths are skipped rather than raising."""
    pre = _make_bare_preprocessor(cif_direct_min_score=0.1)
    input_data = TemplatePreprocessorInputInference(
        query_seq_str="ACDEFGHIKLMNPQRSTVWY",
        template_cif_paths=[Path("/nonexistent/path/does_not_exist.cif")],
    )

    templates = pre._parse_templates_from_cif_files(input_data)

    assert templates == {}
