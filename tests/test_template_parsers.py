# Copyright 2025 AlQuraishi Laboratory
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

from openfold3.core.data.io.sequence.template import (
    A3mParser,
    M8Parser,
    StoParser,
    TemplateData,
)

TEST_DIR = Path("tests/test_data/template_alignments")

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
