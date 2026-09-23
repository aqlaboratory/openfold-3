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

"""Unit tests for the multistate-kinase notebook helpers.

Not auto-collected: the repo pins ``testpaths = ["openfold3/tests"]``, so run
this file by explicit path, e.g.::

    pytest examples/notebooks/2026-multistate_kinase_generation/test_helpers.py
"""

import helpers
import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# sanitize_sequence
# --------------------------------------------------------------------------- #
sanitize_sequence_cases = [
    {"label": "lowercase_uppercased", "raw": "acgt", "expected": "ACGT"},
    {
        "label": "strips_whitespace_and_newlines",
        "raw": "AC GT\nHI",
        "expected": "ACGTHI",
    },
    {"label": "strips_digits_and_punct", "raw": ">seq1\nAC-D9E", "expected": "SEQACDE"},
    {"label": "empty", "raw": "", "expected": ""},
]


@pytest.mark.parametrize("case", sanitize_sequence_cases, ids=lambda c: c["label"])
def test_sanitize_sequence(case):
    assert helpers.sanitize_sequence(case["raw"]) == case["expected"]


# --------------------------------------------------------------------------- #
# sanitize_jobname
# --------------------------------------------------------------------------- #
sanitize_jobname_cases = [
    {"label": "keeps_allowed_chars", "raw": "Job_1-abc", "expected": "Job_1-abc"},
    {
        "label": "spaces_and_slashes_to_underscore",
        "raw": "my job/run",
        "expected": "my_job_run",
    },
    {"label": "special_chars", "raw": "a.b@c!d", "expected": "a_b_c_d"},
]


@pytest.mark.parametrize("case", sanitize_jobname_cases, ids=lambda c: c["label"])
def test_sanitize_jobname(case):
    assert helpers.sanitize_jobname(case["raw"]) == case["expected"]


# --------------------------------------------------------------------------- #
# parse_ffindex_entries
# --------------------------------------------------------------------------- #
parse_ffindex_cases = [
    {
        "label": "three_field_lines_kept",
        "text": "1BLX_A.a3m\t0\t512\n2XYZ_B.a3m\t512\t480\n",
        "expected": ["1BLX_A.a3m", "2XYZ_B.a3m"],
    },
    {
        "label": "short_lines_ignored",
        "text": "1BLX_A.a3m\t0\t512\ngarbage\n2XYZ_B.a3m\t512\n",
        "expected": ["1BLX_A.a3m"],
    },
    {"label": "empty", "text": "", "expected": []},
]


@pytest.mark.parametrize("case", parse_ffindex_cases, ids=lambda c: c["label"])
def test_parse_ffindex_entries(case):
    assert helpers.parse_ffindex_entries(case["text"]) == case["expected"]


# --------------------------------------------------------------------------- #
# normalize_tid
# --------------------------------------------------------------------------- #
normalize_tid_cases = [
    {"label": "strips_a3m_suffix", "raw": "1BLX_A.a3m", "expected": "1BLX_A"},
    {"label": "no_suffix_unchanged", "raw": "1BLX_A", "expected": "1BLX_A"},
    {"label": "strips_any_ext", "raw": "2XYZ_B.hhm", "expected": "2XYZ_B"},
]


@pytest.mark.parametrize("case", normalize_tid_cases, ids=lambda c: c["label"])
def test_normalize_tid(case):
    assert helpers.normalize_tid(case["raw"]) == case["expected"]


# --------------------------------------------------------------------------- #
# parse_template_id
# --------------------------------------------------------------------------- #
parse_template_id_cases = [
    {"label": "four_char_entry", "tid": "1BLX_A", "expected": ("1BLX", "A")},
    {"label": "five_char_entry", "tid": "1ABCD_B", "expected": ("1ABCD", "B")},
    {"label": "numeric_chain", "tid": "1BLX_12", "expected": ("1BLX", "12")},
    {"label": "no_underscore_rejected", "tid": "1BLXA", "expected": None},
    {"label": "too_short_rejected", "tid": "1B_A", "expected": None},
]


@pytest.mark.parametrize("case", parse_template_id_cases, ids=lambda c: c["label"])
def test_parse_template_id(case):
    assert helpers.parse_template_id(case["tid"]) == case["expected"]


# --------------------------------------------------------------------------- #
# extract_protein_sequence
# --------------------------------------------------------------------------- #
_LONG_AA = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 standard residues

extract_sequence_cases = [
    {
        "label": "picks_longest_aa_block",
        "cif_text": f"\n;ACDEF\n;\nfoo\n;{_LONG_AA}\n;\n",
        "expected": _LONG_AA,
    },
    {
        "label": "ignores_non_protein_block",
        "cif_text": f"\n;12345 67890\n;\n;{_LONG_AA}\n;\n",
        "expected": _LONG_AA,
    },
    {
        "label": "rejects_low_std_fraction",
        "cif_text": "\n;XXXXXXXXXXBBZZ\n;\n",
        "expected": "",
    },
    {"label": "no_blocks", "cif_text": "loop_\n_atom_site.id\n1\n2\n", "expected": ""},
]


@pytest.mark.parametrize("case", extract_sequence_cases, ids=lambda c: c["label"])
def test_extract_protein_sequence(case):
    assert helpers.extract_protein_sequence(case["cif_text"]) == case["expected"]


# --------------------------------------------------------------------------- #
# kmer_set
# --------------------------------------------------------------------------- #
kmer_set_cases = [
    {"label": "basic_k4", "seq": "ABCDE", "k": 4, "expected": {"ABCD", "BCDE"}},
    {"label": "exactly_k", "seq": "ABCD", "k": 4, "expected": {"ABCD"}},
    {"label": "shorter_than_k", "seq": "ABC", "k": 4, "expected": set()},
    {"label": "k3", "seq": "AAAA", "k": 3, "expected": {"AAA"}},
]


@pytest.mark.parametrize("case", kmer_set_cases, ids=lambda c: c["label"])
def test_kmer_set(case):
    assert helpers.kmer_set(case["seq"], case["k"]) == case["expected"]


# --------------------------------------------------------------------------- #
# kmer_overlap
# --------------------------------------------------------------------------- #
kmer_overlap_cases = [
    {
        "label": "identical_is_one",
        "query": "ABCDEFGH",
        "target": "ABCDEFGH",
        "expected": 1.0,
    },
    {
        "label": "disjoint_is_zero",
        "query": "AAAAAAAA",
        "target": "CCCCCCCC",
        "expected": 0.0,
    },
    {
        "label": "target_shorter_than_k",
        "query": "ABCDEFGH",
        "target": "ABC",
        "expected": 0.0,
    },
    {
        "label": "partial_overlap",
        "query": "ABCDE",
        "target": "ABCDXYZ",
        "expected": 0.5,
    },
]


@pytest.mark.parametrize("case", kmer_overlap_cases, ids=lambda c: c["label"])
def test_kmer_overlap(case):
    q_kmers = helpers.kmer_set(case["query"], 4)
    assert helpers.kmer_overlap(q_kmers, case["target"], 4) == pytest.approx(
        case["expected"]
    )


# --------------------------------------------------------------------------- #
# rank_templates_by_kmer
# --------------------------------------------------------------------------- #
rank_cases = [
    {
        "label": "closest_homolog_ranks_first",
        "query": "ABCDEFGH",
        "pairs": [("far", "WXYZWXYZ"), ("near", "ABCDEFGH"), ("mid", "ABCDwxyz")],
        "top_n": None,
        "expected_order": ["near", "mid", "far"],
    },
    {
        "label": "top_n_truncates",
        "query": "ABCDEFGH",
        "pairs": [("far", "WXYZWXYZ"), ("near", "ABCDEFGH"), ("mid", "ABCDMMMM")],
        "top_n": 2,
        "expected_order": ["near", "mid"],
    },
    {
        "label": "empty_seqs_skipped",
        "query": "ABCDEFGH",
        "pairs": [("near", "ABCDEFGH"), ("blank", "")],
        "top_n": None,
        "expected_order": ["near"],
    },
]


@pytest.mark.parametrize("case", rank_cases, ids=lambda c: c["label"])
def test_rank_templates_by_kmer(case):
    ranked = helpers.rank_templates_by_kmer(
        case["query"], case["pairs"], k=4, top_n=case["top_n"]
    )
    assert [tid for tid, _ in ranked] == case["expected_order"]


# --------------------------------------------------------------------------- #
# parse_hhr_hits
# --------------------------------------------------------------------------- #
_HHR = (
    "Query         EphA3\n"
    "Match_columns 300\n"
    "\n"
    " No Hit                             Prob E-value P-value  Score\n"
    "  1 1BLX_A.a3m                       99.9 1.2E-30 3.0E-35  120.5\n"
    "  2 2XYZ_B.a3m                       88.5 2.4E-20 6.0E-25   90.1\n"
    "  3 3ABC_C.a3m                       70.0 5.0E-10 1.0E-14   60.0\n"
    "\n"
    "No 1\n"
    ">1BLX_A\n"
)

parse_hhr_cases = [
    {
        "label": "all_hits",
        "max_hits": None,
        "expected": [
            ("1BLX_A", 99.9, "1.2E-30"),
            ("2XYZ_B", 88.5, "2.4E-20"),
            ("3ABC_C", 70.0, "5.0E-10"),
        ],
    },
    {
        "label": "max_hits_caps",
        "max_hits": 2,
        "expected": [("1BLX_A", 99.9, "1.2E-30"), ("2XYZ_B", 88.5, "2.4E-20")],
    },
    {
        "label": "no_table",
        "text": "Query x\nMatch_columns 3\n",
        "max_hits": None,
        "expected": [],
    },
]


@pytest.mark.parametrize("case", parse_hhr_cases, ids=lambda c: c["label"])
def test_parse_hhr_hits(case):
    text = case.get("text", _HHR)
    assert helpers.parse_hhr_hits(text, case["max_hits"]) == case["expected"]


# --------------------------------------------------------------------------- #
# build_query_dict
# --------------------------------------------------------------------------- #
def test_build_query_dict():
    d = helpers.build_query_dict(
        "job1",
        "ACDEF",
        ["a.cif", "b.cif"],
        ["A", "B"],
    )
    chain = d["queries"]["job1"]["chains"][0]
    assert chain["molecule_type"] == "protein"
    assert chain["chain_ids"] == ["A"]
    assert chain["sequence"] == "ACDEF"
    assert chain["template_cif_paths"] == ["a.cif", "b.cif"]
    assert chain["template_cif_chain_ids"] == ["A", "B"]


# --------------------------------------------------------------------------- #
# build_runner_yaml
# --------------------------------------------------------------------------- #
build_runner_cases = [
    {"label": "default_no_low_mem", "use_low_mem": False, "has_low_mem": False},
    {"label": "low_mem_appended", "use_low_mem": True, "has_low_mem": True},
]


@pytest.mark.parametrize("case", build_runner_cases, ids=lambda c: c["label"])
def test_build_runner_yaml(case):
    text = helpers.build_runner_yaml(use_low_mem=case["use_low_mem"])
    assert "structure_format: pdb" in text
    assert "cif_direct_min_score: 0.05" in text
    assert "use_triton_triangle_kernels: true" in text
    assert "use_deepspeed_evo_attention: false" in text
    assert "- predict" in text
    assert ("- low_mem" in text) is case["has_low_mem"]


def test_build_runner_yaml_custom_score():
    text = helpers.build_runner_yaml(cif_direct_min_score=0.1, structure_format="mmcif")
    assert "cif_direct_min_score: 0.1" in text
    assert "structure_format: mmcif" in text


# --------------------------------------------------------------------------- #
# build_inference_command
# --------------------------------------------------------------------------- #
build_cmd_cases = [
    {"label": "msa_server_on_no_flag", "use_msa_server": True, "has_flag": False},
    {"label": "msa_server_off_adds_flag", "use_msa_server": False, "has_flag": True},
]


@pytest.mark.parametrize("case", build_cmd_cases, ids=lambda c: c["label"])
def test_build_inference_command(case):
    cmd = helpers.build_inference_command(
        "q.json", "out", 5, 1, "runner.yml", use_msa_server=case["use_msa_server"]
    )
    assert cmd[:2] == ["run_openfold", "predict"]
    assert "--query-json" in cmd and cmd[cmd.index("--query-json") + 1] == "q.json"
    assert "--num-diffusion-samples" in cmd
    assert ("--use-msa-server=False" in cmd) is case["has_flag"]


# --------------------------------------------------------------------------- #
# extract_plddt / mean_plddt
# --------------------------------------------------------------------------- #
plddt_cases = [
    {
        "label": "plddt_key",
        "conf": {"plddt": [10.0, 20.0]},
        "expected_list": [10.0, 20.0],
        "expected_mean": 15.0,
    },
    {
        "label": "residue_plddts_fallback",
        "conf": {"residue_plddts": [30.0]},
        "expected_list": [30.0],
        "expected_mean": 30.0,
    },
    {
        "label": "atom_plddts_fallback",
        "conf": {"atom_plddts": [40.0, 60.0]},
        "expected_list": [40.0, 60.0],
        "expected_mean": 50.0,
    },
    {
        "label": "missing_is_empty",
        "conf": {"ptm": 0.9},
        "expected_list": [],
        "expected_mean": 0.0,
    },
]


@pytest.mark.parametrize("case", plddt_cases, ids=lambda c: c["label"])
def test_extract_and_mean_plddt(case):
    assert helpers.extract_plddt(case["conf"]) == case["expected_list"]
    assert helpers.mean_plddt(case["conf"]) == pytest.approx(case["expected_mean"])


# --------------------------------------------------------------------------- #
# extract_pae
# --------------------------------------------------------------------------- #
extract_pae_cases = [
    {
        "label": "predicted_aligned_error",
        "conf": {"predicted_aligned_error": [[1.0]]},
        "expected": [[1.0]],
    },
    {"label": "pae_fallback", "conf": {"pae": [[2.0]]}, "expected": [[2.0]]},
    {
        "label": "aligned_confidence_probs_fallback",
        "conf": {"aligned_confidence_probs": [[3.0]]},
        "expected": [[3.0]],
    },
    {"label": "missing_is_none", "conf": {"plddt": [1.0]}, "expected": None},
]


@pytest.mark.parametrize("case", extract_pae_cases, ids=lambda c: c["label"])
def test_extract_pae(case):
    assert helpers.extract_pae(case["conf"]) == case["expected"]


# --------------------------------------------------------------------------- #
# collapse_pae
# --------------------------------------------------------------------------- #
def test_collapse_pae_2d_unchanged():
    pae = [[0.0, 5.0], [5.0, 0.0]]
    out = helpers.collapse_pae(pae)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, np.asarray(pae))


def test_collapse_pae_3d_to_expected():
    # (2, 2, 4) probability tensor -> expected PAE via bin centers.
    probs = np.zeros((2, 2, 4))
    probs[..., 0] = 1.0  # all mass on the first bin
    out = helpers.collapse_pae(probs)
    assert out.shape == (2, 2)
    bin_centers = np.linspace(0.5, 31.5, 4)
    np.testing.assert_allclose(out, np.full((2, 2), bin_centers[0]))
