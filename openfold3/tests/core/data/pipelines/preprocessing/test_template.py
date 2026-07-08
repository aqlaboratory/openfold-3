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

"""Tests for the easily-isolated seams of the template preprocessing pipeline.

Scope (see plan): the side-effect-free helpers, the pydantic validators, the no-IO
instance methods (built via ``object.__new__`` to bypass ``__init__``'s multiprocessing
and file IO), and the legacy TSV log helpers. The heavy ``__call__`` /
``_preprocess_templates_for_query`` orchestrators (``mp.Pool``, ``func_timeout``,
network ``fetch``) are intentionally out of scope.
"""

from datetime import datetime
from pathlib import Path

import pytest

from openfold3.core.data.io.sequence.template import TemplateData
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessor,
    TemplatePreprocessorInputInference,
    TemplatePreprocessorSettings,
    collate_data_logs,
    data_log_to_tsv,
    fails_template_release_date_checks,
    fails_template_sequence_checks,
    match_template_seq_from_aln_to_struc,
    remap_template_chain_id,
)
from openfold3.core.data.primitives.sequence.hash import get_sequence_hash
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Chain,
    InferenceQuerySet,
    Query,
)


def _make_template(
    *,
    entry_id: str = "1abc",
    chain_id: str = "A",
    seq: str = "ACDEFGHIK",
    seq_id: float = 0.5,
    q_cov: float | None = 0.5,
) -> TemplateData:
    """Minimal TemplateData for the pure-logic checks (only a few fields are read)."""
    return TemplateData(
        index=0,
        entry_id=entry_id,
        chain_id=chain_id,
        query_aln_pos=None,
        aln_pos=None,
        seq_id=seq_id,
        q_cov=q_cov,
        seq=seq,
    )


# ---------------------------------------------------------------------------
# Tier A: pure logic functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template, max_seq_id, min_align, min_len, expected",
    [
        pytest.param(
            _make_template(), None, None, None, False, id="all_thresholds_none_passes"
        ),
        pytest.param(
            _make_template(seq_id=0.9),
            0.8,
            None,
            None,
            True,
            id="seq_id_over_max_fails",
        ),
        pytest.param(
            _make_template(seq_id=0.8),
            0.8,
            None,
            None,
            False,
            id="seq_id_equals_max_passes",
        ),
        pytest.param(
            _make_template(q_cov=0.4),
            None,
            0.5,
            None,
            True,
            id="q_cov_below_min_align_fails",
        ),
        pytest.param(
            _make_template(q_cov=0.5),
            None,
            0.5,
            None,
            False,
            id="q_cov_equals_min_align_passes",
        ),
        pytest.param(
            _make_template(seq="ACD"),
            None,
            None,
            5,
            True,
            id="seq_shorter_than_min_len_fails",
        ),
        pytest.param(
            _make_template(seq="ACDEF"),
            None,
            None,
            5,
            False,
            id="seq_equals_min_len_passes",
        ),
        pytest.param(
            _make_template(seq_id=0.9, q_cov=0.4, seq="ACD"),
            0.8,
            0.5,
            5,
            True,
            id="multiple_thresholds_tripped_fails",
        ),
    ],
)
def test_fails_template_sequence_checks(
    template, max_seq_id, min_align, min_len, expected
):
    assert (
        fails_template_sequence_checks(template, max_seq_id, min_align, min_len)
        is expected
    )


_T_2020 = datetime(2020, 1, 1)
_T_2021 = datetime(2021, 1, 1)


@pytest.mark.parametrize(
    "template_date, query_date, max_date, min_diff, expected",
    [
        pytest.param(_T_2020, None, None, None, False, id="no_constraints_passes"),
        pytest.param(_T_2021, None, _T_2020, None, True, id="template_after_max_fails"),
        pytest.param(
            _T_2020, None, _T_2021, None, False, id="template_before_max_passes"
        ),
        pytest.param(
            _T_2020, None, _T_2020, None, False, id="template_equals_max_passes"
        ),
        pytest.param(
            _T_2020,
            datetime(2020, 1, 11),
            None,
            10,
            False,
            id="diff_equals_min_passes",
        ),
        pytest.param(
            _T_2020,
            datetime(2020, 1, 6),
            None,
            10,
            True,
            id="diff_below_min_fails",
        ),
    ],
)
def test_fails_template_release_date_checks(
    template_date, query_date, max_date, min_diff, expected
):
    assert (
        fails_template_release_date_checks(
            template_release_date=template_date,
            query_release_date=query_date,
            max_template_release_date=max_date,
            min_release_date_diff=min_diff,
        )
        is expected
    )


def test_fails_template_release_date_checks_requires_query_date():
    """min_release_date_diff without a query release date is a programming error."""
    with pytest.raises(ValueError, match="Query release date not provided"):
        fails_template_release_date_checks(
            template_release_date=_T_2020,
            query_release_date=None,
            max_template_release_date=None,
            min_release_date_diff=10,
        )


@pytest.mark.parametrize(
    "original_chain_id, seq_from_aln, chain_id_seq_map, expected",
    [
        pytest.param(
            "A",
            "DEF",
            {"A": "DEF", "B": "DEFGHI"},
            "B",
            id="seq_found_in_other_chain",
        ),
        pytest.param(
            "A",
            "DEF",
            {"A": "DEF"},
            None,
            id="only_original_chain_matches_returns_none",
        ),
        pytest.param(
            "A",
            "WYV",
            {"A": "DEF", "B": "GHI"},
            None,
            id="no_chain_matches_returns_none",
        ),
        pytest.param(
            "A",
            "GH",
            {"A": "DEF", "B": "FGHIK"},
            "B",
            id="subsequence_match",
        ),
    ],
)
def test_remap_template_chain_id(
    original_chain_id, seq_from_aln, chain_id_seq_map, expected
):
    assert (
        remap_template_chain_id(original_chain_id, seq_from_aln, chain_id_seq_map)
        == expected
    )


@pytest.mark.parametrize(
    "chain_id, seq, chain_id_seq_map, expected",
    [
        pytest.param(
            "C",
            "DEF",
            {"A": "XYZ", "B": "DEFGHI"},
            "B",
            id="branch_A_chain_absent_remaps",
        ),
        pytest.param(
            "A",
            "GHI",
            {"A": "DEF", "B": "GHIKL"},
            "B",
            id="branch_B_chain_present_seq_mismatch_remaps",
        ),
        pytest.param(
            "A",
            "DEF",
            {"A": "DEFGHI", "B": "XYZ"},
            "A",
            id="branch_C_chain_present_seq_matches_keeps_original",
        ),
        pytest.param(
            "C",
            "WYV",
            {"A": "DEF", "B": "GHI"},
            None,
            id="absent_chain_no_remap_returns_none",
        ),
    ],
)
def test_match_template_seq_from_aln_to_struc(
    chain_id, seq, chain_id_seq_map, expected
):
    template = _make_template(chain_id=chain_id, seq=seq)
    assert match_template_seq_from_aln_to_struc(template, chain_id_seq_map) == expected


# ---------------------------------------------------------------------------
# Tier B: pydantic validators
# ---------------------------------------------------------------------------


def test_input_inference_alignment_only_valid():
    inp = TemplatePreprocessorInputInference(
        aln_path=Path("/some/aln.sto"), query_seq_str="ACDEF"
    )
    assert inp.aln_path == Path("/some/aln.sto")
    assert inp.template_cif_paths is None


def test_input_inference_cif_only_valid():
    inp = TemplatePreprocessorInputInference(
        query_seq_str="ACDEF",
        template_cif_paths=[Path("/some/t.cif")],
        template_cif_chain_ids=["A"],
    )
    assert inp.template_cif_paths == [Path("/some/t.cif")]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param(
            dict(
                aln_path=Path("/a.sto"),
                query_seq_str="ACDEF",
                template_cif_paths=[Path("/t.cif")],
            ),
            "Cannot provide both",
            id="both_aln_and_cif_paths",
        ),
        pytest.param(
            dict(query_seq_str="ACDEF", template_cif_chain_ids=["A"]),
            "requires 'template_cif_paths'",
            id="chain_ids_without_cif_paths",
        ),
        pytest.param(
            dict(
                query_seq_str="ACDEF",
                template_cif_paths=[Path("/t.cif")],
                template_cif_chain_ids=["A", "B"],
            ),
            "Length mismatch",
            id="chain_ids_length_mismatch",
        ),
    ],
)
def test_input_inference_invalid(kwargs, match):
    with pytest.raises(ValueError, match=match):
        TemplatePreprocessorInputInference(**kwargs)


def test_settings_rejects_unsupported_structure_format(tmp_path):
    with pytest.raises(NotImplementedError, match="structure_file_format"):
        TemplatePreprocessorSettings(
            output_directory=tmp_path, structure_file_format="pdb"
        )


def test_settings_derives_default_directories(tmp_path):
    """Only the unconditional sub-dirs are derived under base by default."""
    settings = TemplatePreprocessorSettings(output_directory=tmp_path)

    assert settings.structure_directory == tmp_path / "template_structures"
    assert settings.cache_directory == tmp_path / "template_cache"
    # Conditional directories stay None when their feature flag is off.
    assert settings.precache_directory is None
    assert settings.structure_array_directory is None
    assert settings.log_directory is None
    # Derived directories are created on disk.
    assert settings.structure_directory.is_dir()
    assert settings.cache_directory.is_dir()


def test_settings_derives_conditional_directories(tmp_path):
    settings = TemplatePreprocessorSettings(
        output_directory=tmp_path,
        create_precache=True,
        preparse_structures=True,
        create_logs=True,
    )

    assert settings.precache_directory == tmp_path / "template_precache"
    assert settings.structure_array_directory == tmp_path / "template_structure_arrays"
    assert settings.log_directory == tmp_path / "template_logs"
    assert settings.precache_directory.is_dir()
    assert settings.structure_array_directory.is_dir()
    assert settings.log_directory.is_dir()


def test_settings_respects_explicit_directories(tmp_path):
    explicit_cache = tmp_path / "my_cache"
    settings = TemplatePreprocessorSettings(
        output_directory=tmp_path, cache_directory=explicit_cache
    )
    assert settings.cache_directory == explicit_cache
    assert explicit_cache.is_dir()


# ---------------------------------------------------------------------------
# Tier C: bare-instance methods (no __init__, no mp.Pool)
# ---------------------------------------------------------------------------


def _make_bare_preprocessor(**attrs) -> TemplatePreprocessor:
    """Build a TemplatePreprocessor without running __init__ (no mp.Pool / file IO).

    Mirrors the pattern in test_template_parsers.py; only the attributes the method
    under test reads need to be set.
    """
    pre = object.__new__(TemplatePreprocessor)
    for key, value in attrs.items():
        setattr(pre, key, value)
    return pre


def _write_file(path: Path, content: str = "") -> Path:
    path.write_text(content)
    return path


def test_parse_inference_query_set_alignment_mode_dedup(tmp_path):
    """Two chains sharing an alignment path collapse to one input; moltype-mismatched
    chains are skipped."""
    aln = _write_file(tmp_path / "aln.sto", ">q\nACDEF\n")
    query = Query(
        chains=[
            Chain(
                molecule_type="protein",
                chain_ids=["A"],
                sequence="ACDEF",
                template_alignment_file_path=aln,
                template_entry_chain_ids=["1abc_A"],
            ),
            # Same alignment path -> deduplicated.
            Chain(
                molecule_type="protein",
                chain_ids=["B"],
                sequence="ACDEF",
                template_alignment_file_path=aln,
            ),
            # Non-protein -> skipped before template fields are read.
            Chain(
                molecule_type="dna",
                chain_ids=["C"],
                sequence="ACGT",
                template_alignment_file_path=aln,
            ),
        ]
    )
    iqs = InferenceQuerySet(queries={"q0": query})
    pre = _make_bare_preprocessor(input_set=iqs, moltypes=[MoleculeType.PROTEIN])

    pre._parse_inference_query_set()

    assert len(pre.inputs) == 1
    assert pre.inputs[0].aln_path == Path(aln)
    assert pre.inputs[0].query_seq_str == "ACDEF"
    assert pre.inputs[0].template_entry_chain_ids == ["1abc_A"]


def test_parse_inference_query_set_cif_mode_dedup(tmp_path):
    """CIF-direct chains dedup by (sorted cif paths, chain ids) key."""
    cif1 = _write_file(tmp_path / "t1.cif", "data_")
    cif2 = _write_file(tmp_path / "t2.cif", "data_")
    query = Query(
        chains=[
            Chain(
                molecule_type="protein",
                chain_ids=["A"],
                sequence="ACDEF",
                template_cif_paths=[cif1, cif2],
                template_cif_chain_ids=["A", "B"],
            ),
            # Same path set + chain ids -> deduplicated.
            Chain(
                molecule_type="protein",
                chain_ids=["B"],
                sequence="ACDEF",
                template_cif_paths=[cif1, cif2],
                template_cif_chain_ids=["A", "B"],
            ),
            # Same paths, different chain ids -> distinct key, kept.
            Chain(
                molecule_type="protein",
                chain_ids=["D"],
                sequence="GHIKL",
                template_cif_paths=[cif1, cif2],
                template_cif_chain_ids=["C", "D"],
            ),
        ]
    )
    iqs = InferenceQuerySet(queries={"q0": query})
    pre = _make_bare_preprocessor(input_set=iqs, moltypes=[MoleculeType.PROTEIN])

    pre._parse_inference_query_set()

    assert len(pre.inputs) == 2
    assert all(inp.template_cif_paths is not None for inp in pre.inputs)


def test_parse_inference_query_set_no_template_data_skipped(tmp_path):
    """A chain with neither alignment nor CIF produces no input."""
    query = Query(
        chains=[
            Chain(molecule_type="protein", chain_ids=["A"], sequence="ACDEF"),
        ]
    )
    iqs = InferenceQuerySet(queries={"q0": query})
    pre = _make_bare_preprocessor(input_set=iqs, moltypes=[MoleculeType.PROTEIN])

    pre._parse_inference_query_set()

    assert pre.inputs == []


def test_update_inference_query_set(tmp_path):
    """Chains with a cache entry get the npz path + template ids; missing ones get
    None."""
    seq_present = "ACDEFGHIK"
    seq_missing = "KLMNPQRST"
    hash_present = get_sequence_hash(seq_present)
    # Create the cache entry only for the present sequence.
    _write_file(tmp_path / f"{hash_present}.npz")

    query = Query(
        chains=[
            Chain(molecule_type="protein", chain_ids=["A"], sequence=seq_present),
            Chain(molecule_type="protein", chain_ids=["B"], sequence=seq_missing),
        ]
    )
    iqs = InferenceQuerySet(queries={"q0": query})
    pre = _make_bare_preprocessor(
        input_set=iqs,
        moltypes=[MoleculeType.PROTEIN],
        cache_directory=tmp_path,
        hash_template_id_map={hash_present: ["1abc_A", "2def_B"]},
    )

    pre._update_inference_query_set()

    chains = pre.input_set.queries["q0"].chains
    assert chains[0].template_alignment_file_path == tmp_path / f"{hash_present}.npz"
    assert chains[0].template_entry_chain_ids == ["1abc_A", "2def_B"]
    assert chains[1].template_alignment_file_path is None
    assert chains[1].template_entry_chain_ids == []


# ---------------------------------------------------------------------------
# Tier D: legacy TSV log helpers (characterization)
# ---------------------------------------------------------------------------


def test_data_log_to_tsv_writes_header_once(tmp_path):
    tsv = tmp_path / "data_log_1.tsv"
    data_log_to_tsv({"a": 1, "b": 2}, tsv)
    data_log_to_tsv({"a": 3, "b": 4}, tsv)

    lines = tsv.read_text().splitlines()
    assert lines[0] == "a\tb"  # header written once
    assert lines[1] == "1\t2"
    assert lines[2] == "3\t4"
    assert len(lines) == 3


def test_collate_data_logs_merges_and_unlinks(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    f1 = log_dir / "data_log_1.tsv"
    f2 = log_dir / "data_log_2.tsv"
    data_log_to_tsv({"a": 1, "b": 2}, f1)
    data_log_to_tsv({"a": 3, "b": 4}, f2)

    collate_data_logs(log_dir, output_dir, "combined.tsv")

    out = output_dir / "combined.tsv"
    assert out.exists()
    rows = out.read_text().splitlines()
    assert rows[0] == "a\tb"
    assert set(rows[1:]) == {"1\t2", "3\t4"}
    # Source per-worker logs are consumed.
    assert not f1.exists()
    assert not f2.exists()
