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
and file IO), and the legacy TSV log helpers.

Tier E is the exception: cross-query template source isolation is only observable
end to end, so those tests run the real ``__call__``. They stay offline by pre-seeding
the template structure directory and setting ``fetch_missing_structures=False``.
"""

import getpass
import gzip
import logging
import shutil
from datetime import date, datetime
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path
from unittest.mock import patch

import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
import pytest

import openfold3
from openfold3.core.data.io.sequence.template import TemplateData
from openfold3.core.data.io.structure.cif import _load_ciffile
from openfold3.core.data.pipelines.preprocessing import template as template_module
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePrecachePreprocessor,
    TemplatePreprocessor,
    TemplatePreprocessorInputInference,
    TemplatePreprocessorSettings,
    TemplateStructurePreprocessor,
    build_template_cache_key,
    collate_data_logs,
    data_log_to_tsv,
    fails_template_release_date_checks,
    fails_template_sequence_checks,
    match_template_seq_from_aln_to_struc,
    remap_template_chain_id,
)
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetStructureData,
)
from openfold3.core.data.primitives.sequence.hash import get_sequence_hash
from openfold3.core.data.primitives.structure.metadata import (
    get_asym_id_to_canonical_seq_dict,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.core.data.tools.colabfold_msa_server import (
    remap_colabfold_template_chain_ids,
)
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
            _T_2020, None, _T_2020, None, True, id="template_equals_max_fails"
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


def _visible_content_hash(*parts) -> str:
    """Stand-in for `get_file_content_hash` that echoes what it was handed.

    Keeps these tests free of file IO and, more usefully, makes the argument order
    `build_template_cache_key` builds up visible in the asserted key.
    """
    return "|".join(
        "None" if part is None else part.name if isinstance(part, Path) else str(part)
        for part in parts
    )


_SEQ = "ACDEF"
_SEQ_PART = f"seq-{get_sequence_hash(_SEQ)}"

# Applied per test so no file is ever opened; the stub's echo shows, in the asserted
# key, exactly which parts `build_template_cache_key` hands to the content hasher.
_no_file_io = patch.object(
    template_module, "get_file_content_hash", _visible_content_hash
)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no_source_declared"),
        pytest.param({"cif_paths": []}, id="empty_cif_list"),
        pytest.param({"cif_paths": None}, id="explicit_none"),
    ],
)
def test_build_template_cache_key_without_a_source_is_none(kwargs):
    """A chain that declares no template source has no cache entry to look up."""
    assert build_template_cache_key(sequence=_SEQ, **kwargs) is None


@pytest.mark.parametrize(
    "kwargs, expected_source_part",
    [
        pytest.param(
            {"aln_path": Path("/msas/rep0/colabfold_template.m8")},
            "aln-colabfold_template.m8",
            id="alignment_mode_colabfold_m8",
        ),
        pytest.param(
            {"cif_paths": [Path("/templates/1y57.cif")], "cif_chain_ids": ["A"]},
            "cif-1y57.cif|A",
            id="cif_direct_chain_pinned",
        ),
        pytest.param(
            {"cif_paths": [Path("/templates/1y57.cif")]},
            "cif-1y57.cif|None",
            id="cif_direct_chain_auto_selected",
        ),
        pytest.param(
            {
                "cif_paths": [Path("/t/a.cif"), Path("/t/b.cif")],
                "cif_chain_ids": ["A", "B"],
            },
            "cif-a.cif|A|b.cif|B",
            id="cif_direct_two_files",
        ),
        pytest.param(
            # Declared in the other order, same (file, chain) pairing -> same key.
            {
                "cif_paths": [Path("/t/b.cif"), Path("/t/a.cif")],
                "cif_chain_ids": ["B", "A"],
            },
            "cif-a.cif|A|b.cif|B",
            id="cif_declaration_order_is_irrelevant",
        ),
        pytest.param(
            # Chain IDs are positional, so swapping them is a different request.
            {
                "cif_paths": [Path("/t/a.cif"), Path("/t/b.cif")],
                "cif_chain_ids": ["B", "A"],
            },
            "cif-a.cif|B|b.cif|A",
            id="cif_chain_pairing_is_preserved",
        ),
    ],
)
@_no_file_io
def test_build_template_cache_key_shape(kwargs, expected_source_part):
    """`seq-<hash of the sequence>.<aln|cif>-<hash of the source>`."""
    key = build_template_cache_key(sequence=_SEQ, **kwargs)

    assert key == f"{_SEQ_PART}.{expected_source_part}"


@_no_file_io
def test_build_template_cache_key_separates_sources_for_one_sequence():
    """The regression this key exists for (issue #294 follow-up).

    Queries sharing a sequence but declaring different template sources must not
    share a cache entry, while the shared `seq-` half stays visible so a directory
    listing still shows they are the same sequence.
    """
    keys = [
        build_template_cache_key(sequence=_SEQ, aln_path=Path("/msas/cf.m8")),
        build_template_cache_key(sequence=_SEQ, cif_paths=[Path("/t/1y57.cif")]),
        build_template_cache_key(sequence=_SEQ, cif_paths=[Path("/t/2src.cif")]),
    ]

    assert all(key.startswith(f"{_SEQ_PART}.") for key in keys)
    assert len(set(keys)) == 3

    # ...and the same source under a different sequence is a different entry too.
    other_sequence = build_template_cache_key(
        sequence="GHIKL", cif_paths=[Path("/t/1y57.cif")]
    )
    assert not other_sequence.startswith(f"{_SEQ_PART}.")


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
    with patch(
        "openfold3.core.data.tools.utils.tempfile.gettempdir",
        return_value=str(tmp_path),
    ):
        settings = TemplatePreprocessorSettings()

    output_directory = tmp_path / f"of3-of-{getpass.getuser()}" / "template_data"

    assert settings.output_directory == output_directory
    assert settings.structure_directory == output_directory / "template_structures"
    assert settings.cache_directory == output_directory / "template_cache"
    # Conditional directories stay None when their feature flag is off.
    assert settings.precache_directory is None
    assert settings.structure_array_directory is None
    assert settings.log_directory is None
    assert not output_directory.exists()


def test_template_directories_are_created_when_preprocessing_starts(tmp_path):
    output_directory = tmp_path / "template_data"
    settings = TemplatePreprocessorSettings(
        create_precache=True,
        preparse_structures=True,
        create_logs=True,
    )
    settings._set_inference_output_directory(output_directory)

    assert settings.precache_directory == output_directory / "template_precache"
    assert (
        settings.structure_array_directory
        == output_directory / "template_structure_arrays"
    )
    assert settings.log_directory == output_directory / "template_logs"
    assert not output_directory.exists()

    preprocessor = TemplatePreprocessor(
        input_set=InferenceQuerySet(queries={}),
        config=settings,
    )
    preprocessor._ensure_output_directories()

    assert settings.structure_directory.is_dir()
    assert settings.cache_directory.is_dir()
    assert settings.precache_directory.is_dir()
    assert settings.structure_array_directory.is_dir()
    assert settings.log_directory.is_dir()


def test_inference_replaces_only_implicit_template_paths(tmp_path):
    explicit_cache = tmp_path / "my_cache"
    settings = TemplatePreprocessorSettings(cache_directory=explicit_cache)
    output_directory = tmp_path / "output" / "template_data"

    settings._set_inference_output_directory(output_directory)

    assert settings.output_directory == output_directory
    assert settings.structure_directory == output_directory / "template_structures"
    assert settings.cache_directory == explicit_cache
    assert explicit_cache.is_dir()
    assert not output_directory.exists()


@pytest.mark.parametrize(
    "preprocessor_cls,output_field",
    [
        pytest.param(
            TemplatePrecachePreprocessor,
            "precache_directory",
            id="precache",
        ),
        pytest.param(
            TemplateStructurePreprocessor,
            "structure_array_directory",
            id="structure-arrays",
        ),
    ],
)
def test_standalone_template_consumers_create_output_when_called(
    tmp_path, preprocessor_cls, output_field
):
    structure_directory = tmp_path / "structures"
    structure_directory.mkdir()
    (structure_directory / "1abc.cif").write_text("data_1abc\n")
    system_tmp = tmp_path / "tmp"
    with patch(
        "openfold3.core.data.tools.utils.tempfile.gettempdir",
        return_value=str(system_tmp),
    ):
        if output_field == "precache_directory":
            settings = TemplatePreprocessorSettings(
                structure_directory=structure_directory,
                create_precache=True,
            )
        else:
            settings = TemplatePreprocessorSettings(
                structure_directory=structure_directory,
                preparse_structures=True,
            )
    output_directory = getattr(settings, output_field)
    preprocessor = preprocessor_cls(settings)

    assert output_directory is not None
    assert not output_directory.exists()

    with patch(
        "openfold3.core.data.pipelines.preprocessing.template.mp.Pool"
    ) as pool_cls:
        pool_cls.return_value.__enter__.return_value.imap_unordered.return_value = []
        preprocessor()

    assert output_directory.is_dir()


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
    (template_input,) = pre.inputs.values()
    assert template_input.aln_path == Path(aln)
    assert template_input.query_seq_str == "ACDEF"
    assert template_input.template_entry_chain_ids == ["1abc_A"]
    # The mapping is keyed by cache key, which is what makes it the dedup bookkeeping.
    assert set(pre.inputs) == {template_input.cache_key}


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
    assert all(inp.template_cif_paths is not None for inp in pre.inputs.values())
    assert set(pre.inputs) == {inp.cache_key for inp in pre.inputs.values()}


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

    assert pre.inputs == {}


def test_update_inference_query_set(tmp_path):
    """Chains with a cache entry get the npz path + template ids; missing ones get
    None."""
    seq_present = "ACDEFGHIK"
    seq_missing = "KLMNPQRST"
    aln_present = _write_file(tmp_path / "present.sto")
    aln_missing = _write_file(tmp_path / "missing.sto")
    key_present = build_template_cache_key(sequence=seq_present, aln_path=aln_present)
    # Create the cache entry only for the present sequence.
    _write_file(tmp_path / f"{key_present}.npz")

    query = Query(
        chains=[
            Chain(
                molecule_type="protein",
                chain_ids=["A"],
                sequence=seq_present,
                template_alignment_file_path=aln_present,
            ),
            Chain(
                molecule_type="protein",
                chain_ids=["B"],
                sequence=seq_missing,
                template_alignment_file_path=aln_missing,
            ),
        ]
    )
    iqs = InferenceQuerySet(queries={"q0": query})
    pre = _make_bare_preprocessor(
        input_set=iqs,
        moltypes=[MoleculeType.PROTEIN],
        cache_directory=tmp_path,
        template_ids_by_cache_key={key_present: ["1abc_A", "2def_B"]},
    )

    pre._update_inference_query_set()

    chains = pre.input_set.queries["q0"].chains
    assert chains[0].template_alignment_file_path == tmp_path / f"{key_present}.npz"
    assert chains[0].template_entry_chain_ids == ["1abc_A", "2def_B"]
    assert chains[1].template_alignment_file_path is None
    assert chains[1].template_entry_chain_ids == []


def test_update_inference_query_set_skips_chains_without_template_input(tmp_path):
    """A chain that asked for no templates must not inherit another query's cache
    entry just because the two share a sequence."""
    seq = "ACDEFGHIK"
    cif_path = _write_file(tmp_path / "1abc.cif")
    key = build_template_cache_key(sequence=seq, cif_paths=[cif_path])
    _write_file(tmp_path / f"{key}.npz")

    iqs = InferenceQuerySet(
        queries={
            "with_templates": Query(
                chains=[
                    Chain(
                        molecule_type="protein",
                        chain_ids=["A"],
                        sequence=seq,
                        template_cif_paths=[cif_path],
                    )
                ]
            ),
            "without_templates": Query(
                chains=[Chain(molecule_type="protein", chain_ids=["A"], sequence=seq)]
            ),
        }
    )
    pre = _make_bare_preprocessor(
        input_set=iqs,
        moltypes=[MoleculeType.PROTEIN],
        cache_directory=tmp_path,
        template_ids_by_cache_key={key: ["1abc_A"]},
    )

    pre._update_inference_query_set()

    with_templates = pre.input_set.queries["with_templates"].chains[0]
    without_templates = pre.input_set.queries["without_templates"].chains[0]
    assert with_templates.template_alignment_file_path == tmp_path / f"{key}.npz"
    assert with_templates.template_entry_chain_ids == ["1abc_A"]
    assert without_templates.template_alignment_file_path is None
    assert without_templates.template_entry_chain_ids is None


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


# ---------------------------------------------------------------------------
# Tier E: cross-query template source isolation (integration, runs __call__)
# ---------------------------------------------------------------------------

MMCIFS_DIR = Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs"
TEMPLATE_CIF = "2q2k.cif"


def _two_source_query_set(tmp_path: Path, order: list[str]) -> InferenceQuerySet:
    """Two queries sharing one sequence, each with a different template source.

    2q2k chains B and C are identical 70-residue proteins, so both are valid
    templates for the same query sequence and the two sources stay distinguishable:

      q_custom     CIF-direct, chain B pinned      -> 2q2k_B
      q_colabfold  alignment (.m8) hitting 2q2k_C  -> 2q2k_C

    Each yields exactly that when it is the only query in the set.
    """
    cif_path = tmp_path / TEMPLATE_CIF
    shutil.copy(MMCIFS_DIR / TEMPLATE_CIF, cif_path)
    seq = get_asym_id_to_canonical_seq_dict(_load_ciffile(cif_path))["B"]

    # Minimal ColabFold-style m8: one full-length, identical hit against chain C.
    n = len(seq)
    aln_path = _write_file(
        tmp_path / "colabfold_template.m8",
        f"101\t2q2k_C\t1.000\t{n}\t0\t0\t1\t{n}\t1\t{n}\t1.0E-40\t300\t{n}M\n",
    )

    sources = {
        "q_custom": {"template_cif_paths": [cif_path], "template_cif_chain_ids": ["B"]},
        "q_colabfold": {"template_alignment_file_path": aln_path},
    }
    return InferenceQuerySet(
        queries={
            name: Query(
                chains=[
                    Chain(
                        molecule_type="protein",
                        chain_ids=["A"],
                        sequence=seq,
                        **sources[name],
                    )
                ]
            )
            for name in order
        }
    )


@pytest.mark.parametrize(
    "order",
    [
        pytest.param(["q_custom", "q_colabfold"], id="custom_declared_first"),
        pytest.param(["q_colabfold", "q_custom"], id="colabfold_declared_first"),
    ],
)
def test_template_sources_do_not_collide_across_queries(tmp_path, order):
    """Queries sharing a sequence must each keep their own template source.

    The cache entry file, the "already processed" set and the template-id write-back
    are all keyed by sequence hash alone, so whichever query is declared first claims
    the sequence and the other silently inherits its templates. Which one wins is
    pure declaration order, so this is asserted both ways round.
    """
    iqs = _two_source_query_set(tmp_path, order)
    structure_dir = tmp_path / "template_structures"
    structure_dir.mkdir()
    shutil.copy(MMCIFS_DIR / TEMPLATE_CIF, structure_dir / TEMPLATE_CIF)

    settings = TemplatePreprocessorSettings(
        mode="predict",
        output_directory=tmp_path / "template_data",
        # Pre-seeded structures + no fetching keeps this offline.
        structure_directory=structure_dir,
        fetch_missing_structures=False,
        n_processes=1,
    )

    TemplatePreprocessor(input_set=iqs, config=settings)()

    custom = iqs.queries["q_custom"].chains[0]
    colabfold = iqs.queries["q_colabfold"].chains[0]

    assert custom.template_entry_chain_ids == ["2q2k_B"], (
        "the custom CIF query did not get its own template"
    )
    assert colabfold.template_entry_chain_ids == ["2q2k_C"], (
        "the ColabFold query did not get its own template"
    )
    assert (
        custom.template_alignment_file_path != colabfold.template_alignment_file_path
    ), "both queries were pointed at the same template cache entry"


def test_requeued_query_set_keeps_cached_templates(tmp_path):
    """A reused query set keeps its cached template entry (issue #420).

    Run 1 preprocesses a ColabFold alignment and writes the cache entry path and
    template ids into the chain. The reused batch mixes that chain, deep-copied
    from run 1's query set, with a raw CIF source that run 1 never
    processed, so the write-back still runs while the cached entry is bypassed.
    """
    run_1_qs = _two_source_query_set(tmp_path, ["q_colabfold"])
    structure_dir = tmp_path / "template_structures"
    structure_dir.mkdir()
    shutil.copy(MMCIFS_DIR / TEMPLATE_CIF, structure_dir / TEMPLATE_CIF)
    settings = TemplatePreprocessorSettings(
        mode="predict",
        output_directory=tmp_path / "template_data",
        # Pre-seeded structures + no fetching keeps this offline.
        structure_directory=structure_dir,
        fetch_missing_structures=False,
        n_processes=1,
    )

    TemplatePreprocessor(input_set=run_1_qs, config=settings)()

    cached_chain = run_1_qs.queries["q_colabfold"].chains[0]
    cached_path = cached_chain.template_alignment_file_path
    assert cached_path is not None
    assert cached_chain.template_entry_chain_ids == ["2q2k_C"]

    run_2_qs = run_1_qs.model_copy(deep=True)
    fresh_qs = _two_source_query_set(tmp_path, ["q_custom"])
    run_2_qs.queries |= fresh_qs.queries

    preprocessor = TemplatePreprocessor(input_set=run_2_qs, config=settings)
    preprocessor()

    requeued_cached = run_2_qs.queries["q_colabfold"].chains[0]
    assert requeued_cached.template_alignment_file_path == cached_path, (
        "the cached template entry was dropped when the query set was reused"
    )
    assert requeued_cached.template_entry_chain_ids == ["2q2k_C"], (
        "the template ids were dropped when the query set was reused"
    )
    requeued_raw = run_2_qs.queries["q_custom"].chains[0]
    assert requeued_raw.template_entry_chain_ids == ["2q2k_B"], (
        "the raw source in the reused batch was not processed"
    )
    # Only the raw source needs preprocessing; the cached entry must not be
    # re-parsed as an alignment.
    assert len(preprocessor.inputs) == 1


# ---------------------------------------------------------------------------
# Train mode, end to end: real ColabFold hits -> template cache (1fdl)
#
# Inputs are real data for 1fdl (Fab light chain, Fab heavy chain, lysozyme): cut-down
# ColabFold .m8 hits and the hit structures, offline. See
# test_data/template_preprocessing/1fdl/README.md for provenance. The dataset cache
# holds 1fdl twice: once with its real release date (1991-10-15) and once re-dated to
# 2020. Both copies share the three alignment representatives, so release-date
# filtering must happen per structure, not per representative.
# ---------------------------------------------------------------------------

TRAIN_FIXTURE_DIR = (
    Path(openfold3.__file__).parent
    / "tests"
    / "test_data"
    / "template_preprocessing"
    / "1fdl"
)

# Templates must be released at least this many days before the query structure.
TRAIN_MIN_RELEASE_DATE_DIFF = 60

# chain_id -> (label_asym_id, auth_asym_id, entity_id, alignment_representative_id)
TRAIN_CHAINS = {
    "1": ("A", "L", 1, "1fdl_A"),  # Fab light chain
    "2": ("B", "H", 2, "1fdl_B"),  # Fab heavy chain
    "3": ("C", "Y", 3, "1fdl_C"),  # lysozyme
}

# rep_id -> ColabFold query index (column 1 of raw_pdb70.m8)
TRAIN_REP_ID_TO_M = {"1fdl_A": 101, "1fdl_B": 102, "1fdl_C": 103}

TRAIN_STRUCTURE_RELEASE_DATES = {
    "1fdl": date(1991, 10, 15),
    "1fdl-2020": date(2020, 1, 1),
}

# Expected template_ids per structure and chain, in e-value order (for the light chain,
# deliberately not alphabetical). Template release dates: 1fdl 1991-10-15,
# 1qbl 1998-12-02, 1jhk 2001-10-10, 3hfm 1989-07-12, 1ior 2001-04-11, 7ynv 2022-09-21,
# 5lyz 1977-04-12.
TRAIN_EXPECTED_TEMPLATE_IDS = {
    # 1fdl itself (released the same day) and every later entry are excluded; the
    # light chain has no older hit at all.
    "1fdl": {"1": [], "2": ["3hfm_B"], "3": ["5lyz_A"]},
    # Same sequences, later date: 1fdl is now a valid template; 7ynv (2022) is not.
    "1fdl-2020": {
        "1": ["1fdl_A", "1qbl_A", "1jhk_A"],
        "2": ["1fdl_B", "1qbl_B", "3hfm_B"],
        "3": ["1ior_A", "5lyz_A"],
    },
}


def _train_dataset_cache() -> ClusteredDatasetCache:
    structure_data = {
        pdb_id: ClusteredDatasetStructureData(
            release_date=release_date,
            resolution=2.5,
            chains={
                chain_id: ClusteredDatasetChainData(
                    label_asym_id=label,
                    auth_asym_id=auth,
                    entity_id=entity_id,
                    molecule_type="PROTEIN",
                    reference_mol_id=None,
                    alignment_representative_id=rep_id,
                    template_ids=None,
                    cluster_id="0",
                    cluster_size=1,
                )
                for chain_id, (label, auth, entity_id, rep_id) in TRAIN_CHAINS.items()
            },
            interfaces={},
        )
        for pdb_id, release_date in TRAIN_STRUCTURE_RELEASE_DATES.items()
    }
    return ClusteredDatasetCache(
        name="1fdl-template-train",
        structure_data=structure_data,
        reference_molecule_data={},
    )


def _run_train_template_preprocessor(
    tmp_path: Path, **overrides
) -> tuple[ClusteredDatasetCache, TemplatePreprocessorSettings]:
    """Run train-mode preprocessing offline on the 1fdl fixture.

    ``overrides`` replace the fixture's settings. Returns the updated dataset cache and
    the settings, for their output directories.
    """
    structure_dir = tmp_path / "template_structures"
    structure_dir.mkdir()
    for gz in (TRAIN_FIXTURE_DIR / "template_structures").glob("*.cif.gz"):
        (structure_dir / gz.name.removesuffix(".gz")).write_bytes(
            gzip.decompress(gz.read_bytes())
        )

    settings = TemplatePreprocessorSettings(
        **{
            "mode": "train",
            "output_directory": tmp_path / "template_data",
            "structure_directory": structure_dir,
            "template_alignment_directory": TRAIN_FIXTURE_DIR / "template_alignments",
            "alignment_representatives_fasta": TRAIN_FIXTURE_DIR
            / "representatives.fasta",
            "min_release_date_diff": TRAIN_MIN_RELEASE_DATE_DIFF,
            "fetch_missing_structures": False,
            "preparse_structures": True,
            "n_processes": 1,
            **overrides,
        }
    )
    cache = _train_dataset_cache()
    TemplatePreprocessor(input_set=cache, config=settings)()
    return cache, settings


def _train_template_ids(
    cache: ClusteredDatasetCache,
) -> dict[str, dict[str, list[str]]]:
    return {
        pdb_id: {
            chain_id: list(chain.template_ids or [])
            for chain_id, chain in structure.chains.items()
        }
        for pdb_id, structure in cache.structure_data.items()
    }


def test_template_ids_are_filtered_per_structure_release_date(tmp_path):
    """Each structure gets the e-value-ordered templates its own release date allows.

    Covers the self-template (1fdl for 1fdl), templates newer than the query, and two
    structures sharing representatives but differing in release date.
    """
    cache, _ = _run_train_template_preprocessor(tmp_path)

    assert _train_template_ids(cache) == TRAIN_EXPECTED_TEMPLATE_IDS


def test_cache_entries_match_golden(tmp_path):
    """Every assigned template has a cache entry equal to the golden one.

    The golden entries were produced by predict mode on the same inputs; index,
    release date and residue index map do not depend on the mode.
    """
    cache, settings = _run_train_template_preprocessor(tmp_path)

    for _, chain in _train_assigned_chains(cache):
        rep_id = chain.alignment_representative_id
        with (
            np.load(
                settings.cache_directory / f"{rep_id}.npz", allow_pickle=True
            ) as out,
            np.load(
                TRAIN_FIXTURE_DIR / "golden" / f"{rep_id}.npz", allow_pickle=True
            ) as gold,
        ):
            for template_id in chain.template_ids:
                actual = out[template_id].item()
                expected = gold[template_id].item()
                assert actual["index"] == expected["index"], template_id
                assert actual["release_date"] == expected["release_date"], template_id
                np.testing.assert_array_equal(
                    actual["idx_map"], expected["idx_map"], err_msg=template_id
                )


def test_cache_entries_keep_alignment_rank(tmp_path):
    """Each cache entry's index is its hit's rank in the template alignment."""
    _, settings = _run_train_template_preprocessor(tmp_path)

    for m8 in sorted((TRAIN_FIXTURE_DIR / "template_alignments").glob("*/*.m8")):
        hits = [line.split("\t")[1] for line in m8.read_text().splitlines()]
        with np.load(
            settings.cache_directory / f"{m8.parent.name}.npz", allow_pickle=True
        ) as entries:
            ranks = {
                template_id: entries[template_id].item()["index"]
                for template_id in entries
            }
        assert ranks == {hit: rank for rank, hit in enumerate(hits)}, m8.parent.name


def test_structure_arrays_written_for_assigned_templates(tmp_path):
    """Each assigned template has the preparsed chain array training reads."""
    cache, settings = _run_train_template_preprocessor(tmp_path)

    for _, chain in _train_assigned_chains(cache):
        for template_id in chain.template_ids:
            entry_id = template_id.split("_")[0]
            path = settings.structure_array_directory / entry_id / f"{template_id}.npz"
            assert path.exists(), path


def _fixture_label_to_author(pdb_ids: set[str]) -> dict[str, dict[str, str]]:
    """Stands in for the RCSB chain mapping call: read from the fixture structures."""
    label_to_author = {}
    for pdb_id in pdb_ids:
        with gzip.open(
            TRAIN_FIXTURE_DIR / "template_structures" / f"{pdb_id}.cif.gz", "rt"
        ) as f:
            scheme = pdbx.CIFFile.read(f).block["pdbx_poly_seq_scheme"]
        label_to_author[pdb_id] = dict(
            zip(
                map(str, scheme["asym_id"].as_array()),
                map(str, scheme["pdb_strand_id"].as_array()),
                strict=True,
            )
        )
    return label_to_author


@patch(
    "openfold3.core.data.tools.colabfold_msa_server.fetch_label_to_author_chain_ids",
    side_effect=_fixture_label_to_author,
)
def test_author_chain_hits_remapped_for_train_mode(_fetch, tmp_path):
    """Raw ColabFold hits (author chain IDs) work in train mode once remapped.

    1fdl's antibody chains are author L/H/Y but label A/B/C, so without the remap the
    light and heavy chains' hits could not be found in their structures.
    """
    remapped = remap_colabfold_template_chain_ids(
        template_alignments=pd.read_csv(
            TRAIN_FIXTURE_DIR / "raw_pdb70.m8", sep="\t", header=None
        ),
        m_with_templates={101, 102, 103},
        rep_ids=list(TRAIN_REP_ID_TO_M),
        rep_id_to_m=TRAIN_REP_ID_TO_M,
    )
    aln_dir = tmp_path / "template_alignments"
    for rep_id, df in remapped.items():
        # As ColabFoldQueryRunner writes them
        (aln_dir / rep_id).mkdir(parents=True)
        df.to_csv(
            aln_dir / rep_id / "colabfold_template.m8",
            sep="\t",
            header=False,
            index=False,
        )

    # Remapping renames chains and keeps every hit in its original rank
    for rep_id in TRAIN_REP_ID_TO_M:
        expected = pd.read_csv(
            TRAIN_FIXTURE_DIR
            / "template_alignments"
            / rep_id
            / "colabfold_template.m8",
            sep="\t",
            header=None,
        )
        pd.testing.assert_frame_equal(remapped[rep_id].reset_index(drop=True), expected)

    cache, _ = _run_train_template_preprocessor(
        tmp_path, template_alignment_directory=aln_dir
    )

    assert _train_template_ids(cache) == TRAIN_EXPECTED_TEMPLATE_IDS


def test_template_chain_missing_from_structure_warns(tmp_path, caplog):
    """A hit whose chain ID is not in its structure is dropped with a warning.

    Un-remapped author chain IDs (1fdl_L for the light chain) are the usual cause.
    """
    aln_dir = tmp_path / "template_alignments"
    shutil.copytree(TRAIN_FIXTURE_DIR / "template_alignments", aln_dir)
    light = aln_dir / "1fdl_A" / "colabfold_template.m8"
    light.write_text(light.read_text().replace("_A\t", "_L\t"))  # undo the remap

    # In-process workers, so caplog sees their records
    with patch.object(template_module.mp, "Pool", ThreadPool):
        cache, _ = _run_train_template_preprocessor(
            tmp_path, template_alignment_directory=aln_dir
        )

    # The light chain's hits are unusable; the other chains are unaffected.
    template_ids = _train_template_ids(cache)
    assert template_ids["1fdl-2020"]["1"] == []
    assert (
        template_ids["1fdl-2020"]["2"] == TRAIN_EXPECTED_TEMPLATE_IDS["1fdl-2020"]["2"]
    )
    assert (
        template_ids["1fdl-2020"]["3"] == TRAIN_EXPECTED_TEMPLATE_IDS["1fdl-2020"]["3"]
    )
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "label chain ID" in record.getMessage()
    ]
    for template_id in ("1fdl_L", "1qbl_L", "1jhk_L"):
        assert any(f"{template_id} not found" in w for w in warnings), warnings


def _train_assigned_chains(cache: ClusteredDatasetCache):
    """Yield (pdb_id, chain) for chains with at least one template, failing if none."""
    assigned = [
        (pdb_id, chain)
        for pdb_id, structure in cache.structure_data.items()
        for chain in structure.chains.values()
        if chain.template_ids
    ]
    assert assigned, "no chain was assigned any template"
    yield from assigned
