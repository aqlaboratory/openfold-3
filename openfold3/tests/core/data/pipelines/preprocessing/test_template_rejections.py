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

"""Characterization tests: why TemplatePreprocessor drops templates.

`_preprocess_templates_for_query` drops a hit that fails the sequence or release-date
checks, whose chain or sequence its structure doesn't have, or that comes after
`max_templates` others. A chain that times out gets no templates at all. These tests
pin today's behaviour, including one bug that its docstring points out.

Workers run as threads (see `_settings`), so `unittest.mock.patch` reaches them.
"""

import shutil
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import openfold3
from openfold3.core.data.io.structure.cif import _load_ciffile
from openfold3.core.data.pipelines.preprocessing import template as template_module
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessor,
    TemplatePreprocessorSettings,
)
from openfold3.core.data.primitives.structure.metadata import (
    get_asym_id_to_canonical_seq_dict,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    Chain,
    InferenceQuerySet,
    Query,
)

# Chains B and C of 2q2k are the same 70-residue protein, released on 2008-02-05
TEMPLATE_CIF = Path(openfold3.__file__).parent / "tests/test_data/mmcifs/2q2k.cif"


def _query_seq() -> str:
    """2q2k chain B's sequence, so that chains B and C are identical hits."""
    return get_asym_id_to_canonical_seq_dict(_load_ciffile(TEMPLATE_CIF))["B"]


def _m8(tmp_path: Path, hits: list[str]) -> Path:
    """A ColabFold alignment of `hits` (e.g. "2q2k_C"), in order.

    Each hit is a full-length, identical match.
    """
    n = len(_query_seq())
    aln_path = tmp_path / "hits.m8"
    aln_path.write_text(
        "".join(
            f"101\t{hit}\t1.000\t{n}\t0\t0\t1\t{n}\t1\t{n}\t1.0E-40\t300\t{n}M\n"
            for hit in hits
        )
    )
    return aln_path


def _query_set(aln_path: Path) -> InferenceQuerySet:
    """One query with 2q2k chain B's sequence and the alignment `aln_path`."""
    chain = Chain(
        molecule_type=MoleculeType.PROTEIN,
        chain_ids=["A"],
        sequence=_query_seq(),
        template_alignment_file_path=aln_path,
    )
    return InferenceQuerySet(queries={"query": Query(chains=[chain])})


def _settings(tmp_path: Path, **overrides) -> TemplatePreprocessorSettings:
    """Offline settings with thread workers and 2q2k in the structure directory."""
    structure_dir = tmp_path / "template_structures"
    settings = TemplatePreprocessorSettings(
        **{
            "mode": "predict",
            "output_directory": tmp_path / "template_data",
            "structure_directory": structure_dir,
            "fetch_missing_structures": False,
            "n_processes": 1,
            "worker_pool": "thread",
            **overrides,
        }
    )
    shutil.copy(TEMPLATE_CIF, structure_dir)
    return settings


def _preprocess(query_set: InferenceQuerySet, settings) -> Chain:
    """Preprocesses the query set and returns its only chain."""
    TemplatePreprocessor(input_set=query_set, config=settings)()
    return query_set.queries["query"].chains[0]


@pytest.mark.parametrize(
    ("overrides", "hits", "template_ids"),
    [
        pytest.param({"max_seq_id": 1.0}, ["2q2k_C"], ["2q2k_C"], id="seq_id_at_max"),
        pytest.param({"max_seq_id": 0.99}, ["2q2k_C"], [], id="seq_id_above_max"),
        # A bug, see the docstring
        pytest.param({"min_align": 0.5}, ["2q2k_C"], [], id="min_align"),
        pytest.param(
            {"max_release_date": datetime(2008, 2, 6)},
            ["2q2k_C"],
            ["2q2k_C"],
            id="released_before_max",
        ),
        pytest.param(
            {"max_release_date": datetime(2008, 2, 5)},
            ["2q2k_C"],
            [],
            id="released_on_max",
        ),
        pytest.param(
            {"max_templates": 1},
            ["2q2k_B", "2q2k_C"],
            ["2q2k_B"],
            id="beyond_max_templates",
        ),
        pytest.param({}, ["2q2k_Z", "2q2k_C"], ["2q2k_C"], id="chain_not_in_structure"),
    ],
)
def test_hits_that_fail_a_check_are_dropped(tmp_path, overrides, hits, template_ids):
    """Which ColabFold hits are kept, given the settings in `overrides`.

    Every hit is identical to the query (sequence identity 1.0), and 2q2k was released
    on 2008-02-05.

    `min_align` pins a bug: ColabFold hits have no query coverage until they are
    realigned, which happens after the sequence checks, so comparing their coverage
    with `min_align` raises TypeError and the chain loses every template.
    """
    settings = _settings(tmp_path, **overrides)

    chain = _preprocess(_query_set(_m8(tmp_path, hits)), settings)

    assert chain.template_entry_chain_ids == template_ids


def test_hit_whose_sequence_is_not_in_its_structure_is_dropped(tmp_path):
    seq = _query_seq()
    # No chain of 2q2k has this sequence: its His-tag is swapped out
    mismatched = seq.replace("HHHHHH", "WWWWWW")
    n = len(seq)
    aln_path = tmp_path / "hits.a3m"
    aln_path.write_text(
        f">query_X/1-{n}\n{seq}\n>2q2k_B/1-{n}\n{seq}\n>2q2k_C/1-{n}\n{mismatched}\n"
    )

    chain = _preprocess(_query_set(aln_path), _settings(tmp_path))

    assert chain.template_entry_chain_ids == ["2q2k_B"]


def test_chain_that_times_out_gets_no_templates(tmp_path, capsys):
    settings = _settings(tmp_path, preprocess_timeout=1)

    with patch.object(
        template_module,
        "parse_template_alignment",
        side_effect=lambda *args: time.sleep(2),
    ):
        chain = _preprocess(_query_set(_m8(tmp_path, ["2q2k_C"])), settings)

    assert chain.template_entry_chain_ids == []
    assert "TIMED OUT after 1s" in capsys.readouterr().out
