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

"""Characterization tests: where TemplatePreprocessor gets template structures from.

For each alignment hit, `_preprocess_templates_for_query` reads the structure from the
structure directory or a precache entry, or downloads it, and can write a precache
entry and per-chain structure arrays on the way. An earlier run's cache entry skips
all of that. These tests pin today's behaviour, including two bugs that their
docstrings point out.

Workers run as threads (see `_settings`), so `unittest.mock.patch` reaches them.
"""

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from biotite.database import RequestError

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

# Chains B and C of 2q2k are the same 70-residue protein
TEMPLATE_CIF = Path(openfold3.__file__).parent / "tests/test_data/mmcifs/2q2k.cif"
TEMPLATE_RELEASE_DATE = "2008-02-05"


def _query_set(tmp_path: Path, hits: list[str]) -> InferenceQuerySet:
    """One query with 2q2k chain B's sequence and a ColabFold alignment of `hits`.

    Each hit (e.g. "2q2k_C") is a full-length, identical match, listed in order.
    """
    seq = get_asym_id_to_canonical_seq_dict(_load_ciffile(TEMPLATE_CIF))["B"]
    n = len(seq)
    aln_path = tmp_path / "hits.m8"
    aln_path.write_text(
        "".join(
            f"101\t{hit}\t1.000\t{n}\t0\t0\t1\t{n}\t1\t{n}\t1.0E-40\t300\t{n}M\n"
            for hit in hits
        )
    )
    chain = Chain(
        molecule_type=MoleculeType.PROTEIN,
        chain_ids=["A"],
        sequence=seq,
        template_alignment_file_path=aln_path,
    )
    return InferenceQuerySet(queries={"query": Query(chains=[chain])})


def _settings(
    tmp_path: Path, *, with_structure: bool = True, **overrides
) -> TemplatePreprocessorSettings:
    """Offline settings with thread workers; 2q2k is in the structure directory
    unless `with_structure` is False."""
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
    if with_structure:
        shutil.copy(TEMPLATE_CIF, structure_dir)
    return settings


def _preprocess(query_set: InferenceQuerySet, settings) -> Chain:
    """Preprocesses the query set and returns its only chain."""
    TemplatePreprocessor(input_set=query_set, config=settings)()
    return query_set.queries["query"].chains[0]


def _download_2q2k(*, pdb_ids: str, target_path: Path, **_) -> None:
    """Stands in for a successful download, saving 2q2k's file as `pdb_ids`."""
    shutil.copy(TEMPLATE_CIF, target_path / f"{pdb_ids}.cif")


@pytest.mark.parametrize(
    ("structure_file", "precache_entry", "template_ids"),
    [
        pytest.param(True, False, ["2q2k_C"], id="structure_file"),
        # A bug, see the docstring
        pytest.param(False, True, ["2q2k_C"], id="precache_entry_only"),
        pytest.param(False, False, [], id="neither"),
    ],
)
def test_template_needs_its_structure_file_or_precache_entry(
    tmp_path, structure_file, precache_entry, template_ids
):
    """With fetching off, a template is kept if either is on disk.

    `precache_entry_only` pins a bug: the loop's comment says a precache entry counts
    only together with structure arrays, which aren't checked. Featurization still
    needs the structure file, so this template later raises FileNotFoundError there,
    which invalidates the whole query.
    """
    precache_dir = tmp_path / "precache"
    precache_dir.mkdir()
    if precache_entry:
        np.savez_compressed(
            precache_dir / "2q2k.npz",
            release_date=TEMPLATE_RELEASE_DATE,
            chain_id_seq_map=get_asym_id_to_canonical_seq_dict(
                _load_ciffile(TEMPLATE_CIF)
            ),
        )
    settings = _settings(
        tmp_path, with_structure=structure_file, precache_directory=precache_dir
    )

    with patch.object(template_module, "fetch") as fetch:
        chain = _preprocess(_query_set(tmp_path, ["2q2k_C"]), settings)

    fetch.assert_not_called()
    assert chain.template_entry_chain_ids == template_ids


@pytest.mark.parametrize(
    ("download", "template_ids"),
    [
        pytest.param(_download_2q2k, ["9xyz_C", "2q2k_C"], id="downloaded"),
        pytest.param(RequestError("No entries found"), ["2q2k_C"], id="unknown_pdb_id"),
        # A bug, see the docstring
        pytest.param(ConnectionError("Network unreachable"), [], id="network_error"),
    ],
)
def test_missing_structure_is_downloaded_when_fetching_is_on(
    tmp_path, download, template_ids
):
    """Of the two hits, only 9xyz's structure is missing, so only it is downloaded.

    `network_error` pins a bug: only RequestError is caught around `fetch`, so any
    other download error aborts the chain, losing 2q2k_C too although its structure
    is on disk.
    """
    settings = _settings(tmp_path, fetch_missing_structures=True)

    with patch.object(template_module, "fetch", side_effect=download) as fetch:
        chain = _preprocess(_query_set(tmp_path, ["9xyz_C", "2q2k_C"]), settings)

    fetch.assert_called_once_with(
        pdb_ids="9xyz", format="cif", target_path=settings.structure_directory
    )
    assert chain.template_entry_chain_ids == template_ids


@pytest.mark.parametrize(
    ("setting", "output_directory", "written"),
    [
        pytest.param(
            "create_precache", "precache_directory", ["2q2k.npz"], id="precache_entry"
        ),
        # Only the protein chains: chain A is nucleic acid
        pytest.param(
            "preparse_structures",
            "structure_array_directory",
            ["2q2k/2q2k_B.npz", "2q2k/2q2k_C.npz", "2q2k/chain_id_to_moltype.npz"],
            id="structure_arrays",
        ),
    ],
)
def test_structure_file_is_preprocessed_for_later_runs(
    tmp_path, setting, output_directory, written
):
    settings = _settings(tmp_path, **{setting: True})

    chain = _preprocess(_query_set(tmp_path, ["2q2k_C"]), settings)

    assert chain.template_entry_chain_ids == ["2q2k_C"]
    out = getattr(settings, output_directory)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.npz")) == written


def test_existing_cache_entry_is_reused(tmp_path):
    settings = _settings(tmp_path)
    _preprocess(_query_set(tmp_path, ["2q2k_C"]), settings)
    # From here on, the template can only come from the first run's cache entry
    (settings.structure_directory / "2q2k.cif").unlink()

    chain = _preprocess(_query_set(tmp_path, ["2q2k_C"]), settings)

    assert chain.template_entry_chain_ids == ["2q2k_C"]
