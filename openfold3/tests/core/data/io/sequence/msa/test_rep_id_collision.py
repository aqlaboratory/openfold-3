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

"""Regression tests for gh-340: user-provided MSAs loaded for the wrong chain.

At inference, `MsaSampleParserInference.create_maps` used to derive each
chain's alignment representative from its MSA *paths*: the parent directory
name for .sto/.a3m files, the file stem for .npz files. Chains whose
representatives coincided were treated as copies of one sequence, and only the
first chain's MSA files were read. That was correct for the layout the
ColabFold runner writes (directories/files named by sequence hash), but not for
arbitrary user layouts:

- different sequences whose MSAs live under identically named directories or
  .npz files collapsed onto one representative (the reporter's case), and
- identical sequences whose MSAs live under differently named locations were
  split into separate representatives.

Representatives are now derived from molecule type and sequence. These tests
pin that down for every assembly under both a unique and a colliding layout.
"""

import os
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from openfold3.core.config.msa_pipeline_configs import MsaKind
from openfold3.core.data.primitives.sequence.msa import find_monomer_homomer
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.projects.of3_all_atom.config.inference_query_format import Query
from openfold3.tests.utils import msa_helpers
from openfold3.tests.utils.msa_helpers import (
    LONG,
    MEDIUM,
    MSA_FORMATS,
    MSA_LAYOUTS,
    SHORT,
    ChainSpec,
    build_query,
)

# Valid as both a protein and an RNA sequence
GCA = "GCAGCA"
MAIN = ("uniprot_hits", "uniref90_hits")

# Keep parsing cheap and deterministic: no main-MSA subsampling, and only the
# alignment types these fixtures write.
MSA_SETTINGS = MSASettings(
    max_seq_counts={"uniprot_hits": 100, "uniref90_hits": 100, "colabfold_paired": 100},
    msas_to_pair=["uniprot_hits"],
    aln_order=["uniprot_hits", "uniref90_hits"],
    paired_msa_order=["colabfold_paired"],
    subsample_main=False,
)

# Distinct sequences of distinct lengths stand in for the reporter's heavy chain /
# light chain / lysozyme complex. Each entry's MSA files are its own unless a
# location is shared (see ChainSpec).
ASSEMBLIES = {
    "heteromer_longest_first": (
        ChainSpec(LONG, MAIN),
        ChainSpec(MEDIUM, MAIN),
        ChainSpec(SHORT, MAIN),
    ),
    "heteromer_shortest_first": (
        ChainSpec(SHORT, MAIN),
        ChainSpec(MEDIUM, MAIN),
        ChainSpec(LONG, MAIN),
    ),
    # A2B: a homodimer plus a partner, so a correct grouping has to keep A and B
    # together while still separating C.
    "a2b": (
        ChainSpec(LONG, MAIN, chain_ids="AB"),
        ChainSpec(SHORT, MAIN, chain_ids="C"),
    ),
    "homomer_one_entry": (ChainSpec(MEDIUM, MAIN, chain_ids="AB"),),
    "homomer_two_entries_shared_msas": (
        ChainSpec(MEDIUM, MAIN, location="medium"),
        ChainSpec(MEDIUM, MAIN, location="medium"),
    ),
    "homomer_two_entries_copied_msas": (
        ChainSpec(MEDIUM, MAIN),
        ChainSpec(MEDIUM, MAIN),
    ),
    # Same letters, different molecule types: not copies of one another.
    "protein_rna_same_letters": (
        ChainSpec(GCA, MAIN),
        ChainSpec(GCA, MAIN, molecule_type="rna"),
    ),
    # One copy of a sequence is given MSAs, the other is not.
    "homomer_second_copy_without_msas": (ChainSpec(MEDIUM, MAIN), ChainSpec(MEDIUM)),
    "homomer_first_copy_without_msas": (ChainSpec(MEDIUM), ChainSpec(MEDIUM, MAIN)),
    "heteromer_one_chain_without_msas": (ChainSpec(LONG, MAIN), ChainSpec(SHORT)),
}


def _build_assembly(
    root: Path,
    assembly: str,
    fmt: str,
    layout: str = "unique",
    paired_layout: str | None = None,
) -> Query:
    """Builds the query of `assembly`, with precomputed paired MSAs for the chains
    with main MSAs if `paired_layout` is given."""
    chains = ASSEMBLIES[assembly]
    if paired_layout is not None:
        chains = tuple(
            replace(chain, paired=("colabfold_paired",)) if chain.main else chain
            for chain in chains
        )
    return build_query(root, fmt, chains, layout=layout, paired_layout=paired_layout)


def _expected_sequences(query: Query) -> dict[str, str]:
    return {
        chain_id: sequence
        for chain in query.chains
        if (sequence := chain.sequence) is not None
        for chain_id in chain.chain_ids
    }


def _chain_identities(query: Query) -> dict[str, tuple[str, str | None]]:
    """What makes two chains copies of one another: molecule type and sequence."""
    return {
        chain_id: (chain.molecule_type.name, chain.sequence)
        for chain in query.chains
        for chain_id in chain.chain_ids
    }


def _create_maps(query: Query):
    return msa_helpers.create_maps(query, MSA_SETTINGS)


def _process(query: Query):
    return msa_helpers.process(query, MSA_SETTINGS)


fmt_parametrize = pytest.mark.parametrize("fmt", list(MSA_FORMATS))
layout_parametrize = pytest.mark.parametrize("layout", list(MSA_LAYOUTS))


def assembly_parametrize(*assemblies: str):
    return pytest.mark.parametrize("assembly", assemblies)


# Copies of a sequence with copied MSA files warn that only the first copy's
# files are read; that warning is tested on its own below.
ignore_copied_msa_warning = pytest.mark.filterwarnings(
    "ignore:Chains .* have the same sequence but different:UserWarning"
)

HETEROMERS = ("heteromer_longest_first", "heteromer_shortest_first", "a2b")
HOMOMERS = (
    "homomer_one_entry",
    "homomer_two_entries_shared_msas",
    "homomer_two_entries_copied_msas",
)


@ignore_copied_msa_warning
@fmt_parametrize
@layout_parametrize
@assembly_parametrize(*HETEROMERS, *HOMOMERS, "protein_rna_same_letters")
def test_representatives_group_chains_by_sequence(tmp_path, fmt, layout, assembly):
    """Chains share a representative exactly when they share a molecule type
    and sequence, regardless of how their MSA files are named on disk."""
    query = _build_assembly(tmp_path, assembly, fmt, layout)

    chain_id_to_rep_id = _create_maps(query).chain_id_to_rep_id

    def partition(chain_id_to_key: dict) -> set[frozenset[str]]:
        groups: dict[object, set[str]] = {}
        for chain_id, key in chain_id_to_key.items():
            groups.setdefault(key, set()).add(chain_id)
        return {frozenset(group) for group in groups.values()}

    assert partition(chain_id_to_rep_id) == partition(_chain_identities(query)), (
        f"representatives {chain_id_to_rep_id} do not match sequences"
    )


@fmt_parametrize
@layout_parametrize
@pytest.mark.parametrize("paired_layout", [None, "unique"], ids=["main", "paired"])
def test_identical_sequences_with_different_msa_files_warn(
    tmp_path, fmt, layout, paired_layout
):
    """Copies of a sequence pointing at different MSA files share the first
    copy's files, and the user is told which files are ignored."""
    query = _build_assembly(
        tmp_path,
        "homomer_two_entries_copied_msas",
        fmt,
        layout,
        paired_layout=paired_layout,
    )
    chain_a, chain_b = query.chains

    with pytest.warns(UserWarning) as record:
        maps = _create_maps(query)

    msa_kinds = [MsaKind.MAIN] if paired_layout is None else list(MsaKind)
    messages = [str(w.message) for w in record]
    for msa_kind in msa_kinds:
        assert any(
            f"same sequence but different {msa_kind} MSA files" in m for m in messages
        ), messages
    assert maps.rep_id_to_main_msa_paths["A"] == chain_a.main_msa_file_paths
    assert maps.rep_id_to_paired_msa_paths.get("A") == chain_a.paired_msa_file_paths
    assert chain_b.main_msa_file_paths != chain_a.main_msa_file_paths


@fmt_parametrize
@assembly_parametrize("homomer_one_entry", "homomer_two_entries_shared_msas")
def test_identical_sequences_sharing_msa_files_do_not_warn(tmp_path, fmt, assembly):
    query = _build_assembly(tmp_path, assembly, fmt, paired_layout="unique")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _create_maps(query)


def _relative_and_absolute(files: list[Path], directory: Path) -> tuple[list, list]:
    """The files as absolute paths, and as paths relative to the current working
    directory, which MSA paths are resolved against."""
    return files, [os.path.relpath(p) for p in files]


@pytest.mark.parametrize(
    "fmt, give_paths",
    [
        pytest.param(
            "sto", lambda files, directory: (files, files[::-1]), id="sto-reordered"
        ),
        pytest.param(
            "sto",
            lambda files, directory: ([directory], files),
            id="sto-directory_and_its_files",
        ),
        pytest.param(
            "sto", _relative_and_absolute, id="sto-relative_and_absolute_paths"
        ),
        pytest.param(
            "npz", _relative_and_absolute, id="npz-relative_and_absolute_paths"
        ),
    ],
)
def test_identical_sequences_with_same_files_given_differently_do_not_warn(
    tmp_path, fmt, give_paths
):
    """Copies of a sequence whose MSA paths differ but hold the same alignment
    files share them, with all of their sources, without a warning."""
    query = build_query(tmp_path, fmt, [ChainSpec(MEDIUM, MAIN, location="medium")] * 2)
    chain_a, chain_b = query.chains
    files = chain_a.main_msa_file_paths
    chain_a.main_msa_file_paths, chain_b.main_msa_file_paths = give_paths(
        files, files[0].parent
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        collection = msa_helpers.parse(query, MSA_SETTINGS)

    assert collection.chain_id_to_rep_id == {"A": "A", "B": "A"}
    assert set(collection.rep_id_to_main_msa["A"]) == {"uniprot_hits", "uniref90_hits"}


@fmt_parametrize
def test_identical_sequences_with_split_sources_use_the_first_copys(tmp_path, fmt):
    """Copies of a sequence never pool their MSA files: if they split the
    sources of an MSA between them, only the first copy's are used, and the
    others are reported."""
    query = build_query(
        tmp_path,
        fmt,
        [ChainSpec(MEDIUM, ("uniprot_hits",)), ChainSpec(MEDIUM, ("uniref90_hits",))],
    )

    with pytest.warns(
        UserWarning,
        match=r"Chains A and B in query q have the same sequence but different "
        r"main MSA files",
    ):
        collection = msa_helpers.parse(query, MSA_SETTINGS)

    assert set(collection.rep_id_to_main_msa["A"]) == {"uniprot_hits"}


@fmt_parametrize
@pytest.mark.parametrize(
    "assembly, chain_without_msas, copy_with_msas",
    [
        pytest.param("homomer_second_copy_without_msas", "B", "A", id="second_copy"),
        pytest.param("homomer_first_copy_without_msas", "A", "B", id="first_copy"),
        pytest.param("heteromer_one_chain_without_msas", "B", None, id="heteromer"),
    ],
)
def test_chain_without_msas_warns(
    tmp_path, fmt, assembly, chain_without_msas, copy_with_msas
):
    """A chain without MSA files gets no MSA features. If another chain with the
    same sequence has MSA files, the paths were most likely left out by mistake,
    so the warning points at that chain, whichever comes first in the query."""
    query = _build_assembly(tmp_path, assembly, fmt)

    with pytest.warns(UserWarning, match="no MSA files found") as record:
        maps = _create_maps(query)

    assert chain_without_msas not in maps.chain_id_to_rep_id
    (message,) = [str(w.message) for w in record if "no MSA files" in str(w.message)]
    assert f"chain {chain_without_msas} " in message
    if copy_with_msas is None:
        assert "same sequence" not in message
    else:
        assert f"Chain {copy_with_msas} has the same sequence" in message


@fmt_parametrize
@layout_parametrize
@assembly_parametrize(*HETEROMERS)
def test_each_chain_keeps_its_own_msa(tmp_path, fmt, layout, assembly):
    """Each chain's parsed MSA must be the one its own query entry points to.

    With path-derived representatives, the shared layout resolved every entry
    to "target", so only the first entry's files were read, and the other
    chains got alignments of the first entry's sequence.
    """
    query = _build_assembly(tmp_path, assembly, fmt, layout)

    collection = msa_helpers.parse(query, MSA_SETTINGS)

    first_rows = {
        chain_id: {
            name: "".join(msa.msa[0])
            for name, msa in collection.rep_id_to_main_msa[rep_id].items()
        }
        for chain_id, rep_id in collection.chain_id_to_rep_id.items()
    }
    assert first_rows == {
        chain_id: dict.fromkeys(MAIN, sequence)
        for chain_id, sequence in _expected_sequences(query).items()
    }


@fmt_parametrize
@layout_parametrize
@assembly_parametrize(*HETEROMERS)
def test_msa_features_spell_each_chain_sequence(tmp_path, fmt, layout, assembly):
    """Row 0 of the MSA feature block is the query row, so it must spell the
    assembly's own sequences.

    This is where the reporter's traceback came from: with the shortest chain
    first, the borrowed MSA had fewer columns than the later chains have
    tokens and `map_msas_to_tokens` raised IndexError. With the longest chain
    first there was no error and the feature block quietly carried the wrong
    residues for the later chains.
    """
    query = _build_assembly(tmp_path, assembly, fmt, layout)

    msa_feature_precursor = msa_helpers.features(query, MSA_SETTINGS)

    query_row = "".join(msa_feature_precursor.msa[0])
    assert query_row == "".join(_expected_sequences(query).values())


@fmt_parametrize
@layout_parametrize
@assembly_parametrize(*HETEROMERS)
def test_heteromer_msas_are_paired(tmp_path, fmt, layout, assembly):
    """A heteromer whose chains share a species must produce paired rows.

    Collapsed representatives also suppressed pairing outright:
    `find_monomer_homomer` saw a single representative, called the complex a
    homomer and skipped pairing, so the regression was silent even in the
    chain order that did not crash.
    """
    query = _build_assembly(tmp_path, assembly, fmt, layout)

    msa_array_collection = _process(query)

    n_rows_paired = msa_array_collection.row_counts.n_rows_paired_subsampled
    assert n_rows_paired > 0, "heteromer produced no paired MSA rows"

    for chain_id, sequence in _expected_sequences(query).items():
        paired_msa = msa_array_collection.chain_id_to_paired_msa[chain_id].msa
        assert paired_msa.shape == (n_rows_paired, len(sequence))
        assert np.any(paired_msa != "-"), f"chain {chain_id} paired block is all gaps"


@ignore_copied_msa_warning
@fmt_parametrize
@layout_parametrize
@assembly_parametrize(*HOMOMERS)
def test_homomer_msas_are_not_paired(tmp_path, fmt, layout, assembly):
    """A homomer's main MSAs are implicitly paired, so online species pairing
    must not run, however the copies' MSA files are laid out.

    With path-derived representatives, copied MSAs in differently named
    locations gave the copies separate representatives, `find_monomer_homomer`
    called the complex a heteromer, and the homomer was species-paired with
    itself.
    """
    query = _build_assembly(tmp_path, assembly, fmt, layout)

    msa_array_collection = _process(query)

    assert find_monomer_homomer(msa_array_collection)
    assert msa_array_collection.row_counts.n_rows_paired_subsampled == 0


@fmt_parametrize
@layout_parametrize
@pytest.mark.parametrize("paired_layout", list(MSA_LAYOUTS))
@assembly_parametrize(*HETEROMERS)
def test_precomputed_paired_msas_belong_to_their_chain(
    tmp_path, fmt, layout, paired_layout, assembly
):
    """Each chain's paired block must come from its own precomputed paired MSA.

    With path-derived representatives, only the main MSA paths counted: a
    shared main layout took the precomputed paired MSAs down with it, even
    when those were laid out uniquely.
    """
    query = _build_assembly(tmp_path, assembly, fmt, layout, paired_layout)

    msa_array_collection = _process(query)

    paired_queries = {
        chain_id: "".join(msa_array_collection.chain_id_to_paired_msa[chain_id].msa[0])
        for chain_id in _expected_sequences(query)
    }
    assert paired_queries == _expected_sequences(query)
