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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openfold3.core.config.msa_pipeline_configs import MsaKind
from openfold3.core.data.primitives.featurization.msa import (
    create_msa_feature_precursor_of3,
)
from openfold3.core.data.primitives.sequence.msa import find_monomer_homomer
from openfold3.core.data.primitives.structure.query import (
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import (
    add_token_positions,
    get_token_count,
    tokenize_atom_array,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.projects.of3_all_atom.config.inference_query_format import Query
from openfold3.tests.utils import msa_helpers
from openfold3.tests.utils.msa_helpers import (
    MSA_FORMATS,
    main_alignments,
    paired_alignments,
    write_msas,
)

# Distinct sequences of distinct lengths, standing in for the reporter's heavy
# chain / light chain / lysozyme complex. Lengths differ so that a chain
# borrowing another's MSA is detectable (and, when the borrowed MSA is
# shorter, fatal).
SEQUENCES = {
    "long": "ACDEFGHIKLMNPQRSTVWY",
    "medium": "MKTAYIAKQRQISFVK",
    "short": "GSHMLEDP",
    # Valid as both a protein and an RNA sequence.
    "gca": "GCAGCA",
}

# Keep parsing cheap and deterministic: no main-MSA subsampling, and only the
# alignment types these fixtures write.
MSA_SETTINGS = MSASettings(
    max_seq_counts={"uniprot_hits": 100, "uniref90_hits": 100, "colabfold_paired": 100},
    msas_to_pair=["uniprot_hits"],
    aln_order=["uniprot_hits", "uniref90_hits"],
    paired_msa_order=["colabfold_paired"],
    subsample_main=False,
)


@dataclass(frozen=True)
class Entry:
    """One chain entry of a query.

    `msa_key` names the location of the entry's MSA files, so two entries with
    the same sequence can either share MSA files (same key) or carry copies of
    them in different places (different keys). An entry without an `msa_key`
    has no MSA files at all.
    """

    chain_ids: tuple[str, ...]
    sequence_key: str
    msa_key: str | None
    molecule_type: str = "protein"


def _entries(*specs: tuple[Any, ...]) -> tuple[Entry, ...]:
    """Entries from (chain_ids, sequence_key, msa_key[, molecule_type]) tuples."""
    return tuple(Entry(tuple(spec[0]), *spec[1:]) for spec in specs)


ASSEMBLIES = {
    "heteromer_longest_first": _entries(
        ("A", "long", "long"), ("B", "medium", "medium"), ("C", "short", "short")
    ),
    "heteromer_shortest_first": _entries(
        ("A", "short", "short"), ("B", "medium", "medium"), ("C", "long", "long")
    ),
    # A2B: a homodimer plus a partner, so a correct grouping has to keep A and B
    # together while still separating C.
    "a2b": _entries(("AB", "long", "long"), ("C", "short", "short")),
    "homomer_one_entry": _entries(("AB", "medium", "medium")),
    "homomer_two_entries_shared_msas": _entries(
        ("A", "medium", "medium"), ("B", "medium", "medium")
    ),
    "homomer_two_entries_copied_msas": _entries(
        ("A", "medium", "copy_a"), ("B", "medium", "copy_b")
    ),
    # Same letters, different molecule types: not copies of one another.
    "protein_rna_same_letters": _entries(
        ("A", "gca", "gca_protein", "protein"), ("B", "gca", "gca_rna", "rna")
    ),
    # One copy of a sequence is given MSAs, the other is not.
    "homomer_second_copy_without_msas": _entries(
        ("A", "medium", "medium"), ("B", "medium", None)
    ),
    "homomer_first_copy_without_msas": _entries(
        ("A", "medium", None), ("B", "medium", "medium")
    ),
    "heteromer_one_chain_without_msas": _entries(
        ("A", "long", "long"), ("B", "short", None)
    ),
}

# Where an entry's MSAs live, as a path without suffix. Each format turns it
# into a list of .sto files, the directory containing them, or an .npz file.
#   unique: the layout OF3's own tooling produces, one distinctly named
#           location per MSA key.
#   shared: the reporter's layout, every location is named "target", which
#           made path-derived representatives collide across entries.
LAYOUTS = {
    "unique": lambda root, msa_key: root / f"msas_{msa_key}",
    "shared": lambda root, msa_key: root / msa_key / "target",
}


def _build_query(
    root: Path,
    assembly: str,
    layout: str,
    fmt: str,
    paired_layout: str | None = None,
) -> Query:
    """Builds a query for `assembly`, writing each entry's MSAs to disk.

    Main MSAs are placed according to `layout`. If `paired_layout` is given,
    precomputed paired MSAs are also written, under a separate root.
    """
    chains = []
    for entry in ASSEMBLIES[assembly]:
        sequence = SEQUENCES[entry.sequence_key]
        chain = {
            "molecule_type": entry.molecule_type,
            "chain_ids": list(entry.chain_ids),
            "sequence": sequence,
        }
        if entry.msa_key is None:
            chains.append(chain)
            continue
        chain["main_msa_file_paths"] = write_msas(
            fmt,
            LAYOUTS[layout](root / MsaKind.MAIN, entry.msa_key),
            main_alignments(sequence),
        )
        if paired_layout is not None:
            chain["paired_msa_file_paths"] = write_msas(
                fmt,
                LAYOUTS[paired_layout](root / MsaKind.PAIRED, entry.msa_key),
                paired_alignments(sequence),
            )
        chains.append(chain)
    return Query.model_validate({"query_name": "gh340", "chains": chains})


def _expected_sequences(query: Query) -> dict[str, str]:
    return {
        chain_id: chain.sequence
        for chain in query.chains
        if chain.sequence is not None
        for chain_id in chain.chain_ids
    }


def _chain_identities(query: Query) -> dict[str, tuple[str, str]]:
    """What makes two chains copies of one another: molecule type and sequence."""
    return {
        chain_id: (chain.molecule_type.name, sequence)
        for chain in query.chains
        if (sequence := chain.sequence) is not None
        for chain_id in chain.chain_ids
    }


def _create_maps(query: Query):
    return msa_helpers.create_maps(query, MSA_SETTINGS)


def _process(query: Query):
    return msa_helpers.process(query, MSA_SETTINGS)


def _tokenized_atom_array(query: Query):
    """The tokenized AtomArray the inference dataset would featurize."""
    atom_array, _ = structure_with_ref_mols_from_query(query=query)
    tokenize_atom_array(atom_array)
    add_token_positions(atom_array)
    return atom_array


fmt_parametrize = pytest.mark.parametrize("fmt", list(MSA_FORMATS))
layout_parametrize = pytest.mark.parametrize("layout", list(LAYOUTS))


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
    query = _build_query(tmp_path, assembly, layout, fmt)

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
    query = _build_query(
        tmp_path,
        "homomer_two_entries_copied_msas",
        layout,
        fmt,
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
    query = _build_query(tmp_path, assembly, "unique", fmt, paired_layout="unique")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _create_maps(query)


def _homodimer_query(paths_a: list, paths_b: list) -> Query:
    """Two copies of one sequence in separate entries, with the given MSA paths."""
    return Query.model_validate(
        {
            "query_name": "gh340",
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": [chain_id],
                    "sequence": SEQUENCES["medium"],
                    "main_msa_file_paths": [str(p) for p in paths],
                }
                for chain_id, paths in [("A", paths_a), ("B", paths_b)]
            ],
        }
    )


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
    files = [
        Path(p)
        for p in write_msas(
            fmt, tmp_path / "msas", main_alignments(SEQUENCES["medium"])
        )
    ]
    query = _homodimer_query(*give_paths(files, tmp_path / "msas"))

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
    alignments = main_alignments(SEQUENCES["medium"])
    query = _homodimer_query(
        write_msas(fmt, tmp_path / "a", {"uniprot_hits": alignments["uniprot_hits"]}),
        write_msas(fmt, tmp_path / "b", {"uniref90_hits": alignments["uniref90_hits"]}),
    )

    with pytest.warns(
        UserWarning,
        match=r"Chains A and B in query gh340 have the same sequence but different "
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
    query = _build_query(tmp_path, assembly, "unique", fmt)

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
    to "target", so only the first entry's files were read and the other
    chains reported the first entry's query row as their own.
    """
    query = _build_query(tmp_path, assembly, layout, fmt)

    msa_array_collection = _process(query)

    parsed_queries = {
        chain_id: "".join(msa_array_collection.chain_id_to_query_seq[chain_id].msa[0])
        for chain_id in _expected_sequences(query)
    }
    assert parsed_queries == _expected_sequences(query)


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
    query = _build_query(tmp_path, assembly, layout, fmt)
    msa_array_collection = _process(query)
    atom_array = _tokenized_atom_array(query)

    msa_feature_precursor = create_msa_feature_precursor_of3(
        atom_array=atom_array,
        msa_array_collection=msa_array_collection,
        n_tokens=get_token_count(atom_array),
    )

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
    query = _build_query(tmp_path, assembly, layout, fmt)

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
    query = _build_query(tmp_path, assembly, layout, fmt)

    msa_array_collection = _process(query)

    parsed_queries = {
        chain_id: "".join(msa_array_collection.chain_id_to_query_seq[chain_id].msa[0])
        for chain_id in _expected_sequences(query)
    }
    assert parsed_queries == _expected_sequences(query)
    assert find_monomer_homomer(msa_array_collection)
    assert msa_array_collection.row_counts.n_rows_paired_subsampled == 0


@fmt_parametrize
@layout_parametrize
@pytest.mark.parametrize("paired_layout", list(LAYOUTS))
@assembly_parametrize(*HETEROMERS)
def test_precomputed_paired_msas_belong_to_their_chain(
    tmp_path, fmt, layout, paired_layout, assembly
):
    """Each chain's paired block must come from its own precomputed paired MSA.

    With path-derived representatives, only the main MSA paths counted: a
    shared main layout took the precomputed paired MSAs down with it, even
    when those were laid out uniquely.
    """
    query = _build_query(tmp_path, assembly, layout, fmt, paired_layout=paired_layout)

    msa_array_collection = _process(query)

    paired_queries = {
        chain_id: "".join(msa_array_collection.chain_id_to_paired_msa[chain_id].msa[0])
        for chain_id in _expected_sequences(query)
    }
    assert paired_queries == _expected_sequences(query)
