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

"""Tests for checking the names of user-provided MSA alignments at inference.

Within a chain's MSA files, alignments are selected by name: the file stem for
.sto/.a3m files, the array name for .npz files. Names the MSA settings do not
list used to be dropped silently, and a chain left without a usable alignment
crashed deep in the pipeline (IndexError / "need at least one array to
concatenate"). `MsaSampleParserInference.check_alignment_names` now raises an
actionable error for unusable MSAs and warns about ignored alignments.

All tests run against the default `MSASettings`, as users get them.
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

from openfold3.core.config.msa_pipeline_configs import (
    MsaKind,
    MsaSampleProcessorInputInference,
)
from openfold3.core.data.io.sequence.msa import MsaSampleParserInference
from openfold3.core.data.primitives.quality_control.msa_issues import (
    MsaIssueKind,
    MsaValidationError,
    Severity,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.projects.of3_all_atom.config.inference_query_format import Query
from openfold3.tests.utils.msa_helpers import (
    MSA_FORMATS,
    pairable_alignment,
    parse,
    process,
    write_msas,
)

SEQUENCES = {
    "long": "ACDEFGHIKLMNPQRSTVWY",
    "medium": "MKTAYIAKQRQISFVK",
    "short": "GSHMLEDP",
}

MSA_SETTINGS = MSASettings()


@dataclass(frozen=True)
class ChainMsas:
    """One chain of a query, and the names of the alignments in its MSA files."""

    sequence_key: str
    main: tuple[str, ...] = ()
    paired: tuple[str, ...] = ()


def _chains(*specs: tuple[str, str | tuple[str, ...], str | tuple[str, ...]]):
    """Chains from (sequence_key, main, paired) tuples; a single alignment name
    may be given as a plain string."""

    def names(value: str | tuple[str, ...]) -> tuple[str, ...]:
        return (value,) if isinstance(value, str) else value

    return tuple(
        ChainMsas(seq, names(main), names(paired)) for seq, main, paired in specs
    )


GOOD_MAIN = ("uniref90_hits", "uniprot_hits")


def _build_query(root: Path, fmt: str, chains: tuple[ChainMsas, ...], **flags) -> Query:
    """Builds a query with one entry per chain, writing an alignment for every
    listed name. Every alignment carries species information, so whether a
    chain gets paired depends on its alignment names only. MSA files are stored
    per sequence, so copies of a sequence share them."""
    entries = []
    for chain_id, chain in zip("ABCDEFGH", chains, strict=False):
        sequence = SEQUENCES[chain.sequence_key]
        entry: dict = {
            "molecule_type": "protein",
            "chain_ids": [chain_id],
            "sequence": sequence,
        }
        for msa_kind, names in [
            (MsaKind.MAIN, chain.main),
            (MsaKind.PAIRED, chain.paired),
        ]:
            if names:
                entry[f"{msa_kind}_msa_file_paths"] = write_msas(
                    fmt,
                    root / msa_kind / chain.sequence_key,
                    {name: pairable_alignment(sequence) for name in names},
                )
        entries.append(entry)
    return Query.model_validate({"query_name": "q", "chains": entries, **flags})


fmt_parametrize = pytest.mark.parametrize("fmt", list(MSA_FORMATS))


@fmt_parametrize
@pytest.mark.parametrize(
    "chains, match",
    [
        pytest.param(
            _chains(("long", "chainA", ())),
            r"can be used as a main MSA\. Found alignments: \['chainA'\]",
            id="main_unknown_name",
        ),
        pytest.param(
            _chains(("long", ("chainA", "my_db"), ())),
            r"can be used as a main MSA\. Found alignments: \['chainA', 'my_db'\]",
            id="main_several_unknown_names",
        ),
        pytest.param(
            _chains(("long", "uniprot_hits", ())),
            r"\['uniprot_hits'\] are only used as named in msas_to_pair",
            id="main_pairing_alignment_only",
        ),
        pytest.param(
            _chains(("long", "colabfold_paired", ())),
            r"can be used as a main MSA\. Found alignments: \['colabfold_paired'\]",
            id="main_paired_alignment_given_as_main",
        ),
        pytest.param(
            _chains(("long", GOOD_MAIN, "pair"), ("short", GOOD_MAIN, "pair")),
            r"can be used as a paired MSA\. Found alignments: \['pair'\]",
            id="paired_unknown_name",
        ),
        pytest.param(
            _chains(
                ("long", GOOD_MAIN, "uniprot_hits"),
                ("short", GOOD_MAIN, "uniprot_hits"),
            ),
            r"can be used as a paired MSA\. Found alignments: \['uniprot_hits'\]",
            id="paired_main_alignment_given_as_paired",
        ),
    ],
)
def test_unusable_alignment_names_raise(tmp_path, fmt, chains, match):
    """An MSA without any alignment the settings use fails before parsing, with a
    message naming the alignments found and the settings that select them."""
    query = _build_query(tmp_path, fmt, chains)

    with pytest.raises(ValueError, match=match) as excinfo:
        process(query, MSA_SETTINGS)

    assert "dataset_config_kwargs.msa" in str(excinfo.value)


@fmt_parametrize
def test_raw_alignments_must_be_named_in_max_seq_counts(tmp_path, fmt):
    """ "dummy" is in aln_order but not in max_seq_counts: the ColabFold runner
    only ever writes it to .npz files, and raw .sto/.a3m files are only read if
    named in max_seq_counts."""
    query = _build_query(tmp_path, fmt, _chains(("long", "dummy", ())))

    if fmt == "npz":
        process(query, MSA_SETTINGS)
    else:
        with pytest.raises(
            ValueError,
            match=r"must also be named in max_seq_counts, which \['dummy'\] are not",
        ):
            process(query, MSA_SETTINGS)


def test_messages_name_unnamed_queries(tmp_path):
    """A Query built without a name, rather than loaded from a query file, is
    named as such in MSA messages."""
    query = Query.model_validate(
        {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": SEQUENCES["long"],
                    "main_msa_file_paths": write_msas(
                        "sto",
                        tmp_path / "A",
                        {"chainA": pairable_alignment(SEQUENCES["long"])},
                    ),
                }
            ]
        }
    )
    assert query.query_name is None

    with pytest.raises(ValueError, match=r"in query <unnamed>"):
        process(query, MSA_SETTINGS)


def _settings_with_counts(**counts: int) -> MSASettings:
    """The default MSA settings with some max_seq_counts entries overridden."""
    return MSASettings(max_seq_counts={**MSA_SETTINGS.max_seq_counts, **counts})


@fmt_parametrize
@pytest.mark.parametrize(
    "chains, counts, expected",
    [
        pytest.param(
            _chains(("long", ("uniref90_hits", "mgnify_hits"), ())),
            {"uniref90_hits": 0},
            {"A": {"mgnify_hits"}},
            id="first_alignment_in_aln_order",
        ),
        pytest.param(
            _chains(("long", ("uniref90_hits", "mgnify_hits"), ())),
            {"mgnify_hits": 0},
            {"A": {"uniref90_hits"}},
            id="other_alignment",
        ),
        pytest.param(
            _chains(
                ("long", ("uniref90_hits", "mgnify_hits"), ()),
                ("short", ("uniref90_hits", "mgnify_hits"), ()),
            ),
            {"mgnify_hits": 0, "uniref90_hits": 0},
            None,
            id="all_alignments_of_a_chain",
        ),
    ],
)
def test_zero_max_seq_count_disables_alignment(tmp_path, fmt, chains, counts, expected):
    """A max_seq_counts entry of 0 disables the alignment of that name, for raw
    and .npz files alike, and is not reported as an ignored alignment.

    Raw files used to be truncated to no rows and crash with an IndexError where
    their (empty) first row was read, and .npz files ignored the entry.
    """
    query = _build_query(tmp_path, fmt, chains)
    settings = _settings_with_counts(**counts)

    if expected is None:
        with pytest.raises(
            ValueError,
            match=r"can be used as a main MSA\. .*\['mgnify_hits', 'uniref90_hits'\] "
            r"are disabled by a max_seq_counts of 0",
        ):
            process(query, settings)
        return

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        collection = parse(query, settings)
        process(query, settings)

    assert {
        rep_id: set(msas) for rep_id, msas in collection.rep_id_to_main_msa.items()
    } == expected


@fmt_parametrize
def test_zero_max_seq_count_disables_paired_alignment(tmp_path, fmt):
    query = _build_query(
        tmp_path,
        fmt,
        _chains(
            ("long", GOOD_MAIN, "colabfold_paired"),
            ("short", GOOD_MAIN, "colabfold_paired"),
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"can be used as a paired MSA\. .*\['colabfold_paired'\] are "
        r"disabled by a max_seq_counts of 0",
    ):
        process(query, _settings_with_counts(colabfold_paired=0))


@fmt_parametrize
def test_zero_max_seq_count_of_pairing_alignment_skips_pairing(tmp_path, fmt):
    """Disabling the alignment used for species pairing leaves a heteromer
    without paired rows, which is reported."""
    query = _build_query(
        tmp_path, fmt, _chains(("long", GOOD_MAIN, ()), ("short", GOOD_MAIN, ()))
    )

    with pytest.warns(
        UserWarning,
        match=r"Protein chain\(s\) A, B .* so no species-paired MSA rows are "
        r"generated for this complex",
    ):
        collection = process(query, _settings_with_counts(uniprot_hits=0))

    assert collection.row_counts.n_rows_paired_subsampled == 0


@fmt_parametrize
@pytest.mark.parametrize(
    "chains, match",
    [
        pytest.param(
            _chains(("long", (*GOOD_MAIN, "my_db"), ())),
            r"Ignoring alignments in the main MSA files .* of chain\(s\) A in query "
            r"q: my_db \(not in aln_order or msas_to_pair\)",
            id="main_extra_alignment",
        ),
        pytest.param(
            _chains(
                ("long", GOOD_MAIN, ("colabfold_paired", "extra")),
                ("short", GOOD_MAIN, "colabfold_paired"),
            ),
            r"Ignoring alignments in the paired MSA files .* of chain\(s\) A in "
            r"query q: extra \(not in paired_msa_order\)",
            id="paired_extra_alignment",
        ),
        pytest.param(
            _chains(("long", "uniref90_hits", ()), ("short", "uniref90_hits", ())),
            r"Protein chain\(s\) A, B in query q have no main MSA alignment named in "
            r"msas_to_pair .*, so no species-paired MSA rows are generated for this "
            r"complex",
            id="heteromer_without_pairing_alignments",
        ),
        pytest.param(
            _chains(
                ("long", GOOD_MAIN, ()),
                ("medium", GOOD_MAIN, ()),
                ("short", "uniref90_hits", ()),
            ),
            r"Protein chain\(s\) C in query q have no main MSA alignment named in "
            r"msas_to_pair .*, so these chains get no species-paired MSA rows",
            id="heteromer_chain_without_pairing_alignment",
        ),
    ],
)
def test_unused_alignments_warn(tmp_path, fmt, chains, match):
    """Alignments that are found but not used are reported, and the query still
    runs on the alignments that are used."""
    query = _build_query(tmp_path, fmt, chains)

    with pytest.warns(UserWarning, match=match):
        process(query, MSA_SETTINGS)


@fmt_parametrize
@pytest.mark.parametrize(
    "chains",
    [
        pytest.param(_chains(("long", GOOD_MAIN, ())), id="monomer"),
        pytest.param(
            _chains(("long", "uniref90_hits", ()), ("long", "uniref90_hits", ())),
            id="homomer_without_pairing_alignments",
        ),
        pytest.param(
            _chains(("long", GOOD_MAIN, ()), ("short", GOOD_MAIN, ())),
            id="heteromer",
        ),
        pytest.param(
            _chains(
                ("long", "uniref90_hits", "colabfold_paired"),
                ("short", "uniref90_hits", "colabfold_paired"),
            ),
            id="heteromer_with_precomputed_paired_msas",
        ),
    ],
)
def test_usable_alignments_do_not_warn(tmp_path, fmt, chains):
    """Homomers and heteromers with precomputed paired MSAs are not paired
    online, so they need no alignment named in msas_to_pair."""
    query = _build_query(tmp_path, fmt, chains)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        process(query, MSA_SETTINGS)


@fmt_parametrize
@pytest.mark.parametrize(
    "chains, flags",
    [
        pytest.param(
            _chains(("long", ("uniprot_hits", "my_db"), ())),
            {"use_main_msas": False},
            id="main_disabled",
        ),
        pytest.param(
            _chains(("long", GOOD_MAIN, "pair"), ("short", GOOD_MAIN, "pair")),
            {"use_paired_msas": False},
            id="paired_disabled",
        ),
        pytest.param(
            _chains(("long", "uniref90_hits", ()), ("short", "uniref90_hits", ())),
            {"use_paired_msas": False},
            id="pairing_disabled",
        ),
    ],
)
def test_disabled_msa_types_are_not_checked(tmp_path, fmt, chains, flags):
    query = _build_query(tmp_path, fmt, chains, **flags)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        process(query, MSA_SETTINGS)


@pytest.mark.parametrize("fmt", ["sto", "sto_dir"])
def test_unreadable_alignments_raise_even_when_main_msas_are_disabled(tmp_path, fmt):
    """The query row is read from the main MSA files, so they must hold at least
    one alignment that is parsed at all."""
    query = _build_query(
        tmp_path, fmt, _chains(("long", "chainA", ())), use_main_msas=False
    )

    with pytest.raises(ValueError, match=re.escape("Found alignments: ['chainA']")):
        process(query, MSA_SETTINGS)


@fmt_parametrize
def test_unusable_msas_of_all_chains_are_reported_together(tmp_path, fmt):
    """One error lists every representative with unusable MSA files, rather than
    stopping at the first."""
    query = _build_query(
        tmp_path, fmt, _chains(("long", "chainA", ()), ("short", "my_db", ()))
    )

    with pytest.raises(MsaValidationError) as excinfo:
        process(query, MSA_SETTINGS)

    assert [(i.kind, i.chain_ids) for i in excinfo.value.issues] == [
        (MsaIssueKind.NO_USABLE_ALIGNMENT, ("A",)),
        (MsaIssueKind.NO_USABLE_ALIGNMENT, ("B",)),
    ]
    # The hint is shared, so it is given once
    assert str(excinfo.value).count("dataset_config_kwargs.msa") == 1


@pytest.mark.parametrize("fmt", ["sto", "sto_dir"])
def test_unreadable_query_source_is_reported_as_such(tmp_path, fmt):
    """With main MSAs disabled, the error says the files are needed for the query
    sequence, not that main MSAs are made from aln_order."""
    query = _build_query(
        tmp_path, fmt, _chains(("long", "chainA", ())), use_main_msas=False
    )

    with pytest.raises(MsaValidationError) as excinfo:
        process(query, MSA_SETTINGS)

    (issue,) = excinfo.value.issues
    assert issue.kind is MsaIssueKind.NO_READABLE_ALIGNMENT
    assert "query sequence" in issue.message
    assert "aln_order" not in issue.message


@fmt_parametrize
def test_check_alignment_names_returns_issues(tmp_path, fmt):
    """The check itself only returns issues; create_maps reports them."""
    query = _build_query(
        tmp_path,
        fmt,
        _chains(("long", (*GOOD_MAIN, "my_db"), ()), ("short", "uniref90_hits", ())),
    )
    input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        inference_query=query
    )
    parser = MsaSampleParserInference(config=MSA_SETTINGS)
    with pytest.warns(UserWarning):
        maps = parser.create_maps(input=input)

    issues = parser.check_alignment_names(input=input, maps=maps)

    assert [(i.kind, i.chain_ids, i.severity) for i in issues] == [
        (MsaIssueKind.IGNORED_ALIGNMENTS, ("A",), Severity.WARNING),
        (MsaIssueKind.UNPAIRABLE_CHAINS, ("B",), Severity.WARNING),
    ]
