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

"""Tests for checking user-provided MSAs against the sequences they are used for.

The first row of an MSA is its query sequence, and MSA columns are mapped onto
a chain's residues by position. An MSA built for a sequence of another length
(e.g. a chain pointed at another chain's files) used to crash featurization
with an IndexError, or silently misalign when it was longer.
`MsaSampleProcessorInference.check_query_rows` now raises for those. An MSA of
the right length built for a different sequence is legitimate when reusing a
wild-type MSA for point mutants, so it only warns if a single residue or at
most 10% of residues differ, and raises otherwise.
"""

import re
import warnings
from pathlib import Path

import pytest

from openfold3.core.config.msa_pipeline_configs import MsaKind
from openfold3.core.data.pipelines.sample_processing.msa import (
    find_query_row_mismatches,
    is_tolerated_query_row,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.projects.of3_all_atom.config.inference_query_format import Query
from openfold3.tests.utils.msa_helpers import (
    MSA_FORMATS,
    mutate,
    pairable_alignment,
    process,
    write_msas,
)

LONG = "ACDEFGHIKLMNPQRSTVWY"
SHORT = "GSHMLEDP"
RNA = "ACGUACGUAC"

MSA_SETTINGS = MSASettings()


def _chain(
    sequence: str,
    main: dict[str, str] | None = None,
    paired: dict[str, str] | None = None,
    molecule_type: str = "protein",
) -> dict:
    """A chain whose MSAs map alignment names to the sequence each alignment
    was built for, i.e. its query row."""
    return {
        "sequence": sequence,
        "molecule_type": molecule_type,
        MsaKind.MAIN: main or {},
        MsaKind.PAIRED: paired or {},
    }


def _build_query(root: Path, fmt: str, chains: list[dict], **flags) -> Query:
    entries = []
    for chain_id, chain in zip("ABCD", chains, strict=False):
        entry = {
            "molecule_type": chain["molecule_type"],
            "chain_ids": [chain_id],
            "sequence": chain["sequence"],
        }
        for msa_kind in MsaKind:
            if chain[msa_kind]:
                entry[f"{msa_kind}_msa_file_paths"] = write_msas(
                    fmt,
                    root / msa_kind / chain_id,
                    {
                        name: pairable_alignment(msa_sequence)
                        for name, msa_sequence in chain[msa_kind].items()
                    },
                )
        entries.append(entry)
    return Query.model_validate({"query_name": "q", "chains": entries, **flags})


def _mutate_at(sequence: str, positions: list[int]) -> str:
    for position in positions:
        sequence = mutate(sequence, position, "W")
    return sequence


def _main(msa_sequence: str) -> dict[str, str]:
    return {"uniref90_hits": msa_sequence, "uniprot_hits": msa_sequence}


fmt_parametrize = pytest.mark.parametrize("fmt", list(MSA_FORMATS))


@fmt_parametrize
@pytest.mark.parametrize(
    "chains, match",
    [
        pytest.param(
            [_chain(LONG, _main(LONG)), _chain(SHORT, _main(LONG))],
            r"chain\(s\) B, main 'uniprot_hits' from .*: the query row has 20 "
            r"residues, the chain's sequence 8",
            id="chain_given_another_chains_msas",
        ),
        pytest.param(
            [_chain(LONG, _main(LONG[:-2]))],
            r"chain\(s\) A, main 'uniprot_hits' from .*: the query row has 18 "
            r"residues, the chain's sequence 20",
            id="msa_of_truncated_sequence",
        ),
        pytest.param(
            [_chain(LONG, {"uniref90_hits": LONG, "mgnify_hits": SHORT})],
            r"chain\(s\) A, main 'mgnify_hits' from .*: the query row has 8 "
            r"residues",
            id="one_of_several_alignments_of_another_sequence",
        ),
        pytest.param(
            [
                _chain(LONG, _main(LONG), {"colabfold_paired": LONG}),
                _chain(SHORT, _main(SHORT), {"colabfold_paired": LONG}),
            ],
            r"chain\(s\) B, paired 'colabfold_paired' from .*: the query row has "
            r"20 residues, the chain's sequence 8",
            id="chain_given_another_chains_paired_msa",
        ),
        pytest.param(
            [_chain(LONG, {"uniref90_hits": LONG, "mgnify_hits": "W" * len(LONG)})],
            r"chain\(s\) A, main 'mgnify_hits' from .*: 19 of 20 residues differ "
            r"\(A1W, C2W, D3W, E4W, F5W, \.\.\.\)",
            id="msa_of_unrelated_sequence_of_same_length",
        ),
        pytest.param(
            [_chain(LONG, _main(_mutate_at(LONG, [3, 8, 15])))],
            r"chain\(s\) A, main 'uniprot_hits', main 'uniref90_hits' from .*: 3 of "
            r"20 residues differ \(E4W, K9W, S16W\)",
            id="three_of_20_residues_differ",
        ),
        pytest.param(
            [_chain(SHORT, _main(_mutate_at(SHORT, [1, 5])))],
            r"chain\(s\) A, .*: 2 of 8 residues differ \(S2W, E6W\)",
            id="two_of_8_residues_differ",
        ),
    ],
)
def test_mismatching_query_row_raises(tmp_path, fmt, chains, match):
    query = _build_query(tmp_path, fmt, chains)

    with pytest.raises(ValueError, match=match):
        process(query, MSA_SETTINGS)


@fmt_parametrize
def test_all_query_row_errors_are_reported_at_once(tmp_path, fmt):
    """Swapped MSA paths are reported for both chains in a single error."""
    query = _build_query(
        tmp_path,
        fmt,
        [_chain(LONG, _main(SHORT)), _chain(SHORT, _main(LONG))],
    )

    with pytest.raises(ValueError) as excinfo:
        process(query, MSA_SETTINGS)

    message = str(excinfo.value)
    assert "chain(s) A, main 'uniprot_hits'" in message
    assert "chain(s) B, main 'uniprot_hits'" in message


@fmt_parametrize
@pytest.mark.parametrize(
    "chains, match",
    [
        pytest.param(
            [_chain(LONG, _main(mutate(LONG, 3, "W")))],
            r"for chain\(s\) A in query q differ from the chain's sequence, given "
            r"as <chain residue><position><MSA residue>: main 'uniprot_hits', main "
            r"'uniref90_hits': 1 of 20 residues differ \(E4W\)",
            id="point_mutant_reusing_wild_type_msas",
        ),
        pytest.param(
            [_chain(LONG, _main(_mutate_at(LONG, [3, 8])))],
            r"main 'uniprot_hits', main 'uniref90_hits': 2 of 20 residues differ "
            r"\(E4W, K9W\)",
            id="two_of_20_residues_differ",
        ),
        pytest.param(
            [_chain(SHORT, _main(_mutate_at(SHORT, [1])))],
            r"1 of 8 residues differ \(S2W\)",
            id="single_mutation_in_peptide",
        ),
        pytest.param(
            [
                _chain(LONG, _main(LONG), {"colabfold_paired": mutate(LONG, 0, "M")}),
                _chain(SHORT, _main(SHORT), {"colabfold_paired": SHORT}),
            ],
            r"chain\(s\) A .*: paired 'colabfold_paired': 1 of 20 residues differ "
            r"\(A1M\)",
            id="paired_msa_of_related_sequence",
        ),
    ],
)
def test_tolerated_query_row_warns(tmp_path, fmt, chains, match):
    query = _build_query(tmp_path, fmt, chains)

    with pytest.warns(UserWarning, match=match):
        process(query, MSA_SETTINGS)


@fmt_parametrize
@pytest.mark.parametrize(
    "chains",
    [
        pytest.param([_chain(LONG, _main(LONG))], id="monomer"),
        pytest.param(
            [
                _chain(LONG, _main(LONG), {"colabfold_paired": LONG}),
                _chain(SHORT, _main(SHORT), {"colabfold_paired": SHORT}),
            ],
            id="heteromer_with_precomputed_paired_msas",
        ),
        pytest.param(
            [_chain(LONG, _main(mutate(LONG, 3, "X")))],
            id="unknown_residue_in_msa",
        ),
        pytest.param(
            [_chain(mutate(LONG, 3, "X"), _main(LONG))],
            id="unknown_residue_in_sequence",
        ),
        pytest.param(
            [_chain(RNA, {"rfam_hits": RNA.replace("U", "T")}, molecule_type="rna")],
            id="rna_msa_spelling_uracil_as_thymine",
        ),
        pytest.param(
            # Only aln_order and msas_to_pair alignments are used, and checked
            [_chain(LONG, {**_main(LONG), "my_db": SHORT})],
            id="unused_alignment_of_another_sequence",
        ),
    ],
)
def test_matching_query_rows_do_not_warn(tmp_path, fmt, chains):
    query = _build_query(tmp_path, fmt, chains)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", "Ignoring alignments", UserWarning)
        process(query, MSA_SETTINGS)


@fmt_parametrize
def test_query_rows_are_not_checked_without_msas(tmp_path, fmt):
    query = _build_query(
        tmp_path,
        fmt,
        [_chain(LONG, _main(LONG)), _chain(SHORT, _main(LONG))],
        use_msas=False,
    )

    process(query, MSA_SETTINGS)


@pytest.mark.parametrize(
    "query_row, sequence, molecule_type, expected",
    [
        pytest.param("ACDEF", "ACDEF", MoleculeType.PROTEIN, [], id="identical"),
        pytest.param("ACDEF", "AWDEW", MoleculeType.PROTEIN, [1, 4], id="mismatches"),
        pytest.param("acdef", "ACDEF", MoleculeType.PROTEIN, [], id="case"),
        pytest.param("AXDEF", "ACDEF", MoleculeType.PROTEIN, [], id="x_in_row"),
        pytest.param("ACDEF", "ACXEF", MoleculeType.PROTEIN, [], id="x_in_sequence"),
        pytest.param("ACNEF", "ACDEF", MoleculeType.PROTEIN, [2], id="protein_n"),
        pytest.param("ACGT", "ACGU", MoleculeType.RNA, [], id="rna_t_is_u"),
        pytest.param("ACGN", "ACGU", MoleculeType.RNA, [], id="rna_n_in_row"),
        pytest.param("ACGU", "ACXU", MoleculeType.RNA, [], id="rna_x_in_sequence"),
        pytest.param("ACGT", "ACGU", MoleculeType.PROTEIN, [3], id="protein_t_not_u"),
    ],
)
def test_find_query_row_mismatches(query_row, sequence, molecule_type, expected):
    assert find_query_row_mismatches(query_row, sequence, molecule_type) == expected


def test_find_query_row_mismatches_requires_equal_lengths():
    with pytest.raises(ValueError, match=re.escape("zip()")):
        find_query_row_mismatches("ACDEF", "ACDE", MoleculeType.PROTEIN)


@pytest.mark.parametrize(
    "n_mismatches, length, expected",
    [
        pytest.param(0, 20, True, id="identical"),
        pytest.param(1, 5, True, id="single_mismatch_in_short_chain"),
        pytest.param(2, 20, True, id="exactly_90_percent"),
        pytest.param(3, 20, False, id="below_90_percent"),
        pytest.param(10, 100, True, id="exactly_90_percent_of_100"),
        pytest.param(11, 100, False, id="below_90_percent_of_100"),
        pytest.param(2, 10, False, id="two_mismatches_in_10"),
    ],
)
def test_is_tolerated_query_row(n_mismatches, length, expected):
    assert is_tolerated_query_row(n_mismatches, length) is expected
