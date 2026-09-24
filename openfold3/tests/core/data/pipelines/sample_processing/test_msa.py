"""Regression tests for GH-371: precomputed paired MSAs dropped at inference.

`create_paired_from_precomputed` builds a paired MSA block per chain but never
records its row count via `MsaArrayCollection.set_row_counts`. Downstream,
`vstack_pad_msa_arrays` gates the paired block on that row count, so the block
it just built is silently discarded from the MSA features. It also costs
main-MSA rows: `create_main` dedups main against the paired block regardless
of whether the row count was recorded, so the shared rows are deleted there
too.
"""

import textwrap
from pathlib import Path

import pytest

from openfold3.core.config.msa_pipeline_configs import MsaSampleProcessorInputInference
from openfold3.core.data.pipelines.sample_processing.msa import (
    MsaSampleProcessorInference,
)
from openfold3.core.data.primitives.featurization.msa import vstack_pad_msa_arrays
from openfold3.projects.of3_all_atom.config.dataset_config_components import MSASettings
from openfold3.projects.of3_all_atom.config.inference_query_format import Chain, Query

# Heterodimer fixture: chain A (query ACDEF) and chain B (query GHIKL). Each
# chain's main a3m starts with its own query row (as real colabfold_main.a3m
# files do) followed by two hit rows, one of which also appears in that
# chain's paired block — see the module docstring for why that's the case
# this bug hinges on.
#
#            chain A                          chain B
# main a3m   ACDEF (query row), AADEF, ACDEA   GHIKL (query row), GGIKL, GHIKG
# paired a3m AADEF, ACAEF                      GGIKL, GHGKL
#            ^^^^^ shared with main            ^^^^^ shared with main
#
# create_main dedups main against paired only, not against the standalone
# query row, so the query row legitimately survives inside main too.
# MSASettings defaults to randomly subsampling the main MSA
# (subsample_main=True), so which of the surviving main rows make it into the
# final stack is itself a random draw — the test pins that draw via the
# `seeded_rng` fixture rather than disabling subsampling, to exercise the
# same default config used at inference.
MAIN_A3M = {
    "A": textwrap.dedent(
        """\
        >101
        ACDEF
        >101_1
        AADEF
        >101_2
        ACDEA
        """
    ),
    "B": textwrap.dedent(
        """\
        >102
        GHIKL
        >102_1
        GGIKL
        >102_2
        GHIKG
        """
    ),
}
PAIRED_A3M = {
    "A": textwrap.dedent(
        """\
        >101
        AADEF
        >101_1
        ACAEF
        """
    ),
    "B": textwrap.dedent(
        """\
        >102
        GGIKL
        >102_1
        GHGKL
        """
    ),
}


# A third, RNA representative with only a main MSA (no paired block) — real
# ColabFold runs never produce paired MSAs for RNA, so a rep like this should
# never trigger the "did you forget paired_msa_file_paths?" warning.
RNA_MAIN_A3M = textwrap.dedent(
    """\
    >201
    ACGU
    >201_1
    ACGA
    """
)


def _write_a3m(root: Path, kind: str, rep_id: str, filename: str, content: str) -> Path:
    """Writes `content` to root/<kind>/<rep_id>/<filename> and returns the rep's
    directory (as chain.*_msa_file_paths expects — the a3m file's parent dir
    name is what the parser reads back as the representative ID).
    """
    rep_dir = root / kind / rep_id
    rep_dir.mkdir(parents=True)
    (rep_dir / filename).write_text(content)
    return rep_dir


def _write_heterodimer_a3m_fixtures(root: Path) -> dict[str, dict[str, Path]]:
    """Writes MAIN_A3M/PAIRED_A3M to root/{main,paired}/<rep_id>/colabfold_*.a3m.

    Returns {"main": {"A": path, "B": path}, "paired": {"A": path, "B": path}}.
    """
    paths: dict[str, dict[str, Path]] = {"main": {}, "paired": {}}
    for kind, filename, content_by_rep in (
        ("main", "colabfold_main.a3m", MAIN_A3M),
        ("paired", "colabfold_paired.a3m", PAIRED_A3M),
    ):
        for rep_id, content in content_by_rep.items():
            paths[kind][rep_id] = _write_a3m(root, kind, rep_id, filename, content)
    return paths


def test_msa_sample_processor_inference_includes_precomputed_paired_msa(
    tmp_path, seeded_rng
):
    """Real .a3m fixtures parsed through `MsaSampleProcessorInference`, mirroring
    the default inference path taken when the ColabFold MSA server precomputes
    paired MSAs for a heteromer.
    """
    fixture_paths = _write_heterodimer_a3m_fixtures(tmp_path)

    query = Query(
        query_name="heterodimer",
        chains=[
            Chain(
                molecule_type="protein",
                chain_ids=["A"],
                sequence="ACDEF",
                main_msa_file_paths=[fixture_paths["main"]["A"]],
                paired_msa_file_paths=[fixture_paths["paired"]["A"]],
            ),
            Chain(
                molecule_type="protein",
                chain_ids=["B"],
                sequence="GHIKL",
                main_msa_file_paths=[fixture_paths["main"]["B"]],
                paired_msa_file_paths=[fixture_paths["paired"]["B"]],
            ),
        ],
    )

    processor = MsaSampleProcessorInference(config=MSASettings())
    input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        inference_query=query
    )
    msa_array_collection = processor(input=input)

    # The core GH-371 bug: this stayed 0 because create_paired_from_precomputed
    # never called set_row_counts.
    assert msa_array_collection.row_counts.n_rows_paired_subsampled == 2

    # With seeded_rng pinning the main-MSA subsampling draw, chain A keeps only
    # its query-row hit from main (ACDEA is subsampled out) and chain B keeps
    # only its non-query hit (GHIKL is subsampled out) — both paired blocks
    # survive intact either way, which is what this test guards.
    stack_a, _ = vstack_pad_msa_arrays(msa_array_collection, "A")
    assert ["".join(row) for row in stack_a.msa] == [
        "ACDEF",
        "AADEF",
        "ACAEF",
        "ACDEF",
    ]

    stack_b, _ = vstack_pad_msa_arrays(msa_array_collection, "B")
    assert ["".join(row) for row in stack_b.msa] == [
        "GHIKL",
        "GGIKL",
        "GHGKL",
        "GHIKG",
    ]


def test_missing_paired_msa_warns_when_protein_rep_lacks_one(tmp_path):
    """Chain B is protein but has no paired_msa_file_paths, while chain A does
    — likely a mistake (forgot to supply pairing data for B), so this should
    warn.
    """
    fixture_paths = _write_heterodimer_a3m_fixtures(tmp_path)

    query = Query(
        query_name="protein-missing-pairing",
        chains=[
            Chain(
                molecule_type="protein",
                chain_ids=["A"],
                sequence="ACDEF",
                main_msa_file_paths=[fixture_paths["main"]["A"]],
                paired_msa_file_paths=[fixture_paths["paired"]["A"]],
            ),
            Chain(
                molecule_type="protein",
                chain_ids=["B"],
                sequence="GHIKL",
                main_msa_file_paths=[fixture_paths["main"]["B"]],
                # No paired_msa_file_paths for B.
            ),
        ],
    )

    processor = MsaSampleProcessorInference(config=MSASettings())
    input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        inference_query=query
    )
    with pytest.warns(
        UserWarning, match="Representative B is a protein chain with no precomputed"
    ):
        processor(input=input)


def test_missing_paired_msa_does_not_warn_for_rna_rep(tmp_path, recwarn):
    """Chain C is RNA with no paired_msa_file_paths, alongside a paired protein
    chain A — this is the routine case (ColabFold never pairs RNA), so it
    should not warn.
    """
    fixture_paths = _write_heterodimer_a3m_fixtures(tmp_path)
    rna_main_dir = _write_a3m(tmp_path, "main", "C", "colabfold_main.a3m", RNA_MAIN_A3M)

    query = Query(
        query_name="rna-missing-pairing",
        chains=[
            Chain(
                molecule_type="protein",
                chain_ids=["A"],
                sequence="ACDEF",
                main_msa_file_paths=[fixture_paths["main"]["A"]],
                paired_msa_file_paths=[fixture_paths["paired"]["A"]],
            ),
            Chain(
                molecule_type="rna",
                chain_ids=["C"],
                sequence="ACGU",
                main_msa_file_paths=[rna_main_dir],
                # No paired_msa_file_paths for C — expected for RNA.
            ),
        ],
    )

    processor = MsaSampleProcessorInference(config=MSASettings())
    input = MsaSampleProcessorInputInference.create_from_inference_query_entry(
        inference_query=query
    )
    processor(input=input)

    messages = [str(w.message) for w in recwarn.list]
    assert not any("precomputed paired MSA" in m for m in messages), messages
