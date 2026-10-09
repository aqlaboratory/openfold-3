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

"""Tests for train-mode template preprocessing: dataset cache in, per-structure
template assignments out.

Train mode builds one template cache entry per alignment representative, then filters
and truncates the template IDs per structure. The end-to-end tests run offline on real
1fdl inputs; see test_data/template_preprocessing/1fdl/README.md.
"""

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
from biotite.structure import AtomArray

import openfold3
from openfold3.core.data.io.dataset_cache import read_datacache, write_datacache_to_json
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.pipelines.featurization.template import (
    featurize_template_structures_of3,
)
from openfold3.core.data.pipelines.preprocessing import template as template_module
from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessor,
    TemplatePreprocessorSettings,
)
from openfold3.core.data.pipelines.sample_processing.template import (
    process_template_structures_of3,
)
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ClusteredDatasetChainData,
    ClusteredDatasetStructureData,
    ProteinMonomerChainData,
    ProteinMonomerDatasetCache,
    ProteinMonomerStructureData,
)
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.core.data.tools.colabfold_msa_server import (
    remap_colabfold_template_chain_ids,
)


def _make_bare_preprocessor(**attrs) -> TemplatePreprocessor:
    """Build a TemplatePreprocessor without running __init__ (no mp.Pool / file IO).

    Only the attributes the method under test reads need to be set.
    """
    pre = object.__new__(TemplatePreprocessor)
    for key, value in attrs.items():
        setattr(pre, key, value)
    return pre


def _write_file(path: Path, content: str = "") -> Path:
    path.write_text(content)
    return path


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


def _train_settings_kwargs(tmp_path: Path) -> dict:
    """Offline train-mode settings for the 1fdl fixture (all but `mode`).

    Decompresses the fixture's template structures into ``tmp_path``.
    """
    structure_dir = tmp_path / "template_structures"
    structure_dir.mkdir()
    for gz in (TRAIN_FIXTURE_DIR / "template_structures").glob("*.cif.gz"):
        (structure_dir / gz.name.removesuffix(".gz")).write_bytes(
            gzip.decompress(gz.read_bytes())
        )
    return {
        "output_directory": tmp_path / "template_data",
        "structure_directory": structure_dir,
        "template_alignment_directory": TRAIN_FIXTURE_DIR / "template_alignments",
        "alignment_representatives_fasta": TRAIN_FIXTURE_DIR / "representatives.fasta",
        "min_release_date_diff": TRAIN_MIN_RELEASE_DATE_DIFF,
        "fetch_missing_structures": False,
        "preparse_structures": True,
        "n_processes": 1,
    }


def _run_train_template_preprocessor(
    tmp_path: Path, **overrides
) -> tuple[ClusteredDatasetCache, TemplatePreprocessorSettings]:
    """Run train-mode preprocessing offline on the 1fdl fixture.

    ``overrides`` replace the fixture's settings. Returns the updated dataset cache and
    the settings, for their output directories.
    """
    settings = TemplatePreprocessorSettings(
        mode="train", **{**_train_settings_kwargs(tmp_path), **overrides}
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


# Content of the train-mode outputs, checked against values read off the fixture's
# inputs (template structures, RCSB release dates) rather than golden outputs.


@pytest.fixture(scope="module")
def train_preprocessor_run(tmp_path_factory):
    """One train-mode run on the 1fdl fixture, shared by the content checks."""
    return _run_train_template_preprocessor(
        tmp_path_factory.mktemp("train_preprocessor_run")
    )


def _read_cache_entries(
    settings: TemplatePreprocessorSettings, rep_id: str
) -> dict[str, dict]:
    """Template ID -> cache entry, from the representative's template cache file."""
    assert settings.cache_directory is not None
    with np.load(settings.cache_directory / f"{rep_id}.npz", allow_pickle=True) as npz:
        return {template_id: npz[template_id].item() for template_id in npz}


def _diagonal_idx_map(n_residues: int) -> np.ndarray:
    """idx_map pairing query residue i with template residue i, for i in 1..n."""
    residues = np.arange(1, n_residues + 1)
    return np.stack([residues, residues], axis=1)


def test_idx_map_of_same_sequence_hit_is_diagonal(train_preprocessor_run):
    """A template with the query's exact sequence pairs every residue with itself."""
    _, settings = train_preprocessor_run
    # (rep_id, template_id): query length. Each template is the same protein
    # as its query.
    same_sequence_hits = {
        ("1fdl_A", "1fdl_A"): 214,  # 1fdl light chain is its own top hit
        ("1fdl_B", "1fdl_B"): 218,  # 1fdl heavy chain is its own top hit
        ("1fdl_C", "5lyz_A"): 129,  # hen egg-white lysozyme, another entry
    }

    for (rep_id, template_id), n_residues in same_sequence_hits.items():
        actual = _read_cache_entries(settings, rep_id)[template_id]["idx_map"]
        expected = _diagonal_idx_map(n_residues)
        np.testing.assert_array_equal(actual, expected, err_msg=template_id)


def test_cache_entry_release_dates_match_pdb(train_preprocessor_run):
    """Each cache entry carries its template structure's RCSB release date."""
    _, settings = train_preprocessor_run
    expected = {
        "1fdl_A": {
            "1fdl_A": datetime(1991, 10, 15),
            "1qbl_A": datetime(1998, 12, 2),
            "1jhk_A": datetime(2001, 10, 10),
        },
        "1fdl_B": {
            "1fdl_B": datetime(1991, 10, 15),
            "1qbl_B": datetime(1998, 12, 2),
            "3hfm_B": datetime(1989, 7, 12),
        },
        "1fdl_C": {
            "1ior_A": datetime(2001, 4, 11),
            "7ynv_A": datetime(2022, 9, 21),
            "5lyz_A": datetime(1977, 4, 12),
        },
    }

    actual = {
        rep_id: {
            template_id: entry["release_date"]
            for template_id, entry in _read_cache_entries(settings, rep_id).items()
        }
        for rep_id in expected
    }

    assert actual == expected


def _summarize_structure_array(
    settings: TemplatePreprocessorSettings, template_id: str
) -> dict:
    """The chain IDs and residue counts in a template's preparsed structure array."""
    entry_id = template_id.split("_")[0]
    assert settings.structure_array_directory is not None
    path = settings.structure_array_directory / entry_id / f"{template_id}.npz"
    with np.load(path) as array:
        res_ids = array["res_id"]
        has_coords = ~np.isnan(array["coord"]).any(axis=1)
        return {
            "label_chain_ids": set(array["label_asym_id"].tolist()),
            "author_chain_ids": set(array["auth_asym_id"].tolist()),
            "n_residues": len(np.unique(res_ids)),
            "n_unresolved": len(np.setdiff1d(res_ids, res_ids[has_coords])),
        }


def test_structure_array_holds_the_template_chain(train_preprocessor_run):
    """A hit's structure array holds its label chain, with every residue of it.

    Residues without coordinates in the deposited structure are kept, with NaN
    coordinates, so the residue count is the chain's full sequence length.
    """
    _, settings = train_preprocessor_run
    expected = {
        # Label B, but ColabFold names this hit 1qbl_H by its author chain ID
        "1qbl_B": {
            "label_chain_ids": {"B"},
            "author_chain_ids": {"H"},
            "n_residues": 219,
            "n_unresolved": 0,
        },
        # The C-terminal Cys has no coordinates but is still there
        "1jhk_A": {
            "label_chain_ids": {"A"},
            "author_chain_ids": {"L"},
            "n_residues": 214,
            "n_unresolved": 1,
        },
        # Label and author chain IDs agree
        "5lyz_A": {
            "label_chain_ids": {"A"},
            "author_chain_ids": {"A"},
            "n_residues": 129,
            "n_unresolved": 0,
        },
    }

    actual = {
        template_id: _summarize_structure_array(settings, template_id)
        for template_id in expected
    }

    assert actual == expected


def _tokenized_1fdl(settings: TemplatePreprocessorSettings) -> AtomArray:
    """1fdl's protein chains, chain IDs "1"-"3" as in the dataset cache, tokenized."""
    assert settings.structure_directory is not None
    atom_array = parse_mmcif(
        settings.structure_directory / "1fdl.cif", renumber_chain_ids=True
    ).atom_array
    atom_array = atom_array[atom_array.molecule_type_id == MoleculeType.PROTEIN]
    tokenize_atom_array(atom_array)
    return atom_array


def _residues_covered_per_template(
    template_features: dict, atom_array: AtomArray
) -> dict[str, list[int]]:
    """Chain ID -> number of the chain's residues each template slot covers.

    A residue is covered when its tokens have template_pseudo_beta_mask set. Counted
    per residue, not per token: disulfide-bonded cysteines take one token per atom.
    """
    mask = template_features["template_pseudo_beta_mask"].numpy()
    covered = {}
    for chain_id in np.unique(atom_array.chain_id):
        chain = atom_array[atom_array.chain_id == chain_id]
        covered[str(chain_id)] = [
            len(np.unique(chain.res_id[np.isin(chain.token_id, np.flatnonzero(slot))]))
            for slot in mask
        ]
    return covered


def test_train_mode_outputs_give_training_template_features(train_preprocessor_run):
    """Training's template pipeline turns train-mode outputs into template features.

    Runs the dataset's template steps (sample templates from the cache, align them to
    the query, featurize) on the 1fdl-2020 entry, taking the top templates in order.
    """
    cache, settings = train_preprocessor_run
    atom_array = _tokenized_1fdl(settings)
    # What the dataset reads from the dataset cache for each chain
    assembly_data = {
        chain_id: {
            "alignment_representative_id": chain.alignment_representative_id,
            "template_ids": chain.template_ids,
        }
        for chain_id, chain in cache.structure_data["1fdl-2020"].chains.items()
    }

    template_slices = process_template_structures_of3(
        atom_array=atom_array,
        n_templates=4,
        take_top_k=True,
        min_n_tokens_per_chain=5,
        template_cache_directory=settings.cache_directory,
        assembly_data=assembly_data,
        template_structures_directory=None,
        template_structure_array_directory=settings.structure_array_directory,
        template_file_format="npz",
        ccd=None,
    )
    template_features = featurize_template_structures_of3(
        atom_array=atom_array,
        template_slice_collection=template_slices,
        n_templates=4,
        n_tokens=len(np.unique(atom_array.token_id)),
        min_bin=3.25,
        max_bin=50.75,
        n_bins=39,
    )

    actual = _residues_covered_per_template(template_features, atom_array)
    # One entry per template slot, in template_ids order; 0 for an empty slot.
    expected = {
        # 1fdl_A, 1qbl_A: all 214 residues. 1jhk_A: its C-terminal Cys has no
        # coordinates.
        "1": [214, 214, 213, 0],
        # 1fdl_B, 1qbl_B: all 218 residues. 3hfm_B: aligned to 215 of them.
        "2": [218, 218, 215, 0],
        # 1ior_A, 5lyz_A: all 129 residues; only two templates.
        "3": [129, 129, 0, 0],
    }
    assert actual == expected


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


def _train_chain(
    rep_id: str | None, molecule_type: str = "PROTEIN", template_ids=None
) -> ClusteredDatasetChainData:
    return ClusteredDatasetChainData(
        label_asym_id="A",
        auth_asym_id="A",
        entity_id=1,
        molecule_type=molecule_type,
        reference_mol_id=None,
        alignment_representative_id=rep_id,
        template_ids=template_ids,
        cluster_id="0",
        cluster_size=1,
    )


def _train_cache(
    structures: dict[str, tuple[date, dict[str, ClusteredDatasetChainData]]],
) -> ClusteredDatasetCache:
    return ClusteredDatasetCache(
        name="unit",
        structure_data={
            pdb_id: ClusteredDatasetStructureData(
                release_date=release_date,
                resolution=2.0,
                chains=chains,
                interfaces={},
            )
            for pdb_id, (release_date, chains) in structures.items()
        },
        reference_molecule_data={},
    )


def test_parse_dataset_cache_one_input_per_representative(tmp_path):
    """One input per protein representative with an alignment and a sequence."""
    aln_dir = tmp_path / "template_alignments"
    for rep_id in ("r1", "r3", "d1"):
        _write_file_with_parents(aln_dir / rep_id / "colabfold_template.m8")
    fasta = _write_file(tmp_path / "reps.fasta", ">r1\nACDEF\n>r2\nGHIKL\n>d1\nACGT\n")
    cache = _train_cache(
        {
            "1abc": (
                date(2000, 1, 1),
                {"1": _train_chain("r1"), "2": _train_chain(None, "LIGAND")},
            ),
            "2abc": (
                date(2001, 1, 1),
                {
                    "1": _train_chain("r1"),  # shared representative
                    "2": _train_chain("r2"),  # no alignment file
                    "3": _train_chain("r3"),  # not in the FASTA
                    "4": _train_chain("d1", "DNA"),  # not a requested moltype
                },
            ),
        }
    )
    pre = _make_bare_preprocessor(
        input_set=cache,
        moltypes=[MoleculeType.PROTEIN],
        template_alignment_directory=aln_dir,
        alignment_representatives_fasta=fasta,
    )

    pre._parse_dataset_cache()

    assert list(pre.inputs) == ["r1"]
    assert pre.inputs["r1"].aln_path == aln_dir / "r1" / "colabfold_template.m8"
    assert pre.inputs["r1"].query_seq_str == "ACDEF"
    assert pre.inputs["r1"].cache_key == "r1"


def test_parse_dataset_cache_summarizes_skipped_representatives(tmp_path, caplog):
    """Representatives without template inputs are listed in one warning."""
    aln_dir = tmp_path / "template_alignments"
    for rep_id in ("r1", "r3"):
        _write_file_with_parents(aln_dir / rep_id / "colabfold_template.m8")
    fasta = _write_file(tmp_path / "reps.fasta", ">r1\nACDEF\n>r2\nGHIKL\n")
    cache = _train_cache(
        {
            "1abc": (
                date(2000, 1, 1),
                {
                    "1": _train_chain("r1"),
                    "2": _train_chain("r2"),  # no alignment file
                    "3": _train_chain("r3"),  # not in the FASTA
                },
            )
        }
    )
    pre = _make_bare_preprocessor(
        input_set=cache,
        moltypes=[MoleculeType.PROTEIN],
        template_alignment_directory=aln_dir,
        alignment_representatives_fasta=fasta,
    )

    with caplog.at_level(logging.WARNING):
        pre._parse_dataset_cache()

    actual = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    expected = [
        "2 of 3 alignment representatives have no template inputs and get no "
        f"templates. No alignment in {aln_dir}: ['r2']. Not in {fasta}: ['r3']."
    ]
    assert actual == expected


def test_parse_dataset_cache_fails_when_no_representative_has_an_alignment(tmp_path):
    """No representative with a template alignment means mismatched inputs."""
    aln_dir = tmp_path / "template_alignments"
    aln_dir.mkdir()
    fasta = _write_file(tmp_path / "reps.fasta", ">r1\nACDEF\n>r2\nGHIKL\n")
    cache = _train_cache(
        {
            "1abc": (
                date(2000, 1, 1),
                {"1": _train_chain("r1"), "2": _train_chain("r2")},
            )
        }
    )
    pre = _make_bare_preprocessor(
        input_set=cache,
        moltypes=[MoleculeType.PROTEIN],
        template_alignment_directory=aln_dir,
        alignment_representatives_fasta=fasta,
    )

    with pytest.raises(ValueError, match="None of the 2 alignment representatives"):
        pre._parse_dataset_cache()


def test_parse_dataset_cache_requires_train_settings():
    """Train mode needs the alignment directory and the representatives FASTA."""
    pre = _make_bare_preprocessor(
        input_set=_train_cache({}),
        moltypes=[MoleculeType.PROTEIN],
        template_alignment_directory=None,
        alignment_representatives_fasta=None,
    )

    with pytest.raises(ValueError, match="template_alignment_directory"):
        pre._parse_dataset_cache()


def test_train_mode_rejects_dataset_cache_without_release_dates(tmp_path):
    """Monomer caches carry no release dates or molecule types to filter on."""
    monomer_cache = ProteinMonomerDatasetCache(
        name="monomers",
        structure_data={
            "1abc": ProteinMonomerStructureData(
                chains={
                    "A": ProteinMonomerChainData(
                        alignment_representative_id="1abc_A",
                        template_ids=None,
                        index=0,
                    )
                }
            )
        },
        reference_molecule_data={},
    )
    fasta = _write_file(tmp_path / "reps.fasta", ">1abc_A\nACDEF\n")
    pre = _make_bare_preprocessor(
        input_set=monomer_cache,
        moltypes=[MoleculeType.PROTEIN],
        template_alignment_directory=tmp_path,
        alignment_representatives_fasta=fasta,
    )

    with pytest.raises(TypeError, match="clustered or validation dataset cache"):
        pre._parse_dataset_cache()


def test_update_dataset_cache_truncates_after_date_filtering(tmp_path):
    """max_templates applies to the templates that pass this structure's dates."""
    idx_map = np.zeros((1, 2), dtype=int)
    np.savez_compressed(
        tmp_path / "r1.npz",
        **{
            "new_A": {
                "index": 0,
                "release_date": datetime(2020, 1, 1),
                "idx_map": idx_map,
            },
            "mid_A": {"index": 1, "release_date": "1990-01-01", "idx_map": idx_map},
            "old_A": {
                "index": 2,
                "release_date": datetime(1980, 1, 1),
                "idx_map": idx_map,
            },
        },
    )
    cache = _train_cache(
        {
            "1abc": (
                date(2000, 1, 1),
                {
                    "1": _train_chain("r1"),
                    "2": _train_chain("r9", template_ids=["stale_A"]),
                    "3": _train_chain("d1", "DNA", template_ids=["keep_A"]),
                },
            )
        }
    )
    pre = _make_bare_preprocessor(
        input_set=cache,
        inputs={"r1": object()},
        cache_directory=tmp_path,
        moltypes=[MoleculeType.PROTEIN],
        max_release_date=None,
        min_release_date_diff=60,
        max_templates=1,
    )

    pre._update_dataset_cache()

    chains = cache.structure_data["1abc"].chains
    assert chains["1"].template_ids == ["mid_A"]
    # A protein representative without template inputs gets no templates, not the
    # template_ids the input cache had.
    assert chains["2"].template_ids == []
    # A chain of a molecule type that was not preprocessed keeps what it had.
    assert chains["3"].template_ids == ["keep_A"]


def _write_file_with_parents(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_file(path, content)


def test_train_mode_on_dataset_cache_read_from_json(tmp_path):
    """Train mode works on a dataset cache read from JSON, and its output round-trips.

    The steps of the preprocessing script's train mode. Read from JSON, release dates
    are strings rather than dates.
    """
    input_path = tmp_path / "dataset_cache.json"
    write_datacache_to_json(_train_dataset_cache(), input_path)
    cache = read_datacache(input_path)
    settings = TemplatePreprocessorSettings(
        mode="train", **_train_settings_kwargs(tmp_path)
    )

    TemplatePreprocessor(input_set=cache, config=settings)()
    output_path = tmp_path / "dataset_cache_with_templates.json"
    write_datacache_to_json(cache, output_path)

    actual = _train_template_ids(read_datacache(output_path))
    assert actual == TRAIN_EXPECTED_TEMPLATE_IDS
