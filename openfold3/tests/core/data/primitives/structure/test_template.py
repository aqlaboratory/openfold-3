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

import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pytest

import openfold3
from openfold3.core.data.primitives.structure.component import BiotiteCCDWrapper
from openfold3.core.data.primitives.structure.template import (
    parse_template_structure,
    sample_templates,
)
from openfold3.tests.utils.template_helpers import (
    TEMPLATE_ID,
    make_cache_entry,
    template_structure_array_path,
    write_cache_npz,
)

MMCIFS_DIR = Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs"


def _cache_entry():
    """The single template both written to the cache and expected back out."""
    return make_cache_entry([[1, 1], [2, 2]])


def _write_cache_npz(path: Path) -> Path:
    return write_cache_npz(path, {TEMPLATE_ID: _cache_entry()})


def _assembly_data(cache_npz: Path) -> dict:
    return {
        "A": {
            "template_ids": [TEMPLATE_ID],
            "cache_entry_file_path": cache_npz,
        }
    }


def _cache_none(tmp_path: Path) -> None:
    return None


def _cache_tmp_path(tmp_path: Path) -> Path:
    return tmp_path


def _structure_arrays_none(tmp_path: Path) -> None:
    """No preparsed structure arrays -> the existence filter is skipped."""
    return None


def _structure_arrays_dummy(tmp_path: Path) -> Path:
    """A dummy structure erray (empty inside)"""
    array_dir = tmp_path / "arrays"
    struct_path = template_structure_array_path(array_dir)
    struct_path.parent.mkdir(parents=True, exist_ok=True)
    struct_path.touch()
    return array_dir


@pytest.mark.parametrize(
    "make_cache_directory, make_structure_array_directory, expected",
    [
        pytest.param(
            _cache_none, _structure_arrays_none, {}, id="no_cache__no_arrays__drops"
        ),
        pytest.param(
            _cache_none, _structure_arrays_dummy, {}, id="no_cache__with_arrays__drops"
        ),
        pytest.param(
            _cache_tmp_path,
            _structure_arrays_none,
            {TEMPLATE_ID: _cache_entry()},
            id="cache__no_arrays__loads",
        ),
        pytest.param(
            _cache_tmp_path,
            _structure_arrays_dummy,
            {TEMPLATE_ID: _cache_entry()},
            id="cache__with_arrays__loads",
        ),
    ],
)
def test_sample_templates_cache_directory_gate(
    tmp_path, make_cache_directory, make_structure_array_directory, expected
):
    cache_npz = _write_cache_npz(tmp_path / "chainA.npz")

    actual = sample_templates(
        assembly_data=_assembly_data(cache_npz),
        template_cache_directory=make_cache_directory(tmp_path),
        n_templates=4,
        take_top_k=True,  # deterministic: k = min(len(ids), n_templates)
        chain_id="A",
        template_structure_array_directory=make_structure_array_directory(tmp_path),
        template_file_format="npz",
    )

    np.testing.assert_equal(
        {k: dataclasses.asdict(v) for k, v in actual.items()},
        {k: dataclasses.asdict(v) for k, v in expected.items()},
    )


def test_sample_templates_preserves_cif_path(tmp_path):
    """A CIF-direct cache entry must keep its coordinate source on read-back.

    In CIF-direct mode (a query chain pins `template_cif_paths`) preprocessing records
    the provided CIF in the cache entry, and `get_template_slices` forwards it to
    `parse_template_structure`, which has a dedicated CIF-direct branch. Dropping it
    here makes that branch unreachable: the template silently falls back to
    `<structure_directory>/<stem>.cif`, i.e. the PDB-deposited entry, so user-supplied
    or non-deposited coordinates are never the ones the model sees.

    Regression test for https://github.com/aqlaboratory/openfold-3/issues/406
    """
    cif_path = tmp_path / "6TEL_notpdb.cif"
    cif_path.write_text("data_dummy\n")
    entry = make_cache_entry([[1, 1], [2, 2]], cif_path=cif_path)
    cache_npz = write_cache_npz(tmp_path / "chainA.npz", {TEMPLATE_ID: entry})

    # The cache npz itself carries the path...
    assert np.load(cache_npz, allow_pickle=True)[TEMPLATE_ID].item()["cif_path"] == str(
        cif_path
    )

    sampled = sample_templates(
        assembly_data=_assembly_data(cache_npz),
        template_cache_directory=tmp_path,
        n_templates=4,
        take_top_k=True,
        chain_id="A",
        template_structure_array_directory=None,
        template_file_format="npz",
    )

    # ...so the entry handed to parse_template_structure must carry it too.
    assert sampled[TEMPLATE_ID].cif_path == cif_path


def test_cif_direct_coordinates_reach_the_model(tmp_path):
    """The pinned CIF's own coordinates must be the ones that get featurized.

    The end-to-end version of `test_sample_templates_preserves_cif_path`. Two different
    deposited entries are used rather than an edit of one: the query pins ubiquitin
    while a wholly unrelated structure sits in `structure_directory` under the name the
    template ID resolves to. Whichever file was read is then unmistakable from the atom
    array alone -- when `cif_path` is lost, `parse_template_structure` falls back to
    that directory and returns the wrong protein outright.

    Regression test for https://github.com/aqlaboratory/openfold-3/issues/406
    """
    # The template ID says "1a8q", so a filename-keyed lookup resolves to the decoy...
    template_id = "1a8q_A"
    decoy_dir = tmp_path / "template_structures"
    decoy_dir.mkdir()
    shutil.copy(MMCIFS_DIR / "1a8q.cif", decoy_dir / "1a8q.cif")
    # ...while the query actually pinned a different structure entirely.
    pinned = MMCIFS_DIR / "1ubq.cif"

    entry = make_cache_entry([[1, 1], [2, 2]], cif_path=pinned)
    cache_npz = write_cache_npz(tmp_path / "chainA.npz", {template_id: entry})

    sampled = sample_templates(
        assembly_data={
            "A": {"template_ids": [template_id], "cache_entry_file_path": cache_npz}
        },
        template_cache_directory=tmp_path,
        n_templates=4,
        take_top_k=True,
        chain_id="A",
        template_structure_array_directory=None,
        template_file_format="cif",
    )

    ccd = BiotiteCCDWrapper()
    common = dict(
        template_structure_array_directory=None,
        template_pdb_chain_id=template_id,
        template_file_format="cif",
        ccd=ccd,
    )
    actual = parse_template_structure(
        template_structures_directory=decoy_dir,
        cif_path=sampled[template_id].cif_path,
        **common,
    )
    decoy = parse_template_structure(
        template_structures_directory=decoy_dir, cif_path=None, **common
    )
    expected = parse_template_structure(
        template_structures_directory=None, cif_path=pinned, **common
    )

    assert actual is not None and decoy is not None and expected is not None
    # Guard the premise: the two files really are distinguishable.
    assert decoy.array_length() != expected.array_length()
    assert actual.array_length() == expected.array_length()
    np.testing.assert_array_equal(actual.coord, expected.coord)


def test_cif_direct_template_id_with_underscores_is_parsed(tmp_path):
    """A pinned filename containing underscores must still resolve to a chain.

    Template IDs are `f"{entry_id}_{chain_id}"`, and in CIF-direct mode the entry ID is
    the pinned file's stem -- which users routinely give names like `6TEL_relaxed` or
    `model_1_rank_2`. Splitting on every underscore rather than the last one makes the
    whole query die with "too many values to unpack", after preprocessing has already
    accepted the template.

    Regression test for https://github.com/aqlaboratory/openfold-3/issues/406
    """
    pinned = MMCIFS_DIR / "1ubq.cif"
    template_id = "1ubq_not_a_pdb_id_A"

    parsed = parse_template_structure(
        template_structures_directory=None,
        template_structure_array_directory=None,
        template_pdb_chain_id=template_id,
        template_file_format="cif",
        ccd=BiotiteCCDWrapper(),
        cif_path=pinned,
    )

    assert parsed is not None and parsed.array_length() > 0
