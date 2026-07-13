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
from pathlib import Path

import numpy as np
import pytest

from openfold3.core.data.primitives.structure.template import (
    TemplateCacheEntry,
    sample_templates,
)

TEMPLATE_ID = "1FOO_A"


def _cache_entry() -> TemplateCacheEntry:
    """The single template both written to the cache and expected back out."""
    return TemplateCacheEntry(
        index=0,
        release_date="2000-01-01",
        idx_map=np.array([[1, 1], [2, 2]], dtype=int),
    )


def _write_cache_npz(path: Path) -> Path:
    entry_dict = {
        k: v for k, v in dataclasses.asdict(_cache_entry()).items() if v is not None
    }
    np.savez(path, **{TEMPLATE_ID: np.array(entry_dict, dtype=object)})
    return path


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
    [pdb_id, chain_id] = TEMPLATE_ID.split("_")
    struct_path = array_dir / pdb_id / f"{TEMPLATE_ID}.npz"
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
