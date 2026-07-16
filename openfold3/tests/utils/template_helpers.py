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

"""Shared fixture builders for template tests.

Single source of truth for the on-disk template formats the inference pipeline reads:
the per-chain cache npz (keyed by template id) and the preparsed structure-array npz at
``<dir>/<pdb_id>/<template_id>.npz``. Kept as plain importable functions (not fixtures)
because callers use them at parametrize/collection time.
"""

import dataclasses
from pathlib import Path

import numpy as np

from openfold3.core.data.primitives.structure.template import TemplateCacheEntry

TEMPLATE_ID = "1FOO_A"


def make_cache_entry(
    idx_map, *, index: int = 0, release_date: str = "2000-01-01"
) -> TemplateCacheEntry:
    """Build a TemplateCacheEntry from a query<->template residue index map."""
    return TemplateCacheEntry(
        index=index, release_date=release_date, idx_map=np.asarray(idx_map)
    )


def write_cache_npz(path: Path, entries: dict[str, TemplateCacheEntry]) -> Path:
    npz = {
        template_id: np.array(
            {k: v for k, v in dataclasses.asdict(entry).items() if v is not None},
            dtype=object,
        )
        for template_id, entry in entries.items()
    }
    np.savez(path, **npz)
    return path


def template_structure_array_path(
    array_dir: Path, template_id: str = TEMPLATE_ID
) -> Path:
    pdb_id = template_id.split("_")[0]
    return array_dir / pdb_id / f"{template_id}.npz"
