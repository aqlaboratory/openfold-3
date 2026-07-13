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

import biotite.structure as struc
import numpy as np

from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz
from openfold3.core.data.primitives.structure.template import TemplateCacheEntry
from openfold3.core.data.resources.residues import MoleculeType

TEMPLATE_ID = "1FOO_A"


def make_cache_entry(
    idx_map, *, index: int = 0, release_date: str = "2000-01-01"
) -> TemplateCacheEntry:
    """Build a TemplateCacheEntry from a query<->template residue index map."""
    return TemplateCacheEntry(
        index=index, release_date=release_date, idx_map=np.asarray(idx_map)
    )


def write_cache_npz(path: Path, entries: dict[str, TemplateCacheEntry]) -> Path:
    """Write a template cache npz keyed by template id.

    Mirrors the production on-disk format (preprocessing/template.py:2263-2294): each
    value is a 0-d object array whose ``.item()`` yields a dict with ``index``,
    ``release_date`` and ``idx_map`` (the unused ``cif_path`` is dropped when None so the
    shape matches exactly). Read back by ``sample_templates`` (template.py:232-236).
    """
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
    """Path a preparsed structure array is read from: <dir>/<pdb_id>/<template_id>.npz.

    Matches the layout ``parse_template_structure``/``sample_templates`` expect
    (template.py:189-191, 346-348).
    """
    pdb_id = template_id.split("_")[0]
    return array_dir / pdb_id / f"{template_id}.npz"


def write_template_structure_array(
    array_dir: Path, n_res: int, *, template_id: str = TEMPLATE_ID
) -> Path:
    """Write a preparsed poly-ALA template chain of ``n_res`` residues.

    Each residue carries N/CA/C/CB (pseudo-beta = CB for non-GLY) so the template is
    featurizable; coordinates only need to be finite for the presence masks. Written with
    the production ``write_atomarray_to_npz`` to ``template_structure_array_path(...)``.
    """
    atoms = []
    for res_id in range(1, n_res + 1):
        for i, (atom_name, element) in enumerate(
            [("N", "N"), ("CA", "C"), ("C", "C"), ("CB", "C")]
        ):
            atoms.append(
                struc.Atom(
                    [float(res_id), float(i), 0.0],
                    chain_id="A",
                    res_id=res_id,
                    res_name="ALA",
                    atom_name=atom_name,
                    element=element,
                )
            )
    atom_array = struc.array(atoms)
    atom_array.set_annotation(
        "molecule_type_id",
        np.full(len(atom_array), int(MoleculeType.PROTEIN), dtype=int),
    )
    atom_array.set_annotation("occupancy", np.ones(len(atom_array), dtype=float))

    out = template_structure_array_path(array_dir, template_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_atomarray_to_npz(atom_array, out)
    return out
