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

"""Shared reference-molecule preparation helpers."""

from openfold3.core.data.primitives.structure.component import (
    AnnotatedMol,
    get_reference_molecule_metadata,
)
from openfold3.core.data.primitives.structure.conformer import (
    resolve_and_format_fallback_conformer,
)


def prepare_reference_molecule(
    mol: AnnotatedMol,
    residue_count: int = 1,
) -> tuple[AnnotatedMol, dict]:
    """Prepare a fallback conformer and its cache metadata."""
    mol, conformer_strategy = resolve_and_format_fallback_conformer(mol)
    metadata = get_reference_molecule_metadata(
        mol,
        conformer_strategy,
        residue_count,
    )
    return mol, metadata
