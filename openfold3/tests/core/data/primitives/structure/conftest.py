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

import re
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, Mol

# Element-wise tolerance for the strict, orientation-sensitive snapshot comparison.
# ETKDGv3 is deterministic for a fixed seed within one RDKit build, so a matching
# build reproduces the exact lab frame to ~1e-3 Å.
_SNAPSHOT_ATOL = 5e-3
_SNAPSHOT_RTOL = 5e-3

# Symmetry-aware, rigid-body-aligned RMSD tolerance. RDKit re-orients (and slightly
# re-minimizes) the molecule across minor versions / CPU archs, so the raw coordinates
# rotate even though the conformation is unchanged. Stay ≪ chemistry scale (~0.1 Å) so
# a genuinely different conformer (ring flip, rotamer) still fails.
_SNAPSHOT_RMSD_TOL = 0.05


def _mol_with_coords(template: Mol, coords: np.ndarray) -> Mol:
    """Copy `template`'s topology and attach a single conformer from `coords`."""
    mol = Chem.Mol(template)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    return mol


def _snapshot_path(datadir: Path, node_name: str) -> Path:
    """Reproduce pytest-regressions' npz filename for a parametrized test node.

    `test_compute_conformer_snapshot[paracetamol_default_no_hs]` ->
    `test_compute_conformer_snapshot_paracetamol_default_no_hs_.npz`
    """
    return datadir / (re.sub(r"[\[\]]", "_", node_name) + ".npz")


@pytest.fixture
def assert_conformer_snapshot(
    request: pytest.FixtureRequest, original_datadir: Path
) -> Callable[[Mol, int], None]:
    """Compare a generated conformer against its stored snapshot in three tiers.

    1. Strict element-wise match -> pass silently (same RDKit build / lab frame).
    2. Match only after symmetry-aware rigid-body alignment -> pass *with a warning*
       (the conformation is unchanged; RDKit just rotated the molecule, e.g. an RDKit
       version or CPU-arch difference rather than a regression).
    3. Differs beyond rigid-body alignment -> fail (a real conformational regression).

    The fixture returns a `check(mol, conf_id)` callable; it sources the snapshot path
    from the requesting test node and honours `--force-regen`.
    """

    def check(mol: Mol, conf_id: int) -> None:
        coords = mol.GetConformer(conf_id).GetPositions().astype(np.float64)
        path = _snapshot_path(original_datadir, request.node.name)

        if request.config.getoption("force_regen", default=False) or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, coords=coords)
            pytest.skip(f"Regenerated conformer snapshot: {path.name}")

        expected = np.load(path)["coords"].astype(np.float64)

        # Tier 1: orientation-sensitive, exact within a matching RDKit build.
        if np.allclose(coords, expected, atol=_SNAPSHOT_ATOL, rtol=_SNAPSHOT_RTOL):
            return

        # Tier 2: rigid-body + symmetry aware. GetBestRMS aligns a copy of `mol` onto
        # the reference and minimizes over symmetry-equivalent atom mappings.
        rmsd = AllChem.GetBestRMS(Chem.Mol(mol), _mol_with_coords(mol, expected))
        if rmsd <= _SNAPSHOT_RMSD_TOL:
            warnings.warn(
                f"{request.node.name}: snapshot matched only after rigid-body "
                f"alignment (symmetry-aware RMSD={rmsd:.4f} Å ≤ {_SNAPSHOT_RMSD_TOL} "
                f"Å). The conformation is unchanged but the lab frame rotated — most "
                f"likely an RDKit version/CPU-arch difference, not a regression. "
                f"Regenerate with --force-regen to silence.",
                stacklevel=2,
            )
            return

        # Tier 3: genuine conformational change — fail loudly.
        raise AssertionError(
            f"{request.node.name}: conformer differs beyond rigid-body alignment "
            f"(symmetry-aware RMSD={rmsd:.4f} Å > {_SNAPSHOT_RMSD_TOL} Å). This is a "
            f"real geometry change; inspect it before regenerating with --force-regen."
        )

    return check
