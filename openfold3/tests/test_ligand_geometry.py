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

"""Ligand internal-geometry metrics.

These back the end-to-end steering test
(``openfold3/tests/inference/test_chemical_steering.py``), which needs an
accelerator and weights and so cannot be run on demand. The metrics themselves
are pure geometry, so they are pinned here instead, against constructed
coordinates whose answers are known analytically.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from rdkit import Chem

from openfold3.core.metrics.ligand_geometry import (
    mean_ring_torsion,
    saturated_ring_atom_names,
    torsion_degrees,
    worst_bounds_violation,
)
from openfold3.steering.types import RestraintSet


def _restraints(
    pairs: list[list[int]], lower: list[float], upper: list[float]
) -> RestraintSet:
    return RestraintSet(
        atom_index=torch.tensor(pairs, dtype=torch.int64),
        lower=torch.tensor(lower, dtype=torch.float32),
        upper=torch.tensor(upper, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# torsion_degrees
# ---------------------------------------------------------------------------


def _butane_like(dihedral_degrees: float) -> np.ndarray:
    """Four points whose dihedral is exactly ``dihedral_degrees``.

    Atoms 1 and 2 sit on the x axis; 0 and 3 hang off them in the yz plane,
    rotated apart by the requested angle. Rotating the fourth point is the
    whole construction, so the expected answer is known by build rather than
    by a second implementation of the same formula.
    """
    angle = np.radians(dihedral_degrees)
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.5, np.cos(angle), np.sin(angle)],
        ]
    )


@pytest.mark.parametrize("expected", [0.0, 60.0, 90.0, 120.0, 179.0, -60.0, -120.0])
def test_torsion_degrees_recovers_the_angle_it_was_built_with(expected: float) -> None:
    assert torsion_degrees(_butane_like(expected)) == pytest.approx(expected, abs=1e-4)


def test_torsion_degrees_is_sign_flipped_by_mirroring() -> None:
    """A dihedral is signed: reflecting the four points must negate it. This is
    what distinguishes it from a plain angle, and what makes |torsion| the
    right thing to average over a ring."""
    points = _butane_like(70.0)
    mirrored = points * np.array([1.0, 1.0, -1.0])

    assert torsion_degrees(points) == pytest.approx(70.0, abs=1e-4)
    assert torsion_degrees(mirrored) == pytest.approx(-70.0, abs=1e-4)


def test_torsion_degrees_ignores_rigid_motion() -> None:
    """Translating and rotating the whole system cannot change an internal
    coordinate."""
    points = _butane_like(55.0)
    rotation = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    if np.linalg.det(rotation) < 0:  # keep it a rotation, not a reflection
        rotation[:, 0] *= -1
    moved = points @ rotation.T + np.array([3.0, -7.0, 11.0])

    assert torsion_degrees(moved) == pytest.approx(torsion_degrees(points), abs=1e-4)


# ---------------------------------------------------------------------------
# worst_bounds_violation
# ---------------------------------------------------------------------------


def test_no_violation_when_every_pair_sits_inside_its_window() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    restraints = _restraints([[0, 1]], [1.0], [3.0])

    assert worst_bounds_violation(coords, restraints) == 0.0


@pytest.mark.parametrize(
    ("separation", "expected"),
    [
        pytest.param(0.5, 0.5, id="below_lower_bound"),
        pytest.param(1.0, 0.0, id="exactly_on_lower_bound"),
        pytest.param(3.0, 0.0, id="exactly_on_upper_bound"),
        pytest.param(4.25, 1.25, id="above_upper_bound"),
    ],
)
def test_violation_is_the_distance_outside_the_window(
    separation: float, expected: float
) -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
    restraints = _restraints([[0, 1]], [1.0], [3.0])

    assert worst_bounds_violation(coords, restraints) == pytest.approx(expected)


def test_violation_reports_the_worst_pair_not_the_first_or_the_sum() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.2, 0.0]])
    restraints = _restraints([[0, 1], [0, 2]], [1.0, 1.0], [3.0, 3.0])

    # pair (0,1) is 0.1 short, pair (0,2) is 0.8 short.
    assert worst_bounds_violation(coords, restraints) == pytest.approx(0.8)


def test_infinite_upper_bounds_never_count_as_violated() -> None:
    """Nonbonded pairs carry an infinite upper bound -- they may be arbitrarily
    far apart, and must never register as violated however far apart they
    are."""
    coords = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    restraints = _restraints([[0, 1]], [1.0], [float("inf")])

    assert worst_bounds_violation(coords, restraints) == 0.0


# ---------------------------------------------------------------------------
# saturated_ring_atom_names / mean_ring_torsion
# ---------------------------------------------------------------------------


def _named(smiles: str) -> Chem.Mol:
    """Parse SMILES and give every atom the annotation the pipeline uses."""
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetProp("annot_atom_name", f"{atom.GetSymbol()}{atom.GetIdx()}")
    return mol


@pytest.mark.parametrize(
    ("smiles", "expected_rings"),
    [
        pytest.param("C1CCCCC1", 1, id="cyclohexane"),
        pytest.param("c1ccccc1", 0, id="benzene_is_aromatic"),
        pytest.param("C1=CCCCC1", 0, id="cyclohexene_has_a_double_bond"),
        pytest.param("C1CCCC1", 0, id="cyclopentane_is_five_membered"),
        pytest.param("C1CCNCC1", 0, id="piperidine_is_not_all_carbon"),
        pytest.param("C1CCCCC1C1CCCCC1", 2, id="bicyclohexyl"),
        pytest.param("C1CCc2ccccc2C1", 0, id="tetralin_ring_is_fused_to_an_arene"),
    ],
)
def test_saturated_ring_detection_accepts_only_saturated_carbocycles(
    smiles: str, expected_rings: int
) -> None:
    """The metric is calibrated on a chair, so anything that is not a saturated
    all-carbon six-ring has to be excluded -- an aromatic ring is flat by
    right, and a five-ring puckers to a different amplitude."""
    assert len(saturated_ring_atom_names(_named(smiles))) == expected_rings


def test_saturated_ring_names_are_the_ring_atoms_in_ring_order() -> None:
    mol = _named("C1CCCCC1")
    (ring,) = saturated_ring_atom_names(mol)

    assert len(ring) == 6
    assert set(ring) == {f"C{index}" for index in range(6)}
    # Consecutive entries must be bonded, or the torsions walk the wrong path.
    name_to_index = {f"C{atom.GetIdx()}": atom.GetIdx() for atom in mol.GetAtoms()}
    for position, name in enumerate(ring):
        neighbour = ring[(position + 1) % 6]
        assert mol.GetBondBetweenAtoms(name_to_index[name], name_to_index[neighbour]), (
            f"{name} and {neighbour} are adjacent in the ring but not bonded"
        )


RING_NAMES = tuple(f"C{index}" for index in range(6))


def _hexagon(pucker_angstrom: float) -> dict[str, np.ndarray]:
    """Six carbons on a 1.46 A hexagon, alternating +-``pucker`` in z.

    One builder for both ring fixtures, so a chair and a flat ring differ in
    exactly one number and nothing else.
    """
    coords = {}
    for index, name in enumerate(RING_NAMES):
        angle = index * np.pi / 3
        coords[name] = np.array(
            [
                1.46 * np.cos(angle),
                1.46 * np.sin(angle),
                pucker_angstrom * (-1) ** index,
            ]
        )
    return coords


@pytest.fixture
def chair() -> dict[str, np.ndarray]:
    """An ideal cyclohexane chair: +-0.25 A of alternating pucker."""
    return _hexagon(0.25)


@pytest.fixture
def flat_ring() -> dict[str, np.ndarray]:
    """The same hexagon built with no pucker at all, so every z is 0.

    The badchem failure mode, where a predictor returns a saturated ring
    planar: https://www.rbvi.ucsf.edu/chimerax/data/ligchem-feb2026/badchem.html
    """
    return _hexagon(0.0)


def test_mean_ring_torsion_reports_a_chair_near_55_degrees(
    chair: dict[str, np.ndarray],
) -> None:
    """The number the steering test's floor is set against."""
    assert mean_ring_torsion(chair, [RING_NAMES]) == pytest.approx(55.0, abs=5.0)


def test_mean_ring_torsion_is_zero_for_a_flat_ring(
    flat_ring: dict[str, np.ndarray],
) -> None:
    """A ring returned planar reads as 0 degrees, so the floor in the steering
    test separates the two cases by its whole range."""
    assert mean_ring_torsion(flat_ring, [RING_NAMES]) == pytest.approx(0.0, abs=1e-6)


def test_mean_ring_torsion_averages_over_every_ring_and_bond(
    chair: dict[str, np.ndarray], flat_ring: dict[str, np.ndarray]
) -> None:
    """Two rings, one puckered and one flat, must average rather than report
    either alone -- otherwise a single flattened ring could hide."""
    puckered_group = {f"a_{name}": xyz for name, xyz in chair.items()}
    flat_group = {f"b_{name}": xyz for name, xyz in flat_ring.items()}
    coords = puckered_group | flat_group
    rings = [
        tuple(f"a_{name}" for name in RING_NAMES),
        tuple(f"b_{name}" for name in RING_NAMES),
    ]

    both = mean_ring_torsion(coords, rings)
    puckered_only = mean_ring_torsion(coords, rings[:1])

    assert both == pytest.approx(puckered_only / 2, rel=1e-6)
