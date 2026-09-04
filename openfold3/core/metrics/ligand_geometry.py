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

"""Internal geometry of a predicted ligand.

Answers "is this molecule built correctly?" rather than "is it in the right
place" -- no experimental reference is involved, only the predicted
coordinates and the molecule's own chemistry. That makes these metrics
complementary to :mod:`openfold3.core.metrics.alignment`, which scores
placement against experiment.

Two families, matching the two ways a predicted ligand goes wrong:

* :func:`worst_bounds_violation` -- how far any atom pair sits outside the
  window RDKit's distance-geometry bounds allow. Covers bond lengths, bond
  angles and internal clashes at once, the same way PoseBusters' internal
  validity checks do.
* :func:`saturated_ring_atom_names` and :func:`mean_ring_torsion` -- whether a
  saturated ring kept its pucker. A chair sits near 55 degrees at every ring
  bond and a flattened ring at 0, the failure mode surveyed at
  https://www.rbvi.ucsf.edu/chimerax/data/ligchem-feb2026/badchem.html

Coordinates come in as plain arrays keyed by atom name rather than as an
``AtomArray``, so a caller can map a prediction onto a reference molecule by
name -- which is what makes these safe to use across structures whose atom
ordering differs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from rdkit.Chem.rdchem import Mol

from openfold3.steering.types import RestraintSet


def worst_bounds_violation(coords: torch.Tensor, restraints: RestraintSet) -> float:
    """Largest distance-bounds violation among a molecule's restrained pairs.

    Args:
        coords: ``[n_atoms, 3]`` coordinates, indexed the way
            ``restraints.atom_index`` expects. Atoms no restraint references
            are never read, so a caller may lay a ligand out on a whole
            complex's atom axis and leave the rest at the origin.
        restraints: pairs and their permitted ``[lower, upper]`` windows, as
            built by ``openfold3.steering.featurization.build_context``.

    Returns:
        The largest amount, in Angstrom, by which any pair sits outside its
        window; ``0.0`` when every pair is inside one. An infinite upper bound
        -- what a nonbonded pair carries, since it may be arbitrarily far
        apart -- needs no special case: ``distance - inf`` is ``-inf``, which
        the clamp takes to zero.
    """
    separation = (
        coords[restraints.atom_index[:, 0]] - coords[restraints.atom_index[:, 1]]
    )
    distance = torch.linalg.norm(separation, dim=-1)
    below = (restraints.lower - distance).clamp(min=0)
    above = (distance - restraints.upper).clamp(min=0)
    return float((below + above).max())


def saturated_ring_atom_names(mol: Mol) -> list[tuple[str, ...]]:
    """Atom names of every saturated all-carbon six-ring in ``mol``.

    Restricted to that one ring class because :func:`mean_ring_torsion` is
    calibrated against a chair: an aromatic ring is flat by right, a
    heteroatom ring or one carrying a double bond puckers differently, and a
    five-ring puckers to a smaller amplitude. Including any of them would make
    the mean meaningless.

    Args:
        mol: a molecule whose atoms carry the ``annot_atom_name`` property,
            i.e. one of the pipeline's processed reference molecules.

    Returns:
        One tuple of atom names per qualifying ring, in ring-traversal order
        so consecutive names are bonded. Empty if the molecule has no such
        ring.
    """
    names = {atom.GetIdx(): atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()}
    rings = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        if not all(mol.GetAtomWithIdx(index).GetSymbol() == "C" for index in ring):
            continue
        # Every ring bond single. This is also what excludes aromatic rings,
        # whose bonds are order 1.5 -- including a ring fused to one, since the
        # shared bond is aromatic and such a ring is a half-chair anyway.
        if not all(
            mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % 6]).GetBondTypeAsDouble()
            == 1.0
            for i in range(6)
        ):
            continue
        rings.append(tuple(names[index] for index in ring))
    return rings


def torsion_degrees(points: np.ndarray) -> float:
    """Signed dihedral angle of four points, in degrees.

    Args:
        points: ``[4, 3]`` coordinates in bonded order.

    Returns:
        The angle in ``(-180, 180]``, signed: a molecule and its mirror image
        give opposite signs, which is what makes the dihedral -- unlike a bond
        angle -- able to tell them apart.
    """
    b0, b1, b2 = points[0] - points[1], points[2] - points[1], points[3] - points[2]
    b1 = b1 / np.linalg.norm(b1)
    projected_first = b0 - np.dot(b0, b1) * b1
    projected_last = b2 - np.dot(b2, b1) * b1
    return float(
        np.degrees(
            np.arctan2(
                np.dot(np.cross(b1, projected_first), projected_last),
                np.dot(projected_first, projected_last),
            )
        )
    )


def mean_ring_torsion(
    coords_by_name: Mapping[str, np.ndarray],
    rings: Sequence[Sequence[str]],
) -> float:
    """Mean absolute torsion around one or more six-rings, in degrees.

    Absolute because a chair's six ring torsions alternate in sign; averaging
    them signed would cancel to zero and report a perfect chair as flat.

    Args:
        coords_by_name: coordinates of at least every atom named in ``rings``.
        rings: ring atom names in traversal order, as
            :func:`saturated_ring_atom_names` returns them.

    Returns:
        The mean over every ring and every bond in it: about 55 degrees for an
        ideal chair, 0 for a planar ring. Averaging across rings means one
        flattened ring among several shows up as a partial drop rather than
        not at all.
    """
    angles = [
        abs(
            torsion_degrees(
                np.array(
                    [
                        coords_by_name[ring[(position + offset) % 6]]
                        for offset in range(4)
                    ]
                )
            )
        )
        for ring in rings
        for position in range(6)
    ]
    return float(np.mean(angles))
