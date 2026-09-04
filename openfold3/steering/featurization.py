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

"""Featurization: RDKit chemistry to steering batch features.

The data-side entry point (``maybe_create_steering_features``) and the chemistry
it wraps. Runs once per query on CPU; RDKit never enters the sampling loop,
and everything downstream of this module is index and bounds tensors. This
is the only module in the package that imports RDKit, which is what keeps
it off the model's import path -- see ``batch_features``.

Bounds come from RDKit's distance-geometry bounds matrix on the intact,
H-containing reference molecule, exactly as Boltz computes them
(``src/boltz/model/potentials/potentials.py``, MIT). See
THIRD_PARTY_NOTICES.md. Buffers are applied per pair classification (bond /
1-3 angle / neither) rather than uniformly, and the reference molecule is
never rebuilt from an ``AtomArray`` — a round trip through RDKit valence
perception drops molecules with formal charges RDKit cannot reassign (e.g.
quaternary nitrogen, boron).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np
import torch
from biotite.structure import AtomArray, get_residue_count
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

from openfold3.core.data.pipelines.sample_processing.conformer import (
    ProcessedReferenceMolecule,
)
from openfold3.core.data.primitives.structure.component import PERIODIC_TABLE
from openfold3.core.data.primitives.structure.labels import residue_view_iter
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.steering import defaults
from openfold3.steering.batch_features import context_to_features
from openfold3.steering.config import SteeringSettings
from openfold3.steering.potentials import DistanceBoundsPotential
from openfold3.steering.types import RestraintSet, SteeringContext


class MissingReferenceMoleculeError(ValueError):
    """Raised when reference molecules do not line up 1:1 with the residues
    in the structure.

    Subclasses ``ValueError`` so callers that already catch ``ValueError``
    around featurization keep working unchanged.
    """

    pass


@dataclass
class _DistanceConstraints:
    """Mutable accumulator for one or more ligand chains' distance pairs.

    One entry per field per pair, all lists the same length. Ethanol (``CCO``,
    heavy atoms only, at global offset 18 behind a protein chain) produces
    three entries -- both bonds and the single 1-3 pair across them::

        index            [(18, 19),  (18, 20),  (19, 20)]   # C1-C2, C1..O1, C2-O1
        lower            [1.504,     2.336,     1.384]      # raw bounds-matrix
        upper            [1.524,     2.416,     1.404]      #   values, unbuffered
        bond_mask        [True,      False,     True]
        angle_mask       [False,     True,      False]
        pair_vdw_cutoffs [2.05,      1.975,     1.975]

    ``_tensorize_distance_constraints`` then widens these into the restraint
    set: the two bonded pairs take the bond buffer and the 1-3 pair the angle
    buffer, giving ``[1.316, 1.715]``, ``[2.044, 2.718]`` and
    ``[1.211, 1.579]``. The vdW cutoffs cap the bonded upper bounds, but on a
    molecule this small none of them binds.
    """

    index: list[tuple[int, int]] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)
    bond_mask: list[bool] = field(default_factory=list)
    angle_mask: list[bool] = field(default_factory=list)
    pair_vdw_cutoffs: list[float] = field(default_factory=list)

    def extend(self, other: _DistanceConstraints) -> None:
        for constraint_field in fields(self):
            getattr(self, constraint_field.name).extend(
                getattr(other, constraint_field.name)
            )


def _compute_distance_constraints(
    mol: Mol,
    idx_map: dict[int, int],
    *,
    vdw_pair_cutoff_offset: float,
) -> _DistanceConstraints:
    """RDKit distance-geometry bounds for one ligand, in the model atom axis.

    Args:
        mol: RDKit reference molecule containing the ligand bond graph.
        idx_map: mapping from reference-molecule atom indices to model
            (global) atom indices. Only pairs where both atoms are present
            survive the crop.
        vdw_pair_cutoff_offset: additive offset (Angstrom) on the mean
            pairwise van der Waals radius, used as a per-pair floor/ceiling
            during buffering.

    Raises:
        ValueError: if a mapped atom has an unsupported atomic number.

    Returns:
        Unbuffered distance bounds and pair classifications for every atom
        pair with both atoms present in ``idx_map``, on the model atom axis.
    """
    constraints = _DistanceConstraints()
    if mol.GetNumAtoms() <= 1:
        return constraints

    mapped_atomic_numbers = {
        atom_idx: mol.GetAtomWithIdx(atom_idx).GetAtomicNum() for atom_idx in idx_map
    }
    if any(
        atomic_number < 1 or atomic_number > PERIODIC_TABLE.GetMaxAtomicNumber()
        for atomic_number in mapped_atomic_numbers.values()
    ):
        raise ValueError(
            "Chemical steering requires ligand atoms with supported atomic numbers."
        )

    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    bounds = GetMoleculeBoundsMatrix(
        mol,
        set15bounds=True,
        scaleVDW=True,
        doTriangleSmoothing=True,
        useMacrocycle14config=False,
    )
    bonds = {
        tuple(sorted(match))
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*"))
    }
    angles = {
        tuple(sorted((match[0], match[2])))
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*"))
    }

    for i, j in zip(*np.triu_indices(mol.GetNumAtoms(), k=1), strict=True):
        i_idx, j_idx = int(i), int(j)
        if i_idx not in idx_map or j_idx not in idx_map:
            continue
        atomic_numbers = (mapped_atomic_numbers[i_idx], mapped_atomic_numbers[j_idx])
        atom_pair = tuple(sorted((i_idx, j_idx)))
        constraints.index.append((idx_map[i_idx], idx_map[j_idx]))
        constraints.upper.append(float(bounds[i_idx, j_idx]))
        constraints.lower.append(float(bounds[j_idx, i_idx]))
        constraints.bond_mask.append(atom_pair in bonds)
        constraints.angle_mask.append(atom_pair in angles)
        pair_radii = [PERIODIC_TABLE.GetRvdw(number) for number in atomic_numbers]
        constraints.pair_vdw_cutoffs.append(
            vdw_pair_cutoff_offset + sum(pair_radii) / len(pair_radii)
        )

    return constraints


def _tensorize_distance_constraints(
    constraints: _DistanceConstraints,
    *,
    bond_buffer: float,
    angle_buffer: float,
    clash_buffer: float,
) -> RestraintSet:
    """Apply per-pair-class buffers and the VDW floor/ceiling.

    Four pair classes, each buffered differently:
      - bond only: tighten both bounds by ``bond_buffer``.
      - 1-3 angle only: tighten both bounds by ``angle_buffer``.
      - both (a 3-membered ring): tighten by ``min(bond_buffer, angle_buffer)``.
      - neither (nonbonded): loosen the lower (clash) bound by
        ``clash_buffer`` and drop the upper bound to +inf.
    Then, regardless of class, the lower bound on any non-bonded pair is
    floored at the VDW cutoff carried on ``constraints``, and the upper bound
    on any bonded pair is capped at it.

    Args:
        constraints: unbuffered bounds, pair classifications and per-pair VDW
            cutoffs from ``_compute_distance_constraints``, already expressed
            on the model atom axis.
        bond_buffer: relative tolerance for 1-2 bonded pairs, applied
            multiplicatively as ``[lower * (1 - buffer), upper * (1 + buffer)]``
            so a restraint accommodates the spread of a real conformer
            ensemble instead of pinning one idealized geometry.
        angle_buffer: the same, for 1-3 (angle-defined) pairs. A pair that is
            both bonded and angle-defined, as in a three-membered ring, takes
            the tighter of the two.
        clash_buffer: relative slack on the lower bound of pairs that are
            neither bonded nor angle-defined. These keep only a lower
            (clash) bound; their upper bound is dropped.

    Returns:
        A ``RestraintSet`` with buffered bounds, or an empty one if
        ``constraints`` held no pairs.
    """
    if not constraints.index:
        return RestraintSet(
            atom_index=torch.empty((0, 2), dtype=torch.int64),
            lower=torch.empty((0,), dtype=torch.float32),
            upper=torch.empty((0,), dtype=torch.float32),
        )

    index = torch.tensor(constraints.index, dtype=torch.int64)
    lower = torch.tensor(constraints.lower, dtype=torch.float32)
    upper = torch.tensor(constraints.upper, dtype=torch.float32)
    bond = torch.tensor(constraints.bond_mask, dtype=torch.bool)
    angle = torch.tensor(constraints.angle_mask, dtype=torch.bool)
    pair_vdw_cutoff = torch.tensor(constraints.pair_vdw_cutoffs, dtype=torch.float32)

    lower[bond & ~angle] *= 1.0 - bond_buffer
    upper[bond & ~angle] *= 1.0 + bond_buffer
    lower[~bond & angle] *= 1.0 - angle_buffer
    upper[~bond & angle] *= 1.0 + angle_buffer
    shared_buffer = min(bond_buffer, angle_buffer)
    lower[bond & angle] *= 1.0 - shared_buffer
    upper[bond & angle] *= 1.0 + shared_buffer
    lower[~bond & ~angle] *= 1.0 - clash_buffer
    upper[~bond & ~angle] = float("inf")
    lower[~bond] = torch.maximum(lower[~bond], pair_vdw_cutoff[~bond])
    upper[bond] = torch.minimum(upper[bond], pair_vdw_cutoff[bond])

    return RestraintSet(atom_index=index, lower=lower, upper=upper)


def extract_distance_bounds(
    mol: Mol,
    atom_index: torch.Tensor,
    *,
    bond_buffer: float = defaults.BOND_BUFFER,
    angle_buffer: float = defaults.ANGLE_BUFFER,
    clash_buffer: float = defaults.CLASH_BUFFER,
    vdw_pair_cutoff_offset: float = defaults.VDW_PAIR_CUTOFF_OFFSET,
) -> RestraintSet:
    """Bounds matrix on the intact, H-containing reference molecule.

    Args:
        mol: RDKit reference molecule (``ProcessedReferenceMolecule.mol``).
        atom_index: int64 ``[n_local]``; global (model) atom index per local
            RDKit atom, or ``-1`` for an atom that was cropped out.
        bond_buffer: relative tolerance for 1-2 bonded pairs.
        angle_buffer: relative tolerance for 1-3 (angle-defined) pairs.
        clash_buffer: relative slack on the lower bound of nonbonded pairs.
        vdw_pair_cutoff_offset: additive offset (Angstrom) on the mean
            pairwise van der Waals radius, which floors the lower bound of
            nonbonded pairs and caps the upper bound of bonded ones.

    Returns:
        A ``RestraintSet`` of buffered distance bounds on the model atom axis.
    """
    idx_map = {
        local: int(global_idx)
        for local, global_idx in enumerate(atom_index.tolist())
        if global_idx >= 0
    }
    constraints = _compute_distance_constraints(
        mol, idx_map, vdw_pair_cutoff_offset=vdw_pair_cutoff_offset
    )
    return _tensorize_distance_constraints(
        constraints,
        bond_buffer=bond_buffer,
        angle_buffer=angle_buffer,
        clash_buffer=clash_buffer,
    )


def build_context(
    atom_array: AtomArray,
    processed_reference_molecules: list[ProcessedReferenceMolecule],
    *,
    n_atoms: int,
    bond_buffer: float = defaults.BOND_BUFFER,
    angle_buffer: float = defaults.ANGLE_BUFFER,
    clash_buffer: float = defaults.CLASH_BUFFER,
    vdw_pair_cutoff_offset: float = defaults.VDW_PAIR_CUTOFF_OFFSET,
) -> SteeringContext:
    """Build a SteeringContext from every ligand residue in ``atom_array``.

    The local -> global atom-index bridge (``in_crop_mask`` positionally
    zipped against each ligand residue) is load-bearing: it is what lets
    restraints skip atoms that were cropped out without rebuilding the
    molecule from ``atom_array``.

    Args:
        atom_array: preprocessed structure whose atom axis the model uses.
            Only residues whose atoms are all ligands contribute restraints.
        processed_reference_molecules: reference molecules positionally
            aligned with the residues of ``atom_array``, one per residue.
        n_atoms: size of the model's atom axis, recorded on the context so
            the sampler can reject a batch built for a different structure.
        bond_buffer: relative tolerance for 1-2 bonded pairs.
        angle_buffer: relative tolerance for 1-3 (angle-defined) pairs.
        clash_buffer: relative slack on the lower bound of nonbonded pairs.
        vdw_pair_cutoff_offset: additive offset (Angstrom) on the mean
            pairwise van der Waals radius, which floors the lower bound of
            nonbonded pairs and caps the upper bound of bonded ones.

    Raises:
        MissingReferenceMoleculeError: if the reference molecules do not line
            up 1:1 with the residues in ``atom_array``.
        ValueError: if a processed reference molecule's crop-surviving atom
            count does not match its residue's atom count.

    Returns:
        A ``SteeringContext`` with one ``distance_bounds_potential``
        restraint set spanning every ligand residue in ``atom_array``.
    """
    n_residues = get_residue_count(atom_array)
    if len(processed_reference_molecules) != n_residues:
        raise MissingReferenceMoleculeError(
            "Chemical steering needs one processed reference molecule per residue, "
            f"but got {len(processed_reference_molecules)} reference molecule(s) "
            f"for {n_residues} residue(s)."
        )

    all_constraints = _DistanceConstraints()
    global_atom_indices = np.arange(len(atom_array))
    for residue, processed_mol in zip(
        residue_view_iter(atom_array), processed_reference_molecules, strict=True
    ):
        if not np.all(residue.molecule_type_id == MoleculeType.LIGAND):
            continue

        reference_indices = np.flatnonzero(processed_mol.in_crop_mask)
        residue_indices = global_atom_indices[residue.indices]
        if len(reference_indices) != len(residue_indices):
            raise ValueError(
                "Processed ligand reference atoms must match the OF3 residue."
            )
        idx_map = dict(
            zip(reference_indices.tolist(), residue_indices.tolist(), strict=True)
        )
        mol_constraints = _compute_distance_constraints(
            processed_mol.mol, idx_map, vdw_pair_cutoff_offset=vdw_pair_cutoff_offset
        )
        all_constraints.extend(mol_constraints)

    restraint_set = _tensorize_distance_constraints(
        all_constraints,
        bond_buffer=bond_buffer,
        angle_buffer=angle_buffer,
        clash_buffer=clash_buffer,
    )
    return SteeringContext(
        restraints={DistanceBoundsPotential.name: restraint_set}, n_atoms=n_atoms
    )


def maybe_create_steering_features(
    atom_array: AtomArray,
    processed_reference_molecules: list[ProcessedReferenceMolecule],
    settings: SteeringSettings | None = None,
) -> dict[str, torch.Tensor]:
    """Create inference-time chemical steering features, if there are any.

    ``maybe_`` because three ordinary situations produce no features at all:
    steering off, every term off, or a query with no ligand to restrain.

    Steering is a run-level setting, so this applies to every query in the
    run. Queries with no ligand produce no restraints and are handled by the
    same empty return as a disabled run.

    Args:
        atom_array: preprocessed structure whose atom axis the model uses.
        processed_reference_molecules: reference molecules aligned with the
            residues in ``atom_array``.
        settings: validated run-level settings; defaults to steering off.

    Raises:
        MissingReferenceMoleculeError: if the reference molecules do not line
            up 1:1 with the residues in ``atom_array``.

    Returns:
        Batch features for the sampler, or ``{}`` when steering is disabled,
        every term is disabled, or the query yielded no restraints. Emitting
        no keys at all is what makes a disabled run bit-identical to one
        without steering compiled in: the sampler probes for its enable key
        with ``.get`` and never runs.
    """
    settings = settings if settings is not None else SteeringSettings()
    if not settings.enabled or not settings.active_terms():
        return {}

    ctx = build_context(
        atom_array,
        processed_reference_molecules,
        n_atoms=len(atom_array),
    )
    return context_to_features(ctx, settings)
