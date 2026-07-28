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

"""Scoring predicted structures against experimental references.

Answers "is this consistent with experiment?" rather than "did it run". Scoring goes
through :func:`~openfold3.core.metrics.quality.get_superimpose_metrics` — Kabsch
superposition followed by RMSD — so the tests measure agreement with the same primitive
the model's own validation uses, rather than a second implementation that could drift.

Only protein CA-RMSD lives here for now; ligand RMSD needs symmetry-aware atom matching
and is deliberately left out.
"""

import itertools
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.metrics.quality import get_superimpose_metrics

logger = logging.getLogger(__name__)

#: Guard on the chain-assignment search below, which is factorial in the number of
#: interchangeable copies. Six covers every example query; beyond that the caller wants
#: real permutation alignment, not brute force.
MAX_PERMUTED_CHAINS = 6


def ca_positions_by_chain(cif_path: Path) -> dict[str, dict[int, np.ndarray]]:
    """Map ``chain_id -> {res_id: CA position}`` over the polymer CAs in *cif_path*.

    Keying by residue id (rather than returning a flat array) is what lets a prediction
    be compared against a reference with unmodelled gaps: only residues present in both
    are scored.
    """
    atom_array = parse_mmcif(cif_path).atom_array
    ca = atom_array[(atom_array.atom_name == "CA") & (~atom_array.hetero)]
    by_chain: dict[str, dict[int, np.ndarray]] = {}
    for chain_id, res_id, coord in zip(ca.chain_id, ca.res_id, ca.coord, strict=True):
        residues = by_chain.setdefault(str(chain_id), {})
        if int(res_id) in residues:
            # Insertion codes or altlocs would otherwise silently overwrite, quietly
            # dropping residues from the comparison instead of failing.
            raise ValueError(
                f"{cif_path}: chain {chain_id} has more than one CA for residue "
                f"{res_id}"
            )
        residues[int(res_id)] = coord
    return by_chain


def _paired_positions(
    pred_by_chain: Mapping[str, Mapping[int, np.ndarray]],
    ref_by_chain: Mapping[str, Mapping[int, np.ndarray]],
    assignment: Sequence[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack the CAs shared by each ``(pred_chain, ref_chain)`` pair, in matching order."""
    pred_xyz: list[np.ndarray] = []
    ref_xyz: list[np.ndarray] = []
    for pred_chain, ref_chain in assignment:
        pred_residues = pred_by_chain[pred_chain]
        ref_residues = ref_by_chain[ref_chain]
        for res_id in sorted(set(pred_residues) & set(ref_residues)):
            pred_xyz.append(pred_residues[res_id])
            ref_xyz.append(ref_residues[res_id])
    return np.asarray(pred_xyz, dtype=float), np.asarray(ref_xyz, dtype=float)


def _superimpose_rmsd(pred_xyz: np.ndarray, ref_xyz: np.ndarray) -> dict[str, float]:
    """Kabsch-superimpose *pred_xyz* onto *ref_xyz* and return rmsd/gdt_ts/gdt_ha."""
    metrics = get_superimpose_metrics(
        all_atom_pred_pos=torch.from_numpy(pred_xyz),
        all_atom_gt_pos=torch.from_numpy(ref_xyz),
        all_atom_mask=torch.ones(len(pred_xyz), dtype=torch.float64),
    )
    return {name: float(value) for name, value in metrics.items()}


def best_ca_rmsd(
    pred_cif: Path,
    ref_cif: Path,
    ref_chains: Sequence[str],
    pred_chains: Sequence[str] | None = None,
) -> dict[str, float]:
    """Best superposition CA-RMSD (Å) of *pred_cif* against *ref_cif*.

    ``pred_chains`` and ``ref_chains`` are matched as sets, not pairwise: every bijection
    between them is scored and the best is returned. That is what makes a homomer
    meaningful — with N interchangeable copies the chain labels a prediction happens to
    emit carry no information, so pinning a specific pairing would measure label
    agreement rather than structural agreement.

    ``pred_chains`` defaults to every polymer chain in *pred_cif*, which is what a
    single-chain prediction wants: the chain id the writer emits is then irrelevant.

    Returns the ``rmsd``/``gdt_ts``/``gdt_ha`` of the winning assignment.
    """
    if pred_chains is None:
        pred_chains = sorted(ca_positions_by_chain(pred_cif))
    if len(pred_chains) != len(ref_chains):
        raise ValueError(
            f"Need equal chain counts to pair up, got predicted {list(pred_chains)} "
            f"vs reference {list(ref_chains)}"
        )
    if len(pred_chains) > MAX_PERMUTED_CHAINS:
        raise ValueError(
            f"Refusing to brute-force {len(pred_chains)}! chain assignments; "
            f"limit is {MAX_PERMUTED_CHAINS}"
        )

    pred_by_chain = ca_positions_by_chain(pred_cif)
    ref_by_chain = ca_positions_by_chain(ref_cif)
    for label, wanted, available in (
        ("predicted", pred_chains, pred_by_chain),
        ("reference", ref_chains, ref_by_chain),
    ):
        missing = sorted(set(wanted) - set(available))
        if missing:
            raise ValueError(
                f"{label} structure {pred_cif if label == 'predicted' else ref_cif} "
                f"has no chain(s) {missing}; it holds {sorted(available)}"
            )

    best: dict[str, float] | None = None
    for permuted_ref in itertools.permutations(ref_chains):
        assignment = list(zip(pred_chains, permuted_ref, strict=True))
        pred_xyz, ref_xyz = _paired_positions(pred_by_chain, ref_by_chain, assignment)
        if not len(pred_xyz):
            raise ValueError(
                f"No residues in common between {pred_cif} and {ref_cif} "
                f"for assignment {assignment}"
            )
        metrics = _superimpose_rmsd(pred_xyz, ref_xyz)
        logger.debug("assignment %s -> CA-RMSD %.2f", assignment, metrics["rmsd"])
        if best is None or metrics["rmsd"] < best["rmsd"]:
            best = metrics

    assert best is not None  # permutations() of a non-empty sequence is non-empty
    return best
