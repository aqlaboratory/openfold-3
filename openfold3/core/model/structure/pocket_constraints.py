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

"""Pocket-guided ligand proposal sampling for partial-diffusion refinement.

Consumes the `pocket_sampling_*` batch features produced by
`openfold3.core.data.pipelines.featurization.pocket_constraints.create_pocket_sampling_features`.
Those features are always generated as a complete set, so the accessors here
trust that invariant rather than re-validating batch contents.
"""

import logging
from typing import NamedTuple

import torch

from openfold3.core.model.structure.augmentation import sample_rotations

logger = logging.getLogger(__name__)


def _pocket_sampling_enabled(batch: dict) -> bool:
    enabled = batch.get("pocket_sampling_enabled")
    return enabled is not None and bool(enabled.flatten()[0].item())


def _batch_scalar(batch: dict, name: str, cast=float):
    return cast(batch[name].flatten()[0].item())


def _feature_mask(batch: dict, atom_mask: torch.Tensor, name: str) -> torch.Tensor:
    mask = batch[name].to(device=atom_mask.device).bool()
    batch_dim = atom_mask.shape[0]
    while mask.dim() > 2 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and batch_dim > 1:
        mask = mask.expand(batch_dim, -1)
    return mask


def _score_ligand_pose(
    lig_pose: torch.Tensor,
    protein: torch.Tensor,
    pocket: torch.Tensor,
    lig_vdw: torch.Tensor,
    protein_vdw: torch.Tensor,
    contact_distance: float,
    vdw_buffer: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank a ligand proposal by site entry."""
    d_prot = torch.cdist(lig_pose.unsqueeze(0), protein.unsqueeze(0)).squeeze(0)
    vdw_lower = (lig_vdw[:, None] + protein_vdw[None, :]) * (1.0 - vdw_buffer)
    overlap = torch.relu(vdw_lower - d_prot)
    vdw_overlap = torch.square(overlap).sum()
    min_prot = d_prot.min()

    d_pocket = torch.cdist(lig_pose.unsqueeze(0), pocket.unsqueeze(0)).squeeze(0)
    lig_to_pocket = d_pocket.min(dim=1).values
    contact = torch.square(torch.relu(lig_to_pocket - contact_distance)).mean()
    lig_atoms_in_pocket = (lig_to_pocket < contact_distance).sum().to(lig_pose.dtype)
    pocket_com_dist = torch.linalg.vector_norm(
        lig_pose.mean(dim=0) - pocket.mean(dim=0)
    )

    return pocket_com_dist, vdw_overlap, min_prot, lig_atoms_in_pocket, contact


class _PocketSamplingCandidate(NamedTuple):
    """A ligand pose candidate and its pocket-localization metrics.

    ``lig_atoms_in_pocket`` is the count of ligand atoms within the contact
    threshold. The ascending sort key negates this value so larger counts rank
    ahead of smaller counts.
    """

    pocket_com_dist: torch.Tensor
    vdw_overlap: torch.Tensor
    min_protein_dist: torch.Tensor
    lig_atoms_in_pocket: torch.Tensor
    contact_penalty: torch.Tensor
    pose: torch.Tensor
    parent_slot: int


def _candidate_sort_key(
    candidate: _PocketSamplingCandidate,
) -> tuple[float, float, float, float, float]:
    return (
        float(candidate.pocket_com_dist),
        -float(candidate.lig_atoms_in_pocket),
        float(candidate.contact_penalty),
        float(candidate.vdw_overlap),
        -float(candidate.min_protein_dist),
    )


def _build_pocket_sampling_seeds(
    batch: dict,
    xl_base: torch.Tensor,
    atom_mask: torch.Tensor,
    no_rollout_samples: int,
) -> torch.Tensor:
    """Generate ligand seeds in a user-specified pocket for partial diffusion."""
    batch_dim = atom_mask.shape[0]
    lig_mask = _feature_mask(batch, atom_mask, "pocket_sampling_ligand_atom_mask")
    pocket_mask = _feature_mask(batch, atom_mask, "pocket_sampling_pocket_atom_mask")
    lig_idx = torch.nonzero(lig_mask[0], as_tuple=False).flatten()
    pocket_idx = torch.nonzero(pocket_mask[0], as_tuple=False).flatten()
    protein_idx = torch.nonzero(atom_mask[0].bool() & ~lig_mask[0], as_tuple=False)
    protein_idx = protein_idx.flatten()

    radii = batch["pocket_sampling_vdw_radii"].to(
        device=atom_mask.device, dtype=atom_mask.dtype
    )
    while radii.dim() > 1:
        radii = radii[0]
    lig_vdw = radii[lig_idx]
    protein_vdw = radii[protein_idx]

    conformers = batch.get("pocket_sampling_conformer_rels")
    if conformers is not None:
        conformers = conformers.to(device=atom_mask.device, dtype=atom_mask.dtype)
        while conformers.dim() > 3 and conformers.shape[0] == 1:
            conformers = conformers[0]

    n_parents = max(
        1,
        min(
            no_rollout_samples,
            _batch_scalar(batch, "pocket_sampling_num_parents", int),
        ),
    )
    n_candidates = max(
        no_rollout_samples,
        _batch_scalar(batch, "pocket_sampling_candidates", int),
    )
    contact_distance = _batch_scalar(batch, "pocket_sampling_contact_distance", float)
    center_jitter = _batch_scalar(batch, "pocket_sampling_center_jitter", float)
    surface_jitter = _batch_scalar(batch, "pocket_sampling_surface_jitter", float)
    vdw_buffer = _batch_scalar(batch, "pocket_sampling_vdw_buffer", float)
    diversity_rmsd = _batch_scalar(batch, "pocket_sampling_diversity_rmsd", float)

    all_seed_batches = []
    for b in range(batch_dim):
        parent_scores = []
        for s in range(no_rollout_samples):
            score, _, _, _, _ = _score_ligand_pose(
                xl_base[b, s, lig_idx],
                xl_base[b, s, protein_idx],
                xl_base[b, s, pocket_idx],
                lig_vdw,
                protein_vdw,
                contact_distance,
                vdw_buffer,
            )
            parent_scores.append(score)
        parent_order = torch.argsort(torch.stack(parent_scores))[:n_parents]

        candidates = []
        for i in range(n_candidates):
            parent_slot = int(parent_order[i % n_parents])
            parent = xl_base[b, parent_slot]
            pocket_parent = parent[pocket_idx]
            lig_parent = parent[lig_idx]
            lig_com = lig_parent.mean(dim=0, keepdim=True)
            parent_rel = lig_parent - lig_com

            if i < n_parents:
                pose = lig_parent
            else:
                if conformers is None or conformers.shape[0] == 0:
                    lig_rel = parent_rel
                else:
                    lig_rel = conformers[
                        torch.randint(conformers.shape[0], (), device=atom_mask.device)
                    ]
                rot = sample_rotations(
                    shape=(),
                    dtype=atom_mask.dtype,
                    device=atom_mask.device,
                )
                if torch.rand((), device=atom_mask.device) < 0.5:
                    target = pocket_parent.mean(dim=0, keepdim=True)
                    target = target + center_jitter * torch.randn(
                        (1, 3), dtype=atom_mask.dtype, device=atom_mask.device
                    )
                else:
                    target = pocket_parent[
                        torch.randint(
                            pocket_parent.shape[0], (), device=atom_mask.device
                        )
                    ].unsqueeze(0)
                    target = target + surface_jitter * torch.randn(
                        (1, 3), dtype=atom_mask.dtype, device=atom_mask.device
                    )
                pose = lig_rel @ rot.transpose(-1, -2) + target

            score, vdw_overlap, min_prot, lig_atoms, contact = _score_ligand_pose(
                pose,
                parent[protein_idx],
                parent[pocket_idx],
                lig_vdw,
                protein_vdw,
                contact_distance,
                vdw_buffer,
            )
            candidates.append(
                _PocketSamplingCandidate(
                    pocket_com_dist=score,
                    vdw_overlap=vdw_overlap,
                    min_protein_dist=min_prot,
                    lig_atoms_in_pocket=lig_atoms,
                    contact_penalty=contact,
                    pose=pose,
                    parent_slot=parent_slot,
                )
            )

        candidates.sort(key=_candidate_sort_key)
        selected: list[_PocketSamplingCandidate] = []
        for cand in candidates:
            if all(
                torch.sqrt(
                    torch.square(cand.pose - prev.pose).sum(dim=-1).mean()
                ).item()
                >= diversity_rmsd
                for prev in selected
            ):
                selected.append(cand)
                if len(selected) == no_rollout_samples:
                    break
        if len(selected) < no_rollout_samples:
            selected.extend(candidates[: no_rollout_samples - len(selected)])

        seed_list = []
        for cand in selected[:no_rollout_samples]:
            seed = xl_base[b, cand.parent_slot].clone()
            seed[lig_idx] = cand.pose
            seed_list.append(seed)
        all_seed_batches.append(torch.stack(seed_list, dim=0))

        best = selected[0]
        logger.info(
            "[pocket_sampling] parents=%s/%s candidates=%s "
            "best_pocket_com=%.3g best_vdw=%.3g "
            "best_min_protein=%.3g best_lig_atoms=%s",
            n_parents,
            no_rollout_samples,
            n_candidates,
            float(best.pocket_com_dist),
            float(best.vdw_overlap),
            float(best.min_protein_dist),
            int(best.lig_atoms_in_pocket.item()),
        )

    return torch.stack(all_seed_batches, dim=0)
