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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations, combinations_with_replacement
from typing import Literal

import torch

from openfold3.core.data.resources.lists import (
    AB_AG_CHAIN_PAIR_TYPES,
    AB_AG_CHAIN_TYPES,
)
from openfold3.core.data.resources.token_atom_constants import BACKBONE_ATOMS
from openfold3.core.metrics.confidence import compute_plddt
from openfold3.core.metrics.rasa import compute_rasa_batch
from openfold3.core.utils.atomize_utils import (
    aggregate_atom_feat_to_tokens_nd,
    broadcast_token_feat_to_atoms,
)
from openfold3.core.utils.geometry.kabsch_alignment import (
    apply_transformation,
    get_optimal_transformation,
    kabsch_align,
)
from openfold3.core.utils.tensor_utils import tensor_tree_map


def select_inter_filter_mask(
    inter_mask_atomized: torch.Tensor, mol_type_mask: torch.Tensor, out_shape: list
) -> torch.Tensor:
    """
    Due to MAX INT limit with masked select, we need to select from the
    inter_mask_atomized mask for each sample independently and then stack
    them together.

    Args:
        inter_mask_atomized:
            [*, N_atom, N_atom] Pairwise filter for inter-chain computations
        mol_type_mask:
            [*, N_atom, N_atom] Boolean mask for molecule type to select from
            inter_mask_atomized
        out_shape:
            Shape of output tensor
    Returns:
        Inter-chain mask per sample filtered by molecule type
    """
    n_samples = inter_mask_atomized.shape[-3]

    inter_mask_filtered = []
    for i in range(n_samples):
        inter_mask_filtered_chunk = torch.masked_select(
            inter_mask_atomized[:, i], mol_type_mask[:, i]
        ).reshape(out_shape)
        inter_mask_filtered.append(inter_mask_filtered_chunk)

    return torch.stack(inter_mask_filtered, dim=-3)


def _get_atom_name_mask(
    ref_atom_name_chars: torch.Tensor, names: list[str]
) -> torch.Tensor:
    """
    ref_atom_name_chars: tensor with shape [..., Natom, 4, 64], one-hot per char.
    names: list of atom names (e.g., ["CA"] or ["P", "OP1", "OP2"]).

    Returns:
        Boolean mask with shape [..., Natom], True where name matches any in `names`.
    """
    # Convert one-hot to code indices [..., Natom, 4]
    codes = ref_atom_name_chars.argmax(dim=-1)

    # Build target code rows for the names, right-padded to 4
    def name_to_codes(n: str) -> torch.Tensor:
        s = n.ljust(4)[:4]
        return torch.tensor(
            [ord(c) - 32 for c in s], device=codes.device, dtype=codes.dtype
        )

    # [M, 4]
    targets = torch.stack([name_to_codes(n) for n in names], dim=0)

    # Flatten leading dims so we can broadcast cleanly
    leading = codes.shape[:-2]
    N = codes.shape[-2]
    # [B*, N, 4]
    codes2 = codes.reshape(-1, N, 4)

    # Compare every atom against every target name
    # [B*, N, M, 4]
    eq = codes2.unsqueeze(2) == targets.unsqueeze(0)
    # [B*, N]
    match_any = eq.all(dim=-1).any(dim=-1)

    # Restore original leading dims
    return match_any.reshape(*leading, N)


def _spread_contacts(
    is_contact_atom: torch.Tensor,
    atom_to_res_id: torch.Tensor,
) -> torch.Tensor:
    """
    For each atom, mark True if *any* atom in the same residue has a contact.
    Returns a [B,S,N] bool mask aligned with `is_contact_atom`.

    is_contact_atom: torch.Tensor [*, N_atoms]
    atom_to_res_id: torch.Tensor [*, N_atoms]
    """
    if is_contact_atom.shape != atom_to_res_id.shape:
        raise ValueError(
            f"Shape mismatch: is_contact_atom {is_contact_atom.shape} vs "
            f"atom_to_res_id {atom_to_res_id.shape}"
        )

    bs = is_contact_atom.shape[:-1]
    idx = atom_to_res_id.long()

    # +1 because residue ids start at 1 (slot 0 remains unused)
    max_res_id = int(idx.max().item()) + 1
    per_res_any = torch.zeros(
        bs + (max_res_id,), dtype=torch.int32, device=is_contact_atom.device
    )

    # Reduce with AMAX over atoms per residue
    per_res_any.scatter_reduce_(
        dim=-1,
        index=idx,
        src=is_contact_atom.to(torch.int32),
        reduce="amax",
        include_self=False,
    )

    # Map residue-level flag back to each atom
    mask_atoms = per_res_any.gather(-1, idx) > 0
    return mask_atoms


def gdt(p1, p2, mask, cutoffs):
    n = torch.sum(mask, dim=-1)

    p1 = p1.float()
    p2 = p2.float()

    distances = torch.sqrt(torch.sum((p1 - p2) ** 2, dim=-1))
    scores = [torch.sum((distances <= c) * mask, dim=-1) / n for c in cutoffs]

    return torch.sum(torch.stack(scores, dim=-1), dim=-1) / len(scores)


def gdt_ts(p1, p2, mask):
    return gdt(p1, p2, mask, [1.0, 2.0, 4.0, 8.0])


def gdt_ha(p1, p2, mask):
    return gdt(p1, p2, mask, [0.5, 1.0, 2.0, 4.0])


def rmsd(
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Root Mean Square Deviation of Atomic Positions (RMSD) between a
    set of predicted and ground-truth coordinates.

    Args:
        pred_positions:
            [*, N, 3] the predicted coordinates
        target_positions:
            [*, N, 3] the ground-truth coordinates
        positions_mask:
            [*, N] mask for coordinates that should not be considered

    Returns:
        [*] the RMSDs of the predicted coordinates for each "batch" dimension
        (e.g. actual batch dimension and/or structure module layers)
    """
    squared_error_dists = torch.sum((pred_positions - target_positions) ** 2, dim=-1)

    # mask unobserved atoms
    squared_error_dists = squared_error_dists * positions_mask
    n_observed_atoms = torch.sum(positions_mask, dim=-1)

    # compute RMSD
    msd = torch.sum(squared_error_dists, dim=-1) / n_observed_atoms
    rmsd = torch.sqrt(msd)

    return rmsd


def drmsd(
    pair_dist_pred_pos: torch.Tensor,
    pair_dist_gt_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    eps: float | None = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes drmsds from pair distances

    Args:
        pair_dist_pred_pos: predicted coordinates [*, n_atom, n_atom]
        pair_dist_gt_pos: gt coordinates [*, n_atom, n_atom]
        all_atom_mask: atom mask [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom]
        eps: epsilon
    Returns:
        intra_drmsd: drmsd within chains
        inter_drmsd: drmsd across chains

    Note:
        returns None if inter_drmsd is invalid (ie. single chain)
    """
    # Calculate squared distance differences
    drmsd = pair_dist_pred_pos - pair_dist_gt_pos
    drmsd = drmsd**2

    # Apply mask and exclude diagonal
    mask = all_atom_mask[..., None] * all_atom_mask[..., None, :]
    mask = mask * (1.0 - torch.eye(mask.shape[-1], device=all_atom_mask.device))

    # Create intra and inter chain masks
    intra_mask = torch.where(asym_id[..., None] == asym_id[..., None, :], 1, 0).bool()
    inter_mask = ~intra_mask

    intra_drmsd = None
    intra_mask = intra_mask * mask
    if torch.any(intra_mask):
        intra_drmsd = drmsd * intra_mask
        intra_drmsd = torch.sum(intra_drmsd, dim=(-1, -2))
        n_intra = torch.sum(intra_mask, dim=(-1, -2)) + eps
        intra_drmsd = intra_drmsd * (1 / n_intra)
        intra_drmsd = torch.sqrt(intra_drmsd)

    inter_drmsd = None
    inter_mask = inter_mask * mask
    if torch.any(inter_mask):
        inter_drmsd = drmsd * inter_mask
        inter_drmsd = torch.sum(inter_drmsd, dim=(-1, -2))
        n_inter = torch.sum(inter_mask, dim=(-1, -2)) + eps
        inter_drmsd = inter_drmsd * (1 / n_inter)
        inter_drmsd = torch.sqrt(inter_drmsd)

    return intra_drmsd, inter_drmsd


def lddt(
    pair_dist_pred_pos: torch.Tensor,
    pair_dist_gt_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
    intra_mask_filter: torch.Tensor,
    inter_mask_filter: torch.Tensor,
    asym_id: torch.Tensor,
    threshold: Sequence | None = (0.5, 1.0, 2.0, 4.0),
    cutoff: float | None = 15.0,
    eps: float | None = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates lddt scores from pair distances.

    Performs a disjoint calculation on within-chain and cross-chain distances.

    Args:
        pair_dist_pred_pos:
            Pairwise distance of prediction [*, n_atom, n_atom]
        pair_dist_gt_pos:
            Pairwise distance of gt [*, n_atom, n_atom]
        all_atom_mask:
            Atom level mask (typically for subsetting the calculation to atoms that are
            resolved in the GT) [*, n_atom]
        intra_mask_filter: [*, n_atom]
            Filter for intra chain computations
        inter_mask_filter: [*, n_atom, n_atom]
            Filter for inter chain computations
        asym_id:
            Atomized asym_id feature [*, n_atom]
        threshold:
            A list of thresholds to apply for lddt computation - Standard values: [0.5,
            1., 2., 4.] - lddt_uha (for ligands): [0.25, 0.5, 0.75, 1.]
        cutoff:
            distance cutoff (aka. inclusion radius) - Nucleic Acids (DNA/RNA) 30. -
            Other biomolecules (Protein/Ligands) 15.
        eps:
            Constant for numerical stability.

    Returns:
        intra_score:
            intra lddt scores [*]
        inter_score:
            inter lddt scores [*]

    Note:
        returns None for inter_score if inter_lddt invalid (ie. single chain, no atom
        pair within cutoff)
    """
    # create a mask
    n_atom = pair_dist_gt_pos.shape[-2]
    dists_to_score = (pair_dist_gt_pos < cutoff) * (
        all_atom_mask[..., None]
        * all_atom_mask[..., None, :]
        * (1.0 - torch.eye(n_atom, device=all_atom_mask.device))
    )  # [*, n_atom, n_atom]

    # distinguish intra- and inter- pair indices based on asym_id
    intra_mask = torch.where(asym_id[..., None] == asym_id[..., None, :], 1, 0).bool()
    inter_mask = ~intra_mask  # [*, n_atom, n_atom]

    # update masks with filters
    intra_mask = intra_mask * (
        intra_mask_filter[..., None] * intra_mask_filter[..., None, :]
    )
    inter_mask = inter_mask * inter_mask_filter

    # get lddt scores
    dist_l1 = torch.abs(pair_dist_gt_pos - pair_dist_pred_pos)  # [*, n_atom, n_atom]
    score = torch.zeros_like(dist_l1)
    for distance_threshold in threshold:
        score += (dist_l1 < distance_threshold).type(dist_l1.dtype)
    score = score / len(threshold)

    # Normalize to get intra_lddt scores
    intra_mask = dists_to_score * intra_mask
    intra_score = None
    if torch.any(intra_mask):
        intra_norm = 1.0 / (eps + torch.sum(intra_mask, dim=(-1, -2)))
        intra_score = intra_norm * (torch.sum(intra_mask * score, dim=(-1, -2)))

    # inter_score only applies when there exist atom pairs with
    # different asym_id (inter_mask) and distance threshold (dists_to_score)
    inter_mask = dists_to_score * inter_mask
    inter_score = None
    if torch.any(inter_mask):
        inter_norm = 1.0 / (eps + torch.sum(inter_mask, dim=(-1, -2)))
        inter_score = inter_norm * (torch.sum(inter_mask * score, dim=(-1, -2)))

    return intra_score, inter_score


def interface_lddt(
    all_atom_pred_pos_1: torch.Tensor,
    all_atom_pred_pos_2: torch.Tensor,
    all_atom_gt_pos_1: torch.Tensor,
    all_atom_gt_pos_2: torch.Tensor,
    all_atom_mask1: torch.Tensor,
    all_atom_mask2: torch.Tensor,
    filter_mask: torch.Tensor,
    cutoff: float | None = 15.0,
    eps: float | None = 1e-10,
) -> torch.Tensor:
    """
    Calculates interface_lddt (ilddt) score between two different molecules.

    Note that, unlike the lddt function, this function uses as inputs the flat 3D
    coordinate tensors, not the pairwise distances between coordinates

    Args:
        all_atom_pred_pos_1:
            predicted protein coordinates [*, n_atom1, 3]
        all_atom_pred_pos_2:
            predicted interacting molecule coordinates [*, n_atom2, 3]
        all_atom_gt_pos_1:
            gt protein coordinates [*, n_atom1, 3]
        all_atom_gt_pos_2:
            gt interacting molecule coordinates  [*, n_atom2, 3]
        all_atom_mask1:
            protein atom mask [*, n_atom1]
        all_atom_mask2:
            interacting molecule atom maks [*, n_atom2]
        filter_mask:
            pairwise filter for atom types [*, n_atom1, n_atom2]
        cutoff:
            distance cutoff - Nucleic Acids (DNA/RNA) 30. - Others(Protein/Ligands) 15.
        eps:
            Constant for numerical stability.

    Returns:
        scores: ilddt scores [*]
    """
    # get pairwise distance
    pair_dist_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_gt_pos_1.unsqueeze(-2) - all_atom_gt_pos_2.unsqueeze(-3)) ** 2,
            dim=-1,
        )
    )  # [*, n_atom1, n_atom2]
    pair_dist_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos_1.unsqueeze(-2) - all_atom_pred_pos_2.unsqueeze(-3))
            ** 2,
            dim=-1,
        )
    )  # [*, n_atom1, n_atom2]

    # create a mask
    dists_to_score = (pair_dist_true < cutoff) * (
        all_atom_mask1[..., None] * all_atom_mask2[..., None, :]
    )  # [*, n_atom1, n_atom2]
    dists_to_score = dists_to_score * filter_mask

    score = None
    if torch.any(dists_to_score):
        # get score
        dist_l1 = torch.abs(pair_dist_true - pair_dist_pred)
        score = (
            (dist_l1 < 0.5).type(dist_l1.dtype)
            + (dist_l1 < 1.0).type(dist_l1.dtype)
            + (dist_l1 < 2.0).type(dist_l1.dtype)
            + (dist_l1 < 4.0).type(dist_l1.dtype)
        )
        score = score * 0.25

        # normalize
        norm = 1.0 / (eps + torch.sum(dists_to_score, dim=(-1, -2)))
        score = norm * (torch.sum(dists_to_score * score, dim=(-1, -2)))

    return score


def fnat(contacts_gt: torch.Tensor, contacts_pred: torch.Tensor) -> torch.Tensor:
    """Calculates the fraction of GT contacts reproduced in the prediction.

    Args:
         contacts_gt (torch.Tensor):
              Contacts in the ground truth structure [*, N, N].
         contacts_pred (torch.Tensor):
              Contacts in the predicted structure [*, N, N].

    Returns:
         torch.Tensor:
                Fraction of native contacts reproduced [*].
    """
    contacts_nat = contacts_gt.sum(dim=(-2, -1))
    contacts_nat_recovered = (contacts_gt & contacts_pred).sum(dim=(-2, -1))
    fnat = contacts_nat_recovered.to(torch.float32) / torch.clamp(contacts_nat, min=1.0)
    return fnat


@dataclass
class DockQResult:
    """DockQ results class.

    Attributes:
        chain_pair_to_dockq:
            Chain ID pairs to DockQ score [*].
        chain_pair_to_moltype:
            Chain ID pairs to molecule types.
        chain_pair_to_n_contacts:
            Chain ID pairs to number of residue pairs in contact at the FNAT threshold.
        chain_pair_to_n_if_res:
            Chain ID pairs to number of contacting residues at the FNAT threshold.
    """

    chain_pair_to_dockq: dict = field(default_factory=dict)
    chain_pair_to_moltype: dict = field(default_factory=dict)
    chain_pair_to_n_contacts: dict = field(default_factory=dict)
    chain_pair_to_n_if_res: dict = field(default_factory=dict)

    def iter_pairs(
        self,
    ) -> Iterator[
        tuple[
            tuple[int, int], torch.Tensor, tuple[str, str], torch.Tensor, torch.Tensor
        ]
    ]:
        """
        Yields (chain_pair, dockq, moltype, n_contacts, n_if_res).
        """
        keys = self.chain_pair_to_dockq.keys()
        # Check that all dicts have identical keys:
        if not (
            keys
            == self.chain_pair_to_moltype.keys()
            == self.chain_pair_to_n_contacts.keys()
            == self.chain_pair_to_n_if_res.keys()
        ):
            raise ValueError("DockQResult keys don't match.")

        for cp in keys:
            yield (
                cp,
                self.chain_pair_to_dockq[cp],
                self.chain_pair_to_moltype[cp],
                self.chain_pair_to_n_contacts[cp],
                self.chain_pair_to_n_if_res[cp],
            )


def dockq(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id_atomized: torch.Tensor,
    res_id_atomized: torch.Tensor,
    ref_atom_name_chars_atomized: torch.Tensor,
    inter_filter_atomized: torch.Tensor,
    is_protein_atomized: torch.Tensor,
    is_rna_atomized: torch.Tensor,
    is_dna_atomized: torch.Tensor,
    d_fnat: float = 5.0,
    d_irmsd: float = 10.0,
    d1: float = 8.5,
    d2: float = 1.5,
) -> DockQResult:
    """Calculates per-interface DockQ scores, separately for each diffusion sample.

    Only batch size=1 implemented.

    Args:
        pred_coords (torch.Tensor):
            Predicted coordinates of the complex [B, S, N_atom, 3].
        gt_coords (torch.Tensor):
            Ground truth coordinates of the complex [B, S, N_atom, 3].
        all_atom_mask (torch.Tensor):
            Atom mask for unresolved atoms [B, S, N_atom].
        asym_id_atomized (torch.Tensor):
            Per-atom asym IDs [B, S, N_atom].
        res_id_atomized (torch.Tensor):
            Per-atom residue IDs [B, S, N_atom].
        ref_atom_name_chars_atomized (torch.Tensor):
            Per-atom atom names in the AF3-stype encoding [B, S, N_atom, 4, 64].
        inter_filter_atomized (torch.Tensor):
            Precomputed mask for which atom pairs to consider in the calculation [B, S,
            N_atom, N_atom].
        is_protein_atomized (torch.Tensor):
            Per-atom protein mask [B, S, N_atom].
        is_rna_atomized (torch.Tensor):
            Per-atom RNA mask [B, S, N_atom].
        is_dna_atomized (torch.Tensor):
            Per-atom DNA mask [B, S, N_atom].
        d_fnat (float, optional):
            Contact threshold for FNAT calculation. Defaults to 5.0.
        d_irmsd (float, optional):
            Contact threshold for iRMSD calculation. Defaults to 10.0.
        d1 (float, optional):
            DockQ d1 parameter. Defaults to 8.5.
        d2 (float, optional):
            DockQ d2 parameter. Defaults to 1.5.

    Raises:
        NotImplementedError:
            If batch size is > 1.

    Returns:
        DockQResult:
            Per-interface and per-diffusion-sample DockQ scores and associated metadata
            [B].
    """
    bs = asym_id_atomized.shape[:-1]
    n = asym_id_atomized.shape[-1:]
    dockq_result = DockQResult()
    moltypes = ["protein", "rna", "dna"]
    chain_id_to_moltype = {}

    is_polymer_atomized = is_protein_atomized + is_rna_atomized + is_dna_atomized
    chain_id_polymer = torch.unique(
        asym_id_atomized[all_atom_mask.bool() & is_polymer_atomized.bool()]
    ).to(torch.int32)

    for moltype_mask_atomized, moltype in zip(
        [is_protein_atomized, is_rna_atomized, is_dna_atomized], moltypes
    ):
        asym_id_moltype_atomized = asym_id_atomized[all_atom_mask.bool()][
            moltype_mask_atomized[all_atom_mask.bool()]
        ]
        if torch.any(asym_id_moltype_atomized):
            chain_ids = torch.unique(asym_id_moltype_atomized).to(torch.int32)
            for chain_id in chain_ids:
                chain_id_to_moltype[chain_id.item()] = moltype

    is_backbone = _get_atom_name_mask(ref_atom_name_chars_atomized, BACKBONE_ATOMS)
    if inter_filter_atomized is None:
        inter_filter_atomized = torch.ones(
            bs + n + n, dtype=torch.bool, device=asym_id_atomized.device
        )

    if (len(bs) > 0) and (bs[0] > 1):
        raise NotImplementedError("DockQ for batch size > 1 not implemented.")
    for ci, cj in combinations(chain_id_polymer, 2):
        for sample_idx in range(bs[-1]):
            # Chain masks
            is_ci = (asym_id_atomized[..., sample_idx, :] == ci) & all_atom_mask[
                ..., sample_idx, :
            ].bool()
            is_cj = (asym_id_atomized[..., sample_idx, :] == cj) & all_atom_mask[
                ..., sample_idx, :
            ].bool()

            ni_value = torch.unique(torch.sum(is_ci, axis=-1))
            nj_value = torch.unique(torch.sum(is_cj, axis=-1))
            # Skip if no valid atoms or inconsistent number of atoms across samples for
            # the same chain
            if torch.all(ni_value == 0) or torch.all(nj_value == 0):
                continue

            chain_id_pair = (ci.item(), cj.item(), sample_idx)

            # batch size > 1 not supported, so can just take the first sample's chain
            # mask
            atom_id_i = torch.where(is_ci[0])[0]
            atom_id_j = torch.where(is_cj[0])[0]

            inter_filter_atomized_ij = (
                inter_filter_atomized[..., sample_idx, :, :]
                .index_select(-2, atom_id_i)
                .index_select(-1, atom_id_j)
            )

            # Skip if no metrics are needed for this interface
            if (
                (inter_filter_atomized_ij.shape[-1] == 0)
                or (inter_filter_atomized_ij.shape[-2] == 0)
                or torch.all(torch.sum(inter_filter_atomized_ij, dim=[-1, -2]) == 0)
            ):
                continue

            # homo-multimers can cause some unexpected errors
            try:
                gt_coords_i = gt_coords[..., sample_idx, :, :][is_ci].view(
                    bs[:-1] + (ni_value.item(), 3)
                )
                gt_coords_j = gt_coords[..., sample_idx, :, :][is_cj].view(
                    bs[:-1] + (nj_value.item(), 3)
                )
                pred_coords_i = pred_coords[..., sample_idx, :, :][is_ci].view(
                    bs[:-1] + (ni_value.item(), 3)
                )
                pred_coords_j = pred_coords[..., sample_idx, :, :][is_cj].view(
                    bs[:-1] + (nj_value.item(), 3)
                )
                is_backbone_i = is_backbone[..., sample_idx, :][is_ci].view(
                    bs[:-1] + (ni_value.item(),)
                )
                is_backbone_j = is_backbone[..., sample_idx, :][is_cj].view(
                    bs[:-1] + (nj_value.item(),)
                )
            except RuntimeError as e:
                print(
                    f"DockQ Error: Failed to extract per-chain features for chains {ci}"
                    f" and {cj}:\n{e}"
                )
                continue

            d_gt_ij_squared = torch.sum(
                (gt_coords_i.unsqueeze(-2) - gt_coords_j.unsqueeze(-3)) ** 2, dim=-1
            )
            contacts_gt_d_fnat_atomized = (
                d_gt_ij_squared < d_fnat**2
            ) & inter_filter_atomized_ij

            # Skip if no valid contacts
            if not torch.any(contacts_gt_d_fnat_atomized):
                continue

            d_pred_ij_squared = torch.sum(
                (pred_coords_i.unsqueeze(-2) - pred_coords_j.unsqueeze(-3)) ** 2, dim=-1
            )
            contacts_pred_d_fnat_atomized = (
                d_pred_ij_squared < d_fnat**2
            ) & inter_filter_atomized_ij

            res_id_i = res_id_atomized[..., sample_idx, :][is_ci].view(bs[:-1] + (-1,))
            res_id_j = res_id_atomized[..., sample_idx, :][is_cj].view(bs[:-1] + (-1,))

            # FNAT
            # Aggregate atom-wise to residue-wise contacts [*, ni, nj] -> [* ri, rj]
            contacts_gt_d_fnat_res = aggregate_atom_feat_to_tokens_nd(
                atom_feat=contacts_gt_d_fnat_atomized,
                atom_dims=[-2, -1],
                atom_to_token_index_list=[res_id_i, res_id_j],
                aggregate_fn="any",
            )
            contacts_pred_d_fnat_res = aggregate_atom_feat_to_tokens_nd(
                atom_feat=contacts_pred_d_fnat_atomized,
                atom_dims=[-2, -1],
                atom_to_token_index_list=[res_id_i, res_id_j],
                aggregate_fn="any",
            )

            contacts_nat = contacts_gt_d_fnat_res.sum(dim=(-2, -1))
            contacts_nat_recovered = (
                contacts_gt_d_fnat_res & contacts_pred_d_fnat_res
            ).sum(dim=(-2, -1))
            fnat = contacts_nat_recovered.to(torch.float32) / torch.clamp(
                contacts_nat, min=1.0
            )

            # Add contact data
            dockq_result.chain_pair_to_n_contacts[chain_id_pair] = torch.sum(
                contacts_gt_d_fnat_res, dim=[-1, -2]
            )
            dockq_result.chain_pair_to_n_if_res[chain_id_pair] = torch.sum(
                torch.sum(contacts_gt_d_fnat_res, dim=-1) > 0, dim=-1
            ) + torch.sum(torch.sum(contacts_gt_d_fnat_res, dim=-2) > 0, dim=-1)

            del [
                contacts_gt_d_fnat_atomized,
                contacts_pred_d_fnat_atomized,
                contacts_gt_d_fnat_res,
                contacts_pred_d_fnat_res,
            ]

            # lRMSD
            rec_idx = 0 if ni_value.item() >= nj_value.item() else 1
            lig_idx = 1 - rec_idx
            gt_coords_rec, gt_coords_lig = (
                (gt_coords_i, gt_coords_j)[rec_idx],
                (gt_coords_i, gt_coords_j)[lig_idx],
            )
            pred_coords_rec, pred_coords_lig = (
                (pred_coords_i, pred_coords_j)[rec_idx],
                (pred_coords_i, pred_coords_j)[lig_idx],
            )
            is_backbone_rc, is_backbone_lig = (
                (is_backbone_i, is_backbone_j)[rec_idx],
                (is_backbone_i, is_backbone_j)[lig_idx],
            )

            gt_coords_rec_bb = gt_coords_rec[is_backbone_rc].view(bs[:-1] + (-1, 3))
            gt_coords_lig_bb = gt_coords_lig[is_backbone_lig].view(bs[:-1] + (-1, 3))
            pred_coords_rec_bb = pred_coords_rec[is_backbone_rc].view(bs[:-1] + (-1, 3))
            pred_coords_lig_bb = pred_coords_lig[is_backbone_lig].view(
                bs[:-1] + (-1, 3)
            )

            rec_transform = get_optimal_transformation(
                mobile_positions=pred_coords_rec_bb,
                target_positions=gt_coords_rec_bb,
                positions_mask=torch.ones(
                    pred_coords_rec_bb.shape[:-1], device=pred_coords_rec_bb.device
                ),
            )
            pred_coords_lig_bb_transformed = apply_transformation(
                pred_coords_lig_bb, rec_transform
            )

            lrmsd_score = torch.sqrt(
                torch.mean(
                    torch.sum(
                        (gt_coords_lig_bb - pred_coords_lig_bb_transformed) ** 2,
                        dim=-1,
                    ),
                    dim=-1,
                )
            )

            # iRMSD
            # Get all atoms that belong to an interface residue at d_irmsd
            contacts_gt_d_irmsd_atomized = (
                d_gt_ij_squared < d_irmsd**2
            ) & inter_filter_atomized_ij
            is_contact_gt_d_irmsd_atomized_i = (
                torch.sum(contacts_gt_d_irmsd_atomized, axis=-1) > 0
            )
            is_contact_gt_d_irmsd_atomized_j = (
                torch.sum(contacts_gt_d_irmsd_atomized, axis=-2) > 0
            )
            is_contact_gt_d_irmsd_atomized_i = _spread_contacts(
                is_contact_atom=is_contact_gt_d_irmsd_atomized_i,
                atom_to_res_id=res_id_i,
            )
            is_contact_gt_d_irmsd_atomized_j = _spread_contacts(
                is_contact_atom=is_contact_gt_d_irmsd_atomized_j,
                atom_to_res_id=res_id_j,
            )
            # Get backbone coordinates of interface residues
            is_contact_bb_i = is_backbone_i & is_contact_gt_d_irmsd_atomized_i
            is_contact_bb_j = is_backbone_j & is_contact_gt_d_irmsd_atomized_j

            # Skip if no interface backbone atoms
            if (is_contact_bb_i.sum() == 0) or (is_contact_bb_j.sum() == 0):
                continue

            gt_coords_if_i_bb = gt_coords_i[is_contact_bb_i].view(bs[:-1] + (-1, 3))
            gt_coords_if_j_bb = gt_coords_j[is_contact_bb_j].view(bs[:-1] + (-1, 3))
            pred_coords_if_i_bb = pred_coords_i[is_contact_bb_i].view(bs[:-1] + (-1, 3))
            pred_coords_if_j_bb = pred_coords_j[is_contact_bb_j].view(bs[:-1] + (-1, 3))

            gt_coords_if_bb = torch.cat([gt_coords_if_i_bb, gt_coords_if_j_bb], dim=-2)
            pred_coords_if_bb = torch.cat(
                [pred_coords_if_i_bb, pred_coords_if_j_bb], dim=-2
            )

            if_transform = get_optimal_transformation(
                mobile_positions=pred_coords_if_bb,
                target_positions=gt_coords_if_bb,
                positions_mask=torch.ones(
                    pred_coords_if_bb.shape[:-1], device=pred_coords_if_bb.device
                ),
            )
            pred_coords_if_bb_transformed = apply_transformation(
                pred_coords_if_bb, if_transform
            )

            irmsd_score = torch.sqrt(
                torch.mean(
                    torch.sum(
                        (gt_coords_if_bb - pred_coords_if_bb_transformed) ** 2, dim=-1
                    ),
                    dim=-1,
                )
            )

            # DockQ
            dockq_score = (
                fnat
                + (1.0 / (1.0 + (lrmsd_score / d1) ** 2)).view(bs[:-1])
                + (1.0 / (1.0 + (irmsd_score / d2) ** 2)).view(bs[:-1])
            ) / 3.0

            dockq_result.chain_pair_to_moltype[chain_id_pair] = (
                chain_id_to_moltype[ci.item()],
                chain_id_to_moltype[cj.item()],
            )
            dockq_result.chain_pair_to_dockq[chain_id_pair] = dockq_score
    return dockq_result


def dockq_full_complex(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id_atomized: torch.Tensor,
    res_id_atomized: torch.Tensor,
    ref_atom_name_chars_atomized: torch.Tensor,
    inter_filter_atomized: torch.Tensor,
    is_protein_atomized: torch.Tensor,
    is_rna_atomized: torch.Tensor,
    is_dna_atomized: torch.Tensor,
    d_fnat: float = 5.0,
    d_irmsd: float = 10.0,
    d1: float = 8.5,
    d2: float = 1.5,
    weight_by: Literal["n_contacts", "n_if_res"] = "n_contacts",
    eps: float = 1e-10,
) -> dict[str, torch.Tensor]:
    """Computes whole-complex DockQ score averaged across all interfaces.

    Returns both unweighted DockQ and DockQ weighted by either number of contacts or
    number of interface residues, per molecule type pair.

    Only batch size=1 implemented.

    Note that this function averages values across diffision samples.

    Args:
        pred_coords (torch.Tensor):
            Predicted coordinates of the complex [B, S, N_atom, 3].
        gt_coords (torch.Tensor):
            Ground truth coordinates of the complex [B, S, N_atom, 3].
        all_atom_mask (torch.Tensor):
            Atom mask for unresolved atoms [B, S, N_atom].
        asym_id_atomized (torch.Tensor):
            Per-atom asym IDs [B, S, N_atom].
        res_id_atomized (torch.Tensor):
            Per-atom residue IDs [B, S,, N_atom].
        ref_atom_name_chars_atomized (torch.Tensor):
            Per-atom atom names in the AF3-stype encoding [B, S, N_atom, 4, 64].
        inter_filter_atomized (torch.Tensor):
            Precomputed mask for which atom pairs to consider in the calculation [B, S,
            N_atom, N_atom].
        is_protein_atomized (torch.Tensor):
            Per-atom protein mask [B, S, N_atom].
        is_rna_atomized (torch.Tensor):
            Per-atom RNA mask [B, S, N_atom].
        is_dna_atomized (torch.Tensor):
            Per-atom DNA mask [B, S, N_atom].
        d_fnat (float, optional):
            Contact threshold for FNAT calculation. Defaults to 5.0.
        d_irmsd (float, optional):
            Contact threshold for iRMSD calculation. Defaults to 10.0.
        d1 (float, optional):
            DockQ d1 parameter. Defaults to 8.5.
        d2 (float, optional):
            DockQ d2 parameter. Defaults to 1.5.
        weight_by (Literal['n_contacts', 'n_if_res'], optional):
            Metric to weight by. Defaults to "n_contacts".
        eps (float):
            Constant for numerical stability.

    Returns:
        dict[str, torch.Tensor]:
            Molecule type pair to weighted and unweighted DockQ scores [B, 1].
    """

    dockq_result = dockq(
        pred_coords=pred_coords,
        gt_coords=gt_coords,
        all_atom_mask=all_atom_mask,
        asym_id_atomized=asym_id_atomized,
        res_id_atomized=res_id_atomized,
        ref_atom_name_chars_atomized=ref_atom_name_chars_atomized,
        inter_filter_atomized=inter_filter_atomized,
        is_protein_atomized=is_protein_atomized,
        is_rna_atomized=is_rna_atomized,
        is_dna_atomized=is_dna_atomized,
        d_fnat=d_fnat,
        d_irmsd=d_irmsd,
        d1=d1,
        d2=d2,
    )

    out = {}
    n_sample = pred_coords.shape[-3]
    moltype_pairs = list(combinations_with_replacement(["protein", "rna", "dna"], 2))
    aggregate_items = ["dockq_scores", "n_contacts", "n_if_res"]
    aggregator = {}

    for _, dockq_scores, moltypes, n_contacts, n_if_res in dockq_result.iter_pairs():
        if moltypes not in moltype_pairs:
            moltypes = (moltypes[1], moltypes[0])
        for i, iname in zip(
            [dockq_scores, n_contacts, n_if_res], aggregate_items, strict=True
        ):
            if moltypes not in aggregator:
                aggregator[moltypes] = {k: None for k in aggregate_items}
            if aggregator[moltypes][iname] is None:
                aggregator[moltypes][iname] = i.unsqueeze(-1)
            else:
                aggregator[moltypes][iname] = torch.cat(
                    [aggregator[moltypes][iname], i.unsqueeze(-1)], dim=-1
                )

    for moltype_pair, metrics in aggregator.items():
        dockq_scores = metrics["dockq_scores"]
        dockq_scores_unweighted = torch.mean(dockq_scores, dim=-1)
        weight_metric = metrics[weight_by]
        weights = weight_metric / torch.clamp(
            torch.sum(weight_metric, dim=-1, keepdim=True), min=eps
        )
        dockq_scores_weighted = torch.sum(dockq_scores * weights, dim=-1)

        metric_name = f"dockq_{moltype_pair[0]}_{moltype_pair[1]}"

        # Expand to have shape [B, S] for model selection
        out[f"{metric_name}_uw"] = dockq_scores_unweighted.unsqueeze(-1).expand(
            -1, n_sample
        )
        out[f"{metric_name}_w"] = dockq_scores_weighted.unsqueeze(-1).expand(
            -1, n_sample
        )

    return out


def get_protein_metrics(
    is_protein_atomized: torch.Tensor,
    asym_id: torch.Tensor,
    intra_mask_atomized: torch.Tensor,
    inter_mask_atomized: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
    eps: float | None = 1e-10,
) -> dict[str, torch.Tensor]:
    """
    Compute validation metrics of protein

    Args:
        is_protein_atomized: broadcasted is_protein feature [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom]
        intra_mask_atomized:[*, n_atom] filter for intra chain computations
        inter_mask_atomized: [*, n_atom, n_atom] pairwise interaction filter
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
        eps: epsilon
    Returns:
        out: dictionary containing validation metrics
            'lddt_intra_protein': intra protein lddt
            'lddt_inter_protein_protein: inter protein-protein lddt
            'drmsd_intra_protein: intra protein drmsd
    """
    out = {}

    is_protein_atomized = is_protein_atomized.bool()

    bs = is_protein_atomized.shape[:-1]  # (bs, (n_sample),)

    gt_protein = gt_coords[is_protein_atomized].view(bs + (-1, 3))
    pred_protein = pred_coords[is_protein_atomized].view(bs + (-1, 3))
    asym_id_protein = asym_id[is_protein_atomized].view(bs + (-1,))
    all_atom_mask_protein = all_atom_mask[is_protein_atomized].view(bs + (-1,))
    intra_mask_atomized_protein = intra_mask_atomized[is_protein_atomized].view(
        bs + (-1,)
    )

    # Apply pairwise protein mask to get protein index values for inter_chain_mask
    is_protein_atomized_pair = (
        is_protein_atomized[..., None] * is_protein_atomized[..., None, :]
    )  # (1, n_protein, n_protein)

    n_protein_atoms = all_atom_mask_protein.shape[-1]

    inter_mask_atomized_protein = select_inter_filter_mask(
        inter_mask_atomized=inter_mask_atomized,
        mol_type_mask=is_protein_atomized_pair,
        out_shape=(bs[:-1] + (n_protein_atoms, n_protein_atoms)),
    )

    # (bs,(n_sample), n_prot, n_prot)
    gt_protein_pair = torch.sqrt(
        eps
        + torch.sum((gt_protein.unsqueeze(-2) - gt_protein.unsqueeze(-3)) ** 2, dim=-1)
    )
    pred_protein_pair = torch.sqrt(
        eps
        + torch.sum(
            (pred_protein.unsqueeze(-2) - pred_protein.unsqueeze(-3)) ** 2, dim=-1
        )
    )

    intra_lddt, inter_lddt = lddt(
        pair_dist_pred_pos=pred_protein_pair,
        pair_dist_gt_pos=gt_protein_pair,
        all_atom_mask=all_atom_mask_protein,
        intra_mask_filter=intra_mask_atomized_protein,
        inter_mask_filter=inter_mask_atomized_protein,
        asym_id=asym_id_protein,
        cutoff=15.0,
        eps=eps,
    )
    out["lddt_intra_protein"] = intra_lddt
    out["lddt_inter_protein_protein"] = inter_lddt

    intra_drmsd, _ = drmsd(
        pair_dist_pred_pos=pred_protein_pair,
        pair_dist_gt_pos=gt_protein_pair,
        all_atom_mask=all_atom_mask_protein,
        asym_id=asym_id_protein,
        eps=eps,
    )
    out["drmsd_intra_protein"] = intra_drmsd

    intra_clash, inter_clash = steric_clash(
        pred_pair=pred_protein_pair,
        all_atom_mask=all_atom_mask_protein,
        asym_id=asym_id_protein,
        threshold=1.1,
        eps=eps,
    )
    out["clash_intra_protein"] = intra_clash
    out["clash_inter_protein_protein"] = inter_clash

    return out


def get_ab_ag_metrics(
    intra_ab_ag_type_atomized: torch.Tensor,
    intra_ab_ag_type_atomized_filtered: torch.Tensor,
    inter_ab_ag_type_atomized_filtered: torch.Tensor,
    gt_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id_atomized: torch.Tensor,
    eps: float = 1e-10,
) -> dict[str, torch.Tensor]:
    """Compute AB/AG metrics.

    Args:
        intra_ab_ag_type_atomized (torch.Tensor):
            Per-atom int tensor of AB/AG chain types. [*, n_atom]
        intra_ab_ag_type_atomized_filtered (torch.Tensor):
            Per-atom int tensor of AB/AG chain types with the validation filter applied.
            [*, n_atom]
        inter_ab_ag_type_atomized_filtered (torch.Tensor):
            Per-atom-pair int tensor of AB/AG chain pair types with the validation
            filter applied. [*, n_atom, n_atom]
        gt_coords (torch.Tensor):
            Ground truth coordinates for the whole complex. [*, n_atom, 3]
        pred_coords (torch.Tensor):
            Predicted coordinates for the whole complex. [*, n_atom, 3]
        all_atom_mask (torch.Tensor):
            Atom level mask (typically for subsetting the calculation to atoms that are
            resolved in the GT) [*, n_atom]
        asym_id_atomized (torch.Tensor):
            Per-atom int tensor of the asym ID for the chain each atom belongs to. [*,
            n_atom]
        eps (float, optional):
            Constant for numerical stability. Defaults to 1e-10.

    Returns:
        dict[str, torch.Tensor]:
            Chain lDDT scores for heavy, light and antigen chains. Chain pair lDDT
            scores for pairwise combinations of these chain types.

    """
    out = {}

    ab_ag_type_to_chain_id = {t: i for (i, t) in enumerate(AB_AG_CHAIN_TYPES, start=1)}
    ab_ag_type_to_chain_id_pair = {
        t: i for (i, t) in enumerate(AB_AG_CHAIN_PAIR_TYPES, start=1)
    }
    pred_coords_pair = torch.sqrt(
        eps
        + torch.sum(
            (pred_coords.unsqueeze(-2) - pred_coords.unsqueeze(-3)) ** 2, dim=-1
        )
    )
    gt_coords_pair = torch.sqrt(
        eps
        + torch.sum((gt_coords.unsqueeze(-2) - gt_coords.unsqueeze(-3)) ** 2, dim=-1)
    )
    # intra chain lddts
    for t, i in ab_ag_type_to_chain_id.items():
        intra_ab_ag_mask_atomized = intra_ab_ag_type_atomized_filtered == i
        if torch.any(intra_ab_ag_mask_atomized):
            out[f"lddt_intra_{t}"], _ = lddt(
                pair_dist_pred_pos=pred_coords_pair,
                pair_dist_gt_pos=gt_coords_pair,
                all_atom_mask=all_atom_mask,
                intra_mask_filter=intra_ab_ag_mask_atomized,
                inter_mask_filter=torch.zeros_like(pred_coords_pair),
                asym_id=asym_id_atomized,
            )

    # inter chain lddts
    for t, ij in ab_ag_type_to_chain_id_pair.items():
        ti, tj = t
        inter_ab_ag_mask_atomized_ij = inter_ab_ag_type_atomized_filtered == ij
        if torch.any(inter_ab_ag_mask_atomized_ij):
            intra_ab_ag_type_atomized_i = (
                intra_ab_ag_type_atomized == ab_ag_type_to_chain_id[ti]
            )
            intra_ab_ag_type_atomized_j = (
                intra_ab_ag_type_atomized == ab_ag_type_to_chain_id[tj]
            )

            bs_i = intra_ab_ag_type_atomized_i.shape[:-1]
            bs_j = intra_ab_ag_type_atomized_j.shape[:-1]

            pred_coords_i = pred_coords[intra_ab_ag_type_atomized_i].view(
                (bs_i) + (-1, 3)
            )
            pred_coords_j = pred_coords[intra_ab_ag_type_atomized_j].view(
                (bs_j) + (-1, 3)
            )
            gt_coords_i = gt_coords[intra_ab_ag_type_atomized_i].view((bs_i) + (-1, 3))
            gt_coords_j = gt_coords[intra_ab_ag_type_atomized_j].view((bs_j) + (-1, 3))
            all_atom_mask_i = all_atom_mask[intra_ab_ag_type_atomized_i].view(
                (bs_i) + (-1,)
            )
            all_atom_mask_j = all_atom_mask[intra_ab_ag_type_atomized_j].view(
                (bs_j) + (-1,)
            )

            intra_ab_ag_type_atomized_pair = (
                intra_ab_ag_type_atomized_i[..., None]
                * intra_ab_ag_type_atomized_j[..., None, :]
            )
            inter_mask_atomized_subset = select_inter_filter_mask(
                inter_mask_atomized=inter_ab_ag_mask_atomized_ij,
                mol_type_mask=intra_ab_ag_type_atomized_pair,
                out_shape=(
                    bs_i[:-1] + (all_atom_mask_i.shape[-1], all_atom_mask_j.shape[-1])
                ),
            )

            out[f"lddt_inter_{ti}_{tj}"] = interface_lddt(
                all_atom_pred_pos_1=pred_coords_i,
                all_atom_pred_pos_2=pred_coords_j,
                all_atom_gt_pos_1=gt_coords_i,
                all_atom_gt_pos_2=gt_coords_j,
                all_atom_mask1=all_atom_mask_i,
                all_atom_mask2=all_atom_mask_j,
                filter_mask=inter_mask_atomized_subset,
            )
    return out


def get_nucleic_acid_metrics(
    is_nucleic_acid_atomized: torch.Tensor,
    asym_id: torch.Tensor,
    intra_mask_atomized: torch.Tensor,
    inter_mask_atomized: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
    is_protein_atomized: torch.Tensor,
    substrate: str,
    eps: float | None = 1e-10,
) -> dict[str, torch.Tensor]:
    """
    Compute validation metrics of nucleic acids (dna/rna)

    Args:
        is_nucleic_acid_atomized: broadcasted is_dna/rna feature [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom]
        intra_mask_atomized:[*, n_atom] filter for intra chain computations
        inter_mask_atomized: [*, n_atom, n_atom] pairwise interaction filter
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
        is_protein_atomized: broadcasted is_protein feature [*, n_atom]
        substrate: 'rna', 'dna'
        eps: epsilon
    Returns:
        out: dictionary containing validation metrics
            'lddt_intra_f'{dna/rna}': intra dna/rna lddt
            'lddt_inter_f'{dna/rna}'_f'{dna/rna}': inter dna/rna lddt
            'drmsd_intra_f'{dna/rna}': intra dna/rna drmsd
            'lddt_inter_protein_f'{dna/rna}': inter protein-dna/rna lddt
            'lddt_intra_{dna/rna}_15': intra dna/rna lddt with 15 A radius
            'lddt_inter_{dna/rna}_{dna/rna}_15': inter lddt with 15 A radius
            'lddt_inter_protein_{dna/rna}_15': inter protein-dna/rna lddt

    Notes:
        if there exists no appropriate substrate: returns an empty dict {}
        function is compatible with multiple samples,
            not compatible with batch with different number of atoms/substrates
    """
    out = {}

    is_nucleic_acid_atomized = is_nucleic_acid_atomized.bool()
    is_protein_atomized = is_protein_atomized.bool()

    bs = is_nucleic_acid_atomized.shape[:-1]  # (bs, (n_sample),)

    # getting appropriate atoms of shape (bs, (n_sample), n_na, (3)),
    gt_protein = gt_coords[is_protein_atomized].view((bs) + (-1, 3))
    gt_na = gt_coords[is_nucleic_acid_atomized].view((bs) + (-1, 3))
    pred_protein = pred_coords[is_protein_atomized].view((bs) + (-1, 3))
    pred_na = pred_coords[is_nucleic_acid_atomized].view((bs) + (-1, 3))
    asym_id_na = asym_id[is_nucleic_acid_atomized].view((bs) + (-1,))

    all_atom_mask_protein = all_atom_mask[is_protein_atomized].view((bs) + (-1,))
    all_atom_mask_na = all_atom_mask[is_nucleic_acid_atomized].view((bs) + (-1,))
    intra_mask_atomized_na = intra_mask_atomized[is_nucleic_acid_atomized].view(
        bs + (-1,)
    )

    # Apply pairwise na mask to get intra na interactions
    is_nucleic_acid_atomized_pair = (
        is_nucleic_acid_atomized[..., None] * is_nucleic_acid_atomized[..., None, :]
    )

    n_nucleic_acid_atoms = all_atom_mask_na.shape[-1]

    inter_mask_atomized_na = select_inter_filter_mask(
        inter_mask_atomized=inter_mask_atomized,
        mol_type_mask=is_nucleic_acid_atomized_pair,
        out_shape=(bs[:-1] + (n_nucleic_acid_atoms, n_nucleic_acid_atoms)),
    )

    # Apply protein x na masks to select protein - na interactions
    is_protein_na_pair = (
        is_protein_atomized[..., None] * is_nucleic_acid_atomized[..., None, :]
    )

    n_protein_atoms = all_atom_mask_protein.shape[-1]
    inter_filter_mask = select_inter_filter_mask(
        inter_mask_atomized=inter_mask_atomized,
        mol_type_mask=is_protein_na_pair,
        out_shape=(bs[:-1] + (n_protein_atoms, n_nucleic_acid_atoms)),
    )

    # (bs,(n_sample), n_na, n_na)
    gt_na_pair = torch.sqrt(
        eps + torch.sum((gt_na.unsqueeze(-2) - gt_na.unsqueeze(-3)) ** 2, dim=-1)
    )
    pred_na_pair = torch.sqrt(
        eps + torch.sum((pred_na.unsqueeze(-2) - pred_na.unsqueeze(-3)) ** 2, dim=-1)
    )

    intra_lddt, inter_lddt = lddt(
        pair_dist_pred_pos=pred_na_pair,
        pair_dist_gt_pos=gt_na_pair,
        all_atom_mask=all_atom_mask_na,
        intra_mask_filter=intra_mask_atomized_na,
        inter_mask_filter=inter_mask_atomized_na,
        asym_id=asym_id_na,
        cutoff=30.0,
        eps=eps,
    )
    out["lddt_intra_" + substrate] = intra_lddt
    out["lddt_inter_" + substrate + "_" + substrate] = inter_lddt

    intra_drmsd, _ = drmsd(
        pair_dist_pred_pos=pred_na_pair,
        pair_dist_gt_pos=gt_na_pair,
        all_atom_mask=all_atom_mask_na,
        asym_id=asym_id_na,
        eps=eps,
    )
    out["drmsd_intra_" + substrate] = intra_drmsd

    intra_lddt_15, inter_lddt_15 = lddt(
        pair_dist_pred_pos=pred_na_pair,
        pair_dist_gt_pos=gt_na_pair,
        all_atom_mask=all_atom_mask_na,
        intra_mask_filter=intra_mask_atomized_na,
        inter_mask_filter=inter_mask_atomized_na,
        asym_id=asym_id_na,
        cutoff=15.0,
        eps=eps,
    )
    out["lddt_intra_" + substrate + "_15"] = intra_lddt_15
    out["lddt_inter_" + substrate + "_" + substrate + "_15"] = inter_lddt_15

    # ilddt with protein
    inter_lddt_protein_na = interface_lddt(
        all_atom_pred_pos_1=pred_protein,
        all_atom_pred_pos_2=pred_na,
        all_atom_gt_pos_1=gt_protein,
        all_atom_gt_pos_2=gt_na,
        all_atom_mask1=all_atom_mask_protein,
        all_atom_mask2=all_atom_mask_na,
        filter_mask=inter_filter_mask,
        cutoff=30.0,
        eps=eps,
    )
    out["lddt_inter_protein_" + substrate] = inter_lddt_protein_na

    inter_lddt_protein_na_15 = interface_lddt(
        all_atom_pred_pos_1=pred_protein,
        all_atom_pred_pos_2=pred_na,
        all_atom_gt_pos_1=gt_protein,
        all_atom_gt_pos_2=gt_na,
        all_atom_mask1=all_atom_mask_protein,
        all_atom_mask2=all_atom_mask_na,
        filter_mask=inter_filter_mask,
        cutoff=15.0,
        eps=eps,
    )
    out["lddt_inter_protein_" + substrate + "_15"] = inter_lddt_protein_na_15

    intra_clash, inter_clash = steric_clash(
        pred_pair=pred_na_pair,
        all_atom_mask=all_atom_mask_na,
        asym_id=asym_id_na,
        threshold=1.1,
    )
    out["clash_intra_" + substrate] = intra_clash
    out["clash_inter_" + substrate + "_" + substrate] = inter_clash

    interface_clash = interface_steric_clash(
        pred_protein=pred_protein,
        pred_substrate=pred_na,
        all_atom_mask_protein=all_atom_mask_protein,
        all_atom_mask_substrate=all_atom_mask_na,
        threshold=1.1,
        eps=eps,
    )
    out["clash_inter_protein_" + substrate] = interface_clash

    return out


def get_ligand_metrics(
    is_ligand_atomized: torch.Tensor,
    asym_id: torch.Tensor,
    intra_mask_atomized: torch.Tensor,
    inter_mask_atomized: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
    is_protein_atomized: torch.Tensor,
    eps: float | None = 1e-10,
) -> dict[str, torch.Tensor]:
    """
    Compute validation metrics of a ligand

    Args:
        is_ligand_atomized: broadcasted is_ligand feature [*, n_atom]
        asym_id: atomized asym_id feature [*, n_atom]
        intra_mask_atomized:
        inter_mask_atomized:
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
        is_protein_atomized: broadcasted is_protein feature [*, n_atom]
        eps: epsilon
    Returns:
        out: dictionary containing validation metrics
            'lddt_intra_ligand: intra ligand lddt
            'lddt_inter_ligand_ligand: inter ligand-ligand lddt
            'lddt_inter_protein_ligand': inter protein-ligand lddt
            'drmsd_intra_ligand': intra ligand drmsd

            'lddt_intra_ligand_uha': intra ligand lddt with [0.25, 0.5, 0.75, 1.]
            'lddt_inter_ligand_ligand_uha': inter ligand lddt with above threshold

    Notes:
        If there exists no appropriate substrate: returns an empty dict {}

        Function is compatible with multiple samples, not compatible with batch with
        different number of atoms/substrates
    """
    out = {}

    is_ligand_atomized = is_ligand_atomized.bool()
    is_protein_atomized = is_protein_atomized.bool()

    bs = is_ligand_atomized.shape[:-1]  # (bs, (n_sample),)

    # getting appropriate atoms of shape (bs, (n_sample), n_protein/ligand, (3)),
    gt_protein = gt_coords[is_protein_atomized].view((bs) + (-1, 3))
    gt_ligand = gt_coords[is_ligand_atomized].view((bs) + (-1, 3))
    pred_protein = pred_coords[is_protein_atomized].view((bs) + (-1, 3))
    pred_ligand = pred_coords[is_ligand_atomized].view((bs) + (-1, 3))
    asym_id_ligand = asym_id[is_ligand_atomized].view((bs) + (-1,))

    all_atom_mask_protein = all_atom_mask[is_protein_atomized].view((bs) + (-1,))
    all_atom_mask_ligand = all_atom_mask[is_ligand_atomized].view((bs) + (-1,))
    intra_mask_atomized_ligand = intra_mask_atomized[is_ligand_atomized].view(
        (bs) + (-1,)
    )

    # Apply pairwise na mask to get intra na interactions
    is_ligand_atomized_pair = (
        is_ligand_atomized[..., None] * is_ligand_atomized[..., None, :]
    )

    n_ligand_atoms = all_atom_mask_ligand.shape[-1]

    inter_mask_atomized_ligand = select_inter_filter_mask(
        inter_mask_atomized=inter_mask_atomized,
        mol_type_mask=is_ligand_atomized_pair,
        out_shape=(bs[:-1] + (n_ligand_atoms, n_ligand_atoms)),
    )

    # Apply protein x na masks to select protein - na interactions
    is_protein_ligand_pair = (
        is_protein_atomized[..., None] * is_ligand_atomized[..., None, :]
    )

    n_protein_atoms = all_atom_mask_protein.shape[-1]
    inter_filter_mask = select_inter_filter_mask(
        inter_mask_atomized=inter_mask_atomized,
        mol_type_mask=is_protein_ligand_pair,
        out_shape=(bs[:-1] + (n_protein_atoms, n_ligand_atoms)),
    )

    # (bs,(n_sample), n_lig, n_lig)
    gt_ligand_pair = torch.sqrt(
        eps
        + torch.sum((gt_ligand.unsqueeze(-2) - gt_ligand.unsqueeze(-3)) ** 2, dim=-1)
    )
    pred_ligand_pair = torch.sqrt(
        eps
        + torch.sum(
            (pred_ligand.unsqueeze(-2) - pred_ligand.unsqueeze(-3)) ** 2, dim=-1
        )
    )

    intra_lddt, inter_lddt = lddt(
        pair_dist_pred_pos=pred_ligand_pair,
        pair_dist_gt_pos=gt_ligand_pair,
        all_atom_mask=all_atom_mask_ligand,
        intra_mask_filter=intra_mask_atomized_ligand,
        inter_mask_filter=inter_mask_atomized_ligand,
        asym_id=asym_id_ligand,
        cutoff=15.0,
        eps=eps,
    )
    out["lddt_intra_ligand"] = intra_lddt
    out["lddt_inter_ligand_ligand"] = inter_lddt

    # get tighter threshold lddts
    intra_lddt_uha, inter_lddt_uha = lddt(
        pair_dist_pred_pos=pred_ligand_pair,
        pair_dist_gt_pos=gt_ligand_pair,
        all_atom_mask=all_atom_mask_ligand,
        intra_mask_filter=intra_mask_atomized_ligand,
        inter_mask_filter=inter_mask_atomized_ligand,
        asym_id=asym_id_ligand,
        threshold=[0.25, 0.5, 0.75, 1.0],
        cutoff=15.0,
        eps=eps,
    )
    out["lddt_intra_ligand_uha"] = intra_lddt_uha
    out["lddt_inter_ligand_ligand_uha"] = inter_lddt_uha

    # ilddt with protein
    inter_lddt_protein_ligand = interface_lddt(
        all_atom_pred_pos_1=pred_protein,
        all_atom_pred_pos_2=pred_ligand,
        all_atom_gt_pos_1=gt_protein,
        all_atom_gt_pos_2=gt_ligand,
        all_atom_mask1=all_atom_mask_protein,
        all_atom_mask2=all_atom_mask_ligand,
        filter_mask=inter_filter_mask,
        cutoff=15.0,
        eps=eps,
    )
    out["lddt_inter_protein_ligand"] = inter_lddt_protein_ligand

    intra_drmsd, _ = drmsd(
        pair_dist_pred_pos=pred_ligand_pair,
        pair_dist_gt_pos=gt_ligand_pair,
        all_atom_mask=all_atom_mask_ligand,
        asym_id=asym_id_ligand,
        eps=eps,
    )
    out["drmsd_intra_ligand"] = intra_drmsd

    intra_clash, inter_clash = steric_clash(
        pred_pair=pred_ligand_pair,
        all_atom_mask=all_atom_mask_ligand,
        asym_id=asym_id_ligand,
        threshold=1.1,
        eps=eps,
    )
    out["clash_intra_ligand"] = intra_clash
    out["clash_inter_ligand_ligand"] = inter_clash

    interface_clash = interface_steric_clash(
        pred_protein=pred_protein,
        pred_substrate=pred_ligand,
        all_atom_mask_protein=all_atom_mask_protein,
        all_atom_mask_substrate=all_atom_mask_ligand,
        threshold=1.1,
        eps=eps,
    )
    out["clash_inter_protein_ligand"] = interface_clash

    return out


def steric_clash(
    pred_pair: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id: torch.Tensor,
    threshold: float | None = 1.1,
    eps: float | None = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes steric clash score

    Args:
        pred_pair: pairwise distance of predicted positions [*, n_atom, n_atom]
        all_atom_mask: atom mask [*, n_atom]
        asym_id: asym id [*, n_atom]
        threshold: threshold to define if there is steric clash
            Based on AF3 (SI 5.9.), define threshold as 1.1 Angstrom
            By no means perfect, a good threshold to capture any heavy atoms clashes
        eps: epsilon
    Returns:
        intra_clash_score: steric clash for atoms with same asym_id (intra-chain)
        inter_clash_score: steric clash for atoms with different asym_id (inter-chain)

    Note:
        clash_scores in range (0, 1) s.t.
            0 (no atom pair having distance less than threshold) to
            1 (all atoms having same coordinate)
    """
    # Create mask
    n_atom = pred_pair.shape[-2]
    mask = (1 - torch.eye(n_atom).to(all_atom_mask.device)) * (
        all_atom_mask.unsqueeze(-1) * all_atom_mask.unsqueeze(-2)
    )

    intra = torch.where(asym_id[..., None] == asym_id[..., None, :], 1, 0).bool()
    inter = ~intra

    # Compute the clash
    clash = torch.relu(threshold - pred_pair)

    intra_mask = mask * intra
    intra_clash = None
    if torch.any(intra_mask):
        intra_clash = torch.sum(clash * intra_mask, dim=(-1, -2)) / torch.sum(
            intra_mask, dim=(-1, -2)
        )
        intra_clash = intra_clash / threshold

    inter_mask = mask * inter
    inter_clash = None
    if torch.any(inter_mask):
        inter_clash = torch.sum(clash * inter_mask, dim=(-1, -2)) / (
            torch.sum(inter_mask, dim=(-1, -2)) + eps
        )
        inter_clash = inter_clash / threshold

    return intra_clash, inter_clash


def interface_steric_clash(
    pred_protein: torch.Tensor,
    pred_substrate: torch.Tensor,
    all_atom_mask_protein: torch.Tensor,
    all_atom_mask_substrate: torch.Tensor,
    threshold: float | None = 1.1,
    eps: float | None = 1e-10,
) -> torch.Tensor:
    """
    Computes steric clash score across protein and substrate

    Args:
        pred_protein: predicted protein coordinates [*, n_protein, 3]
        pred_substrate: predicted substrate coordinates [*, n_substrate, 3]
        all_atom_mask_protein: protein atom mask
        all_atom_mask_substrate: substrate atom mask
        threshold: threshold definiing if two atoms have any steric clash
        eps: epsilon
    Returns:
        interface_clash: clash between protein and substrate interface

    Note:
        interface_clash score in range (0, 1) s.t.
            0 (no atom pair having distance less than threshold) to
            1 (all atoms having same coordinate)
    """
    # pair distance
    pair_dist = torch.sqrt(
        eps
        + torch.sum(
            (pred_protein.unsqueeze(-2) - pred_substrate.unsqueeze(-3)) ** 2, dim=-1
        )
    )

    clash = torch.relu(threshold - pair_dist)
    mask = all_atom_mask_protein.unsqueeze(-1) * all_atom_mask_substrate.unsqueeze(-2)

    interface_clash = None
    if torch.any(mask):
        interface_clash = torch.sum(clash * mask, dim=(-1, -2)) / (
            torch.sum(mask, dim=(-1, -2)) + eps
        )
        interface_clash = interface_clash / threshold

    return interface_clash


def get_superimpose_metrics(
    all_atom_pred_pos: torch.Tensor,
    all_atom_gt_pos: torch.Tensor,
    all_atom_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Computes superimposition metrics

    Args:
        all_atom_pred_pos: pred coordinates [*, n_atom, 3]
        all_atom_gt_pos: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
    Returns:
        out: a dictionary containing following metrics
            superimpose_rmsd: rmsd after superimposition [*]
            gdt_ts: gdt_ts [*]
            gdt_ha: gdt_ha [*]
    """
    out = {}

    all_atom_pred_pos_aligned = kabsch_align(
        mobile_positions=all_atom_pred_pos,
        target_positions=all_atom_gt_pos,
        positions_mask=all_atom_mask,
    )

    out["rmsd"] = rmsd(
        pred_positions=all_atom_pred_pos_aligned,
        target_positions=all_atom_gt_pos,
        positions_mask=all_atom_mask,
    )

    out["gdt_ts"] = gdt_ts(
        p1=all_atom_pred_pos_aligned,
        p2=all_atom_gt_pos,
        mask=all_atom_mask,
    )

    out["gdt_ha"] = gdt_ha(
        p1=all_atom_pred_pos_aligned,
        p2=all_atom_gt_pos,
        mask=all_atom_mask,
    )

    return out


def get_full_complex_lddt(
    asym_id: torch.Tensor,
    intra_filter_atomized: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    all_atom_mask: torch.Tensor,
    eps: float | None = 1e-10,
) -> dict[str, torch.Tensor]:
    """
    Computes lddt for the full complex, subject to intra chain filters

    Args:
        asym_id: atomized asym_id feature [*, n_atom]
        intra_filter_atomized:[*, n_atom] filter for intra chain computations
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        all_atom_mask: atom mask [*, n_atom]
        eps: epsilon
    Returns:
        out: dictionary containing validation metrics
            'lddt_complex': full complex lddt score
    """
    out = {}

    # Do the whole complex lddt
    gt_pair = torch.sqrt(
        eps
        + torch.sum((gt_coords.unsqueeze(-2) - gt_coords.unsqueeze(-3)) ** 2, dim=-1)
    )
    pred_pair = torch.sqrt(
        eps
        + torch.sum(
            (pred_coords.unsqueeze(-2) - pred_coords.unsqueeze(-3)) ** 2, dim=-1
        )
    )

    # mask out all inter chain computations
    inter_filter_atomized_zeros = torch.zeros(
        (intra_filter_atomized.shape[-1], intra_filter_atomized.shape[-1])
    ).to(asym_id.device)

    complex_lddt, _ = lddt(
        pair_dist_pred_pos=pred_pair,
        pair_dist_gt_pos=gt_pair,
        all_atom_mask=all_atom_mask,
        intra_mask_filter=intra_filter_atomized,
        inter_mask_filter=inter_filter_atomized_zeros,
        asym_id=asym_id,
        eps=eps,
    )

    out["lddt_intra_complex"] = complex_lddt

    return out


def get_plddt_metrics(
    is_protein_atomized: torch.Tensor,
    is_ligand_atomized: torch.Tensor,
    is_rna_atomized: torch.Tensor,
    is_dna_atomized: torch.Tensor,
    intra_filter_atomized: torch.Tensor,
    plddt_logits: torch.Tensor,
    eps: float | None = 1e-10,
) -> dict[str, torch.Tensor]:
    """
    Compute plddt metric and report for different atom types.
    Args:
        is_protein_atomized: broadcasted is_protein feature [*, n_atom]
        is_ligand_atomized: broadcasted is_ligand feature [*, n_atom]
        is_rna_atomized: broadcasted is_rna feature [*, n_atom]
        is_dna_atomized: broadcasted is_dna feature [*, n_atom]
        intra_filter_atomized:[*, n_atom] filter for intra chain computations
        plddt_logits: [*, n_atom, 50] prediction output of lddt from model
        eps: epsilon
    Returns:
        out: dictionary containing validation metrics
            'lddt_intra_protein': intra protein lddt
            'lddt_inter_protein_protein: inter protein-protein lddt
            'drmsd_intra_protein: intra protein drmsd
    """

    out = {}

    # Report plddt scaled to 0-1
    plddt_complex = compute_plddt(plddt_logits)

    out["plddt_complex"] = torch.sum(plddt_complex * intra_filter_atomized, dim=-1) / (
        torch.sum(intra_filter_atomized, dim=-1) + eps
    )

    is_protein_atomized = is_protein_atomized * intra_filter_atomized
    is_ligand_atomized = is_ligand_atomized * intra_filter_atomized
    is_rna_atomized = is_rna_atomized * intra_filter_atomized
    is_dna_atomized = is_dna_atomized * intra_filter_atomized

    if torch.any(is_protein_atomized):
        plddt_logits_protein = plddt_complex * is_protein_atomized
        out["plddt_protein"] = torch.sum(plddt_logits_protein, dim=-1) / (
            torch.sum(is_protein_atomized, dim=-1) + eps
        )

    if torch.any(is_ligand_atomized):
        plddt_logits_ligand = plddt_complex * is_ligand_atomized
        out["plddt_ligand"] = torch.sum(plddt_logits_ligand, dim=-1) / (
            torch.sum(is_ligand_atomized, dim=-1) + eps
        )

    if torch.any(is_rna_atomized):
        plddt_logits_rna = plddt_complex * is_rna_atomized
        out["plddt_rna"] = torch.sum(plddt_logits_rna, dim=-1) / (
            torch.sum(is_rna_atomized, dim=-1) + eps
        )

    if torch.any(is_dna_atomized):
        plddt_logits_dna = plddt_complex * is_dna_atomized
        out["plddt_dna"] = torch.sum(plddt_logits_dna, dim=-1) / (
            torch.sum(is_dna_atomized, dim=-1) + eps
        )

    return out


def get_validation_lddt_metrics(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    is_ligand_atomized: torch.Tensor,
    is_rna_atomized: torch.Tensor,
    is_dna_atomized: torch.Tensor,
    is_modified_residue_atomized: torch.Tensor,
    all_atom_mask: torch.Tensor,
    asym_id_atomized: torch.Tensor,
    intra_filter_atomized: torch.Tensor,
    inter_filter_atomized: torch.Tensor,
    eps: float | None = 1e-10,
):
    """Compute lddt metrics for ligand-RNA, ligand-DNA and modified residues.
    These extra metrics are required for model selection metric.

    Args:
        pred_coords: predicted coordinates [*, n_atom, 3]
        gt_coords: gt coordinates [*, n_atom, 3]
        is_ligand_atomized: broadcasted is_ligand feature [*, n_atom]
        is_rna_atomized: broadcasted is_rna feature [*, n_atom]
        is_dna_atomized: broadcasted is_dna feature [*, n_atom]
        is_modified_residue_atomized: broadcasted is_modified_residue [*, n_atom]
        all_atom_mask: atom mask [*, n_atom]
        asym_id_atomized: atomized asym_id feature [*, n_atom]
        intra_filter_atomized:[*, n_atom] filter for intra chain computations
        inter_filter_atomized: [*, n_atom, n_atom] pairwise interaction filter
        eps: epsilon
    Returns:
        out: dictionary containing validation metrics, if applicable
            'lddt_inter_ligand_dna': inter ligand dna lddt
            'lddt_inter_ligand_rna': inter ligand rna lddt
            'lddt_intra_modified_residue': intra modified residue lddt

    Notes:
        if there exists no appropriate substrate: returns an empty dict {}
        function is compatible with multiple samples,
            not compatible with batch with different number of atoms/substrates

    """
    metrics = {}
    bs = is_ligand_atomized.shape[:-1]  # (bs, (n_sample),)

    if torch.any(is_ligand_atomized) and torch.any(is_rna_atomized):
        is_rna_ligand_pair = (
            is_rna_atomized[..., None] * is_ligand_atomized[..., None, :]
        )

        n_rna_atoms = torch.max(torch.sum(is_rna_atomized, dim=-1))
        n_ligand_atoms = torch.max(torch.sum(is_ligand_atomized, dim=-1))

        inter_filter_mask_rna_ligand = select_inter_filter_mask(
            inter_mask_atomized=inter_filter_atomized,
            mol_type_mask=is_rna_ligand_pair,
            out_shape=(bs[:-1] + (n_rna_atoms, n_ligand_atoms)),
        )

        lddt_inter_ligand_rna = interface_lddt(
            all_atom_pred_pos_1=pred_coords[is_rna_atomized].view(bs + (-1, 3)),
            all_atom_pred_pos_2=pred_coords[is_ligand_atomized].view(bs + (-1, 3)),
            all_atom_gt_pos_1=gt_coords[is_rna_atomized].view(bs + (-1, 3)),
            all_atom_gt_pos_2=gt_coords[is_ligand_atomized].view(bs + (-1, 3)),
            all_atom_mask1=all_atom_mask[is_rna_atomized].view(bs + (-1,)),
            all_atom_mask2=all_atom_mask[is_ligand_atomized].view(bs + (-1,)),
            filter_mask=inter_filter_mask_rna_ligand,
            cutoff=30.0,
            eps=eps,
        )
        metrics.update({"lddt_inter_ligand_rna": lddt_inter_ligand_rna})

    if torch.any(is_ligand_atomized) and torch.any(is_dna_atomized):
        is_dna_ligand_pair = (
            is_dna_atomized[..., None] * is_ligand_atomized[..., None, :]
        )

        n_dna_atoms = torch.max(torch.sum(is_dna_atomized, dim=-1))
        n_ligand_atoms = torch.max(torch.sum(is_ligand_atomized, dim=-1))
        inter_filter_mask_dna_ligand = select_inter_filter_mask(
            inter_mask_atomized=inter_filter_atomized,
            mol_type_mask=is_dna_ligand_pair,
            out_shape=(bs[:-1] + (n_dna_atoms, n_ligand_atoms)),
        )

        lddt_inter_ligand_dna = interface_lddt(
            all_atom_pred_pos_1=pred_coords[is_dna_atomized].view(bs + (-1, 3)),
            all_atom_pred_pos_2=pred_coords[is_ligand_atomized].view(bs + (-1, 3)),
            all_atom_gt_pos_1=gt_coords[is_dna_atomized].view(bs + (-1, 3)),
            all_atom_gt_pos_2=gt_coords[is_ligand_atomized].view(bs + (-1, 3)),
            all_atom_mask1=all_atom_mask[is_dna_atomized].view(bs + (-1,)),
            all_atom_mask2=all_atom_mask[is_ligand_atomized].view(bs + (-1,)),
            filter_mask=inter_filter_mask_dna_ligand,
            cutoff=30.0,
            eps=eps,
        )

        metrics["lddt_inter_ligand_dna"] = lddt_inter_ligand_dna

    if torch.any(is_modified_residue_atomized):
        pred_mr = pred_coords[is_modified_residue_atomized].view(bs + (-1, 3))
        gt_mr = gt_coords[is_modified_residue_atomized].view(bs + (-1, 3))

        intra_mask_atomized_mr = intra_filter_atomized[
            is_modified_residue_atomized
        ].view(bs + (-1,))

        is_mr_atomized_pair = (
            is_modified_residue_atomized[..., None]
            * is_modified_residue_atomized[..., None, :]
        )

        n_mr_atoms = torch.max(torch.sum(is_modified_residue_atomized, dim=-1))
        inter_mask_atomized_mr = select_inter_filter_mask(
            inter_mask_atomized=inter_filter_atomized,
            mol_type_mask=is_mr_atomized_pair,
            out_shape=(bs[:-1] + (n_mr_atoms, n_mr_atoms)),
        )

        pred_mr_pair = torch.sqrt(
            eps
            + torch.sum(
                (pred_mr.unsqueeze(-2) - pred_mr.unsqueeze(-3)) ** 2,
                dim=-1,
            )
        )

        gt_mr_pair = torch.sqrt(
            eps
            + torch.sum(
                (gt_mr.unsqueeze(-2) - gt_mr.unsqueeze(-3)) ** 2,
                dim=-1,
            )
        )

        lddt_intra_modified_residues, _ = lddt(
            pair_dist_pred_pos=pred_mr_pair,
            pair_dist_gt_pos=gt_mr_pair,
            all_atom_mask=all_atom_mask[is_modified_residue_atomized].view(bs + (-1,)),
            intra_mask_filter=intra_mask_atomized_mr,
            inter_mask_filter=inter_mask_atomized_mr,
            asym_id=asym_id_atomized[is_modified_residue_atomized].view(bs + (-1,)),
            eps=eps,
        )

        metrics["lddt_intra_modified_residues"] = lddt_intra_modified_residues

    return metrics


def get_metrics(
    batch,
    outputs,
    compute_lig_diffusion_metrics=False,
    compute_extra_val_metrics=False,
) -> dict[str, torch.Tensor]:
    """
    Compute validation metrics on all substrates

    Args:
        batch: ground truth and permutation applied features
        outputs: model outputs
        compute_lig_diffusion_metrics: computes ligand metrics not only for the
            mini-rollout result, but also the diffusion training prediction. Note that
            for computational and memory efficiency reasons, only the first of the 48
            diffusion training samples is used here.
        compute_extra_val_metrics: computes extra lddt metrics needed
            for model selection
    Returns:
        metrics: A dict containing validation metrics. The presence of specific
            keys depends on the contents of the batch and the function arguments.

        **Protein Metrics:**
            'lddt_intra_protein': Intra-chain lDDT for protein.
            'lddt_inter_protein_protein': Inter-chain lDDT between proteins.
            'drmsd_intra_protein': Intra-chain dRMSD for protein.
            'clash_intra_protein': Intra-chain steric clash score for protein.
            'clash_inter_protein_protein': Inter-chain steric clash between proteins.

        **Ligand Metrics:**
            'lddt_intra_ligand': Intra-chain lDDT for ligand.
            'lddt_inter_ligand_ligand': Inter-chain lDDT between ligands.
            'lddt_inter_protein_ligand': Inter-chain lDDT between protein and ligand.
            'drmsd_intra_ligand': Intra-chain dRMSD for ligand.
            'lddt_intra_ligand_uha': Intra-ligand lDDT with tighter thresholds.
            'lddt_inter_ligand_ligand_uha': Inter-ligand lDDT with tighter thresholds.
            'clash_intra_ligand': Intra-chain steric clash for ligand.
            'clash_inter_ligand_ligand': Inter-chain clash between ligands.
            'clash_inter_protein_ligand': Inter-chain clash between protein and ligand.

        **Diffusion Metrics (`compute_lig_diffusion_metrics=True`):**
            Keys from the "Ligand Metrics" section appended with '_diffusion'.

        **Nucleic Acid Metrics (rna/dna):**
            'lddt_intra_{rna/dna}': Intra-chain lDDT.
            'lddt_inter_{rna/dna}_{rna/dna}': Inter-chain lDDT between same NA types.
            'drmsd_intra_{rna/dna}': Intra-chain dRMSD.
            'lddt_inter_protein_{rna/dna}': Inter-chain lDDT between protein and NA.
            'lddt_intra_{rna/dna}_15': Intra-chain lDDT with 15A cutoff.
            'lddt_inter_{rna/dna}_{rna/dna}_15': Inter-chain lDDT with 15A cutoff.
            'lddt_inter_protein_{rna/dna}_15':
                Inter-chain lDDT between protein and NA with 15A cutoff.
            'clash_intra_{rna/dna}': Intra-chain steric clash.
            'clash_inter_{rna/dna}_{rna/dna}': Inter-chain clash between same NA types.
            'clash_inter_protein_{rna/dna}': Inter-chain clash between protein and NA.

        **Extra Validation Metrics (`compute_extra_val_metrics=True`):**
            'lddt_intra_complex': lDDT for the entire complex (intra-chain only).
            'plddt_complex': Predicted lDDT for the complex.
            'plddt_protein': Predicted lDDT for protein atoms.
            'plddt_ligand': Predicted lDDT for ligand atoms.
            'plddt_rna': Predicted lDDT for RNA atoms.
            'plddt_dna': Predicted lDDT for DNA atoms.
            'lddt_inter_ligand_dna': Inter-chain lDDT between ligand and DNA.
            'lddt_inter_ligand_rna': Inter-chain lDDT between ligand and RNA.
            'lddt_intra_modified_residues': Intra-chain lDDT for modified residues.
            'rmsd': RMSD after Kabsch alignment.
            'gdt_ts': Global Distance Test (Total Score).
            'gdt_ha': Global Distance Test (High Accuracy).
            'rasa':
                Relative accessible surface area for proteins with unresolved residues.
            'lddt_intra_{AB-AG chain types}': Intra-chain lDDT for atoms in AB heavy
                chains, AB light chains or AG chains.
            'lddt_inter_{AB-AG chain pair types}': Inter-chain lDDT between pairwise
                combinations of AB-AG chain types.

    Note:
        if no appropriate substrates, no corresponding metrics will be included
    """
    metrics = {}

    gt_coords = batch["ground_truth"]["atom_positions"]
    pred_coords = outputs["atom_positions_predicted"]

    token_mask = batch["token_mask"]
    atom_padding_mask = batch["atom_mask"]
    num_atoms_per_token = batch["num_atoms_per_token"]
    no_samples = pred_coords.shape[1]
    # getting rid of modified residues
    is_protein = batch["is_protein"]
    is_rna = batch["is_rna"]
    is_dna = batch["is_dna"]
    not_modified_res = 1 - batch["is_atomized"]
    is_protein = is_protein * not_modified_res
    is_rna = is_rna * not_modified_res
    is_dna = is_dna * not_modified_res

    def expand_sample_dim(t: torch.tensor) -> torch.tensor:
        if t.shape[1] == no_samples:
            return t

        feat_dims = t.shape[2:]
        t = t.expand(-1, no_samples, *((-1,) * len(feat_dims)))
        return t

    all_atom_mask = expand_sample_dim(
        batch["ground_truth"]["atom_resolved_mask"]
    ).bool()

    # broadcast token level features to atom level features
    is_protein_atomized = expand_sample_dim(
        broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=is_protein,
        )
    ).bool()

    is_ligand_atomized = expand_sample_dim(
        broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=batch["is_ligand"],
        )
    ).bool()

    is_rna_atomized = expand_sample_dim(
        broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=is_rna,
        )
    ).bool()

    is_dna_atomized = expand_sample_dim(
        broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=is_dna,
        )
    ).bool()

    is_modified_residue = batch["is_atomized"]
    is_modified_residue = is_modified_residue * (1 - batch["is_ligand"])
    is_modified_residue_atomized = expand_sample_dim(
        broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=is_modified_residue,
        )
    ).bool()

    asym_id_atomized = expand_sample_dim(
        broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=batch["asym_id"],
        )
    )

    # set up filters for validation metrics if present, otherwise pass ones
    intra_mask_base = expand_sample_dim(atom_padding_mask)
    inter_mask_base = expand_sample_dim(
        atom_padding_mask[..., None] * atom_padding_mask[..., None, :]
    )
    intra_filter_atomized = batch["ground_truth"].get(
        "intra_filter_atomized", intra_mask_base
    )
    inter_filter_atomized = batch["ground_truth"].get(
        "inter_filter_atomized",
        inter_mask_base,
    )

    # Set up AB/AG filters for validation
    intra_ab_ag_type_atomized = batch["ground_truth"].get("intra_ab_ag_type_atomized")
    if intra_ab_ag_type_atomized is not None:
        intra_ab_ag_type_atomized = expand_sample_dim(intra_ab_ag_type_atomized)
    else:
        intra_ab_ag_type_atomized = torch.zeros_like(intra_mask_base)
    intra_ab_ag_type_atomized_filtered = (
        intra_ab_ag_type_atomized * intra_filter_atomized.to(torch.int32)
    )
    inter_ab_ag_type_atomized = batch["ground_truth"].get("inter_ab_ag_type_atomized")
    if inter_ab_ag_type_atomized is not None:
        inter_ab_ag_type_atomized = expand_sample_dim(inter_ab_ag_type_atomized)
    else:
        inter_ab_ag_type_atomized = torch.zeros_like(inter_mask_base)
    inter_ab_ag_type_atomized_filtered = (
        inter_ab_ag_type_atomized * inter_filter_atomized.to(torch.int32)
    )

    if torch.any(is_protein_atomized):
        protein_validation_metrics = get_protein_metrics(
            is_protein_atomized=is_protein_atomized,
            asym_id=asym_id_atomized,
            intra_mask_atomized=intra_filter_atomized,
            inter_mask_atomized=inter_filter_atomized,
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            all_atom_mask=all_atom_mask,
        )
        metrics = metrics | protein_validation_metrics

    if torch.any(is_ligand_atomized):
        ligand_validation_metrics = get_ligand_metrics(
            is_ligand_atomized=is_ligand_atomized,
            asym_id=asym_id_atomized,
            intra_mask_atomized=intra_filter_atomized,
            inter_mask_atomized=inter_filter_atomized,
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            all_atom_mask=all_atom_mask,
            is_protein_atomized=is_protein_atomized,
        )
        metrics = metrics | ligand_validation_metrics

        pred_coords_diffusion = outputs.get("atom_positions_diffusion")
        if compute_lig_diffusion_metrics and pred_coords_diffusion is not None:
            # Take only first sample for computational efficiency
            pred_coords_diffusion = pred_coords_diffusion[:, 0, ...].unsqueeze(1)

            ligand_metrics_diffusion = get_ligand_metrics(
                is_ligand_atomized=is_ligand_atomized,
                asym_id=asym_id_atomized,
                intra_mask_atomized=intra_filter_atomized,
                inter_mask_atomized=inter_filter_atomized,
                pred_coords=pred_coords_diffusion,
                gt_coords=gt_coords,
                all_atom_mask=all_atom_mask,
                is_protein_atomized=is_protein_atomized,
            )
            # Rename the metrics to indicate they are from diffusion sample
            ligand_metrics_diffusion = {
                f"{k}_diffusion": v for k, v in ligand_metrics_diffusion.items()
            }
            metrics = metrics | ligand_metrics_diffusion

    if torch.any(is_rna_atomized):
        rna_validation_metrics = get_nucleic_acid_metrics(
            is_nucleic_acid_atomized=is_rna_atomized,
            asym_id=asym_id_atomized,
            intra_mask_atomized=intra_filter_atomized,
            inter_mask_atomized=inter_filter_atomized,
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            all_atom_mask=all_atom_mask,
            is_protein_atomized=is_protein_atomized,
            substrate="rna",
        )
        metrics = metrics | rna_validation_metrics

    if torch.any(is_dna_atomized):
        dna_validation_metrics = get_nucleic_acid_metrics(
            is_nucleic_acid_atomized=is_dna_atomized,
            asym_id=asym_id_atomized,
            intra_mask_atomized=intra_filter_atomized,
            inter_mask_atomized=inter_filter_atomized,
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            all_atom_mask=all_atom_mask,
            is_protein_atomized=is_protein_atomized,
            substrate="dna",
        )
        metrics = metrics | dna_validation_metrics

    if compute_extra_val_metrics:
        if torch.any(intra_filter_atomized):
            full_complex_lddt_metrics = get_full_complex_lddt(
                asym_id=asym_id_atomized,
                intra_filter_atomized=intra_filter_atomized,
                pred_coords=pred_coords,
                gt_coords=gt_coords,
                all_atom_mask=all_atom_mask,
            )
            metrics = metrics | full_complex_lddt_metrics

            plddt_logits = expand_sample_dim(outputs["plddt_logits"])
            plddt_metrics = get_plddt_metrics(
                is_protein_atomized=is_protein_atomized,
                is_ligand_atomized=is_ligand_atomized,
                is_rna_atomized=is_rna_atomized,
                is_dna_atomized=is_dna_atomized,
                intra_filter_atomized=intra_filter_atomized,
                plddt_logits=plddt_logits,
            )
            metrics = metrics | plddt_metrics

        extra_lddt = get_validation_lddt_metrics(
            pred_coords=pred_coords,
            gt_coords=gt_coords,
            is_ligand_atomized=is_ligand_atomized,
            is_rna_atomized=is_rna_atomized,
            is_dna_atomized=is_dna_atomized,
            is_modified_residue_atomized=is_modified_residue_atomized,
            all_atom_mask=all_atom_mask,
            asym_id_atomized=asym_id_atomized,
            intra_filter_atomized=intra_filter_atomized,
            inter_filter_atomized=inter_filter_atomized,
        )
        metrics = metrics | extra_lddt

        superimpose_metrics = get_superimpose_metrics(
            all_atom_pred_pos=pred_coords,
            all_atom_gt_pos=gt_coords,
            all_atom_mask=all_atom_mask,
        )
        metrics = metrics | superimpose_metrics

        # Compute RASA (Relative ASA) metric
        if torch.any(is_protein_atomized):
            rasa = compute_rasa_batch(batch, outputs)
            # RASA is only computed for proteins with unresolved residues,
            # otherwise NaN is returned
            if not torch.isnan(rasa).any():
                metrics["rasa"] = rasa

        # Compute AB/AG metrics
        ab_ag_metrics = get_ab_ag_metrics(
            intra_ab_ag_type_atomized=intra_ab_ag_type_atomized,
            intra_ab_ag_type_atomized_filtered=intra_ab_ag_type_atomized_filtered,
            inter_ab_ag_type_atomized_filtered=inter_ab_ag_type_atomized_filtered,
            gt_coords=gt_coords,
            pred_coords=pred_coords,
            all_atom_mask=all_atom_mask,
            asym_id_atomized=asym_id_atomized,
        )
        metrics = metrics | ab_ag_metrics

        # Only compute DockQ if there are at least 2 polymer chains
        if (
            len(
                torch.unique(
                    asym_id_atomized[
                        all_atom_mask.bool() & (~is_ligand_atomized.bool())
                    ]
                ).to(torch.int32)
            )
            > 1
        ):
            res_id_atomized = expand_sample_dim(
                broadcast_token_feat_to_atoms(
                    token_mask=token_mask,
                    num_atoms_per_token=num_atoms_per_token,
                    token_feat=batch["residue_index"],
                )
            )
            ref_atom_name_chars_atomized = expand_sample_dim(
                batch["ref_atom_name_chars"]
            )
            dockq_metrics = dockq_full_complex(
                pred_coords=pred_coords,
                gt_coords=gt_coords,
                all_atom_mask=all_atom_mask,
                asym_id_atomized=asym_id_atomized,
                res_id_atomized=res_id_atomized,
                ref_atom_name_chars_atomized=ref_atom_name_chars_atomized,
                inter_filter_atomized=inter_filter_atomized,
                is_protein_atomized=is_protein_atomized,
                is_rna_atomized=is_rna_atomized,
                is_dna_atomized=is_dna_atomized,
            )
            metrics = metrics | dockq_metrics

    valid_metrics = {
        name: value for name, value in metrics.items() if value is not None
    }

    return valid_metrics


def get_metrics_chunked(
    batch,
    outputs,
    compute_extra_val_metrics=False,
) -> dict[str, torch.Tensor]:
    """
    Compute validation metrics per predicted sample on all substrates.
    If a metric is valid for some samples and not others, it will be masked
    to zero as in the batched version get_metrics(). For example, for
    interface lDDts with ions, it's often the case that some samples do not
    pass the thresholds, thus making the lDDt for that sample zero.

    Args:
        batch: ground truth and permutation applied features
        outputs: model outputs
        compute_extra_val_metrics: computes extra lddt metrics needed
            for model selection
    Returns:
        metrics: dict containing validation metrics across all substrates

    Note:
        if no appropriate substrates, no corresponding metrics will be included
    """
    atom_positions_predicted = outputs["atom_positions_predicted"]
    batch_dims = atom_positions_predicted.shape[:-2]
    num_samples = batch_dims[-1]

    metrics_per_sample_list = []
    for idx in range(num_samples):

        def fetch_cur_sample(t):
            if t.shape[1] != num_samples:
                return t
            return t[:, idx : idx + 1]  # noqa: B023

        cur_batch = tensor_tree_map(fetch_cur_sample, batch, strict_type=False)
        cur_outputs = tensor_tree_map(fetch_cur_sample, outputs, strict_type=False)
        metrics_per_sample_list.append(
            get_metrics(
                cur_batch,
                cur_outputs,
                compute_extra_val_metrics=compute_extra_val_metrics,
            )
        )

    metrics_per_sample = {}
    all_metric_keys = set().union(*(m.keys() for m in metrics_per_sample_list))

    for metric_name in all_metric_keys:
        metric_values = []
        for sample in metrics_per_sample_list:
            metric_values.append(
                sample.get(
                    metric_name,
                    torch.zeros(
                        (*batch_dims[:-1], 1),
                        device=atom_positions_predicted.device,
                        dtype=atom_positions_predicted.dtype,
                    ),
                )
            )
        metrics_per_sample[metric_name] = torch.concat(metric_values, dim=1)

    return metrics_per_sample
