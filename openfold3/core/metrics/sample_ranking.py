from typing import Any, Optional

import torch

from openfold3.core.metrics.confidence import compute_ptm
from openfold3.core.metrics.rasa import compute_disorder
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
)


def _expand_sample_dim(t: torch.Tensor, no_samples: int) -> torch.Tensor:
    """
    Broadcast a tensor to have sample dimension = `no_samples` at axis 1 if needed.
    Works with token- or atom-level trailing dims.
    """
    if t.shape[1] == no_samples:
        return t
    feat_dims = t.shape[2:]
    return t.expand(-1, no_samples, *((-1,) * len(feat_dims)))


def full_complex_sample_ranking_metric(
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    has_frame: Optional[torch.Tensor] = None,
    ptm_weight: float = 0.2,
    iptm_weight: float = 0.8,
    disorder_weight: float = 0.5,
    has_clash_weight: float = -100.0,
    disorder_threshold: float = 0.581,
    **kwargs,
) -> torch.Tensor:
    """
    AlphaFold3 sample ranking metric for the full complex (SI §5.9.3, item 1).

    Computes: 0.8·ipTM + 0.2·pTM + 0.5·disorder − 100·has_clash

    Args:
        batch: model input features (post permutation alignment)
        output: model outputs

    Returns:
        sample_ranking_metric: [*, n_sample] score used for sample ranking
    """
    # inputs
    pred_pos = output[
        "atom_positions_predicted"
    ]  # [*, n_sample, n_atom, 3] or [*, n_atom, 3]
    atom_mask = batch["atom_mask"]  # [*, (n_sample), n_atom]
    token_mask = batch["token_mask"]  # [*, (n_sample), n_tok]
    asym_id = batch["asym_id"]  # [*, (n_sample), n_tok]
    is_protein = batch["is_protein"]
    is_rna = batch["is_rna"]
    is_dna = batch["is_dna"]
    n_atoms_per_tok = batch["num_atoms_per_token"]

    # normalize sample dim for masks
    no_samples = pred_pos.shape[1] if pred_pos.ndim == 4 else 1
    atom_mask = _expand_sample_dim(atom_mask, no_samples).bool()
    token_mask = _expand_sample_dim(token_mask, no_samples)
    asym_id = _expand_sample_dim(asym_id, no_samples)
    is_protein = _expand_sample_dim(is_protein, no_samples)
    is_rna = _expand_sample_dim(is_rna, no_samples)
    is_dna = _expand_sample_dim(is_dna, no_samples)

    # aggregated ipTM / pTM (Eqs. 17–18)
    iptm = compute_ptm(
        logits=output["pae_logits"],
        has_frame=has_frame,
        D_mask=token_mask,
        asym_id=asym_id,
        interface=True,
        **kwargs,
    )
    ptm = compute_ptm(
        logits=output["pae_logits"],
        has_frame=has_frame,
        D_mask=token_mask,
        asym_id=asym_id,
        interface=False,
        **kwargs,
    )

    # atomize features for clash/disorder
    is_protein_atomized = broadcast_token_feat_to_atoms(
        token_mask, n_atoms_per_tok, is_protein
    ).bool()
    is_rna_atomized = broadcast_token_feat_to_atoms(
        token_mask, n_atoms_per_tok, is_rna
    ).bool()
    is_dna_atomized = broadcast_token_feat_to_atoms(
        token_mask, n_atoms_per_tok, is_dna
    ).bool()
    asym_id_atomized = broadcast_token_feat_to_atoms(
        token_mask, n_atoms_per_tok, asym_id
    ).bool()

    is_polymer = is_protein_atomized | is_rna_atomized | is_dna_atomized
    has_clash = compute_has_clash(
        asym_id=asym_id_atomized,
        all_atom_pred_pos=pred_pos,
        atom_mask=atom_mask,
        is_polymer=is_polymer,
    )

    if torch.any(is_protein):
        disorder = compute_disorder(
            batch=batch, outputs=output, disorder_threshold=disorder_threshold
        )
    else:
        disorder = torch.zeros(
            pred_pos.shape[:-2], device=pred_pos.device, dtype=pred_pos.dtype
        )

    scores = {}
    scores["iptm"] = iptm.detach().clone()
    scores["ptm"] = ptm.detach().clone()
    scores["disorder"] = disorder
    scores["has_clash"] = has_clash
    scores["sample_ranking_score"] = (
        (
            iptm_weight * iptm
            + ptm_weight * ptm
            + disorder_weight * disorder
            - has_clash_weight * has_clash
        )
        .detach()
        .clone()
    )

    return scores


def compute_all_pTM(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    has_frame: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Compute all-pTM (OF3 §5.9.3, item 1) by considering all tokens.

    Args:
        batch: model input features
        outputs: model outputs

    Returns:
        pTM: [*, n_sample] all-pTM
    """
    token_mask = batch["token_mask"].bool()

    ptm_by_asym_id = {}
    for asym_id in batch["asym_id"].unique():
        D_mask = token_mask & (batch["asym_id"] == asym_id)
        chain_ptm = compute_ptm(
            outputs["pae_logits"],
            has_frame=has_frame,
            D_mask=D_mask,
            asym_id=batch["asym_id"],
            interface=False,
            **kwargs,
        )
        ptm_by_asym_id[asym_id.item()] = chain_ptm.detach().clone()

    return {"pTM_by_asym_id": ptm_by_asym_id}


def compute_has_clash(
    asym_id: torch.Tensor,  # [B, (S), N_atom]
    all_atom_pred_pos: torch.Tensor,  # [B, (S), N_atom, 3] or [B, N_atom, 3]
    atom_mask: torch.Tensor,  # [B, (S), N_atom]
    is_polymer: torch.Tensor,  # [B, (S), N_atom]
    threshold: float = 1.1,
    violation_abs: int = 100,
    violation_frac: float = 0.5,
) -> torch.Tensor:
    """
    Clash indicator per (batch, sample) following OF3 SI §5.9.2.

    Returns:
        has_clash: [B, S] with 1.0 if any pair of distinct chains clashes, else 0.0.
    """
    device = all_atom_pred_pos.device
    dtype = all_atom_pred_pos.dtype

    # Normalize coords to [B, S, N, 3]
    if all_atom_pred_pos.ndim == 3:  # [B, N, 3]
        all_atom_pred_pos = all_atom_pred_pos.unsqueeze(1)
    B, S, N, _ = all_atom_pred_pos.shape

    valid = atom_mask & is_polymer  # [B,S,N]
    has_clash = torch.zeros((B, S), dtype=dtype, device=device)

    # Use S=0 for chain identity (sample-invariant)
    asym_base = asym_id[:, 0, :]  # [B,N]
    valid_base = valid[:, 0, :]  # [B,N]

    for b in range(B):
        vb = valid_base[b]  # [N]
        if not torch.any(vb):
            continue

        chain_ids_b = asym_base[b][vb]  # [N_valid]
        uniq = torch.unique(chain_ids_b)
        if uniq.numel() <= 1:
            continue

        # Precompute masks within the valid slice
        chain_masks = [(chain_ids_b == cid) for cid in uniq]

        for s in range(S):
            vs = valid[b, s]  # [N]
            if not torch.any(vs):
                continue

            pos = all_atom_pred_pos[b, s][vs]  # [N_valid, 3]
            if pos.numel() == 0:
                continue

            # pairwise checks across distinct chains
            clashing = False
            for i in range(len(chain_masks)):
                for j in range(i + 1, len(chain_masks)):
                    m1, m2 = chain_masks[i], chain_masks[j]
                    n1, n2 = int(m1.sum()), int(m2.sum())
                    if n1 == 0 or n2 == 0:
                        continue

                    d = torch.cdist(pos[m1], pos[m2], p=2)  # [n1, n2]
                    vcount = int((d < threshold).sum())
                    min_len = min(n1, n2)

                    if (vcount > violation_abs) or (
                        min_len > 0 and (vcount / min_len) > violation_frac
                    ):
                        has_clash[b, s] = 1.0
                        clashing = True
                        break
                if clashing:
                    break

    return has_clash


def build_chain_interface_scores(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    has_frame: Optional[torch.Tensor] = None,
    **ptm_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """
    Compute the chain-level interface scores (ipTM) for all chain pairs in the complex 
    and the interface scores (bespoke ipTM) for each chain pair.
      Returns:
        all_ipTM_scores: {
          "iptm": {<chain_pair_key>: [B, S]},
          "bespoke_iptm": {<chain_pair_key>: [B, S]}
        }
    """
    pae_logits = outputs["pae_logits"]  # [B,S,N,N,Bins]
    B, S, N, _, _ = pae_logits.shape

    asym_3d = _expand_sample_dim(batch["asym_id"].long(), S)  # [B,S,N]
    token_3d = _expand_sample_dim(batch["token_mask"].bool(), S)  # [B,S,N]
    islig_3d = _expand_sample_dim(batch["is_ligand"].bool(), S)  # [B,S,N]
    has_frame = (
        has_frame.bool()
        if has_frame is not None
        else torch.ones_like(token_3d, dtype=torch.bool)
    )

    device, dtype = pae_logits.device, pae_logits.dtype
    # Per-batch compute (compact), then pad to C_max
    results = [
        compute_chain_interface_scores_for_batch(
            pae_logits[b],
            asym_3d[b],
            token_3d[b],
            islig_3d[b],
            has_frame[b],
            **ptm_kwargs,
        )
        for b in range(B)
    ]
    chain_lists: list[torch.Tensor] = [r[3] for r in results]
    C_max = max((int(c.numel()) for c in chain_lists), default=0)

    M = torch.full((B, S, C_max, C_max), float("nan"), dtype=dtype, device=device)
    R = torch.full((B, S, C_max), float("nan"), dtype=dtype, device=device)
    I = torch.full((B, S, C_max, C_max), float("nan"), dtype=dtype, device=device)

    for b, (M_b, R_b, I_b, chain_ids_b) in enumerate(results):
        C_b = int(chain_ids_b.numel())
        if C_b == 0:
            continue
        M[b, :, :C_b, :C_b] = M_b
        R[b, :, :C_b] = R_b
        I[b, :, :C_b, :C_b] = I_b
    # writing of the scores
    all_iptm_scores = {"iptm": {}, "bespoke_iptm": {}}
    if C_max > 1 and len(chain_lists) > 0:
        for i in range(C_max):
            for j in range(i + 1, C_max):
                try:
                    key = str((chain_lists[0][i].item(), chain_lists[0][j].item()))
                except Exception:
                    continue
                all_iptm_scores["iptm"][key] = M[:, :, i, j].detach().clone()
                all_iptm_scores["bespoke_iptm"][key] = I[:, :, i, j].detach().clone()

    return {
        "all_ipTM_scores": all_iptm_scores,
    }


def compute_chain_interface_scores_for_batch(
    pae_logits_b: torch.Tensor,  # [S, N, N, Bins]
    asym_id_b: torch.Tensor,  # [S, N]
    token_mask_b: torch.Tensor,  # [S, N]
    is_ligand_b: torch.Tensor,  # [S, N]
    has_frame_b: torch.Tensor,  # [S, N] or None
    *,
    compact_submatrix: bool = True,  # NEW: slice to A∪B for ipTM
    **ptm_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      M_b  : [S, C_b, C_b]  ipTM per chain pair (NaN on diagonal)
      R_b  : [S, C_b]       per-chain score (mean of chain pair ipTM with row gating)
      I_b  : [S, C_b, C_b]  interface score (ligand ⇒ R(ligand); else 0.5*(R(A)+R(B)))
      ids  : [C_b]          chain IDs in the used order
    """
    S, N, _, _ = pae_logits_b.shape
    device, dtype = pae_logits_b.device, pae_logits_b.dtype

    # Chain IDs from sample 0 (restricted to present tokens)
    present_s0 = token_mask_b[0]
    chain_ids = torch.unique(asym_id_b[0][present_s0])  # [C_b]
    C = int(chain_ids.numel())

    M_b = torch.full((S, C, C), float("nan"), dtype=dtype, device=device)
    R_b = torch.full((S, C), float("nan"), dtype=dtype, device=device)
    I_b = torch.full((S, C, C), float("nan"), dtype=dtype, device=device)
    if C <= 1:
        return M_b, R_b, I_b, chain_ids

    # Chain-level ligand flags (sample 0)
    lig_s0 = is_ligand_b[0]
    chain_is_ligand = torch.zeros(C, dtype=torch.bool, device=device)
    for c in range(C):
        c_mask = (asym_id_b[0] == chain_ids[c]) & present_s0
        chain_is_ligand[c] = (lig_s0 & c_mask).any()

    lig_i = chain_is_ligand.unsqueeze(1).expand(C, C)
    lig_j = chain_is_ligand.unsqueeze(0).expand(C, C)
    any_lig_pair = lig_i | lig_j
    diag_mask = torch.eye(C, dtype=torch.bool, device=device)

    # Precompute pair union indices ONCE from sample 0
    chain_masks0 = [(asym_id_b[0] == cid) & present_s0 for cid in chain_ids]
    i_ix, j_ix = torch.triu_indices(C, C, offset=1, device=device)  # [P], [P]
    P = i_ix.numel()
    pair_token_idx = []
    for k in range(P):
        i, j = i_ix[k].item(), j_ix[k].item()
        idx = torch.nonzero(chain_masks0[i] | chain_masks0[j], as_tuple=True)[
            0
        ]  # [n_pair]
        pair_token_idx.append(idx)

    # 1) Pair-by-pair ipTM (M_b), computed only on A∪B (compact submatrix)
    for s in range(S):
        logits_s = pae_logits_b[s]  # [N, N, Bins]
        asym_s = asym_id_b[s]  # [N]
        frame_s = has_frame_b[s]

        if compact_submatrix:
            for k in range(P):
                idx = pair_token_idx[k]
                if idx.numel() < 2:
                    continue
                # Slice logits/asym/has_frame to A∪B
                sub_logits = logits_s.index_select(0, idx).index_select(1, idx)
                sub_asym = asym_s.index_select(0, idx)
                sub_frame = frame_s.index_select(0, idx)

                iptm = compute_ptm(
                    logits=sub_logits,
                    has_frame=sub_frame,
                    D_mask=None,  # already compacted to A∪B
                    asym_id=sub_asym,
                    interface=True,
                    **ptm_kwargs,
                )
                i = i_ix[k].item()
                j = j_ix[k].item()
                M_b[s, i, j] = iptm
                M_b[s, j, i] = iptm
        else:
            # Fallback: original full-N masking path (kept for completeness)
            chain_token_masks = [(asym_s == cid) & token_mask_b[s] for cid in chain_ids]
            for i in range(C):
                for j in range(i + 1, C):
                    pair_mask = chain_token_masks[i] | chain_token_masks[j]
                    if pair_mask.sum() < 2:
                        continue
                    iptm = compute_ptm(
                        logits=logits_s,
                        has_frame=frame_s,
                        D_mask=pair_mask,
                        asym_id=asym_s,
                        interface=True,
                        **ptm_kwargs,
                    )
                    M_b[s, i, j] = iptm
                    M_b[s, j, i] = iptm

        M_b[s, diag_mask] = float("nan")

    # 2) Per-chain score R_b (row-gated mean of touching pairs)
    for s in range(S):
        asym_s = asym_id_b[s]
        token_mask_s = token_mask_b[s]
        frame_s = has_frame_b[s]

        row_has_frame = torch.tensor(
            [
                ((asym_s == chain_ids[c]) & token_mask_s & frame_s).any()
                for c in range(C)
            ],
            dtype=torch.bool,
            device=device,
        )

        for c in range(C):
            vals = []
            if row_has_frame[c]:
                vals.append(M_b[s, c, :C])  # row c
            vals.append(M_b[s, :C, c][row_has_frame])  # column c from rows with frames
            v = torch.cat(vals) if len(vals) == 2 else vals[0]
            finite = ~torch.isnan(v)
            if finite.any():
                R_b[s, c] = v[finite].mean()

    # 3) Interface score I_b (ligand vs polymer–polymer)
    for s in range(S):
        Ri = R_b[s, :C].unsqueeze(1).expand(C, C)
        Rj = R_b[s, :C].unsqueeze(0).expand(C, C)
        polymer_polymer = 0.5 * (Ri + Rj)
        ligand_pick = torch.where(lig_i, Ri, Rj)
        I = torch.where(any_lig_pair, ligand_pick, polymer_polymer)
        I[diag_mask] = float("nan")
        I_b[s] = I

    return M_b, R_b, I_b, chain_ids


def compute_modified_residue_plddt(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    plddt: torch.Tensor,
    eps: Optional[float] = 1e-10,
) -> list[dict[tuple[int, int], torch.Tensor]]:
    """
    For every modified residue (is_atomized & ~is_ligand),
    compute the mean per-atom pLDDT per sample.

    Returns:
        A list of length B. For each batch item:
            {(chain_id, residue_id): Tensor[S] in [0,1] or NaN if no atoms}
    """

    plddt_atom_01 = plddt / 100
    batch_size, n_samples, _ = plddt_atom_01.shape
    device, dtype = plddt_atom_01.device, plddt_atom_01.dtype

    token_mask = _expand_sample_dim(
        batch["token_mask"], n_samples
    ).bool()  # [B, S, N_tok]
    chain_ids_tokens = _expand_sample_dim(
        batch["asym_id"], n_samples
    ).long()  # [B, S, N_tok]
    residue_ids_tokens = _expand_sample_dim(
        batch["residue_index"], n_samples
    ).long()  # [B, S, N_tok]

    is_modified_token = (
        _expand_sample_dim(batch["is_atomized"], n_samples).bool()
        & ~_expand_sample_dim(batch["is_ligand"], n_samples).bool()
        & token_mask
    )  # [B, S, N_tok]

    atoms_per_residue = batch["num_atoms_per_token"]

    atom_mask = _expand_sample_dim(
        batch["atom_mask"], n_samples
    ).bool()  # [B, S, N_atom]
    chain_ids_atoms = broadcast_token_feat_to_atoms(
        token_mask, atoms_per_residue, chain_ids_tokens
    ).long()  # [B, S, N_atom]
    residue_ids_atoms = broadcast_token_feat_to_atoms(
        token_mask, atoms_per_residue, residue_ids_tokens
    ).long()  # [B, S, N_atom]
    is_modified_atom = broadcast_token_feat_to_atoms(
        token_mask, atoms_per_residue, is_modified_token
    ).bool()  # [B, S, N_atom]

    per_batch_results = []

    for b in range(batch_size):
        modified_tokens_b0 = is_modified_token[b, 0]  # [N_tok]
        if not torch.any(modified_tokens_b0):
            per_batch_results.append({})
            continue

        chain_ids_b0 = chain_ids_tokens[b, 0, modified_tokens_b0]  # [R]
        residue_ids_b0 = residue_ids_tokens[b, 0, modified_tokens_b0]  # [R]
        unique_pairs = torch.unique(
            torch.stack([chain_ids_b0, residue_ids_b0], dim=1), dim=0, sorted=True
        )  # [R_unique, 2]

        result_b = {}
        plddt_b = plddt_atom_01[b]  # [S, N_atom]
        atom_mask_b = atom_mask[b]  # [S, N_atom]
        is_modified_atom_b = is_modified_atom[b]  # [S, N_atom]
        chain_ids_atoms_b = chain_ids_atoms[b]  # [S, N_atom]
        residue_ids_atoms_b = residue_ids_atoms[b]  # [S, N_atom]

        for idx in range(unique_pairs.size(0)):
            chain_id = int(unique_pairs[idx, 0].item())
            residue_id = int(unique_pairs[idx, 1].item())

            select_atoms = (
                (chain_ids_atoms_b == chain_id)
                & (residue_ids_atoms_b == residue_id)
                & is_modified_atom_b
                & atom_mask_b
            )  # [S, N_atom]

            atom_counts = select_atoms.sum(dim=-1)  # [S]
            plddt_sum = (plddt_b * select_atoms.to(plddt_b.dtype)).sum(dim=-1)  # [S]

            values = torch.full((n_samples,), float("nan"), device=device, dtype=dtype)
            valid = atom_counts > 0
            values[valid] = plddt_sum[valid] / (atom_counts[valid].to(dtype) + eps)

            result_b[(chain_id, residue_id)] = values.unsqueeze(0)

        per_batch_results.append(result_b)

    return {"modified_residues_plddts": per_batch_results}
