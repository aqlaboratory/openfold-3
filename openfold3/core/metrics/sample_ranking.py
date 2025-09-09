from typing import Optional

import torch

from openfold3.core.metrics.confidence import compute_ptm
from openfold3.core.metrics.rasa import compute_disorder
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_frame_atoms,
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
        disorder = 0.0

    scores = {}
    scores["iptm"] = iptm
    scores["ptm"] = ptm
    scores["disorder"] = disorder
    scores["has_clash"] = has_clash
    scores["sample_ranking_score"] = (
        iptm_weight * iptm
        + ptm_weight * ptm
        + disorder_weight * disorder
        - has_clash_weight * has_clash
    )

    return scores


def compute_chain_pTM(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    chain_asym_id: int,
    **kwargs,
) -> torch.Tensor:
    """
    Chain pTM (AF3 §5.9.3, item 2) computed by restricting the token subset.

    Args:
        batch: model input features
        outputs: model outputs
        chain_asym_id: chain identifier to retain in D_mask

    Returns:
        pTM: [*, n_sample] chain-restricted pTM
    """
    pred_pos = outputs["atom_positions_predicted"]
    atom_mask = batch["atom_mask"]
    token_mask = batch["token_mask"].bool()
    no_samples = pred_pos.shape[1] if pred_pos.ndim == 4 else 1

    _, has_frame = get_token_frame_atoms(
        batch=batch, x=pred_pos, atom_mask=_expand_sample_dim(atom_mask, no_samples)
    )
    d_mask = batch["asym_id"] == chain_asym_id
    d_mask = d_mask & token_mask
    d_mask = _expand_sample_dim(d_mask, no_samples)

    return compute_ptm(
        outputs["pae_logits"],
        has_frame=has_frame,
        D_mask=d_mask,
        interface=False,
        **kwargs,
    )


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
        ptm_by_asym_id[asym_id.item()] = compute_ptm(
            outputs["pae_logits"],
            has_frame=has_frame,
            D_mask=D_mask,
            asym_id=batch["asym_id"],
            interface=False,
            **kwargs,
        )

    return {"pTM_by_asym_id": ptm_by_asym_id}


def compute_interface_pTM(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    interface_pair: tuple[int, int],
    **kwargs,
) -> torch.Tensor:
    """
    Interface ipTM (OF3 §5.9.3, item 3) computed by restricting tokens to A∪B.

    Args:
        batch: model input features
        outputs: model outputs
        interface_pair: (asym_id_A, asym_id_B)

    Returns:
        ipTM: [*, n_sample] ipTM over tokens belonging to either chain in the pair
    """
    pred_pos = outputs["atom_positions_predicted"]
    atom_mask = batch["atom_mask"]
    asym_id = batch["asym_id"]
    token_mask = batch["token_mask"].bool()

    no_samples = pred_pos.shape[1] if pred_pos.ndim == 4 else 1
    _, has_frame = get_token_frame_atoms(
        batch=batch, x=pred_pos, atom_mask=_expand_sample_dim(atom_mask, no_samples)
    )

    pair_ids = torch.tensor(interface_pair, device=pred_pos.device)
    d_mask = torch.isin(asym_id, pair_ids)
    d_mask = d_mask & token_mask
    d_mask = _expand_sample_dim(d_mask, no_samples)
    asym_id = _expand_sample_dim(asym_id, no_samples)

    return compute_ptm(
        outputs["pae_logits"],
        has_frame=has_frame,
        D_mask=d_mask,
        asym_id=asym_id,
        interface=True,
        **kwargs,
    )


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


def build_all_interface_ipTM_and_rankings(
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    has_frame: torch.Tensor,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """
    Vectorized construction of:
      - M[b, s, i, j]     : ipTM for every chain pair (i, j) across samples
      - score[b, s, i, j] : OF3 interface score per pair
      - ranking[b, i, j, s]: argsort over samples (descending) for each pair

    Shapes:
      B: batch size, S: n_sample, C_b: chains in batch b (padded to C_max), N: tokens

    Returns:
      {
        "M":       [B, S, C_max, C_max]  (NaN padded),
        "score":   [B, S, C_max, C_max],
        "ranking": [B, C_max, C_max, S],  # per-pair argsort over samples
        "chains":  list[Tensor],          # per-batch chain IDs (length C_b)
      }
    """
    logits = output["pae_logits"]  # [B, (S), N, N, Bins] or [B, N, N, Bins]
    has_frame = has_frame.bool()  # [B, (S), N]
    B, S, N, N2, Bins = logits.shape

    device, dtype = logits.device, logits.dtype

    asym_id = _expand_sample_dim(batch["asym_id"], S).long()  # [B,S,N]
    token_mask = _expand_sample_dim(batch["token_mask"], S).bool()
    is_protein = _expand_sample_dim(batch["is_protein"], S).bool()
    is_rna = _expand_sample_dim(batch["is_rna"], S).bool()
    is_dna = _expand_sample_dim(batch["is_dna"], S).bool()
    is_polymer = is_protein | is_rna | is_dna

    # chain IDs per batch (from S=0)
    chains: list[torch.Tensor] = []
    for b in range(B):
        ids_b = asym_id[b, 0][token_mask[b, 0]]
        chains.append(torch.unique(ids_b))
    C_max = max((int(c.numel()) for c in chains), default=0)

    M = torch.full((B, S, C_max, C_max), float("nan"), dtype=dtype, device=device)
    R = torch.full((B, S, C_max), float("nan"), dtype=dtype, device=device)
    score = torch.full((B, S, C_max, C_max), float("nan"), dtype=dtype, device=device)

    eye_cache: dict[int, torch.Tensor] = {}

    for b in range(B):
        chains_b = chains[b]
        C_b = int(chains_b.numel())
        if C_b <= 1:
            continue

        # token membership per chain on S=0
        chain_masks0 = [
            (asym_id[b, 0] == cid) & token_mask[b, 0] for cid in chains_b
        ]  # [N] each

        # all unordered pairs
        i_ix, j_ix = torch.triu_indices(C_b, C_b, offset=1, device=device)
        P = i_ix.numel()

        # union D for each pair → [P,N] then broadcast to S
        D_pairs0 = torch.stack(
            [
                chain_masks0[i] | chain_masks0[j]
                for i, j in zip(i_ix.tolist(), j_ix.tolist())
            ],
            dim=0,
        )  # [P,N]
        D_pairs = D_pairs0.unsqueeze(1).expand(P, S, N)  # [P,S,N]

        # batched ipTM over L = P*S
        log_PS = logits[b].unsqueeze(0).expand(P, S, N, N, Bins)
        aid_PS = asym_id[b].unsqueeze(0).expand(P, S, N)
        hfr_PS = has_frame[b].unsqueeze(0).expand(P, S, N)

        L = P * S
        iptm_PS = compute_ptm(
            logits=log_PS.reshape(L, N, N, Bins),
            has_frame=hfr_PS.reshape(L, N),
            D_mask=D_pairs.reshape(L, N),
            asym_id=aid_PS.reshape(L, N),
            interface=True,
            **kwargs,
        ).view(P, S)  # [P,S]

        # scatter symmetrically
        M[b, :, i_ix, j_ix] = iptm_PS.T
        M[b, :, j_ix, i_ix] = iptm_PS.T

        eye_b = eye_cache.get(C_b)
        if eye_b is None:
            eye_b = torch.eye(C_b, dtype=torch.bool, device=device)
            eye_cache[C_b] = eye_b
        M[b, :, eye_b] = float("nan")

        # R(b,s,c): mean over partners with ≥1 valid frame in sample s
        valid_sc = torch.zeros(S, C_b, dtype=torch.bool, device=device)
        for c in range(C_b):
            cmask = (asym_id[b] == chains_b[c]) & token_mask[b]  # [S,N]
            valid_sc[:, c] = (has_frame[b] & cmask).any(dim=-1)

        Mb = M[b, :, :C_b, :C_b]  # [S,C_b,C_b]
        self_mask = eye_b.unsqueeze(0).expand(S, C_b, C_b)
        partner_ok = valid_sc.unsqueeze(1).expand(S, C_b, C_b)
        take_mask = (~self_mask) & partner_ok

        Mb_masked = torch.where(
            take_mask, Mb, torch.tensor(float("nan"), dtype=dtype, device=device)
        )
        R[b, :, :C_b] = torch.nanmean(Mb_masked, dim=-1)  # [S,C_b]

        poly_chain = torch.zeros(C_b, dtype=torch.bool, device=device)
        for c in range(C_b):
            cm0 = chain_masks0[c]
            poly_chain[c] = is_polymer[b, 0][cm0].any() if cm0.any() else False

        Ri = R[b, :, :C_b].unsqueeze(2).expand(S, C_b, C_b)
        Rj = R[b, :, :C_b].unsqueeze(1).expand(S, C_b, C_b)
        base_score = 0.5 * (Ri + Rj)

        pi = poly_chain.unsqueeze(1).expand(C_b, C_b)
        pj = poly_chain.unsqueeze(0).expand(C_b, C_b)
        nonpoly_pair = (~pi) | (~pj)

        cstar_ix = torch.where(
            ~pi,
            torch.arange(C_b, device=device).unsqueeze(1).expand(C_b, C_b),
            torch.arange(C_b, device=device).unsqueeze(0).expand(C_b, C_b),
        )  # [C_b,C_b]

        R_b = R[b, :, :C_b]  # [S,C_b]
        idx = cstar_ix.unsqueeze(0).expand(S, C_b, C_b)  # [S,C_b,C_b]
        R_exp = R_b.unsqueeze(1).expand(S, C_b, C_b)  # [S,C_b,C_b]
        R_sel = torch.gather(R_exp, dim=2, index=idx)  # [S,C_b,C_b]

        score[b, :, :C_b, :C_b] = torch.where(
            nonpoly_pair.unsqueeze(0), R_sel, base_score
        )
    all_iptm_scores = {"iptm": {}, "bespoke_iptm": {}}
    for pair in [(i, j) for i in range(C_max) for j in range(C_max) if i < j]:
        i, j = pair
        all_iptm_scores["iptm"][pair] = M[:, :, i, j]
        all_iptm_scores["bespoke_iptm"][pair] = score[:, :, i, j]

    return {"all_ipTM_scores": all_iptm_scores}


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

            result_b[(chain_id, residue_id)] = values

        per_batch_results.append(result_b)

    return {"modified residues plddts": per_batch_results}
