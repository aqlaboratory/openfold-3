from typing import Optional

import torch

from openfold3.core.metrics.confidence import compute_ptm
from openfold3.core.metrics.rasa import compute_disorder
from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
)


def full_complex_sample_ranking_metric(
    batch: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    has_frame: Optional[torch.Tensor] = None,
    ptm_weight: float = 0.2,
    iptm_weight: float = 0.8,
    disorder_weight: float = 0.5,
    has_clash_weight: float = -100.0,
    disorder_threshold: float = 0.581,
    **ptm_bin_kwargs,
) -> torch.Tensor:
    """
    AlphaFold3 sample ranking metric for the full complex (SI §5.9.3, item 1).
    Computes: 0.8·ipTM + 0.2·pTM + 0.5·disorder - 100·has_clash
    This is per-batch operation, expects batch dimension dropped.

    Args:
        batch: model input features (post permutation alignment)
        output: model outputs

    Returns:
        sample_ranking_metric: [num_samplesample] score used for sample ranking
    """
    # inputs
    atom_positions_predicted = output["atom_positions_predicted"]

    num_atoms_per_token = batch["num_atoms_per_token"]
    atom_mask = batch["atom_mask"].bool()
    token_mask = batch["token_mask"]
    asym_id = batch["asym_id"]

    # aggregated ipTM / pTM (Eqs. 17–18)
    iptm = compute_ptm(
        logits=output["pae_logits"],
        has_frame=has_frame,
        mask_i=token_mask,
        asym_id=asym_id,
        interface=True,
        **ptm_bin_kwargs,
    )
    ptm = compute_ptm(
        logits=output["pae_logits"],
        has_frame=has_frame,
        mask_i=token_mask,
        asym_id=asym_id,
        interface=False,
        **ptm_bin_kwargs,
    )

    # atomize features for clash/disorder
    is_polymer = batch["is_protein"] | batch["is_rna"] | batch["is_dna"]
    is_polymer_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, is_polymer
    ).bool()
    asym_id_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, asym_id
    ).bool()

    has_clash = compute_has_clash(
        asym_id=asym_id_atomized,
        atom_positions_predicted=atom_positions_predicted,
        atom_mask=atom_mask,
        is_polymer=is_polymer_atomized,
    )

    if torch.any(batch["is_protein"]):
        disorder = compute_disorder(
            batch=batch, outputs=output, disorder_threshold=disorder_threshold
        )
    else:
        disorder = torch.zeros(
            atom_positions_predicted.shape[:-2],
            device=atom_positions_predicted.device,
            dtype=atom_positions_predicted.dtype,
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


def compute_has_clash(
    asym_id: torch.Tensor,
    atom_positions_predicted: torch.Tensor,
    atom_mask: torch.Tensor,
    is_polymer: torch.Tensor,
    threshold: float = 1.1,
    violation_abs: int = 100,
    violation_frac: float = 0.5,
) -> torch.Tensor:
    """
    Compute

    Returns:
        has_clash: [num_samples] with 1.0 if any pair of distinct polymer chains
        clashes, else 0.0.
    """
    device = atom_positions_predicted.device
    dtype = atom_positions_predicted.dtype
    unique_chains = torch.unique(asym_id).tolist()
    num_samples = atom_positions_predicted.size(0)
    polymer_chains = list(
        filter(lambda aid: ((asym_id != aid) | is_polymer).all(), unique_chains)
    )
    num_chains = len(polymer_chains)

    chain_masks: list[torch.Tensor] = [
        (asym_id == aid) & atom_mask for aid in polymer_chains
    ]

    has_clash = torch.zeros(num_samples, dtype=dtype, device=device)
    for s in range(num_samples):
        clashing = False
        for i in range(num_chains):
            ni = chain_masks[i].sum()
            if ni == 0:
                continue
            for j in range(i + 1, num_chains):
                nj = chain_masks[j].sum()
                if nj == 0:
                    continue
                chain_i = atom_positions_predicted[s, chain_masks[i], :]
                chain_j = atom_positions_predicted[s, chain_masks[j], :]
                distance = torch.cdist(chain_i, chain_j, p=2)
                num_clashes = (distance < threshold).sum().item()
                if (num_clashes > violation_abs) or (
                    (num_clashes / min(ni, nj)) > violation_frac
                ):
                    has_clash[s] = 1.0
                    clashing = True
                    break
            if clashing:
                break

    return has_clash


def compute_chain_ptm(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    has_frame: torch.Tensor,
    **ptm_bin_kwargs,
) -> torch.Tensor:
    """
    Compute chain pTM. Implements AF3 SI §5.9.3, item 2

    Args:
        batch: model input features
        outputs: model outputs

    Returns:
        pTM: [*, n_chains]
    """
    token_mask = batch["token_mask"].bool()

    ptm_by_asym_id = {}
    for asym_id in batch["asym_id"].unique():
        mask_i = token_mask & (batch["asym_id"] == asym_id)
        chain_ptm = compute_ptm(
            outputs["pae_logits"],
            has_frame=has_frame,
            mask_i=mask_i,
            interface=False,
            **ptm_bin_kwargs,
        )
        ptm_by_asym_id[asym_id.item()] = chain_ptm.detach().clone()

    return {"chain_ptm": ptm_by_asym_id}


def compute_chain_pair_iptm(
    batch: dict[str, torch.Tensor],
    logits: torch.Tensor,
    has_frame: torch.Tensor,
    **bin_kwargs,
) -> dict[str, torch.Tensor]:
    """
    Chain-pair interface predicted TM (ipTM) and bespoke ipTM
    Implements AF3 SI §5.9.3 item 3.

    Args:
        batch: Model input features
        logits: Pair-distance logits with bins in
            [num_samples, num_tokens, num_tokens, no_bins]
        has_frame: [num_samples, num_tokens] boolean mask of tokens with valid frames
        **bin_kwargs: Keyword args for bin (i.e., bin_min, bin_max, no_bins).
    """
    token_mask = batch["token_mask"].bool()
    asym_id = batch["asym_id"].long()
    is_ligand = batch["is_ligand"].bool()
    device = logits.device
    dtype = logits.dtype

    unique_chains = torch.unique(asym_id).tolist()
    num_chains = len(unique_chains)
    num_samples = logits.size(0)

    chain_masks: list[torch.Tensor] = [
        (asym_id == aid) & token_mask for aid in unique_chains
    ]

    # Pairwise ipTM for every chain pair
    chain_pair_iptm = torch.zeros(
        (num_samples, num_chains, num_chains), device=device, dtype=dtype
    )
    for i in range(num_chains):
        for j in range(i + 1, num_chains):
            pair_mask = chain_masks[i] | chain_masks[j]
            iptm = compute_ptm(
                logits=logits,
                has_frame=has_frame,
                mask_i=pair_mask,
                asym_id=asym_id,
                interface=True,
                **bin_kwargs,
            )  # [num_samples]
            chain_pair_iptm[:, i, j] = iptm
            chain_pair_iptm[:, j, i] = iptm

    chain_has_frame = torch.tensor(
        [(chain_mask & has_frame).any().item() for chain_mask in chain_masks],
        device=device,
    )
    chain_is_ligand = torch.tensor(
        [
            (chain_mask & is_ligand).sum() * 2 >= chain_mask.sum()
            for chain_mask in chain_masks
        ],
        device=device,
    )

    chain_mean_iptm = torch.zeros((num_samples, num_chains), device=device, dtype=dtype)
    for i in range(num_chains):
        values = [
            chain_pair_iptm[:, i, j]
            for j in range(num_chains)
            if j != i and chain_has_frame[i]
        ]
        values.extend(
            [
                chain_pair_iptm[:, j, i]
                for j in range(num_chains)
                if j != i and chain_has_frame[j]
            ]
        )

        if values:
            chain_mean_iptm[:, i] = torch.stack(values, dim=-1).mean(dim=-1)
        else:
            chain_mean_iptm[:, i] = 0.0

    bespoke_iptm = torch.zeros_like(chain_pair_iptm)
    for i in range(num_chains):
        for j in range(num_chains):
            if i == j:
                continue
            if chain_is_ligand[i]:
                bespoke_iptm[:, i, j] = chain_mean_iptm[:, i]
            elif chain_is_ligand[j]:
                bespoke_iptm[:, i, j] = chain_mean_iptm[:, j]
            else:
                bespoke_iptm[:, i, j] = 0.5 * (
                    chain_mean_iptm[:, i] + chain_mean_iptm[:, j]
                )

    chain_pair_iptm_map: dict[str, torch.Tensor] = {}
    bespoke_iptm_map: dict[str, torch.Tensor] = {}
    for i in range(num_chains):
        for j in range(num_chains):
            if i == j:
                continue
            key = f"({unique_chains[i]},{unique_chains[j]})"
            chain_pair_iptm_map[key] = chain_pair_iptm[:, i, j]
            bespoke_iptm_map[key] = bespoke_iptm[:, i, j]

    return {
        "chain_pair_iptm": chain_pair_iptm_map,
        "bespoke_iptm": bespoke_iptm_map,
    }
