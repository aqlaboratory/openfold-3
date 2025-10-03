# ruff: noqa: E501
import copy
import pickle
from functools import partial
from pathlib import Path
from typing import Optional, Union

import torch
from tqdm import tqdm

from openfold3.core.metrics.aggregate_confidence_ranking import get_confidence_scores
from openfold3.core.utils.atomize_utils import get_token_frame_atoms
from openfold3.core.utils.tensor_utils import dict_multimap, tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry


def to_gpu(x):
    return x.to("cuda:0")


# All functions are adapted from Protenix
# https://github.com/bytedance/Protenix/blob/main/protenix/model/sample_confidence.py
def _compute_full_data_and_summary(
    pae_logits: torch.Tensor,
    plddt_logits: torch.Tensor,
    pde_logits: torch.Tensor,
    contact_probs: torch.Tensor,
    token_asym_id: torch.Tensor,
    token_has_frame: torch.Tensor,
    token_is_ligand: torch.Tensor,
) -> tuple[list[dict], list[dict]]:
    """
    Compute full data and summary confidence scores for the given inputs.

    Args:
        pae_logits (torch.Tensor): Logits for PAE (Predicted Aligned Error).
        plddt_logits (torch.Tensor): Logits for pLDDT (Predicted Local Distance Difference Test).
        pde_logits (torch.Tensor): Logits for PDE (Predicted Distance Error).
        contact_probs (torch.Tensor): Contact probabilities.
        token_asym_id (torch.Tensor): Asymmetric ID for tokens.
        token_has_frame (torch.Tensor): Indicator for tokens having a frame.

    Returns:
        tuple[list[dict], list[dict]]:
            - summary_confidence: List of dictionaries containing summary confidence scores.
            - full_data: List of dictionaries containing full data if `return_full_data` is True.
    """

    full_data = {}
    full_data["atom_plddt"] = logits_to_score(
        plddt_logits, min_bin=0.0, max_bin=1.0, no_bins=50
    )  # [N_s, N_atom]
    # Cpu offload for saving cuda memory
    pde_logits = pde_logits.to(plddt_logits.device)
    full_data["token_pair_pde"] = logits_to_score(
        pde_logits, min_bin=0, max_bin=32, no_bins=64
    )  # [N_s, N_token, N_token]
    del pde_logits
    full_data["contact_probs"] = contact_probs.clone()  # [N_token, N_token]
    pae_logits = pae_logits.to(plddt_logits.device)
    full_data["token_pair_pae"], pae_prob = logits_to_score(
        pae_logits, min_bin=0, max_bin=32, no_bins=64, return_prob=True
    )  # [N_s, N_token, N_token]
    del pae_logits

    summary_confidence = {}
    summary_confidence["plddt"] = full_data["atom_plddt"].mean(dim=-1) * 100  # [N_s, ]
    summary_confidence["gpde"] = (
        full_data["token_pair_pde"] * full_data["contact_probs"]
    ).sum(dim=[-1, -2]) / full_data["contact_probs"].sum(dim=[-1, -2])

    summary_confidence["ptm"] = calculate_ptm(
        pae_prob, has_frame=token_has_frame, min_bin=0, max_bin=32, no_bins=64
    )  # [N_s, ]
    summary_confidence["iptm"] = calculate_iptm(
        pae_prob,
        has_frame=token_has_frame,
        asym_id=token_asym_id,
        min_bin=0,
        max_bin=32,
        no_bins=64,
    )  # [N_s, ]

    # Add: 'chain_gpde', 'chain_pair_gpde'
    summary_confidence.update(
        calculate_chain_based_gpde(
            token_pair_pde=full_data["token_pair_pde"],
            contact_probs=full_data["contact_probs"],
            asym_id=token_asym_id,
        )
    )
    # Add: 'chain_pair_iptm', 'chain_pair_iptm_global' 'chain_iptm', 'chain_ptm'
    summary_confidence.update(
        calculate_chain_based_ptm(
            pae_prob,
            has_frame=token_has_frame,
            asym_id=token_asym_id,
            token_is_ligand=token_is_ligand,
            min_bin=0,
            max_bin=32,
            no_bins=64,
        )
    )
    del pae_prob

    return summary_confidence


def compute_contact_prob(
    distogram_logits: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres=8.0,
) -> torch.Tensor:
    """
    Compute the contact probability from distogram logits.

    Args:
        distogram_logits (torch.Tensor): Logits for the distogram.
            Shape: [N_token, N_token, N_bins]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        thres (float): Threshold distance for contact probability. Defaults to 8.0.

    Returns:
        torch.Tensor: Contact probability.
            Shape: [N_token, N_token]
    """
    distogram_prob = torch.nn.functional.softmax(
        distogram_logits, dim=-1
    )  # [N_token, N_token, N_bins]
    distogram_bins = get_bin_centers(min_bin, max_bin, no_bins)
    thres_idx = (distogram_bins < thres).sum()
    contact_prob = distogram_prob[..., :thres_idx].sum(-1)
    return contact_prob


def get_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    """
    Calculate the centers of the bins for a given range and number of bins.

    Args:
        min_bin (float): The minimum value of the bin range.
        max_bin (float): The maximum value of the bin range.
        no_bins (int): The number of bins.

    Returns:
        torch.Tensor: The centers of the bins.
            Shape: [no_bins]
    """
    bin_width = (max_bin - min_bin) / no_bins
    boundaries = torch.linspace(
        start=min_bin,
        end=max_bin - bin_width,
        steps=no_bins,
    )
    bin_centers = boundaries + 0.5 * bin_width
    return bin_centers


def logits_to_prob(logits: torch.Tensor, dim=-1) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=dim)


def logits_to_score(
    logits: torch.Tensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    return_prob=False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert logits to a score using bin centers.

    Args:
        logits (torch.Tensor): Logits tensor.
            Shape: [..., no_bins]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        return_prob (bool): Whether to return the probability distribution. Defaults to False.

    Returns:
        score (torch.Tensor): Converted score.
            Shape: [...]
        prob (torch.Tensor, optional): Probability distribution if `return_prob` is True.
            Shape: [..., no_bins]
    """
    prob = logits_to_prob(logits, dim=-1)
    bin_centers = get_bin_centers(min_bin, max_bin, no_bins).to(logits.device)
    score = prob @ bin_centers
    if return_prob:
        return score, prob
    else:
        return score


def calculate_normalization(N):
    # TM-score normalization constant
    return 1.24 * (max(N, 19) - 15) ** (1 / 3) - 1.8


def calculate_ptm(
    pae_prob: torch.Tensor,
    has_frame: torch.BoolTensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    """Compute pTM score

    Args:
        pae_prob (torch.Tensor): Predicted probability from PAE loss head.
            Shape: [..., N_token, N_token, N_bins]
        has_frame (torch.BoolTensor): Indicator for tokens having a frame.
            Shape: [N_token, ]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        token_mask (Optional[torch.BoolTensor]): Mask for tokens.
            Shape: [N_token, ] or None

    Returns:
        torch.Tensor: pTM score. Higher values indicate better ranking.
            Shape: [...]
    """
    has_frame = has_frame.bool()

    if token_mask is not None:
        token_mask = token_mask.bool()
        pae_prob = pae_prob[..., token_mask, :, :][
            ..., :, token_mask, :
        ]  # [..., N_d, N_d, N_bins]
        has_frame = has_frame[token_mask]  # [N_d, ]

    if has_frame.sum() == 0:
        return torch.zeros(size=pae_prob.shape[:-3], device=pae_prob.device)

    N_d = has_frame.shape[-1]
    ptm_norm = calculate_normalization(N_d)

    bin_center = get_bin_centers(min_bin, max_bin, no_bins)
    per_bin_weight = (1 / (1 + (bin_center / ptm_norm) ** 2)).to(
        pae_prob.device
    )  # [N_bins]

    token_token_ptm = (pae_prob * per_bin_weight).sum(dim=-1)  # [..., N_d, N_d]
    ptm = token_token_ptm.mean(dim=-1)[..., has_frame].max(dim=-1).values
    return ptm


def calculate_chain_based_ptm(
    pae_prob: torch.Tensor,
    has_frame: torch.BoolTensor,
    asym_id: torch.LongTensor,
    token_is_ligand: torch.BoolTensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
) -> dict[str, torch.Tensor]:
    """
    Compute chain-based pTM scores.

    Args:
        pae_prob (torch.Tensor): Predicted probability from PAE loss head.
            Shape: [..., N_token, N_token, N_bins]
        has_frame (torch.BoolTensor): Indicator for tokens having a frame.
            Shape: [N_token, ]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.
            Shape: [N_token, ]
        token_is_ligand (torch.BoolTensor): Indicator for tokens being ligands.
            Shape: [N_token, ]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.

    Returns:
        dict: Dictionary containing chain-based pTM scores.
            - chain_ptm (torch.Tensor): pTM scores for each chain.
            - chain_iptm (torch.Tensor): ipTM scores for chain interface.
            - chain_pair_iptm (torch.Tensor): Pairwise ipTM scores between chains.
            - chain_pair_iptm_global (torch.Tensor): Global pairwise ipTM scores between chains.
    """

    has_frame = has_frame.bool()
    asym_id = asym_id.long()
    asym_id_to_asym_mask = {aid.item(): asym_id == aid for aid in torch.unique(asym_id)}
    chain_is_ligand = {
        aid.item(): token_is_ligand[asym_id == aid].sum() >= (asym_id == aid).sum() // 2
        for aid in torch.unique(asym_id)
    }

    batch_shape = pae_prob.shape[:-3]

    # Chain_pair_iptm
    N_chain = len(asym_id_to_asym_mask)
    chain_pair_iptm = torch.zeros(size=batch_shape + (N_chain, N_chain)).to(
        pae_prob.device
    )
    for aid_1 in range(N_chain):
        for aid_2 in range(N_chain):
            if aid_1 == aid_2:
                continue
            if aid_1 > aid_2:
                chain_pair_iptm[:, aid_1, aid_2] = chain_pair_iptm[:, aid_2, aid_1]
                continue
            pair_mask = asym_id_to_asym_mask[aid_1] + asym_id_to_asym_mask[aid_2]
            chain_pair_iptm[:, aid_1, aid_2] = calculate_iptm(
                pae_prob,
                has_frame,
                asym_id,
                min_bin,
                max_bin,
                no_bins,
                token_mask=pair_mask,
            )

    # chain_ptm
    chain_ptm = torch.zeros(size=batch_shape + (N_chain,)).to(pae_prob.device)
    for aid, asym_mask in asym_id_to_asym_mask.items():
        chain_ptm[:, aid] = calculate_ptm(
            pae_prob,
            has_frame,
            min_bin,
            max_bin,
            no_bins,
            token_mask=asym_mask,
        )

    # Chain iptm
    chain_has_frame = [
        (asym_id_to_asym_mask[i] * has_frame).any() for i in range(N_chain)
    ]

    chain_iptm = torch.zeros(size=batch_shape + (N_chain,)).to(pae_prob.device)
    for aid, _ in asym_id_to_asym_mask.items():
        pairs = [
            (i, j)
            for i in range(N_chain)
            for j in range(N_chain)
            if (i == aid or j == aid) and (i != j) and chain_has_frame[i]
        ]
        vals = [chain_pair_iptm[:, i, j] for (i, j) in pairs]
        if len(vals) > 0:
            chain_iptm[:, aid] = torch.stack(vals, dim=-1).mean(dim=-1)

    # Chain_pair_iptm_global
    chain_pair_iptm_global = torch.zeros(size=batch_shape + (N_chain, N_chain)).to(
        pae_prob.device
    )
    for aid_1 in range(N_chain):
        for aid_2 in range(N_chain):
            if aid_1 == aid_2:
                continue
            if chain_is_ligand[aid_1]:
                chain_pair_iptm_global[:, aid_1, aid_2] = chain_iptm[:, aid_1]
            elif chain_is_ligand[aid_2]:
                chain_pair_iptm_global[:, aid_1, aid_2] = chain_iptm[:, aid_2]
            else:
                chain_pair_iptm_global[:, aid_1, aid_2] = (
                    chain_iptm[:, aid_1] + chain_iptm[:, aid_2]
                ) * 0.5

    return {
        "chain_ptm": chain_ptm,
        "chain_iptm": chain_iptm,
        "chain_pair_iptm": chain_pair_iptm,
        "chain_pair_iptm_global": chain_pair_iptm_global,
    }


def calculate_chain_based_gpde(
    token_pair_pde: torch.Tensor,
    contact_probs: torch.Tensor,
    asym_id: torch.LongTensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Calculate chain-based gPDE values.

    Args:
        token_pair_pde (torch.Tensor): PDE (Predicted Distance Error) of token-token pairs.
            [..., N_token, N_token]
        contact_probs (torch.Tensor): Contact probabilities.
            [..., N_token, N_token]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing chain-based gPDE values.
            - chain_gpde (torch.Tensor): Intra-chain gPDE.
            - chain_pair_gpde (torch.Tensor): Interface gPDE.
    """

    asym_id = asym_id.long()
    unique_asym_ids = torch.unique(asym_id)
    N_chain = len(unique_asym_ids)
    assert N_chain == asym_id.max() + 1  # make sure it is from 0 to N_chain-1

    batch_shape = token_pair_pde.shape[:-2]
    device = token_pair_pde.device

    def _cal_gpde(token_mask_1, token_mask_2):
        masked_contact_probs = contact_probs[..., token_mask_1, :][..., token_mask_2]
        masked_pde = token_pair_pde[..., token_mask_1, :][..., token_mask_2]
        return (masked_pde * masked_contact_probs).sum(dim=(-1, -2)) / (
            masked_contact_probs.sum(dim=(-1, -2)) + eps
        )

    # Chain_gpde
    chain_gpde = torch.zeros(size=batch_shape + (N_chain,), device=device)
    for aid in range(N_chain):
        chain_gpde[..., aid] = _cal_gpde(
            token_mask_1=asym_id == aid,
            token_mask_2=asym_id == aid,
        )

    # Chain_pair_pde
    chain_pair_gpde = torch.zeros(size=batch_shape + (N_chain, N_chain), device=device)
    for aid_1 in range(N_chain):
        for aid_2 in range(N_chain):
            if aid_1 == aid_2:
                continue
            if aid_2 < aid_1:
                chain_pair_gpde[..., aid_1, aid_2] = chain_pair_gpde[..., aid_2, aid_1]
                continue
            chain_pair_gpde[..., aid_1, aid_2] = _cal_gpde(
                token_mask_1=asym_id == aid_1,
                token_mask_2=asym_id == aid_2,
            )

    return {"chain_gpde": chain_gpde, "chain_pair_gpde": chain_pair_gpde}


def calculate_iptm(
    pae_prob: torch.Tensor,
    has_frame: torch.BoolTensor,
    asym_id: torch.LongTensor,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    token_mask: Optional[torch.BoolTensor] = None,
    eps: float = 1e-8,
):
    """
    Compute ipTM score.

    Args:
        pae_prob (torch.Tensor): Predicted probability from PAE loss head.
            Shape: [..., N_token, N_token, N_bins]
        has_frame (torch.BoolTensor): Indicator for tokens having a frame.
            Shape: [N_token, ]
        asym_id (torch.LongTensor): Asymmetric ID for tokens.
            Shape: [N_token, ]
        min_bin (float): Minimum bin value.
        max_bin (float): Maximum bin value.
        no_bins (int): Number of bins.
        token_mask (Optional[torch.BoolTensor]): Mask for tokens.
            Shape: [N_token, ] or None
        eps (float): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: ipTM score. Higher values indicate better ranking.
            Shape: [...]
    """
    has_frame = has_frame.bool()
    if token_mask is not None:
        token_mask = token_mask.bool()
        pae_prob = pae_prob[..., token_mask, :, :][
            ..., :, token_mask, :
        ]  # [..., N_d, N_d, N_bins]
        has_frame = has_frame[token_mask]  # [N_d, ]
        asym_id = asym_id[token_mask]  # [N_d, ]

    if has_frame.sum() == 0:
        return torch.zeros(size=pae_prob.shape[:-3], device=pae_prob.device)

    N_d = has_frame.shape[-1]
    ptm_norm = calculate_normalization(N_d)

    bin_center = get_bin_centers(min_bin, max_bin, no_bins)
    per_bin_weight = (1 / (1 + (bin_center / ptm_norm) ** 2)).to(
        pae_prob.device
    )  # [N_bins]

    token_token_ptm = (pae_prob * per_bin_weight).sum(dim=-1)  # [..., N_d, N_d]

    is_diff_chain = asym_id[None, :] != asym_id[:, None]  # [N_d, N_d]

    iptm = (token_token_ptm * is_diff_chain).sum(dim=-1) / (
        eps + is_diff_chain.sum(dim=-1)
    )  # [..., N_d]
    iptm = iptm[..., has_frame].max(dim=-1).values

    return iptm


@torch.no_grad()
def compute_full_data_and_summary(
    pae_logits,
    plddt_logits,
    pde_logits,
    contact_probs,
    token_asym_id,
    token_has_frame,
    token_is_ligand,
):
    """Wrapper of `_compute_full_data_and_summary` by enumerating over N samples"""

    N_sample = pae_logits.size(0)
    if contact_probs.dim() == 2:
        # Convert to [N_sample, N_token, N_token]
        contact_probs = contact_probs.unsqueeze(dim=0).expand(N_sample, -1, -1)
    else:
        assert contact_probs.dim() == 3
    assert (
        contact_probs.size(0) == plddt_logits.size(0) == pde_logits.size(0) == N_sample
    )

    summary_confidence = []
    for i in range(N_sample):
        summary_confidence_i = _compute_full_data_and_summary(
            pae_logits=pae_logits[i : i + 1],
            plddt_logits=plddt_logits[i : i + 1],
            pde_logits=pde_logits[i : i + 1],
            contact_probs=contact_probs[i],
            token_asym_id=token_asym_id,
            token_has_frame=token_has_frame[i],
            token_is_ligand=token_is_ligand,
        )
        summary_confidence.append(summary_confidence_i)
    return summary_confidence


if __name__ == "__main__":
    batch_dir = Path("/pscratch/sd/m/ml5045/MyQuota/val_batch_after")
    output_dir = Path("/pscratch/sd/m/ml5045/MyQuota/val_outputs")
    batches = list(batch_dir.glob("*.pkl"))
    for batch_path in tqdm(batches[42:]):
        with open(batch_path, "rb") as file:
            batch = pickle.load(file)
        output_path = output_dir / batch_path.name
        with open(output_path, "rb") as file:
            outputs = pickle.load(file)

        batch = tensor_tree_map(to_gpu, batch, strict_type=False)
        outputs = tensor_tree_map(to_gpu, outputs, strict_type=False)

        x = outputs["atom_positions_predicted"]

        num_samples = x.size(1)
        device = x.device
        dtype = x.dtype

        def repeat_sample_dim(x):
            reps = (1, num_samples, *([1] * (x.ndim - 2)))  # noqa: B023
            return x.repeat(reps)

        batch_for_frame_mask = copy.deepcopy(batch)
        batch_for_frame_mask = tensor_tree_map(
            repeat_sample_dim, batch_for_frame_mask, strict_type=False
        )
        _, valid_frame_mask = get_token_frame_atoms(
            batch=batch_for_frame_mask,
            x=x,
            atom_mask=batch_for_frame_mask["atom_mask"],
        )

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()
        config.confidence.distogram.return_contact_probs = True
        config.architecture.heads.pae.enabled = True
        config.confidence.sample_ranking.chain_ptm.enabled = True

        # Protenix expects asym_id to be contiguous 0 - len(chains)-1 so map it
        _, asym_id = torch.unique(batch["asym_id"], return_inverse=True)
        batch["asym_id"] = asym_id

        of3_confidence = get_confidence_scores(batch, outputs, config)

        # Protenix expects no batch dimension
        protenix_confidences = compute_full_data_and_summary(
            pae_logits=outputs["pae_logits"][0],
            plddt_logits=outputs["plddt_logits"][0],
            pde_logits=outputs["pde_logits"][0],
            contact_probs=of3_confidence["contact_probs"][0, 0],
            token_asym_id=asym_id[0, 0],
            token_has_frame=valid_frame_mask[0],
            token_is_ligand=batch["is_ligand"][0, 0],
        )

        # Cat sample dimension
        protenix_confidences = dict_multimap(
            partial(torch.concat, dim=0), protenix_confidences
        )
        for key in ["gpde", "plddt", "iptm", "ptm"]:
            assert torch.allclose(of3_confidence[key][0], protenix_confidences[key])

        # Need chain order used to build the pair keys "(aid_i,aid_j)"
        unique_chains = torch.unique(batch["asym_id"]).tolist()
        num_chains = len(unique_chains)

        for of3_key, protenix_key in [
            ["chain_pair_iptm", "chain_pair_iptm"],
            ["bespoke_iptm", "chain_pair_iptm_global"],
        ]:
            chain_pair_from_of3 = torch.zeros(
                (num_samples, num_chains, num_chains), device=device, dtype=dtype
            )
            for i, aid_i in enumerate(unique_chains):
                for j, aid_j in enumerate(unique_chains):
                    if i == j:
                        continue
                    key = f"({aid_i},{aid_j})"
                    chain_pair_from_of3[:, i, j] = of3_confidence[of3_key][key]
            assert torch.allclose(
                chain_pair_from_of3, protenix_confidences[protenix_key]
            )

        ptm_from_of3 = torch.zeros(
            (num_samples, num_chains), device=device, dtype=dtype
        )
        for i, aid in enumerate(unique_chains):
            ptm_from_of3[:, i] = of3_confidence["chain_ptm"][aid]
        assert torch.allclose(
            ptm_from_of3,
            protenix_confidences["chain_ptm"],
        )
