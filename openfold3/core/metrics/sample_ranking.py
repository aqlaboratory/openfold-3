import torch

from openfold3.core.metrics.confidence import compute_ptm
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms


def full_complex_sample_ranking_metric(
    batch: dict,
    output: dict,
):
    """
    Implements AF3 sample ranking (Section 5.9.3, part 1)

    Args:
        batch: Updated batch dictionary post permutation alignment
        output: Output dictionary from model

    Returns:
        sample_ranking_metric: score used for sample ranking
    """

    # get aggregated pTM score
    ipTM = output["confidence_scores"]["iPTM"]
    pTM = output["confidence_scores"]["PTM"]

    # compute has_clash
    pred_pos = output["atom_positions_predicted"]

    token_mask = batch["token_mask"]
    asym_id = batch["asym_id"]
    atom_mask = batch["atom_mask"]
    is_protein = batch["is_protein"]
    is_rna = batch["is_rna"]
    is_dna = batch["is_dna"]
    num_atoms_per_token = batch["num_atoms_per_token"]

    is_protein_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, is_protein
    )
    is_rna_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, is_rna
    )
    is_dna_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, is_dna
    )
    asym_id_atomized = broadcast_token_feat_to_atoms(
        token_mask, num_atoms_per_token, asym_id
    )
    is_polymer = is_protein_atomized + is_rna_atomized + is_dna_atomized

    has_clash = compute_has_clash(
        asym_id_atomized,
        pred_pos,
        atom_mask,
        is_polymer,
    )

    sample_ranking_metric = 0.8 * ipTM + 0.2 * pTM - 100 * has_clash

    ranking = torch.argsort(sample_ranking_metric, descending=True)

    return ranking


def single_chain_sample_ranking_metric(
    batch: dict,
    output: dict,
    considered_chain_id: int,
):
    """
    Implements AF3 sample ranking (Section 5.9.3, part 2)

    Args:
        batch: Updated batch dictionary post permutation alignment
        output: Output dictionary from model

    Returns:
        sample_ranking_metric: score used for sample ranking
    """
    # get the considered chain pTM score
    chain_pTM = compute_chain_pTM(batch, output, considered_chain_id)  # [N_samples, 1]

    # rank the samples
    ranking = torch.argsort(chain_pTM, descending=True)

    return ranking


def interface_sample_ranking_metric(
    batch: dict,
    output: dict,
    considered_interface: tuple[int, int],
):
    """
    Implements AF3 sample ranking (Section 5.9.3, part 3)

    Args:
        batch: Updated batch dictionary post permutation alignment
        output: Output dictionary from model

    Returns:
        sample_ranking_metric: score used for sample ranking
    """
    # get the considered interface pTM score
    ipTM = compute_interface_pTM(batch, output, considered_interface)  # [N_samples, 1]

    # rank the samples
    ranking = torch.argsort(ipTM, descending=True)

    return ranking


def modified_residue_sample_ranking_metric(
    batch: dict,
    output: dict,
    chain_id: int,
    residue_id: int,
):
    """
    Implements AF3 sample ranking (Section 5.9.3, part 4)

    Args:
        batch: Updated batch dictionary post permutation alignment
        output: Output dictionary from model

    Returns:
        sample_ranking_metric: score used for sample ranking
    """
    # get the considered residue plddt scores
    plddt = compute_residue_plddt(batch, output, chain_id, residue_id)  # [N_samples, 1]

    # rank the samples
    ranking = torch.argsort(plddt, descending=True)

    return ranking


def compute_residue_plddt(
    batch: dict,
    output: dict,
    chain_id: int,
    residue_id: int,
):
    """
    Computes the plddt score for a given residue

    Args:
        batch: Updated batch dictionary post permutation alignment
        output: Output dictionary from model
        chain_id: chain id of the residue
        residue_id: residue id of the residue

    Returns:
        plddt: plddt score for the given residue
    """

    plddt = output["confidence_scores"]["plddt"]
    asym_id = batch["asym_id"]
    residue_mask = batch["residue_mask"]
    atom_mask = batch["atom_mask"]
    num_atoms_per_residue = batch["num_atoms_per_residue"]

    residue_mask = broadcast_token_feat_to_atoms(
        residue_mask, num_atoms_per_residue, asym_id
    )
    atom_mask = broadcast_token_feat_to_atoms(atom_mask, num_atoms_per_residue, asym_id)

    valid_residues = torch.logical_and(residue_mask == residue_id, atom_mask == 1)

    plddt = plddt[valid_residues]

    return plddt


def compute_chain_pTM(
    batch: dict,
    output: dict,
    chain_id: int,
):
    """
    Computes the pTM score for a given chain

    Args:
        batch: Updated batch dictionary post permutation alignment
        output: Output dictionary from model
        chain_id: chain id of the chain

    Returns:
        pTM: pTM score for the given chain
    """
    pae_logits = output["pae_logits"][:, batch["asym_id"] == chain_id]
    chain_pTM = compute_ptm(
        pae_logits,
    )

    return chain_pTM


def compute_interface_pTM(
    batch: dict,
    output: dict,
    considered_interface: tuple[int, int],
):
    """
    Computes the pTM score for a given interface

    Args:
        batch: Updated batch dictionary post permutation alignment
        output: Output dictionary from model
        considered_interface: interface between two chains

    Returns:
        ipTM: pTM score for the given interface
    """
    considered_interface = torch.tensor(considered_interface)
    pae_logits = output["pae_logits"][batch["asym_id"].isin(considered_interface)]
    asym_id = batch["asym_id"][batch["asym_id"].isin(considered_interface)]
    interface_pTM = compute_ptm(
        pae_logits,
        asym_id=asym_id,
        interface=True,
    )

    return interface_pTM


def compute_has_clash(
    asym_id: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    atom_mask: torch.Tensor,
    is_polymer: torch.Tensor,
    threshold=1.1,
):
    """
    Implements AF3 has_clash (Section 5.9.2)

    Args:
        asym_id: asym_id atomized feature [*, n_atom]
        all_atom_pred_pos: predicted coordinates [*, n_atom, 3]
        atom_mask: atom mask [*, n_atom]
        is_polymer: feature combining is_protein, is_rna, and is_dna [*, n_atom]
        threshold: threshold determining if two atoms are clashing (1.1 A)
    Return:
        has_clash:
            - 1 if a complex contains at least two chains that are clashing
            - 0 if no clash observed for any pair of chains in a given complex
    """
    valid_atoms = torch.logical_and(atom_mask == 1, is_polymer == 1)
    all_atom_pred_pos = all_atom_pred_pos[valid_atoms]
    asym_id = asym_id[valid_atoms]
    unique_chain_ids = torch.unique(asym_id)

    # no clash for single chain
    if unique_chain_ids.shape[-1] == 1:
        return 0

    # check all the chain pairs
    for i in range(unique_chain_ids.shape[-1]):
        for j in range(i + 1, unique_chain_ids.shape[-1]):
            c1, c2 = unique_chain_ids[i], unique_chain_ids[j]
            pair_dist = torch.cdist(
                all_atom_pred_pos[asym_id == c1],
                all_atom_pred_pos[asym_id == c2],
            )
            violation_count = torch.sum(pair_dist < threshold).item()
            min_length = min(torch.sum(asym_id == c1), torch.sum(asym_id == c2))
            if violation_count > 100 or violation_count / min_length > 0.5:
                return 1

    return 0
