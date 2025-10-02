from functools import partial

import torch
from ml_collections import ConfigDict

from openfold3.core.metrics.confidence import (
    compute_global_predicted_distance_error,
    probs_to_expected_error,
)
from openfold3.core.metrics.sample_ranking import (
    build_chain_interface_scores,
    compute_all_pTM,
    full_complex_sample_ranking_metric,
)
from openfold3.core.utils.atomize_utils import get_token_frame_atoms
from openfold3.core.utils.tensor_utils import dict_multimap, tensor_tree_map


def get_confidence_scores(batch: dict, outputs: dict, config: ConfigDict) -> dict:
    confidence_scores = {}
    confidence_scores["plddt"] = probs_to_expected_error(
        torch.softmax(outputs["plddt_logits"], dim=-1),
        **config.confidence.plddt
    ).mean(dim=-1) * 100.0
    
    pde_probs = torch.softmax(outputs["pde_logits"], dim=-1)
    confidence_scores["pde"] = probs_to_expected_error(
        pde_probs,
        **config.confidence.pde
    )
    if config.confidence.pde.return_probs:
        confidence_scores["pde_probs"] = pde_probs
    else:
        del pde_probs
    
    confidence_scores["gpde"], contact_probs = compute_global_predicted_distance_error(
        pde=confidence_scores["pde"],
        logits=outputs["distogram_logits"], 
        **config.confidence.distogram
    )
    if config.confidence.distogram.return_contact_probs:
        confidence_scores["contact_probs"] = contact_probs
    else:
        del contact_probs

    if config.architecture.heads.pae.enabled:
        pae_probs = torch.softmax(outputs["pae_logits"], dim=-1)
        confidence_scores["pae"] = probs_to_expected_error(
            pae_probs,
            **config.confidence.pae
        )
        if config.confidence.pae.return_probs:
            confidence_scores["pae_probs"] = pae_probs
        else:
            del pae_probs

        _, valid_frame_mask = get_token_frame_atoms(
            batch=batch,
            x=outputs["atom_positions_predicted"],
            atom_mask=batch["atom_mask"],
        )
        valid_frame_mask = valid_frame_mask.bool()

        # Compute weighted pTM score
        # Uses pae_logits (SI pg. 27)
        sample_ranking = full_complex_sample_ranking_metric(
            batch=batch,
            output=outputs,
            has_frame=valid_frame_mask,
            **config.confidence.sample_ranking.full_complex,
            **config.confidence.ptm,
        )
        confidence_scores.update(sample_ranking)

        if config.confidence.sample_ranking.all_ipTM.enabled:
            ipTM_scores = build_chain_interface_scores(
                batch=batch,
                outputs=outputs,
                has_frame=valid_frame_mask,
                **config.confidence.ptm,
            )
            confidence_scores.update(ipTM_scores)

        if config.confidence.sample_ranking.all_pTM.enabled:
            pTM_scores = compute_all_pTM(
                batch=batch,
                outputs=outputs,
                has_frame=valid_frame_mask,
                **config.confidence.ptm,
            )
            confidence_scores.update(pTM_scores)

    return confidence_scores


def get_confidence_scores_chunked(
    batch: dict,
    outputs: dict,
    config: ConfigDict,
) -> dict[str, torch.Tensor]:
    atom_positions_predicted = outputs["atom_positions_predicted"]
    batch_dims = atom_positions_predicted.shape[:-2]
    num_samples = batch_dims[-1]

    metrics_per_sample_list = []
    for idx in range(num_samples):

        def fetch_cur_sample(t):
            if t.ndim < 2 or t.shape[1] != num_samples:
                return t
            return t[:, idx : idx + 1]  # noqa: B023

        cur_batch = tensor_tree_map(fetch_cur_sample, batch, strict_type=False)
        cur_outputs = tensor_tree_map(fetch_cur_sample, outputs, strict_type=False)
        metrics_per_sample_list.append(
            get_confidence_scores(batch=cur_batch, outputs=cur_outputs, config=config)
        )

    cat_fn = partial(torch.concat, dim=1)
    metrics_per_sample = dict_multimap(cat_fn, metrics_per_sample_list)
    return metrics_per_sample
