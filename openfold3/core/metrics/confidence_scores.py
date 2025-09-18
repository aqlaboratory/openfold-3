from functools import partial

import torch
from ml_collections import ConfigDict

from openfold3.core.metrics.confidence import (
    compute_global_predicted_distance_error,
    compute_plddt,
    compute_predicted_aligned_error,
    compute_predicted_distance_error,
)
from openfold3.core.metrics.sample_ranking import (
    build_all_interface_ipTM_and_rankings_chunked_compact,
    compute_all_pTM,
    compute_modified_residue_plddt,
    full_complex_sample_ranking_metric,
)
from openfold3.core.utils.atomize_utils import get_token_frame_atoms
from openfold3.core.utils.tensor_utils import dict_multimap, tensor_tree_map


def get_confidence_scores(batch: dict, outputs: dict, config: ConfigDict) -> dict:
    # Used in modified residue ranking
    confidence_scores = {}
    confidence_scores["plddt"] = compute_plddt(outputs["plddt_logits"])
    confidence_scores.update(
        compute_predicted_distance_error(
            outputs["pde_logits"],
            **config.confidence.pde,
        )
    )
    confidence_scores["global_predicted_distance_error"] = (
        compute_global_predicted_distance_error(
            pde=confidence_scores["predicted_distance_error"],
            distogram_probs=torch.softmax(outputs["distogram_logits"], dim=-1),
        )
    )

    if config.architecture.heads.pae.enabled:
        confidence_scores.update(
            compute_predicted_aligned_error(
                outputs["pae_logits"],
                **config.confidence.pae,
            )
        )

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
            ipTM_scores = build_all_interface_ipTM_and_rankings_chunked_compact(
                batch=batch,
                output=outputs,
                has_frame=valid_frame_mask,
                pair_chunk=config.confidence.sample_ranking.pair_chunk,
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

        if config.confidence.sample_ranking.modified_residue_plddt.enabled:
            modified_residue_scores = compute_modified_residue_plddt(
                batch=batch,
                outputs=outputs,
                plddt=confidence_scores["plddt"],
            )
            confidence_scores.update(modified_residue_scores)

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
