# Copyright 2025 AlQuraishi Laboratory
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

from functools import partial

import torch
from ml_collections import ConfigDict

from openfold3.core.metrics.confidence import (
    compute_global_predicted_distance_error,
    probs_to_expected_error,
)
from openfold3.core.metrics.sample_ranking import (
    compute_chain_pair_iptm,
    compute_chain_ptm,
    full_complex_sample_ranking_metric,
)
from openfold3.core.utils.atomize_utils import get_token_frame_atoms
from openfold3.core.utils.tensor_utils import dict_multimap, tensor_tree_map


def _get_confidence_scores(batch: dict, outputs: dict, config: ConfigDict) -> dict:
    confidence_scores = {}
    confidence_scores["plddt"] = (
        probs_to_expected_error(
            torch.softmax(outputs["plddt_logits"], dim=-1), **config.confidence.plddt
        )
        * 100.0
    )

    pde_probs = torch.softmax(outputs["pde_logits"], dim=-1)
    confidence_scores["pde"] = probs_to_expected_error(
        pde_probs, **config.confidence.pde
    )
    if config.confidence.pde.return_probs:
        confidence_scores["pde_probs"] = pde_probs
    else:
        del pde_probs

    confidence_scores["gpde"], contact_probs = compute_global_predicted_distance_error(
        pde=confidence_scores["pde"],
        logits=outputs["distogram_logits"],
        **config.confidence.distogram,
    )
    if config.confidence.distogram.return_contact_probs:
        confidence_scores["contact_probs"] = contact_probs
    else:
        del contact_probs

    if config.architecture.heads.pae.enabled:
        pae_probs = torch.softmax(outputs["pae_logits"], dim=-1)
        confidence_scores["pae"] = probs_to_expected_error(
            pae_probs, **config.confidence.pae
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

        confidence_scores.update(
            full_complex_sample_ranking_metric(
                batch=batch,
                output=outputs,
                has_frame=valid_frame_mask,
                **config.confidence.sample_ranking.full_complex,
                **config.confidence.ptm,
            )
        )

        if config.confidence.sample_ranking.chain_pair_iptm.enabled:
            confidence_scores.update(
                compute_chain_pair_iptm(
                    batch=batch,
                    logits=outputs["pae_logits"],
                    has_frame=valid_frame_mask,
                    **config.confidence.ptm,
                )
            )

        if config.confidence.sample_ranking.chain_ptm.enabled:
            confidence_scores.update(
                compute_chain_ptm(
                    batch=batch,
                    outputs=outputs,
                    has_frame=valid_frame_mask,
                    **config.confidence.ptm,
                )
            )

    return confidence_scores


def get_confidence_scores(
    batch: dict,
    outputs: dict,
    config: "ConfigDict",
    compute_per_sample: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Wrapper for `_get_confidence_scores`.
    Compute confidence scores by iterating batch (dim 0).
    Optionally iterate the sample (dim 1) if compute_per_sample=True.
    """
    atom_positions_predicted = outputs["atom_positions_predicted"]
    batch_size = atom_positions_predicted.size(0)
    num_samples = atom_positions_predicted.size(1)

    def slice_batch(t: torch.Tensor, i: int):
        if isinstance(t, torch.Tensor) and t.ndim >= 1:
            return t[i]  # squeeze batch dim for downstream functions
        return t

    def slice_sample(t: torch.Tensor, j: int):
        ## skip tensors with batch size 1
        if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[0] != 1:
            return t[j : j + 1]  # keep sample dim
        return t

    per_batch_metrics = []

    for bi in range(batch_size):
        cur_batch_b = tensor_tree_map(
            lambda x: slice_batch(x, bi).squeeze(0),  # noqa: B023
            batch,
            strict_type=False,
        )  # squeeze sample dim
        cur_batch_b["atom_array"] = cur_batch_b["atom_array"][bi]
        cur_outputs_b = tensor_tree_map(
            lambda x: slice_batch(x, bi),  # noqa: B023
            outputs,
            strict_type=False,
        )

        if compute_per_sample and num_samples is not None and num_samples > 1:
            # Iterate samples inside this batch item
            per_sample_metrics_list = []
            for sj in range(num_samples):
                cur_outputs_bs = tensor_tree_map(
                    lambda x: slice_sample(x, sj),  # noqa: B023
                    cur_outputs_b,
                    strict_type=False,
                )
                per_sample_metrics_list.append(
                    _get_confidence_scores(
                        batch=cur_batch_b, outputs=cur_outputs_bs, config=config
                    )
                )
            # Concat samples back on dim=0 for this batch item
            cat_samples = partial(torch.concat, dim=0)
            metrics_b = dict_multimap(cat_samples, per_sample_metrics_list)
        else:
            # Compute once for all samples of this batch item
            metrics_b = _get_confidence_scores(
                batch=cur_batch_b, outputs=cur_outputs_b, config=config
            )

        per_batch_metrics.append(metrics_b)

    metrics = dict_multimap(torch.stack, per_batch_metrics)
    return metrics
