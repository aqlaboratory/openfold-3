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

import copy
import logging
import os
from itertools import cycle, islice

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info

from openfold3.core.data.framework.data_module import openfold_batch_collator
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.core.utils.logging_utils import suppress_warnings
from openfold3.core.utils.permutation_alignment import (
    multi_chain_permutation_alignment,
    naive_alignment,
)

worker_seed_log = logging.getLogger(f"{__name__}.worker_seed")


def pad_to_world_size(df: pd.DataFrame, world_size: int | None = None) -> pd.DataFrame:
    """Pads a dataframe containing examples to match the world size.

    To avoid the default DistributedSampler behavior of repeating samples
    to match the world size, artificially inflate the dataset and flag the
    repeated samples so that they are ignored in the metrics.

    Args:
        df: starting collection of examples
        world_size: world_size in a distributed setting
    Returns:
        collection of examples padded to world size, with the first examples repeated.
    """
    num_examples = len(df)

    if not world_size or num_examples % world_size == 0:
        padded_df = df.copy()
        padded_df["repeated_sample"] = [False] * num_examples
        return padded_df

    # otherwise we need to pad the dataframe
    num_repeated_examples = world_size - num_examples % world_size
    repeated_indices = list(islice(cycle(range(num_examples)), num_repeated_examples))
    padded_df = pd.concat([df, df.iloc[repeated_indices]], ignore_index=True)
    padded_df["repeated_sample"] = [False] * num_examples + [
        True
    ] * num_repeated_examples

    return padded_df


def check_invalid_feature_dict(features: dict):
    """
    Validate the feature dictionary for a single datapoint. Raises a ValueError
    if a check fails. Error handling in each training dataset class should catch
    these in debug_mode and skip/log the invalid samples.

    Args:
        features (dict):
            Feature dictionary for a single datapoint
    """
    # Check that the sum of the number of atoms per token is equal to the
    # number of atoms in the reference conformer
    num_atoms_sum = torch.max(torch.sum(features["num_atoms_per_token"], dim=-1)).int()
    num_ref_atoms = features["ref_pos"].shape[-2]
    if num_atoms_sum != num_ref_atoms:
        raise ValueError(
            f"Size mismatch between sum of 'num_atoms_per_token' {num_atoms_sum} atoms "
            f"and reference conformer {num_ref_atoms} atoms."
        )

    # Check that for overlapping token indices, the total number of
    # atoms in the ground truth is equal to the number of atoms in the
    # reference conformer
    gt_token_ids_atomized = broadcast_token_feat_to_atoms(
        token_mask=features["ground_truth"]["token_mask"].bool(),
        num_atoms_per_token=features["ground_truth"]["num_atoms_per_token"],
        token_feat=features["ground_truth"]["token_index"],
    )
    atom_selection_mask = torch.isin(gt_token_ids_atomized, features["token_index"])
    gt_atom_indices = torch.nonzero(atom_selection_mask, as_tuple=True)[0]
    if gt_atom_indices.shape[0] != num_ref_atoms:
        raise ValueError(
            f"Size mismatch between ground truth {gt_atom_indices.shape[0]} atoms and "
            f"crop {num_ref_atoms} atoms for the same token indices."
        )

    # Check that the crop has some resolved atoms
    if not features["ground_truth"]["atom_resolved_mask"].any():
        raise ValueError("No resolved atoms")

    # Check that the ligand and atomized masks are consistent
    if (features["is_ligand"] & ~features["is_atomized"]).any():
        raise ValueError("Contains non-atomized ligands")

    # Check that the number of tokens per atom is less than the maximum expected
    if (features["num_atoms_per_token"] > 23).any():
        raise ValueError("Token contains number of atoms > max expected (23)")

    # Check that all input features are finite
    for k, v in features.items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            raise ValueError(f"Non-finite values for {k}")
        if isinstance(v, dict):
            for i, j in v.items():
                if isinstance(j, torch.Tensor) and not torch.isfinite(j).all():
                    raise ValueError(f"Non-finite values for {i}")

    # Run the permutation alignment to skip over samples that may fail in the model
    # This could throw an exception that is handled in the __getitem__
    feats_perm = openfold_batch_collator([copy.deepcopy(features)])
    with suppress_warnings(
        logger_name="openfold3.core.utils.geometry.kabsch_alignment"
    ):
        multi_chain_permutation_alignment(
            batch=feats_perm,
            atom_positions_predicted=torch.randn_like(feats_perm["ref_pos"]),
        )
        naive_alignment(
            batch=feats_perm,
            atom_positions_predicted=torch.randn_like(feats_perm["ref_pos"]),
        )


def getitem_debug_log(dataset_name: str = "") -> None:
    wi = get_worker_info()
    worker_id = wi.id if wi is not None else 0
    wi_seed = wi.seed if wi else None
    wi_base_seed = (wi_seed - worker_id) if wi_seed is not None else None
    torch_seed = torch.initial_seed()
    if dist.is_available() and dist.is_initialized():
        global_rank = dist.get_rank()
    else:
        global_rank = int(os.environ.get("RANK", 0))
    local_rank = os.environ.get("LOCAL_RANK", 0)
    worker_seed_log.debug(
        f"__getitem__ {dataset_name}: rank={global_rank} local_rank={local_rank} "
        f"pid={os.getpid()} worker_id={worker_id} wi.seed={wi_seed} "
        f"wi.base_seed={wi_base_seed} torch.initial_seed={torch_seed}",
    )
