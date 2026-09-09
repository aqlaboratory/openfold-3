#!/usr/bin/env python
"""Factorial repeatability check for foundation changes.

Compares the impact of:
1) per-datapoint feature seeding in InferenceDataset
2) segmented atom aggregation in inference mode

Each cell runs two retrievals/forwards with deliberate global RNG pollution
between them and checks bitwise equality of input features and model outputs.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402

from openfold3.core.config import config_utils  # noqa: E402
from openfold3.core.utils.tensor_utils import tensor_tree_map  # noqa: E402
from openfold3.entry_points.experiment_runner import InferenceExperimentRunner  # noqa: E402
from openfold3.entry_points.validator import InferenceExperimentConfig  # noqa: E402
from openfold3.projects.of3_all_atom.config.inference_query_format import (  # noqa: E402
    InferenceQuerySet,
)


FEATURE_KEYS = (
    "ref_pos",
    "ref_mask",
    "ref_element",
    "ref_charge",
    "ref_atom_name_chars",
    "ref_space_uid",
    "msa",
    "msa_mask",
)
OUTPUT_KEYS = (
    "atom_positions_predicted",
    "plddt_logits",
    "pae_logits",
)


@dataclass(frozen=True)
class PatchState:
    feature_seeding: bool
    segmented_atom_agg: bool

    @property
    def name(self) -> str:
        feat = "seeded_features" if self.feature_seeding else "upstream_features"
        agg = "segmented_agg" if self.segmented_atom_agg else "scatter_agg"
        return f"{feat}+{agg}"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pollute_rng() -> None:
    for _ in range(512):
        random.random()
    np.random.randn(16)
    torch.randn(16)


def flatten_tensors(value, prefix: str = "") -> dict[str, torch.Tensor]:
    tensors = {}
    if torch.is_tensor(value):
        tensors[prefix or "output"] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            tensors.update(flatten_tensors(child, name))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            name = f"{prefix}.{index}" if prefix else str(index)
            tensors.update(flatten_tensors(child, name))
    return tensors


def compare_tensor_dict(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> tuple[int, int, list[str]]:
    keys = sorted(set(first) & set(second))
    equal = 0
    mismatches: list[str] = []
    for key in keys:
        a, b = first[key], second[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            mismatches.append(f"{key}: shape/dtype differ")
            continue
        if torch.equal(a, b):
            equal += 1
        else:
            max_diff = (a.float() - b.float()).abs().max().item()
            mismatches.append(f"{key}: max_abs_diff={max_diff:.6g}")
    return equal, len(keys), mismatches


def apply_patches(state: PatchState) -> None:
    import openfold3.core.data.framework.single_datasets.inference as inference_mod
    import openfold3.core.model.layers.sequence_local_atom_attention as atom_attention

    if not hasattr(inference_mod, "_CHECK_ORIGINAL_GETITEM"):
        inference_mod._CHECK_ORIGINAL_GETITEM = inference_mod.InferenceDataset.__getitem__

    original_getitem = inference_mod._CHECK_ORIGINAL_GETITEM

    if state.feature_seeding:

        def getitem(self, index):
            return original_getitem(self, index)

    else:

        def getitem(self, index):
            datapoint = self.datapoint_cache.iloc[index]
            query_id = datapoint["query_id"]
            query = self.query_cache[query_id]
            seed = datapoint["seed"]
            is_repeated_sample = bool(datapoint["repeated_sample"])
            try:
                features = self.create_all_features(query)
                features["query_id"] = query_id
                features["seed"] = torch.tensor([seed])
                features["repeated_sample"] = torch.tensor(
                    [is_repeated_sample], dtype=torch.bool
                )
                features["valid_sample"] = torch.tensor([True], dtype=torch.bool)
                return features
            except Exception as exc:
                import traceback

                inference_mod.logger.warning(
                    f"Failed to process {query_id}: {type(exc).__name__}\n"
                    f"{traceback.format_exc()}"
                )
                return {
                    "query_id": query_id,
                    "repeated_sample": torch.tensor(
                        [is_repeated_sample], dtype=torch.bool
                    ),
                    "valid_sample": torch.tensor([False], dtype=torch.bool),
                }

    inference_mod.InferenceDataset.__getitem__ = getitem

    if not hasattr(atom_attention, "_CHECK_ORIGINAL_ENCODER_FORWARD"):
        atom_attention._CHECK_ORIGINAL_ENCODER_FORWARD = (
            atom_attention.AtomAttentionEncoder.forward
        )

    original_encoder_forward = atom_attention._CHECK_ORIGINAL_ENCODER_FORWARD

    if state.segmented_atom_agg:
        atom_attention.AtomAttentionEncoder.forward = original_encoder_forward
    else:

        def encoder_forward_with_scatter(self, *args, **kwargs):
            was_training = self.training
            self.training = True
            try:
                return original_encoder_forward(self, *args, **kwargs)
            finally:
                self.training = was_training

        atom_attention.AtomAttentionEncoder.forward = encoder_forward_with_scatter


def build_runner(
    query_json: Path,
    runner_yaml: Path,
    seed: int,
) -> InferenceExperimentRunner:
    runner_args = config_utils.load_yaml(runner_yaml)
    runner_args.setdefault("experiment_settings", {})["seeds"] = [seed]
    runner_args.setdefault("data_module_args", {})["num_workers"] = 0
    runner = InferenceExperimentRunner(
        InferenceExperimentConfig(**runner_args),
        num_diffusion_samples=1,
        use_msa_server=False,
        use_templates=False,
    )
    memory = runner.model_config.settings.memory.eval
    memory.offload_inference.token_cutoff = 10_000_000
    memory.use_deepspeed_evo_attention = False
    memory.use_triton_triangle_kernels = False
    memory.use_cueq_triangle_kernels = True
    runner.setup()
    runner.inference_query_set = InferenceQuerySet.from_json(query_json)
    return runner


def get_batch(runner: InferenceExperimentRunner) -> dict:
    data_module = runner.lightning_data_module
    data_module.prepare_data()
    data_module.setup()
    for batch in data_module.predict_dataloader():
        if batch.get("valid_sample") and not batch.get("repeated_sample"):
            return tensor_tree_map(lambda t: t.to("cuda"), batch)
    raise RuntimeError("No valid inference sample")


def capture_features(batch: dict) -> dict[str, torch.Tensor]:
    return {k: batch[k].detach().clone() for k in FEATURE_KEYS if k in batch}


def capture_outputs(output) -> dict[str, torch.Tensor]:
    flat = flatten_tensors(output)
    captured = {}
    for key, tensor in flat.items():
        for wanted in OUTPUT_KEYS:
            if key.endswith(wanted) or key == wanted:
                captured[wanted] = tensor.detach().clone()
                break
    return captured


def run_forward(runner: InferenceExperimentRunner, batch: dict) -> dict:
    module = runner.lightning_module.to("cuda").eval()
    seed_everything(int(batch["seed"][0].item()))
    with torch.inference_mode():
        return module(batch)


def check_configuration(
    state: PatchState,
    query_json: Path,
    runner_yaml: Path,
    seed: int,
) -> dict:
    apply_patches(state)
    runner = build_runner(query_json, runner_yaml, seed)

    pollute_rng()
    features_a = capture_features(get_batch(runner))

    pollute_rng()
    features_b = capture_features(get_batch(runner))

    feat_equal, feat_total, feat_mismatches = compare_tensor_dict(
        features_a, features_b
    )

    # Model repeats use the same batch and the same predict_step reseed contract.
    pollute_rng()
    seed_everything(seed)
    batch = get_batch(runner)
    outputs_a = capture_outputs(run_forward(runner, batch))

    pollute_rng()
    seed_everything(seed)
    outputs_b = capture_outputs(run_forward(runner, batch))

    out_equal, out_total, out_mismatches = compare_tensor_dict(outputs_a, outputs_b)

    return {
        "configuration": state.name,
        "feature_seeding": state.feature_seeding,
        "segmented_atom_agg": state.segmented_atom_agg,
        "features": {
            "bitwise_equal_keys": feat_equal,
            "total_keys": feat_total,
            "all_bitwise_equal": feat_equal == feat_total and not feat_mismatches,
            "ref_pos_bitwise_equal": bool(
                torch.equal(features_a["ref_pos"], features_b["ref_pos"])
            ),
            "mismatches": feat_mismatches,
        },
        "outputs": {
            "bitwise_equal_keys": out_equal,
            "total_keys": out_total,
            "all_bitwise_equal": out_equal == out_total and not out_mismatches,
            "atom_positions_predicted_bitwise_equal": bool(
                torch.equal(
                    outputs_a["atom_positions_predicted"],
                    outputs_b["atom_positions_predicted"],
                )
            ),
            "mismatches": out_mismatches,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query-json",
        type=Path,
        default=Path(
            "examples/example_inference_inputs/query_single_protein_single_ligand.json"
        ),
    )
    parser.add_argument(
        "--runner-yaml",
        type=Path,
        default=Path("examples/example_runner_yamls/smoke_inference.yml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(False)

    configs = [
        PatchState(feature_seeding=False, segmented_atom_agg=False),
        PatchState(feature_seeding=True, segmented_atom_agg=False),
        PatchState(feature_seeding=False, segmented_atom_agg=True),
        PatchState(feature_seeding=True, segmented_atom_agg=True),
    ]

    results = [
        check_configuration(cfg, args.query_json, args.runner_yaml, args.seed)
        for cfg in configs
    ]
    print(json.dumps(results, indent=2))

    print(
        textwrap.dedent(
            f"""
            Matrix (production math, seed={args.seed}):
            - feature repeats: global RNG polluted before each featurization, no pre-seed
            - output repeats: same batch, predict_step-style reseed before each forward

            configuration                         | features equal | outputs equal
            --------------------------------------|----------------|---------------
            """
        ).rstrip()
    )
    for row in results:
        print(
            f"{row['configuration']:<37} | "
            f"{str(row['features']['all_bitwise_equal']):<14} | "
            f"{str(row['outputs']['all_bitwise_equal'])}"
        )


if __name__ == "__main__":
    main()
