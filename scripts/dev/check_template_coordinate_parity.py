#!/usr/bin/env python
"""End-to-end parity: precomputed template pairs vs coordinate-derived features.

Uses one loaded model. Builds the legacy distogram/unit-vector batch from the
same compact coordinates so featurization cannot drift, then compares a
deterministic forward of each representation.

Synthetic module tests aim for ~1e-5 agreement. On real templates the online
local-frame path (normalize/cross) differs from ``create_template_unit_vector``
(Rigid/Vec3Array) by ~1e-6 per component, which can amplify through the trunk;
structure coords should still stay within ``--atom-atol``.

Example:
  source scripts/activate_of3.sh
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \\
    python scripts/dev/check_template_coordinate_parity.py --query 7cnx
"""

from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402

from openfold3.core.config import config_utils  # noqa: E402
from openfold3.core.data.primitives.featurization.template import (  # noqa: E402
    create_template_distogram,
    create_template_unit_vector,
)
from openfold3.core.utils.tensor_utils import tensor_tree_map  # noqa: E402
from openfold3.entry_points.experiment_runner import (  # noqa: E402
    InferenceExperimentRunner,
)
from openfold3.entry_points.validator import InferenceExperimentConfig  # noqa: E402
from openfold3.projects.of3_all_atom.config.inference_query_format import (  # noqa: E402
    InferenceQuerySet,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER = REPO / "data/inference_benchmark_msa_template/runner.yml"
KNOWN_QUERIES = {
    "7cnx": REPO / "data/inference_benchmark_msa_template/queries/7cnx.json",
    "ubiquitin": REPO
    / "data/inference_benchmark_msa_template/queries/ubiquitin.json",
    "mcl1": REPO / "data/inference_benchmark_msa_template/queries/mcl1.json",
}

OUTPUT_KEYS = (
    "atom_positions_predicted",
    "plddt_logits",
    "pae_logits",
    "pde_logits",
    "distogram_logits",
    "si_trunk",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_tensors(value, prefix: str = "") -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
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


def capture_outputs(output) -> dict[str, torch.Tensor]:
    flat = flatten_tensors(output)
    captured: dict[str, torch.Tensor] = {}
    for key, tensor in flat.items():
        for wanted in OUTPUT_KEYS:
            if key.endswith(wanted) or key == wanted:
                captured[wanted] = tensor.detach().float().cpu()
                break
    return captured


def clone_batch(batch: dict) -> dict:
    return {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def build_runner(
    query_json: Path,
    runner_yaml: Path,
    seed: int,
) -> InferenceExperimentRunner:
    runner_args = deepcopy(config_utils.load_yaml(runner_yaml))
    runner_args.setdefault("experiment_settings", {})["seeds"] = [seed]
    runner_args.setdefault("experiment_settings", {})["use_templates"] = True
    runner_args.setdefault("data_module_args", {})["num_workers"] = 0
    template_cfg = runner_args.setdefault("dataset_config_kwargs", {}).setdefault(
        "template", {}
    )
    template_cfg["use_coordinate_pair_features"] = True

    runner = InferenceExperimentRunner(
        InferenceExperimentConfig(**runner_args),
        num_diffusion_samples=1,
        use_msa_server=False,
        use_templates=True,
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
            return tensor_tree_map(
                lambda t: t.to("cuda") if torch.is_tensor(t) else t, batch
            )
    raise RuntimeError("No valid inference sample")


def legacy_batch_from_coordinates(coord_batch: dict) -> dict:
    """Materialize distogram/unit-vector features from compact coordinates."""
    pseudo_beta = coord_batch["template_pseudo_beta_coords"][0].detach().cpu().numpy()
    frame = coord_batch["template_frame_atom_coords"][0].detach().cpu().numpy()
    pb_mask = coord_batch["template_pseudo_beta_mask"][0].detach().cpu()
    bb_mask = coord_batch["template_backbone_frame_mask"][0].detach().cpu()
    asym = coord_batch["asym_id"][0].detach().cpu()
    multichain = (asym[:, None] == asym[None, :])[None, ..., None]

    legacy = {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in coord_batch.items()
        if key
        not in ("template_pseudo_beta_coords", "template_frame_atom_coords")
    }
    legacy["template_distogram"] = create_template_distogram(
        pseudo_beta, pb_mask, multichain
    )[None].to(device=coord_batch["asym_id"].device)
    legacy["template_unit_vector"] = create_template_unit_vector(
        frame, bb_mask, multichain
    )[None].to(device=coord_batch["asym_id"].device)
    return legacy


def run_forward(module, batch: dict, seed: int) -> dict:
    seed_everything(seed)
    with torch.inference_mode():
        return module(clone_batch(batch))


def summarize_diff(
    legacy: dict[str, torch.Tensor], coordinate: dict[str, torch.Tensor]
) -> dict:
    report = {}
    for key in OUTPUT_KEYS:
        if key not in legacy or key not in coordinate:
            report[key] = {"present": False}
            continue
        a, b = legacy[key], coordinate[key]
        diff = (a - b).abs()
        report[key] = {
            "present": True,
            "shape": list(a.shape),
            "bitwise_equal": bool(torch.equal(a, b)),
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        choices=sorted(KNOWN_QUERIES),
        default="7cnx",
        help="Benchmark query with real templates",
    )
    parser.add_argument("--query-json", type=Path, default=None)
    parser.add_argument("--runner-yaml", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--atom-atol",
        type=float,
        default=2e-4,
        help="Max abs diff allowed for atom_positions_predicted",
    )
    parser.add_argument(
        "--logit-atol",
        type=float,
        default=5e-3,
        help="Max abs diff allowed for confidence/distogram logits",
    )
    parser.add_argument(
        "--trunk-atol",
        type=float,
        default=0.1,
        help="Max abs diff allowed for si_trunk (amplifies UV FP noise)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO
        / "data/inference_outputs/profiling/template_coordinate_e2e_parity.json",
    )
    parser.add_argument(
        "--fused-template-coord",
        choices=("1", "0"),
        default="1",
        help="OPENFOLD3_FUSED_TEMPLATE_COORD for the coordinate path",
    )
    args = parser.parse_args()

    query_json = args.query_json or KNOWN_QUERIES[args.query]
    if not query_json.exists():
        raise FileNotFoundError(query_json)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["OPENFOLD3_FUSED_TEMPLATE_COORD"] = args.fused_template_coord

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    print(f"Loading coordinate batch + model for {query_json}...")
    seed_everything(args.seed)
    runner = build_runner(query_json, args.runner_yaml, args.seed)
    seed_everything(args.seed)
    coord_batch = get_batch(runner)
    assert "template_pseudo_beta_coords" in coord_batch
    assert "template_distogram" not in coord_batch
    legacy_batch = legacy_batch_from_coordinates(coord_batch)
    module = runner.lightning_module.to("cuda").eval()

    print("Coordinate forward...")
    coord_out = capture_outputs(run_forward(module, coord_batch, args.seed))
    print("Legacy (materialized from same coords) forward...")
    legacy_out = capture_outputs(run_forward(module, legacy_batch, args.seed))

    report = summarize_diff(legacy_out, coord_out)
    thresholds = {
        "atom_positions_predicted": args.atom_atol,
        "plddt_logits": args.logit_atol,
        "pae_logits": args.logit_atol,
        "pde_logits": args.logit_atol,
        "distogram_logits": args.logit_atol,
        "si_trunk": args.trunk_atol,
    }
    failures = []
    for key, limit in thresholds.items():
        info = report.get(key, {})
        if not info.get("present"):
            failures.append(f"{key}: missing")
            continue
        if info["max_abs"] > limit:
            failures.append(
                f"{key}: max_abs={info['max_abs']:.6g} > atol={limit:g}"
            )

    payload = {
        "query": str(query_json),
        "seed": args.seed,
        "fused_template_coord": args.fused_template_coord,
        "thresholds": thresholds,
        "ok": not failures,
        "failures": failures,
        "keys": report,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit("Parity failed:\n  - " + "\n  - ".join(failures))
    print("Parity OK")


if __name__ == "__main__":
    main()
