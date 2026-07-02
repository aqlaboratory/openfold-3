#!/usr/bin/env python
"""Cold-process inference benchmark for local OF3 optimization work.

The parent process launches one fresh Python subprocess per query/config cell.
That charges imports, model construction, checkpoint load, featurization,
first-use Triton/cuEq compile or autotune, and the first model forward.  There
is intentionally no warm-up forward.

By default it runs the full Lightning predict loop and verifies ``summary.txt``
so failed queries cannot be mistaken for benchmark results.  Pass
``--run-mode forward`` to measure only the first ``lightning_module(batch)``
call without output file writing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER_YAML = REPO / "examples/example_runner_yamls/cuequivariance.yml"
DEFAULT_OUTPUT = REPO / "data/inference_outputs/profiling/cold_benchmark.json"
DEFAULT_RUN_OUTPUT_DIR = REPO / "data/inference_outputs/profiling/cold_benchmark_runs"
QUERY_PATHS = {
    "ubiquitin": REPO / "examples/example_inference_inputs/query_ubiquitin.json",
    "multimer": REPO / "examples/example_inference_inputs/query_multimer.json",
    "protein_ligand": (
        REPO / "examples/example_inference_inputs/query_protein_ligand.json"
    ),
    "homo_1200": REPO / "data/inference_outputs/profiling/queries/homo_1200.json",
    "homo_3000": REPO / "data/inference_outputs/profiling/queries/homo_3000.json",
    "homo_4000": REPO / "data/inference_outputs/profiling/queries/homo_4000.json",
}
DEFAULT_QUERIES = ["multimer", "protein_ligand", "homo_1200"]

BASE_OPTIMIZED_ENV = {
    "OPENFOLD3_FUSED_LN_LINEAR": "1",
    "OPENFOLD3_FUSED_SWIGLU_TRANSITION": "1",
    "OPENFOLD3_FUSED_TRIMUL": "1",
    "OPENFOLD3_FUSED_DIFFUSION_ATTN": "1",
    "OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE": "1",
    "OPENFOLD3_FUSED_TEMPLATE_EMBED": "1",
}

OPTIMIZED_ENV = {
    **BASE_OPTIMIZED_ENV,
    "OPENFOLD3_FUSED_TRI_ATTN_V1": "1",
    "OPENFOLD3_RESIDUAL_FUSED_TRI_ATTN": "1",
    "OPENFOLD3_TRI_ATTN_CHUNK_CAP": "128",
}

CUEQ_CAP128_TRI_ATTN_ENV = {
    **BASE_OPTIMIZED_ENV,
    "OPENFOLD3_FUSED_TRI_ATTN_V1": "0",
    "OPENFOLD3_TRI_ATTN_CHUNK_CAP": "128",
}

OPTIMIZED_CUEQ_TRIMUL_ENV = {
    **OPTIMIZED_ENV,
    "OPENFOLD3_FUSED_TRIMUL": "0",
}

SWEETSPOT_ENV = {
    **OPTIMIZED_ENV,
    "OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE": "0",
    "OPENFOLD3_TRIMUL_CHUNK_CAP": "128",
}

TRIATTN_CUEQ_UNCAPPED_ENV = {
    **CUEQ_CAP128_TRI_ATTN_ENV,
    "OPENFOLD3_TRI_ATTN_CHUNK_CAP": "",
}

TRIATTN_CUEQ_CAP128_ENV = {
    **CUEQ_CAP128_TRI_ATTN_ENV,
}

DISABLED_ENV = {
    "OPENFOLD3_FUSED_LN_LINEAR": "0",
    "OPENFOLD3_FUSED_SWIGLU_TRANSITION": "0",
    "OPENFOLD3_FUSED_TRI_ATTN_V1": "0",
    "OPENFOLD3_FUSED_TRIMUL": "0",
    "OPENFOLD3_FUSED_DIFFUSION_ATTN": "0",
    "OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE": "0",
    "OPENFOLD3_FUSED_TEMPLATE_EMBED": "0",
    "OPENFOLD3_RESIDUAL_FUSED_TRI_ATTN": "0",
    "OPENFOLD3_TRI_ATTN_CHUNK_CAP": "",
    "OPENFOLD3_TRIMUL_CHUNK_CAP": "",
}


def _child_main(args: argparse.Namespace) -> None:
    child_env = _env_for_config(args.config, os.environ)
    os.environ.clear()
    os.environ.update(child_env)

    from openfold3.entry_points.import_utils import _torch_gpu_setup

    _torch_gpu_setup()

    import torch

    from openfold3.core.config import config_utils
    from openfold3.core.utils.tensor_utils import tensor_tree_map
    from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
    from openfold3.entry_points.validator import InferenceExperimentConfig
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    t0 = time.perf_counter()
    runner_args = config_utils.load_yaml(args.runner_yaml)
    runner_args.setdefault("data_module_args", {})
    runner_args["data_module_args"]["num_workers"] = 0
    runner_args.setdefault("experiment_settings", {})
    runner_args["experiment_settings"]["skip_existing"] = False
    runner_args["experiment_settings"]["output_dir"] = str(args.run_output_dir)
    runner_args.setdefault("output_writer_settings", {})
    runner_args["output_writer_settings"]["write_full_confidence_scores"] = (
        args.write_full_confidence
    )
    expt_config = InferenceExperimentConfig(**runner_args)
    runner = InferenceExperimentRunner(
        expt_config,
        num_diffusion_samples=args.samples,
        use_msa_server=False,
    )

    cfg = runner.model_config
    mem = cfg.settings.memory.eval
    if args.config == "sweetspot":
        # Sweetspot is the low-memory config: offload MSA / template / confidence
        # heads whenever the query is nontrivial.  Cutoff at 512 tokens engages
        # offload for every real target (multimer N=590 and up) while leaving
        # tiny targets (ubiquitin N=76) unoffloaded so their launch-bound
        # rollout doesn't take a hit.
        mem.offload_inference.token_cutoff = 512
    else:
        mem.offload_inference.token_cutoff = 10_000_000
    mem.use_deepspeed_evo_attention = False
    mem.use_triton_triangle_kernels = False
    # Keep cuEq triangle kernels on for every config; local_default only disables
    # the optional fused Triton/memory paths via DISABLED_ENV.
    mem.use_cueq_triangle_kernels = True

    inference_query_set = InferenceQuerySet.from_json(args.query_json)
    runner.setup()
    setup_s = time.perf_counter() - t0

    data_s = None
    first_forward_s = None
    predict_run_s = None
    success_count = None
    failed_count = None
    final_allocated_bytes = None
    peak_reserved_bytes = None
    model_batch_baseline_bytes = None
    model_only_bytes = None

    if args.run_mode == "forward":
        runner.inference_query_set = inference_query_set
        lightning_module = runner.lightning_module.to("cuda").eval()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        model_only_bytes = torch.cuda.memory_allocated()

        t_data = time.perf_counter()
        data_module = runner.lightning_data_module
        data_module.prepare_data()
        data_module.setup()
        batch = None
        for candidate in data_module.predict_dataloader():
            if candidate.get("valid_sample") and not candidate.get("repeated_sample"):
                batch = tensor_tree_map(lambda t: t.to("cuda"), candidate)
                break
        if batch is None:
            raise RuntimeError(f"No valid sample in {args.query_json}")
        data_s = time.perf_counter() - t_data

        n_tok = int(batch["token_mask"].shape[-1])
        n_atom = int(batch["atom_mask"].shape[-1]) if "atom_mask" in batch else -1
        c_z = int(lightning_module.model.config.architecture.shared.c_z)
        u_bytes = n_tok * n_tok * c_z * 4

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        resident_baseline_bytes = torch.cuda.memory_allocated()
        model_batch_baseline_bytes = resident_baseline_bytes

        t_forward = time.perf_counter()
        with torch.inference_mode():
            lightning_module(batch)
        torch.cuda.synchronize()
        first_forward_s = time.perf_counter() - t_forward
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_reserved_bytes = torch.cuda.max_memory_reserved()
        final_allocated_bytes = torch.cuda.memory_allocated()
    elif args.run_mode == "predict":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        resident_baseline_bytes = torch.cuda.memory_allocated()

        t_predict = time.perf_counter()
        runner.run(inference_query_set)
        torch.cuda.synchronize()
        predict_run_s = time.perf_counter() - t_predict
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_reserved_bytes = torch.cuda.max_memory_reserved()
        final_allocated_bytes = torch.cuda.memory_allocated()
        c_z = int(runner.lightning_module.model.config.architecture.shared.c_z)

        summary_path = args.run_output_dir / "summary.txt"
        if not summary_path.exists():
            raise RuntimeError(f"Predict run did not write {summary_path}")
        success_count, failed_count = _parse_summary_counts(summary_path.read_text())
        if success_count < 1 or failed_count != 0:
            raise RuntimeError(
                f"Predict run did not complete cleanly: "
                f"success={success_count} failed={failed_count}"
            )

        # Read dimensions from the DataModule after predict so predict-mode rows
        # report the same shape fields as forward-mode rows without a second run.
        n_tok = -1
        n_atom = -1
        for batch in runner.lightning_data_module.predict_dataloader():
            if batch.get("valid_sample") and not batch.get("repeated_sample"):
                n_tok = int(batch["token_mask"].shape[-1])
                n_atom = (
                    int(batch["atom_mask"].shape[-1]) if "atom_mask" in batch else -1
                )
                break
        if n_tok < 0:
            raise RuntimeError("Could not recover token count after predict")
        u_bytes = n_tok * n_tok * c_z * 4
    else:
        raise ValueError(f"Unknown run mode {args.run_mode}")

    activation_u = (
        (peak_bytes - resident_baseline_bytes) / u_bytes
        if args.run_mode == "forward"
        else None
    )
    peak_over_resident_u = (peak_bytes - resident_baseline_bytes) / u_bytes
    final_over_resident_u = (final_allocated_bytes - resident_baseline_bytes) / u_bytes
    peak_over_model_batch_u = (
        (peak_bytes - model_batch_baseline_bytes) / u_bytes
        if model_batch_baseline_bytes is not None
        else None
    )
    total_peak_above_params_u = (
        (peak_bytes - model_only_bytes) / u_bytes
        if model_only_bytes is not None
        else None
    )

    result = {
        "query": args.query,
        "query_json": str(args.query_json),
        "config": args.config,
        "run_mode": args.run_mode,
        "samples": args.samples,
        "n_tokens": n_tok,
        "n_atoms": n_atom,
        "c_z": c_z,
        "U_bytes": u_bytes,
        # Backward-compatible key; this is the resident CUDA baseline captured
        # before the measured forward/predict, not necessarily pure parameters.
        "model_params_bytes": resident_baseline_bytes,
        "resident_baseline_bytes": resident_baseline_bytes,
        "model_batch_baseline_bytes": model_batch_baseline_bytes,
        "max_memory_allocated_bytes": peak_bytes,
        "max_memory_reserved_bytes": peak_reserved_bytes,
        "model_only_bytes": model_only_bytes,
        "final_memory_allocated_bytes": final_allocated_bytes,
        "peak_over_resident_U": peak_over_resident_u,
        "final_over_resident_U": final_over_resident_u,
        "peak_over_model_batch_U": peak_over_model_batch_u,
        "total_peak_above_params_U": total_peak_above_params_u,
        # Meaningful only for forward mode: the baseline is captured after the
        # batch is resident on GPU. Predict mode measures before runner.run(),
        # so it includes DataModule/predict-time resident tensors in the delta.
        "activation_U": activation_u,
        "setup_s": setup_s,
        "data_s": data_s,
        "first_forward_s": first_forward_s,
        "predict_run_s": predict_run_s,
        "success_count": success_count,
        "failed_count": failed_count,
        "run_output_dir": str(args.run_output_dir),
        "env": {
            "OPENFOLD3_FUSED_LN_LINEAR": os.environ.get("OPENFOLD3_FUSED_LN_LINEAR"),
            "OPENFOLD3_FUSED_SWIGLU_TRANSITION": os.environ.get(
                "OPENFOLD3_FUSED_SWIGLU_TRANSITION"
            ),
            "OPENFOLD3_FUSED_TRI_ATTN_V1": os.environ.get(
                "OPENFOLD3_FUSED_TRI_ATTN_V1"
            ),
            "OPENFOLD3_FUSED_TRIMUL": os.environ.get("OPENFOLD3_FUSED_TRIMUL"),
            "OPENFOLD3_FUSED_DIFFUSION_ATTN": os.environ.get(
                "OPENFOLD3_FUSED_DIFFUSION_ATTN"
            ),
            "OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE": os.environ.get(
                "OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE"
            ),
            "OPENFOLD3_FUSED_TEMPLATE_EMBED": os.environ.get(
                "OPENFOLD3_FUSED_TEMPLATE_EMBED"
            ),
            "OPENFOLD3_RESIDUAL_FUSED_TRI_ATTN": os.environ.get(
                "OPENFOLD3_RESIDUAL_FUSED_TRI_ATTN"
            ),
            "OPENFOLD3_TRI_ATTN_CHUNK_CAP": os.environ.get(
                "OPENFOLD3_TRI_ATTN_CHUNK_CAP"
            ),
            "OPENFOLD3_TRIMUL_CHUNK_CAP": os.environ.get(
                "OPENFOLD3_TRIMUL_CHUNK_CAP"
            ),
        },
    }
    print("OF3_COLD_BENCH_RESULT=" + json.dumps(result, sort_keys=True))


def _parse_summary_counts(text: str) -> tuple[int, int]:
    success = re.search(r"Successful Queries:\s+(\d+)", text)
    failed = re.search(r"Failed Queries:\s+(\d+)", text)
    if success is None or failed is None:
        raise RuntimeError(f"Could not parse prediction summary:\n{text}")
    return int(success.group(1)), int(failed.group(1))


def _env_for_config(config: str, base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    if config == "local_default":
        updates = DISABLED_ENV
    elif config == "optimized":
        updates = OPTIMIZED_ENV
    elif config == "optimized_cueqtrimul":
        updates = OPTIMIZED_CUEQ_TRIMUL_ENV
    elif config == "optimized_cueqtriattn_uncapped":
        updates = TRIATTN_CUEQ_UNCAPPED_ENV
    elif config == "optimized_cueqtriattn_cap128":
        updates = TRIATTN_CUEQ_CAP128_ENV
    elif config == "sweetspot":
        updates = SWEETSPOT_ENV
    else:
        raise ValueError(f"Unknown config {config!r}")

    for key, value in updates.items():
        if value == "":
            env.pop(key, None)
        else:
            env[key] = value
    return env


def _run_cell(
    query: str,
    query_json: Path,
    config: str,
    args: argparse.Namespace,
    repeat_idx: int,
) -> dict:
    run_output_dir = (
        args.run_output_dir
        / f"{query}_{config}_s{args.samples}_r{repeat_idx}_{args.run_mode}"
    )
    if run_output_dir.exists():
        shutil.rmtree(run_output_dir)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--query",
        query,
        "--query-json",
        str(query_json),
        "--config",
        config,
        "--runner-yaml",
        str(args.runner_yaml),
        "--samples",
        str(args.samples),
        "--run-mode",
        args.run_mode,
        "--run-output-dir",
        str(run_output_dir),
    ]
    if args.write_full_confidence:
        cmd.append("--write-full-confidence")
    env = _env_for_config(config, os.environ)
    if args.triton_cache_dir is not None:
        env["TRITON_CACHE_DIR"] = str(args.triton_cache_dir)

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    total_wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed for {query}/{config} repeat {repeat_idx}\n"
            f"{proc.stdout}"
        )

    marker = "OF3_COLD_BENCH_RESULT="
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            payload = json.loads(line[len(marker):])
    if payload is None:
        raise RuntimeError(
            f"Missing benchmark result for {query}/{config} repeat {repeat_idx}\n"
            f"{proc.stdout}"
        )
    payload["subprocess_wall_s"] = total_wall_s
    payload["repeat_idx"] = repeat_idx
    return payload


def _summarize(raw: list[dict]) -> dict[str, dict]:
    summary = {}
    for row in raw:
        key = f"{row['query']}_{row['config']}_s{row['samples']}"
        bucket = summary.setdefault(key, {"rows": []})
        bucket["rows"].append(row)

    for bucket in summary.values():
        rows = bucket.pop("rows")
        first = rows[0]
        bucket.update(
            {
                "query": first["query"],
                "config": first["config"],
                "run_mode": first["run_mode"],
                "samples": first["samples"],
                "n_tokens": first["n_tokens"],
                "n_repeats": len(rows),
                "subprocess_wall_s_mean": mean(r["subprocess_wall_s"] for r in rows),
                "first_forward_s_mean": (
                    mean(r["first_forward_s"] for r in rows)
                    if first["first_forward_s"] is not None
                    else None
                ),
                "predict_run_s_mean": (
                    mean(r["predict_run_s"] for r in rows)
                    if first["predict_run_s"] is not None
                    else None
                ),
                "max_memory_allocated_gib": max(
                    r["max_memory_allocated_bytes"] for r in rows
                )
                / 1024**3,
                "max_memory_reserved_gib": max(
                    r["max_memory_reserved_bytes"] for r in rows
                )
                / 1024**3,
                "resident_baseline_gib_mean": mean(
                    r["resident_baseline_bytes"] for r in rows
                )
                / 1024**3,
                "final_memory_allocated_gib_max": max(
                    r["final_memory_allocated_bytes"] for r in rows
                )
                / 1024**3,
                "peak_over_resident_U_max": max(
                    r["peak_over_resident_U"] for r in rows
                ),
                "final_over_resident_U_max": max(
                    r["final_over_resident_U"] for r in rows
                ),
                "peak_over_model_batch_U_max": (
                    max(r["peak_over_model_batch_U"] for r in rows)
                    if first["peak_over_model_batch_U"] is not None
                    else None
                ),
                "activation_U_max": (
                    max(r["activation_U"] for r in rows)
                    if first["activation_U"] is not None
                    else None
                ),
                "total_peak_above_params_U_max": (
                    max(r["total_peak_above_params_U"] for r in rows)
                    if first.get("total_peak_above_params_U") is not None
                    else None
                ),
                "model_only_gib": (
                    mean(r["model_only_bytes"] for r in rows) / 1024**3
                    if first.get("model_only_bytes") is not None
                    else None
                ),
            }
        )
    return summary


def _parent_main(args: argparse.Namespace) -> None:
    queries = args.queries
    configs = args.configs
    raw = []
    for repeat_idx in range(args.repeats):
        for query in queries:
            query_json = QUERY_PATHS.get(query)
            if query_json is None:
                query_json = Path(query)
            if not query_json.exists():
                raise FileNotFoundError(f"Query {query!r} not found at {query_json}")
            for config in configs:
                print(f"Running {query} / {config} repeat {repeat_idx}...", flush=True)
                raw.append(_run_cell(query, query_json, config, args, repeat_idx))

    result = {
        "method": {
            "description": (
                "Fresh subprocess per query/config cell. No warmup. Parent "
                "subprocess_wall_s includes Python imports, setup, checkpoint "
                "load, featurization, first-use compile/autotune, and first "
                "model work. In predict mode it also includes Lightning predict "
                "callbacks and output writing, and summary.txt must report "
                "success=1 failed=0. In forward mode output writing is not "
                "included."
            ),
            "runner_yaml": str(args.runner_yaml),
            "queries": queries,
            "configs": configs,
            "samples": args.samples,
            "repeats": args.repeats,
            "run_mode": args.run_mode,
            "triton_cache_dir": (
                str(args.triton_cache_dir) if args.triton_cache_dir else None
            ),
        },
        "summary": _summarize(raw),
        "raw": raw,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved {args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--query", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--query-json", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--run-output-dir",
        type=Path,
        default=DEFAULT_RUN_OUTPUT_DIR,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["local_default", "optimized"],
        choices=[
            "local_default",
            "optimized",
            "optimized_cueqtrimul",
            "optimized_cueqtriattn_uncapped",
            "optimized_cueqtriattn_cap128",
            "sweetspot",
        ],
    )
    parser.add_argument("--runner-yaml", type=Path, default=DEFAULT_RUNNER_YAML)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--run-mode",
        choices=["forward", "predict"],
        default="predict",
        help=(
            "predict measures the full Lightning predict loop and validates "
            "summary.txt; forward measures a controlled first model forward "
            "without output writing."
        ),
    )
    parser.add_argument(
        "--write-full-confidence",
        action="store_true",
        help="Keep full confidence JSON/NPZ writing in predict mode.",
    )
    parser.add_argument("--triton-cache-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.child:
        if args.query is None or args.query_json is None or args.config is None:
            parser.error("--child requires --query, --query-json, and --config")
        _child_main(args)
    else:
        _parent_main(args)


if __name__ == "__main__":
    main()
