#!/usr/bin/env python
"""Stage timing and top-level memory profiling for OF3 inference.

Protocol:
1. warm-up forward
2. unhooked measured forward (overall peak + wall)
3. hooked forward (top-level / trunk-substage peaks + nested timing)

Only sequential memory-tracked stages reset CUDA peak stats. Nested hooks are
timing-only so they do not corrupt enclosing peaks.

1U = N² × c_z × 4 bytes. The gate metric is peak CUDA allocation above live
model parameter (+ buffer) bytes — not peak minus ambient allocator residency
and not raw ``max_memory_allocated()`` alone.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from pathlib import Path

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402

from openfold3.core.config import config_utils  # noqa: E402
from openfold3.core.utils.tensor_utils import tensor_tree_map  # noqa: E402
from openfold3.entry_points.experiment_runner import (  # noqa: E402
    InferenceExperimentRunner,
)
from openfold3.entry_points.validator import InferenceExperimentConfig  # noqa: E402
from openfold3.projects.of3_all_atom.config.inference_query_format import (  # noqa: E402
    InferenceQuerySet,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNNER_YAML = REPO / "examples/example_runner_yamls/cuequivariance.yml"

KNOWN_QUERIES = {
    "ubiquitin": REPO / "examples/example_inference_inputs/query_ubiquitin.json",
    "homo_1200": REPO / "data/inference_outputs/profiling/queries/homo_1200.json",
    "mcl1": REPO / "data/inference_benchmark_msa_template/queries/mcl1.json",
}
KNOWN_RUNNERS = {
    "mcl1": REPO / "data/inference_benchmark_msa_template/runner.yml",
}


def _gib(n_bytes: int | float) -> float:
    return n_bytes / 1024**3


def _mib(n_bytes: int | float) -> float:
    return n_bytes / 1024**2


class FastStageProfiler:
    def __init__(self):
        self.stats: OrderedDict[str, dict] = OrderedDict()
        self._patches: list[tuple[object, str, object]] = []

    def wrap(self, obj, attr: str, name: str, track_mem: bool = False) -> None:
        orig = getattr(obj, attr)

        def hook(*args, **kwargs):
            if track_mem:
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            else:
                before = 0

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            out = orig(*args, **kwargs)
            end_ev.record()
            torch.cuda.synchronize()

            cuda_ms = start_ev.elapsed_time(end_ev)
            peak = torch.cuda.max_memory_allocated() if track_mem else 0
            after = torch.cuda.memory_allocated() if track_mem else 0

            stat = self.stats.setdefault(
                name,
                {
                    "abs_peak_bytes": 0,
                    "before_bytes": before,
                    "after_bytes": after,
                    "peak_over_before_bytes": 0,
                    "cuda_ms": 0.0,
                    "total_calls": 0,
                    "memory_tracked": track_mem,
                },
            )
            stat["total_calls"] += 1
            stat["cuda_ms"] += cuda_ms
            if track_mem and peak > stat["abs_peak_bytes"]:
                stat["abs_peak_bytes"] = peak
                stat["before_bytes"] = before
                stat["after_bytes"] = after
                stat["peak_over_before_bytes"] = peak - before
            return out

        setattr(obj, attr, hook)
        self._patches.append((obj, attr, orig))

    def unwrap_all(self) -> None:
        for obj, attr, orig in reversed(self._patches):
            setattr(obj, attr, orig)
        self._patches.clear()


def install_hooks(
    model, prof: FastStageProfiler, track_trunk_substages: bool
) -> None:
    prof.wrap(model, "run_trunk", "TRUNK", track_mem=not track_trunk_substages)
    prof.wrap(model.sample_diffusion, "forward", "DIFFUSION", track_mem=True)
    prof.wrap(model.aux_heads, "forward", "CONFIDENCE", track_mem=True)

    prof.wrap(
        model.input_embedder,
        "forward",
        "trunk.input_embedder",
        track_mem=track_trunk_substages,
    )
    prof.wrap(
        model.template_embedder,
        "forward",
        "trunk.template_embedder",
        track_mem=track_trunk_substages,
    )
    prof.wrap(
        model.msa_module,
        "forward",
        "trunk.msa_module",
        track_mem=track_trunk_substages,
    )
    prof.wrap(
        model.pairformer_stack,
        "forward",
        "trunk.pairformer",
        track_mem=track_trunk_substages,
    )

    # Timing-only nested hooks for first-pass localization.
    prof.wrap(model.diffusion_module, "forward", "diff.per_step")
    if hasattr(model.aux_heads, "pairformer_embedding"):
        prof.wrap(
            model.aux_heads.pairformer_embedding, "forward", "conf.pairformer_emb"
        )


def resolve_paths(args) -> tuple[Path, Path]:
    if args.query_json is not None:
        query_json = args.query_json
    elif args.query in KNOWN_QUERIES:
        query_json = KNOWN_QUERIES[args.query]
    else:
        raise FileNotFoundError(
            f"Unknown query {args.query!r}; pass --query-json or one of "
            f"{sorted(KNOWN_QUERIES)}"
        )
    if not query_json.exists():
        raise FileNotFoundError(query_json)

    if args.runner_yaml is not None:
        runner_yaml = args.runner_yaml
    elif args.query in KNOWN_RUNNERS:
        runner_yaml = KNOWN_RUNNERS[args.query]
    else:
        runner_yaml = DEFAULT_RUNNER_YAML
    if not runner_yaml.exists():
        raise FileNotFoundError(runner_yaml)
    return query_json, runner_yaml


def build_runner(
    query_json: Path,
    runner_yaml: Path,
    num_samples: int,
    offload_token_cutoff: int,
) -> InferenceExperimentRunner:
    runner_args = config_utils.load_yaml(runner_yaml)
    runner_args.setdefault("data_module_args", {})["num_workers"] = 0
    runner = InferenceExperimentRunner(
        InferenceExperimentConfig(**runner_args),
        num_diffusion_samples=num_samples,
        use_msa_server=False,
    )
    memory = runner.model_config.settings.memory.eval
    memory.offload_inference.token_cutoff = offload_token_cutoff
    memory.use_deepspeed_evo_attention = False
    memory.use_triton_triangle_kernels = False
    memory.use_cueq_triangle_kernels = True
    runner.setup()
    runner.inference_query_set = InferenceQuerySet.from_json(query_json)
    return runner


def get_batch(runner: InferenceExperimentRunner, device: torch.device) -> dict:
    data_module = runner.lightning_data_module
    data_module.prepare_data()
    data_module.setup()
    for batch in data_module.predict_dataloader():
        if batch.get("valid_sample") and not batch.get("repeated_sample"):
            return tensor_tree_map(lambda t: t.to(device), batch)
    raise RuntimeError("No valid inference sample")


def model_parameter_bytes(model: torch.nn.Module) -> dict[str, int]:
    """Exact live parameter/buffer payload on the current device."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        "model_params_bytes": param_bytes,
        "model_buffer_bytes": buffer_bytes,
        "model_params_and_buffers_bytes": param_bytes + buffer_bytes,
    }


def profile(args) -> dict:
    query_json, runner_yaml = resolve_paths(args)
    device = torch.device("cuda")
    print(f"Building runner for {query_json} (samples={args.samples})...")

    runner = build_runner(
        query_json=query_json,
        runner_yaml=runner_yaml,
        num_samples=args.samples,
        offload_token_cutoff=args.offload_token_cutoff,
    )
    lightning_module = runner.lightning_module.to(device).eval()
    model = lightning_module.model
    param_info = model_parameter_bytes(model)
    model_params_bytes = param_info["model_params_and_buffers_bytes"]

    batch = get_batch(runner, device)
    n_tok = int(batch["token_mask"].shape[-1])
    n_atom = int(batch["atom_mask"].shape[-1]) if "atom_mask" in batch else -1
    c_z = int(model.config.architecture.shared.c_z)
    u_bytes = n_tok * n_tok * c_z * 4
    del batch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cuda_allocated_after_load = torch.cuda.memory_allocated()

    print(f"n_tokens={n_tok}, n_atoms={n_atom}, c_z={c_z}")
    print(f"1U = {_mib(u_bytes):.1f} MiB")
    print(
        f"Model params+buffers: {_mib(model_params_bytes):.1f} MiB "
        f"(cuda allocated after load: {_mib(cuda_allocated_after_load):.1f} MiB)"
    )

    print("Warm-up forward...")
    batch = get_batch(runner, device)
    with torch.inference_mode():
        lightning_module(batch)
    del batch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    print("Overall measured forward...")
    batch = get_batch(runner, device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        lightning_module(batch)
    torch.cuda.synchronize()
    overall_wall_s = time.perf_counter() - t0
    overall_peak_bytes = torch.cuda.max_memory_allocated()
    del batch
    torch.cuda.empty_cache()

    print("Hooked stage forward...")
    batch = get_batch(runner, device)
    profiler = FastStageProfiler()
    install_hooks(
        model, profiler, track_trunk_substages=args.track_trunk_substages
    )
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        with torch.inference_mode():
            lightning_module(batch)
        torch.cuda.synchronize()
        stage_wall_s = time.perf_counter() - t0
    finally:
        profiler.unwrap_all()
        del batch

    peak_above_params = overall_peak_bytes - model_params_bytes
    return {
        "query_json": str(query_json),
        "runner_yaml": str(runner_yaml),
        "n_tokens": n_tok,
        "n_atoms": n_atom,
        "n_diffusion_samples": args.samples,
        "c_z": c_z,
        "U_bytes": u_bytes,
        **param_info,
        "cuda_allocated_after_load_bytes": cuda_allocated_after_load,
        "overall_peak_bytes": overall_peak_bytes,
        "peak_above_params_bytes": peak_above_params,
        "peak_above_params_U": peak_above_params / u_bytes,
        "overall_wall_s": round(overall_wall_s, 2),
        "stage_wall_s": round(stage_wall_s, 2),
        "offload_token_cutoff": args.offload_token_cutoff,
        "track_trunk_substages": args.track_trunk_substages,
        "tri_attn_chunk_cap": os.environ.get("OPENFOLD3_TRI_ATTN_CHUNK_CAP"),
        "trimul_chunk_cap": os.environ.get("OPENFOLD3_TRIMUL_CHUNK_CAP"),
        "transition_chunk_cap": os.environ.get("OPENFOLD3_TRANSITION_CHUNK_CAP"),
        "stages": profiler.stats,
    }


def print_report(result: dict) -> None:
    params = result["model_params_and_buffers_bytes"]
    peak = result["overall_peak_bytes"]
    u_bytes = result["U_bytes"]
    above = peak - params

    print()
    print("=" * 100)
    print(
        f"n_tok={result['n_tokens']} n_atom={result['n_atoms']} "
        f"samples={result['n_diffusion_samples']} 1U={_mib(u_bytes):.1f} MiB"
    )
    print(
        f"model_params+buffers={_gib(params):.2f} GiB  "
        f"total_peak={_gib(peak):.2f} GiB"
    )
    print(f"peak_above_params={_gib(above):.2f} GiB = {above / u_bytes:.2f}U")
    print(
        f"overall_wall={result['overall_wall_s']:.2f}s  "
        f"stage_wall={result['stage_wall_s']:.2f}s"
    )
    print("=" * 100)
    print(
        f"{'stage':<28s} {'above_U':>8s} {'trans_U':>8s} "
        f"{'ms':>10s} {'calls':>7s} {'memory':>8s}"
    )
    print("-" * 100)
    for name, stage in result["stages"].items():
        if stage.get("memory_tracked", False):
            above_u = (stage["abs_peak_bytes"] - params) / u_bytes
            trans_u = stage["peak_over_before_bytes"] / u_bytes
            above_s = f"{above_u:8.2f}"
            trans_s = f"{trans_u:8.2f}"
            mem_s = "yes"
        else:
            above_s = f"{'-':>8s}"
            trans_s = f"{'-':>8s}"
            mem_s = "timing"
        print(
            f"{name:<28s} {above_s} {trans_s} "
            f"{stage['cuda_ms']:10.1f} {stage['total_calls']:7d} {mem_s:>8s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="ubiquitin",
        help=f"Known query key: {sorted(KNOWN_QUERIES)}",
    )
    parser.add_argument("--query-json", type=Path, default=None)
    parser.add_argument("--runner-yaml", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--offload-token-cutoff",
        type=int,
        default=10_000_000,
        help="Default disables offload for practical gate sizes",
    )
    parser.add_argument(
        "--track-trunk-substages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Track input/template/MSA/Pairformer memory separately (default: on)",
    )
    parser.add_argument(
        "--triangle-chunk-cap",
        type=int,
        default=None,
        help=(
            "Set OPENFOLD3_TRI_ATTN_CHUNK_CAP and OPENFOLD3_TRIMUL_CHUNK_CAP "
            "(shared MSA/template/pairformer/confidence triangle ops)"
        ),
    )
    parser.add_argument(
        "--transition-chunk-cap",
        type=int,
        default=None,
        help=(
            "Set OPENFOLD3_TRANSITION_CHUNK_CAP to force row-chunked "
            "pair/MSA transitions (and OPM) after the chunk-size tuner"
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    if args.triangle_chunk_cap is not None:
        os.environ["OPENFOLD3_TRI_ATTN_CHUNK_CAP"] = str(args.triangle_chunk_cap)
        os.environ["OPENFOLD3_TRIMUL_CHUNK_CAP"] = str(args.triangle_chunk_cap)
    if args.transition_chunk_cap is not None:
        os.environ["OPENFOLD3_TRANSITION_CHUNK_CAP"] = str(args.transition_chunk_cap)

    result = profile(args)
    print_report(result)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))
        print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
