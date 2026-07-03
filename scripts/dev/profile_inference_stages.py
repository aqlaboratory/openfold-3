#!/usr/bin/env python
"""Fast stage timing and top-level memory profiling for OF3 inference.

This script is intended for quick performance iteration. It runs:

1. one warm-up forward,
2. one overall measured forward,
3. one hooked forward with top-level memory peaks and nested timing.

Only the sequential top-level stages track CUDA memory. Nested hooks are
timing-only because resetting CUDA peak stats inside nested hooks would corrupt
the enclosing top-level peak measurement.
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
from openfold3.core.model.primitives.fused_swiglu_transition import (  # noqa: E402
    is_fused_swiglu_transition_enabled,
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
QUERY_DIR = REPO / "examples/example_inference_inputs"
DEFAULT_RUNNER_YAML = REPO / "examples/example_runner_yamls/cuequivariance.yml"

QUERY_FILES = {
    "ubiquitin": "query_ubiquitin.json",
    "multimer": "query_multimer.json",
    "protein_ligand": "query_protein_ligand.json",
    "dna_ptm": "query_dna_ptm.json",
    "homomer": "query_homomer.json",
    "single_protein_single_ligand": "query_single_protein_single_ligand.json",
}


def _gib(n_bytes: int | float) -> float:
    return n_bytes / 1024**3


def _mib(n_bytes: int | float) -> float:
    return n_bytes / 1024**2


class FastStageProfiler:
    """Single-forward profiler with top-level memory and nested timing."""

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


def _pair_transition_hook_attr(pair_block) -> str:
    if (
        is_fused_swiglu_transition_enabled()
        and hasattr(pair_block.pair_transition, "_transition_inplace")
    ):
        return "_transition_inplace"
    return "forward"


def _wrap_tri_pair_block(prof: FastStageProfiler, pair_block, prefix: str) -> None:
    prof.wrap(pair_block, "tri_mul_out_in", f"{prefix}.tri_mul")
    prof.wrap(pair_block, "tri_att_start_end", f"{prefix}.tri_att")
    prof.wrap(
        pair_block.pair_transition,
        _pair_transition_hook_attr(pair_block),
        f"{prefix}.pair_trans",
    )


def install_fast_hooks(model, prof: FastStageProfiler, fine: bool = False) -> None:
    """Install stage hooks for one profiled forward."""
    prof.wrap(model, "run_trunk", "TRUNK", track_mem=True)
    prof.wrap(model.sample_diffusion, "forward", "DIFFUSION", track_mem=True)
    prof.wrap(model.aux_heads, "forward", "CONFIDENCE", track_mem=True)

    prof.wrap(model.input_embedder, "forward", "trunk.input_embedder")
    prof.wrap(model.template_embedder, "forward", "trunk.template_embedder")
    prof.wrap(model.msa_module, "forward", "trunk.msa_module")
    prof.wrap(model.pairformer_stack, "forward", "trunk.pairformer")

    template_embedder = model.template_embedder
    prof.wrap(
        template_embedder.template_pair_embedder,
        "forward",
        "template.pair_embedder",
    )
    prof.wrap(
        template_embedder.template_pair_embedder,
        "_embed_feats",
        "template.pair_embedder.feats",
    )
    prof.wrap(
        template_embedder.template_pair_stack,
        "forward",
        "template.pair_stack",
    )
    if (
        hasattr(template_embedder.template_pair_stack, "blocks")
        and len(template_embedder.template_pair_stack.blocks) > 0
    ):
        block = template_embedder.template_pair_stack.blocks[0]
        prof.wrap(block, "forward", "template.b0.total")
        _wrap_tri_pair_block(prof, block, "template.b0")

    if hasattr(model.msa_module, "blocks"):
        for i, block in enumerate(model.msa_module.blocks):
            prefix = f"msa.b{i}"
            prof.wrap(block.outer_product_mean, "forward", f"{prefix}.opm")
            if block.msa_att_row is not None:
                prof.wrap(block.msa_att_row, "forward", f"{prefix}.msa_att_row")
            if block.msa_transition is not None:
                prof.wrap(
                    block.msa_transition, "forward", f"{prefix}.msa_transition"
                )
            prof.wrap(block.pair_stack, "forward", f"{prefix}.pair_stack")
            _wrap_tri_pair_block(prof, block.pair_stack, prefix)

    if (
        hasattr(model.pairformer_stack, "blocks")
        and len(model.pairformer_stack.blocks) > 0
    ):
        block = model.pairformer_stack.blocks[0]
        prof.wrap(block, "forward", "pf.b0.total")
        _wrap_tri_pair_block(prof, block.pair_stack, "pf.b0")
        prof.wrap(block.attn_pair_bias, "forward", "pf.b0.attn_pair_bias")

    diffusion_module = model.diffusion_module
    prof.wrap(diffusion_module, "forward", "diff.per_step")
    if fine:
        prof.wrap(diffusion_module.atom_attn_enc, "forward", "diff.atom_enc")
        prof.wrap(
            diffusion_module.diffusion_transformer, "forward", "diff.transformer"
        )
        prof.wrap(diffusion_module.atom_attn_dec, "forward", "diff.atom_dec")
        if (
            hasattr(diffusion_module.diffusion_transformer, "blocks")
            and len(diffusion_module.diffusion_transformer.blocks) > 0
        ):
            block = diffusion_module.diffusion_transformer.blocks[0]
            prof.wrap(
                block.attention_pair_bias,
                "forward",
                "diff.transformer.b0.attn_pair_bias",
            )
            prof.wrap(
                block.attention_pair_bias,
                "_prep_bias",
                "diff.transformer.b0.attn_pair_bias.prep_bias",
            )
            prof.wrap(
                block.attention_pair_bias.mha,
                "_prep_qkv",
                "diff.transformer.b0.attn_pair_bias.prep_qkv",
            )
            prof.wrap(
                block.attention_pair_bias.mha,
                "_wrap_up",
                "diff.transformer.b0.attn_pair_bias.wrap_up",
            )
            prof.wrap(
                block.conditioned_transition,
                "forward",
                "diff.transformer.b0.conditioned_transition",
            )

    prof.wrap(model.aux_heads.pairformer_embedding, "forward", "conf.pairformer_emb")
    conf_pairformer = model.aux_heads.pairformer_embedding.pairformer_stack
    if hasattr(conf_pairformer, "blocks") and len(conf_pairformer.blocks) > 0:
        block = conf_pairformer.blocks[0]
        prof.wrap(block.pair_stack, "tri_mul_out_in", "conf.pf.b0.tri_mul")
        prof.wrap(block.pair_stack, "tri_att_start_end", "conf.pf.b0.tri_att")
    prof.wrap(model.aux_heads.distogram, "forward", "conf.distogram")
    prof.wrap(model.aux_heads.plddt, "forward", "conf.plddt")
    prof.wrap(model.aux_heads.pde, "forward", "conf.pde")
    if hasattr(model.aux_heads, "pae"):
        prof.wrap(model.aux_heads.pae, "forward", "conf.pae")


def resolve_query_path(query: str | None, query_json: Path | None) -> Path:
    if query_json is not None:
        return query_json
    if query is None:
        query = "ubiquitin"
    filename = QUERY_FILES.get(query, f"{query}.json")
    path = QUERY_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Query {query!r} not found at {path}; pass --query-json for custom inputs"
        )
    return path


def build_runner(
    query_json: Path,
    runner_yaml: Path,
    num_samples: int,
    use_cueq: bool,
    offload_token_cutoff: int = 10_000_000,
) -> InferenceExperimentRunner:
    runner_args = config_utils.load_yaml(runner_yaml)
    runner_args.setdefault("data_module_args", {})
    runner_args["data_module_args"]["num_workers"] = 0

    expt_config = InferenceExperimentConfig(**runner_args)
    runner = InferenceExperimentRunner(
        expt_config,
        num_diffusion_samples=num_samples,
        use_msa_server=False,
    )
    cfg = runner.model_config
    cfg.settings.memory.eval.offload_inference.token_cutoff = offload_token_cutoff
    cfg.settings.memory.eval.use_deepspeed_evo_attention = False
    cfg.settings.memory.eval.use_cueq_triangle_kernels = use_cueq
    cfg.settings.memory.eval.use_triton_triangle_kernels = False
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
    raise RuntimeError("No valid sample")


def profile(args) -> dict:
    query_json = resolve_query_path(args.query, args.query_json)
    device = torch.device("cuda")
    print(f"Building runner for {query_json} (samples={args.samples})...")

    runner = build_runner(
        query_json=query_json,
        runner_yaml=args.runner_yaml,
        num_samples=args.samples,
        use_cueq=not args.no_cueq,
        offload_token_cutoff=args.offload_token_cutoff,
    )
    lightning_module = runner.lightning_module.to(device).eval()
    model = lightning_module.model
    batch = get_batch(runner, device)

    n_tok = int(batch["token_mask"].shape[-1])
    n_atom = int(batch["atom_mask"].shape[-1]) if "atom_mask" in batch else -1
    c_z = int(model.config.architecture.shared.c_z)
    u_bytes = n_tok * n_tok * c_z * 4

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    resident_baseline_bytes = torch.cuda.memory_allocated()
    del batch

    print(f"n_tokens={n_tok}, n_atoms={n_atom}, c_z={c_z}")
    print(f"1U = {_mib(u_bytes):.1f} MiB")
    print(f"Resident baseline: {_mib(resident_baseline_bytes):.0f} MiB")

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
    install_fast_hooks(model, profiler, fine=args.fine)
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

    return {
        "query_json": str(query_json),
        "runner_yaml": str(args.runner_yaml),
        "n_tokens": n_tok,
        "n_atoms": n_atom,
        "n_diffusion_samples": args.samples,
        "c_z": c_z,
        "U_bytes": u_bytes,
        # Backward-compatible key; this is the resident CUDA baseline captured
        # after model/batch setup, not a pure parameter-only payload.
        "model_params_bytes": resident_baseline_bytes,
        "resident_baseline_bytes": resident_baseline_bytes,
        "overall_peak_bytes": overall_peak_bytes,
        "overall_wall_s": round(overall_wall_s, 2),
        "stage_wall_s": round(stage_wall_s, 2),
        "env_flags": {
            "use_cueq_triangle_kernels": "0" if args.no_cueq else "1",
            "OPENFOLD3_FUSED_TRI_ATTN_V1": os.environ.get(
                "OPENFOLD3_FUSED_TRI_ATTN_V1", "0"
            ),
            "OPENFOLD3_FUSED_TRIMUL": os.environ.get(
                "OPENFOLD3_FUSED_TRIMUL", "0"
            ),
            "OPENFOLD3_FUSED_SWIGLU_TRANSITION": os.environ.get(
                "OPENFOLD3_FUSED_SWIGLU_TRANSITION", "0"
            ),
            "OPENFOLD3_FUSED_LN_LINEAR": os.environ.get(
                "OPENFOLD3_FUSED_LN_LINEAR", "0"
            ),
            "OPENFOLD3_FUSED_DIFFUSION_ATTN": os.environ.get(
                "OPENFOLD3_FUSED_DIFFUSION_ATTN", "0"
            ),
            "OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE": os.environ.get(
                "OPENFOLD3_DIFFUSION_PAIR_BIAS_CACHE", "0"
            ),
            "OPENFOLD3_FUSED_TEMPLATE_EMBED": os.environ.get(
                "OPENFOLD3_FUSED_TEMPLATE_EMBED", "0"
            ),
            "OPENFOLD3_RESIDUAL_FUSED_TRI_ATTN": os.environ.get(
                "OPENFOLD3_RESIDUAL_FUSED_TRI_ATTN", "1"
            ),
            "OPENFOLD3_TRI_ATTN_CHUNK_CAP": os.environ.get(
                "OPENFOLD3_TRI_ATTN_CHUNK_CAP"
            ),
            "OPENFOLD3_TRIMUL_CHUNK_CAP": os.environ.get(
                "OPENFOLD3_TRIMUL_CHUNK_CAP"
            ),
            "offload_token_cutoff": args.offload_token_cutoff,
        },
        "stages": profiler.stats,
    }


def print_report(result: dict) -> None:
    resident_baseline = result.get(
        "resident_baseline_bytes", result["model_params_bytes"]
    )
    peak = result["overall_peak_bytes"]
    u_bytes = result["U_bytes"]
    act_peak = peak - resident_baseline

    print()
    print("=" * 116)
    print(
        f"n_tok={result['n_tokens']} n_atom={result['n_atoms']} "
        f"samples={result['n_diffusion_samples']} 1U={_mib(u_bytes):.1f} MiB"
    )
    print(
        f"resident_baseline={_gib(resident_baseline):.2f} GiB "
        f"peak={_gib(peak):.2f} GiB"
    )
    print(f"activation={_gib(act_peak):.2f} GiB = {act_peak / u_bytes:.2f}U")
    print(
        f"overall_wall={result['overall_wall_s']:.2f}s "
        f"stage_wall={result['stage_wall_s']:.2f}s"
    )
    print("=" * 116)
    print(
        f"{'stage':<28s} {'act_U':>7s} {'trans_U':>8s} "
        f"{'ms':>10s} {'calls':>7s} {'memory':>8s}"
    )
    print("-" * 116)
    for name, stage in result["stages"].items():
        tracked = stage.get("memory_tracked", False)
        if tracked:
            act_u = (stage["abs_peak_bytes"] - resident_baseline) / u_bytes
            trans_u = stage["peak_over_before_bytes"] / u_bytes
            act_s = f"{act_u:7.2f}"
            trans_s = f"{trans_u:8.2f}"
            mem_s = "yes"
        else:
            act_s = f"{'-':>7s}"
            trans_s = f"{'-':>8s}"
            mem_s = "timing"
        print(
            f"{name:<28s} {act_s} {trans_s} "
            f"{stage['cuda_ms']:10.1f} {stage['total_calls']:7d} {mem_s:>8s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="ubiquitin", help="Known example query key")
    parser.add_argument("--query-json", type=Path, default=None)
    parser.add_argument("--runner-yaml", type=Path, default=DEFAULT_RUNNER_YAML)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--no-cueq", action="store_true")
    parser.add_argument(
        "--offload-token-cutoff",
        type=int,
        default=10_000_000,
        help=(
            "``settings.memory.eval.offload_inference.token_cutoff`` override. "
            "Default 10_000_000 = offload OFF for every practical target. "
            "Set to 512 to enable offload for anything above N=512 (sweetspot). "
            "Confidence-head offload is independent (always on via ``predict`` "
            "preset's ``confidence_token_cutoff=0``)."
        ),
    )
    parser.add_argument(
        "--fine",
        action="store_true",
        help="Add diffusion submodule timing hooks; still memory-tracks top-level only",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = profile(args)
    print_report(result)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))
        print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
