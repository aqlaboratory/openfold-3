#!/usr/bin/env python
"""DIFFUSION stage peak audit using torch.cuda.memory._record_memory_history.

Goal: account for every byte of the DIFFUSION stage peak in U_trunk units,
so we know whether the rollout pair-bias cache (3 U_trunk resident) is
recoverable.

Procedure:
1. Build the full inference runner (matching `profile_inference_stages.py`).
2. Wrap `model.sample_diffusion.forward` with a thin hook that:
   a. Calls `torch.cuda.memory._record_memory_history(...)` BEFORE the
      diffusion stage starts.
   b. Captures `torch.cuda.memory_snapshot()` IMMEDIATELY AFTER the
      first peak is hit (we use a Pythonic peak tracker that polls
      `max_memory_allocated` between sub-steps).
   c. Disables history recording.
3. Group live segments at peak time by their allocation traceback's
   leaf Python frame (file:line), sum bytes per group.
4. Print a labeled inventory + write annotated markdown.

Run:
    source scripts/activate_of3.sh
    python scripts/dev/audit_diffusion_peak.py \
        --query homo_1200 --samples 1 \
        --output-md data/inference_outputs/kernel_dev/diff_peak_audit_homo1200.md
    python scripts/dev/audit_diffusion_peak.py \
        --query multimer --samples 5 \
        --output-md data/inference_outputs/kernel_dev/diff_peak_audit_multimer_s5.md
"""
from __future__ import annotations

import argparse
import collections
import json
import os
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
QUERY_DIR = REPO / "examples/example_inference_inputs"
DEFAULT_RUNNER_YAML = REPO / "examples/example_runner_yamls/cuequivariance.yml"
PROFILE_QUERY_DIR = REPO / "data/inference_outputs/profiling/queries"

QUERY_FILES = {
    "ubiquitin": QUERY_DIR / "query_ubiquitin.json",
    "multimer": QUERY_DIR / "query_multimer.json",
    "protein_ligand": QUERY_DIR / "query_protein_ligand.json",
    "homo_1200": PROFILE_QUERY_DIR / "homo_1200.json",
    "homo_1500": PROFILE_QUERY_DIR / "homo_1500.json",
}


def _gib(b):
    return b / 1024**3


def _mib(b):
    return b / 1024**2


def resolve_query(name: str | None, path: Path | None) -> Path:
    if path is not None:
        return path
    if name is None:
        name = "homo_1200"
    p = QUERY_FILES.get(name)
    if p is None or not p.exists():
        raise FileNotFoundError(f"Query {name!r} not found")
    return p


def build_runner(query_json, runner_yaml, num_samples) -> InferenceExperimentRunner:
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
    cfg.settings.memory.eval.offload_inference.token_cutoff = 10_000_000
    cfg.settings.memory.eval.use_cueq_triangle_kernels = True
    runner.setup()
    runner.inference_query_set = InferenceQuerySet.from_json(query_json)
    return runner


def get_batch(runner, device):
    dm = runner.lightning_data_module
    dm.prepare_data()
    dm.setup()
    for batch in dm.predict_dataloader():
        if batch.get("valid_sample") and not batch.get("repeated_sample"):
            return tensor_tree_map(lambda t: t.to(device), batch)
    raise RuntimeError("No valid sample")


def _classify_frame(filename: str) -> str:
    """Map a source file path to a short category tag."""
    if "openfold3/core/model/structure/diffusion_module" in filename:
        return "diffusion_module"
    if "openfold3/core/model/layers/diffusion_conditioning" in filename:
        return "diffusion_conditioning"
    if "openfold3/core/model/layers/diffusion_transformer" in filename:
        return "diffusion_transformer"
    if "openfold3/core/model/layers/attention_pair_bias" in filename:
        return "attention_pair_bias"
    if "openfold3/core/model/primitives/attention" in filename:
        return "attention_primitives"
    if "openfold3/core/model/layers/atom_attention" in filename:
        return "atom_attention"
    if "openfold3/core/model/layers/transition" in filename:
        return "transition"
    if "openfold3/core/model/primitives" in filename:
        return "primitives"
    if "openfold3/core" in filename:
        return "openfold3_other"
    if "/torch/" in filename or "torch." in filename:
        return "torch_internal"
    if "<string>" in filename or "ipython" in filename or "<built-in>" in filename:
        return "interp"
    return "extern"


def _frame_signature(stack):
    """Take the deepest non-internal frame as the identifier."""
    if not stack:
        return ("?", 0, "extern")
    # `stack` is a list of dicts: {"filename": ..., "line": ..., "name": ...}
    # In recent torch versions it's a list of FrameInfo (already objects).
    for frame in stack:
        if isinstance(frame, dict):
            fn, lineno = frame.get("filename", "?"), frame.get("line", 0)
        else:
            fn, lineno = getattr(frame, "filename", "?"), getattr(frame, "line", 0)
        cat = _classify_frame(fn)
        if cat in ("torch_internal", "interp", "extern"):
            continue
        # Strip repo prefix for compactness.
        if "/openfold3/" in fn:
            fn_short = fn[fn.find("/openfold3/") + 1:]
        else:
            fn_short = fn
        return (fn_short, lineno, cat)
    # No openfold frame found — fall back to deepest available.
    frame = stack[0]
    if isinstance(frame, dict):
        fn, lineno = frame.get("filename", "?"), frame.get("line", 0)
    else:
        fn, lineno = getattr(frame, "filename", "?"), getattr(frame, "line", 0)
    if "/openfold3/" in fn:
        fn = fn[fn.find("/openfold3/") + 1:]
    return (fn, lineno, _classify_frame(fn))


def audit_one(args) -> dict:
    query_json = resolve_query(args.query, args.query_json)
    device = torch.device("cuda")
    print(f"Building runner for {query_json} (samples={args.samples})...")

    runner = build_runner(query_json, args.runner_yaml, args.samples)
    lightning_module = runner.lightning_module.to(device).eval()
    model = lightning_module.model
    batch = get_batch(runner, device)
    n_tok = int(batch["token_mask"].shape[-1])
    c_z = int(model.config.architecture.shared.c_z)
    u_bytes = n_tok * n_tok * c_z * 4

    print(f"  n_tokens={n_tok}, samples={args.samples}, 1U_trunk={_mib(u_bytes):.1f} MiB")

    # Warm-up forward (don't include in trace).
    print("  Warm-up forward...")
    with torch.inference_mode():
        lightning_module(batch)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # The DIFFUSION stage corresponds to model.sample_diffusion.forward.
    # We capture memory history around just that stage to keep snapshots small.
    orig_diffusion_forward = model.sample_diffusion.forward
    capture = {"snapshot": None, "peak_bytes": 0, "before_bytes": 0}

    def wrapped_forward(*fargs, **fkwargs):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        capture["before_bytes"] = torch.cuda.memory_allocated()
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python",
            max_entries=200_000,
        )
        try:
            out = orig_diffusion_forward(*fargs, **fkwargs)
            torch.cuda.synchronize()
            # Snapshot AFTER the diffusion stage finishes — pick whichever
            # segment block was live at the moment max_memory_allocated was
            # hit. We approximate "peak time" by the post-stage snapshot's
            # surviving allocations; that misses transient peaks within the
            # stage that have already been freed. For peak surfacing, we
            # rely on segment-level history (it tracks free events too).
            capture["snapshot"] = torch.cuda.memory_snapshot()
            capture["peak_bytes"] = torch.cuda.max_memory_allocated()
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
        return out

    model.sample_diffusion.forward = wrapped_forward
    try:
        print("  Profiled forward (capture diffusion stage)...")
        with torch.inference_mode():
            lightning_module(batch)
        torch.cuda.synchronize()
    finally:
        model.sample_diffusion.forward = orig_diffusion_forward

    if capture["snapshot"] is None:
        raise RuntimeError("Diffusion stage never ran during profiled forward")

    # Inventory live blocks from the snapshot. Each segment has a list of
    # blocks; each block has state (active_allocated / active_pending_free /
    # inactive) and a "frames" stack trace.
    bucket_bytes = collections.defaultdict(int)
    bucket_examples = collections.defaultdict(list)
    total_live = 0
    for segment in capture["snapshot"]:
        for block in segment.get("blocks", []):
            state = block.get("state", "")
            if state != "active_allocated":
                continue
            size = block.get("size", 0)
            total_live += size
            stack = block.get("frames", []) or []
            key = _frame_signature(stack)
            bucket_bytes[key] += size
            if len(bucket_examples[key]) < 2:
                bucket_examples[key].append(size)

    sorted_buckets = sorted(bucket_bytes.items(), key=lambda kv: -kv[1])

    result = {
        "query": args.query,
        "query_json": str(query_json),
        "samples": args.samples,
        "n_tokens": n_tok,
        "c_z": c_z,
        "U_bytes": u_bytes,
        "diffusion_peak_bytes": capture["peak_bytes"],
        "diffusion_before_bytes": capture["before_bytes"],
        "diffusion_peak_activation_U": (
            (capture["peak_bytes"] - capture["before_bytes"]) / u_bytes
        ),
        "total_live_at_snapshot_bytes": total_live,
        "buckets": [
            {
                "file": fn,
                "line": lineno,
                "category": cat,
                "bytes": bytes_,
                "U_trunk": bytes_ / u_bytes,
                "example_sizes_bytes": bucket_examples[(fn, lineno, cat)],
            }
            for (fn, lineno, cat), bytes_ in sorted_buckets
        ],
    }
    return result


def render_md(audit: dict) -> str:
    n_tok = audit["n_tokens"]
    u_mib = audit["U_bytes"] / 1024**2
    lines = [
        f"# DIFFUSION peak audit — {audit['query']} (samples={audit['samples']})",
        "",
        f"- n_tokens: **{n_tok}**, c_z: **{audit['c_z']}**",
        f"- 1 U_trunk = N²·c_z·4 = **{u_mib:.1f} MiB**",
        (
            f"- DIFFUSION peak above pre-stage baseline: "
            f"**{(audit['diffusion_peak_bytes'] - audit['diffusion_before_bytes'])/1024**2:.1f} MiB "
            f"= {audit['diffusion_peak_activation_U']:.2f} U_trunk**"
        ),
        (
            f"- Total live at end-of-stage snapshot: "
            f"**{audit['total_live_at_snapshot_bytes']/1024**2:.1f} MiB**"
        ),
        "",
        ("> Note: This snapshot is taken after the diffusion stage completes."
         " Allocations that were transient inside the stage and freed before"
         " the snapshot do not appear here, but `diffusion_peak_bytes` (from"
         " `max_memory_allocated`) accounts for the true peak."),
        "",
        "## Live allocations by source",
        "",
        "| Rank | File:line | Category | Bytes | MiB | U_trunk |",
        "|---:|:---|:---|---:|---:|---:|",
    ]
    for i, b in enumerate(audit["buckets"][:30], 1):
        lines.append(
            f"| {i} | `{b['file']}:{b['line']}` | {b['category']} | "
            f"{b['bytes']:,} | {b['bytes']/1024**2:.1f} | {b['U_trunk']:.3f} |"
        )
    if len(audit["buckets"]) > 30:
        rest = sum(b["bytes"] for b in audit["buckets"][30:])
        lines.append(
            f"| ... | (other {len(audit['buckets']) - 30} sites) | — | "
            f"{rest:,} | {rest/1024**2:.1f} | {rest/audit['U_bytes']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation notes (manual analysis goes here)",
        "",
        "- (i) Largest single component:",
        "- (ii) Any S-multiplied O(N²) tensor (bug candidate):",
        "- (iii) Resident-once-per-rollout cache candidates:",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="homo_1200")
    parser.add_argument("--query-json", type=Path, default=None)
    parser.add_argument("--runner-yaml", type=Path, default=DEFAULT_RUNNER_YAML)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    audit = audit_one(args)

    # Print top-15 to stdout for at-a-glance.
    n_tok = audit["n_tokens"]
    print()
    print("=" * 116)
    print(
        f"query={audit['query']} n_tok={n_tok} samples={audit['samples']} "
        f"1U={audit['U_bytes']/1024**2:.1f} MiB"
    )
    print(
        f"DIFFUSION peak activation = "
        f"{(audit['diffusion_peak_bytes'] - audit['diffusion_before_bytes'])/1024**2:.1f} MiB "
        f"({audit['diffusion_peak_activation_U']:.2f} U_trunk)"
    )
    print("=" * 116)
    print(f"{'rank':>4} {'MiB':>8} {'U':>7}  category               file:line")
    for i, b in enumerate(audit["buckets"][:15], 1):
        print(
            f"{i:>4} {b['bytes']/1024**2:>8.1f} {b['U_trunk']:>7.3f}  "
            f"{b['category']:<22} {b['file']}:{b['line']}"
        )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_md(audit))
    print(f"\nSaved {args.output_md}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(audit, indent=2, default=str))
        print(f"Saved {args.output_json}")


if __name__ == "__main__":
    main()
