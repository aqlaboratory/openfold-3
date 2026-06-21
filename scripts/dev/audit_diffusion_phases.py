#!/usr/bin/env python
"""Decompose DIFFUSION peak: which sub-phase actually drives it?

Hooks each phase of `sample_diffusion.forward` with reset_peak between:
  1. prepare_diffusion_conditioning_cache  (one-time, before rollout)
  2. prepare_atom_rep_cache                 (one-time, before rollout)
  3. diffusion_module.forward — first call  (one rollout step)
  4. diffusion_module.forward — typical 5th call

This tells us:
- How much memory is RESIDENT after the one-time setup (caches).
- How much memory the PER-STEP transient adds on top.
- Whether the first step has higher transient than later steps
  (often does because cuBLAS/cuEq autotune fires there).
"""
from __future__ import annotations

import argparse
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


def build(query_json: Path, samples: int):
    runner_args = config_utils.load_yaml(REPO / "examples/example_runner_yamls/cuequivariance.yml")
    runner_args.setdefault("data_module_args", {})
    runner_args["data_module_args"]["num_workers"] = 0
    expt = InferenceExperimentConfig(**runner_args)
    runner = InferenceExperimentRunner(expt, num_diffusion_samples=samples, use_msa_server=False)
    cfg = runner.model_config
    cfg.settings.memory.eval.offload_inference.token_cutoff = 10_000_000
    cfg.settings.memory.eval.use_cueq_triangle_kernels = True
    runner.setup()
    runner.inference_query_set = InferenceQuerySet.from_json(query_json)
    return runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-json", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    runner = build(args.query_json, args.samples)
    lm = runner.lightning_module.to("cuda").eval()
    model = lm.model

    dm = runner.lightning_data_module
    dm.prepare_data()
    dm.setup()
    batch = None
    for b in dm.predict_dataloader():
        if b.get("valid_sample") and not b.get("repeated_sample"):
            batch = tensor_tree_map(lambda t: t.to("cuda"), b)
            break

    n_tok = int(batch["token_mask"].shape[-1])
    c_z = int(model.config.architecture.shared.c_z)
    u_bytes = n_tok * n_tok * c_z * 4
    print(f"n_tok={n_tok} samples={args.samples} 1U={u_bytes/1024**2:.1f} MiB")

    # Warm-up.
    print("\nWarm-up...")
    with torch.inference_mode():
        lm(batch)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Hook each phase of sample_diffusion.
    sd = model.sample_diffusion
    dm_mod = model.sample_diffusion.diffusion_module
    orig_prep_cond = dm_mod.prepare_diffusion_conditioning_cache
    orig_prep_atom = dm_mod.prepare_atom_rep_cache
    orig_diff_fwd = dm_mod.forward

    call_count = {"diff": 0}
    measurements = []

    def measure(name, fn, *args, **kwargs):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        after = torch.cuda.memory_allocated()
        transient = peak - before
        resident = after - before
        measurements.append({
            "name": name,
            "before_mib": before / 1024**2,
            "after_mib": after / 1024**2,
            "peak_mib": peak / 1024**2,
            "transient_mib": transient / 1024**2,
            "resident_added_mib": resident / 1024**2,
            "transient_U": transient / u_bytes,
            "resident_added_U": resident / u_bytes,
        })
        return out

    def hooked_prep_cond(*a, **kw):
        return measure("prepare_diffusion_conditioning_cache", orig_prep_cond, *a, **kw)

    def hooked_prep_atom(*a, **kw):
        return measure("prepare_atom_rep_cache", orig_prep_atom, *a, **kw)

    def hooked_diff_fwd(*a, **kw):
        call_count["diff"] += 1
        n = call_count["diff"]
        if n in (1, 2, 5, 100):
            return measure(f"diffusion_module.forward (step #{n})", orig_diff_fwd, *a, **kw)
        return orig_diff_fwd(*a, **kw)

    dm_mod.prepare_diffusion_conditioning_cache = hooked_prep_cond
    dm_mod.prepare_atom_rep_cache = hooked_prep_atom
    dm_mod.forward = hooked_diff_fwd

    try:
        # Wrap the WHOLE sample_diffusion to also catch its overall peak.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        before_sd = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            lm(batch)
        torch.cuda.synchronize()
        overall_peak = torch.cuda.max_memory_allocated()
        print(f"\nOverall peak: {overall_peak/1024**2:.1f} MiB "
              f"= {(overall_peak - before_sd)/u_bytes:.2f}U above pre-DIFFUSION baseline")
    finally:
        dm_mod.prepare_diffusion_conditioning_cache = orig_prep_cond
        dm_mod.prepare_atom_rep_cache = orig_prep_atom
        dm_mod.forward = orig_diff_fwd

    print()
    print(f"{'phase':<50}{'before':>9}{'peak':>9}{'after':>9}"
          f"{'transient_MiB':>15}{'trans_U':>9}{'resident_add_U':>16}")
    print("-" * 117)
    for m in measurements:
        print(
            f"{m['name']:<50}{m['before_mib']:>9.1f}{m['peak_mib']:>9.1f}{m['after_mib']:>9.1f}"
            f"{m['transient_mib']:>15.1f}{m['transient_U']:>9.3f}{m['resident_added_U']:>16.3f}"
        )


if __name__ == "__main__":
    main()
