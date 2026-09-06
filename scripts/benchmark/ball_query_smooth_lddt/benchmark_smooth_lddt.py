"""Benchmark dense and ball-query smooth-lDDT implementations on point clouds."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from openfold3.core.kernels.triton.smooth_lddt_ball_query import (
    DEFAULT_RADIUS_NUCLEOTIDE,
    DEFAULT_RADIUS_PROTEIN,
    _checkpointed_type_group_sums,
    _finish_weighted_lddt,
    _legacy_ball_query_smooth_lddt_loss,
    _prepare_sparse_inputs,
    _query_group_with_pred,
    _type_aware_ball_query_smooth_lddt_loss,
    _type_group_indices,
    _validate_top_k,
    ball_query_smooth_lddt_loss,
)
from openfold3.core.loss.diffusion import smooth_lddt_loss

EPS = 1e-8
BENCHMARK_BACKENDS = (
    "dense",
    "ball_query_unweighted",
    "ball_query",
)
OPTIONAL_BACKENDS = (
    "ball_query_legacy",
    "ball_query_type_aware",
)


def _dense_batch(
    x_gt: torch.Tensor, atom_mask: torch.Tensor, is_nucleotide: torch.Tensor
) -> dict:
    n_atom = x_gt.shape[-2]
    return {
        "ground_truth": {
            "atom_positions": x_gt,
            "atom_resolved_mask": atom_mask,
        },
        "token_mask": torch.ones((1, n_atom), device=x_gt.device),
        "num_atoms_per_token": torch.ones(
            (1, n_atom), device=x_gt.device, dtype=torch.long
        ),
        "is_dna": is_nucleotide,
        "is_rna": torch.zeros_like(is_nucleotide),
    }


def _matched_unweighted_ball_query_smooth_lddt_loss(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask_gt: torch.Tensor,
    is_nucleotide: torch.Tensor,
    loss_atom_mask: torch.Tensor,
    eps: float,
    top_k: int | None = 512,
    seed: int = 0,
    radius_protein: float = DEFAULT_RADIUS_PROTEIN,
    radius_nucleotide: float = DEFAULT_RADIUS_NUCLEOTIDE,
) -> torch.Tensor:
    """Run the current ball query and reduction without row reweighting."""
    k = _validate_top_k(top_k, x.shape[-2])
    if k == 0:
        return torch.ones(x.shape[:-2], device=x.device, dtype=x.dtype) + x.sum() * 0
    flat_x, flat_x_gt, valid_atom, flat_is_nucleotide = _prepare_sparse_inputs(
        x, x_gt, atom_mask_gt, is_nucleotide, loss_atom_mask
    )
    group = _query_group_with_pred(
        flat_x,
        flat_x_gt,
        valid_atom,
        flat_is_nucleotide,
        valid_atom,
        k,
        radius_protein,
        radius_nucleotide,
        seed,
    )
    type_group_index = _type_group_indices(group[3], group[5], k)
    score_by_type, count_by_type = _checkpointed_type_group_sums(
        group[0],
        group[1],
        group[2],
        group[5],
        type_group_index,
        eps,
        radius_protein,
        radius_nucleotide,
    )
    score_sum = score_by_type.sum(dim=-1)
    pair_count = count_by_type.sum(dim=-1)
    return _finish_weighted_lddt(score_sum, pair_count, valid_atom, eps, x.shape[:-2])


def _calculate_loss(
    name: str,
    x: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
    seed: int,
    top_k: int,
    nucleotide_scale: float,
) -> torch.Tensor:
    if name == "dense":
        return smooth_lddt_loss(
            x=x,
            batch=_dense_batch(x_gt, atom_mask, is_nucleotide),
            loss_token_mask=atom_mask,
            eps=EPS,
        )

    kwargs = {
        "x": x,
        "x_gt": x_gt,
        "atom_mask_gt": atom_mask,
        "is_nucleotide": is_nucleotide,
        "loss_atom_mask": atom_mask,
        "eps": EPS,
        "seed": seed,
    }
    if name == "ball_query_unweighted":
        return _matched_unweighted_ball_query_smooth_lddt_loss(**kwargs, top_k=top_k)
    if name == "ball_query":
        return ball_query_smooth_lddt_loss(**kwargs, top_k=top_k)
    if name == "ball_query_legacy":
        return _legacy_ball_query_smooth_lddt_loss(**kwargs, top_k=top_k)
    if name == "ball_query_type_aware":
        return _type_aware_ball_query_smooth_lddt_loss(
            **kwargs, top_k=top_k, nucleotide_scale=nucleotide_scale
        )
    raise ValueError(name)


def _run_backend(
    name: str,
    x_base: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
    seed: int,
    top_k: int,
    nucleotide_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_base.detach().clone().requires_grad_(True)
    loss = _calculate_loss(
        name,
        x,
        x_gt,
        atom_mask,
        is_nucleotide,
        seed,
        top_k,
        nucleotide_scale,
    )
    loss.sum().backward()
    return loss.detach(), x.grad.detach()


@torch.inference_mode()
def _run_inference_backend(
    name: str,
    x_base: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
    seed: int,
    top_k: int,
    nucleotide_scale: float,
) -> torch.Tensor:
    x = x_base.detach().clone()
    return _calculate_loss(
        name,
        x,
        x_gt,
        atom_mask,
        is_nucleotide,
        seed,
        top_k,
        nucleotide_scale,
    ).detach()


def _timed_run(
    name: str,
    x: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
    warmup: int,
    repeats: int,
    top_k: int,
    nucleotide_scale: float,
) -> tuple[float, int]:
    for _ in range(warmup):
        _run_backend(
            name,
            x,
            x_gt,
            atom_mask,
            is_nucleotide,
            seed=0,
            top_k=top_k,
            nucleotide_scale=nucleotide_scale,
        )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(repeats):
        start.record()
        _run_backend(
            name,
            x,
            x_gt,
            atom_mask,
            is_nucleotide,
            seed=0,
            top_k=top_k,
            nucleotide_scale=nucleotide_scale,
        )
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))
    peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    return float(np.median(elapsed)), int(peak)


def _timed_inference_run(
    name: str,
    x: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask: torch.Tensor,
    is_nucleotide: torch.Tensor,
    warmup: int,
    repeats: int,
    top_k: int,
    nucleotide_scale: float,
) -> tuple[float, int]:
    for _ in range(warmup):
        _run_inference_backend(
            name,
            x,
            x_gt,
            atom_mask,
            is_nucleotide,
            seed=0,
            top_k=top_k,
            nucleotide_scale=nucleotide_scale,
        )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed = []
    for _ in range(repeats):
        start.record()
        _run_inference_backend(
            name,
            x,
            x_gt,
            atom_mask,
            is_nucleotide,
            seed=0,
            top_k=top_k,
            nucleotide_scale=nucleotide_scale,
        )
        end.record()
        torch.cuda.synchronize()
        elapsed.append(start.elapsed_time(end))
    peak = max(0, torch.cuda.max_memory_allocated() - baseline)
    return float(np.median(elapsed)), int(peak)


def _gradient_metrics(
    gradient: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float, float]:
    delta = gradient.float() - reference.float()
    relative_l2 = delta.norm() / reference.float().norm().clamp_min(EPS)
    cosine = torch.nn.functional.cosine_similarity(
        gradient.float().flatten(), reference.float().flatten(), dim=0
    )
    return float(relative_l2), float(cosine), float(delta.abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--reservoir-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--nucleotide-scale", type=float, default=4.0)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BENCHMARK_BACKENDS + OPTIONAL_BACKENDS,
        default=list(BENCHMARK_BACKENDS),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    output_paths = (args.output, args.output.with_suffix(".csv"))
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs:
        parser.error(
            "Refusing to overwrite existing output: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text())
    if args.max_cases is not None:
        manifest = manifest[: args.max_cases]

    backends = args.backends
    rows: list[dict[str, object]] = []
    for case in manifest:
        data = np.load(args.dataset_dir / case["file"])
        device = torch.device("cuda")
        x_gt = torch.from_numpy(data["x_gt"]).to(device)[None]
        atom_mask = torch.from_numpy(data["atom_mask"]).to(device)[None].float()
        is_nucleotide = torch.from_numpy(data["is_nucleotide"]).to(device)[None].float()
        for noise_index, noise in enumerate(data["noise_levels"]):
            x = torch.from_numpy(data["x_pred"][noise_index]).to(device)[None]
            try:
                reference_loss, reference_grad = _run_backend(
                    "dense",
                    x,
                    x_gt,
                    atom_mask,
                    is_nucleotide,
                    seed=0,
                    top_k=args.top_k,
                    nucleotide_scale=args.nucleotide_scale,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                reference_loss = reference_grad = None

            for backend in backends:
                seeds = [0] if backend == "dense" else args.reservoir_seeds
                for seed in seeds:
                    started = time.time()
                    loss_value = dense_loss_value = None
                    try:
                        loss, gradient = _run_backend(
                            backend,
                            x,
                            x_gt,
                            atom_mask,
                            is_nucleotide,
                            seed,
                            args.top_k,
                            args.nucleotide_scale,
                        )
                        loss_value = float(loss.item())
                        dense_loss_value = (
                            float(reference_loss.item())
                            if reference_loss is not None
                            else None
                        )
                        if reference_loss is None:
                            loss_error = grad_l2 = grad_cosine = grad_max = None
                        else:
                            loss_error = float((loss - reference_loss).abs().max())
                            grad_l2, grad_cosine, grad_max = _gradient_metrics(
                                gradient, reference_grad
                            )
                        runtime_ms, peak_memory = _timed_run(
                            backend,
                            x,
                            x_gt,
                            atom_mask,
                            is_nucleotide,
                            args.warmup,
                            args.repeats,
                            args.top_k,
                            args.nucleotide_scale,
                        )
                        (
                            inference_runtime_ms,
                            inference_peak_memory,
                        ) = _timed_inference_run(
                            backend,
                            x,
                            x_gt,
                            atom_mask,
                            is_nucleotide,
                            args.warmup,
                            args.repeats,
                            args.top_k,
                            args.nucleotide_scale,
                        )
                        error = None
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        loss_error = grad_l2 = grad_cosine = grad_max = None
                        runtime_ms = peak_memory = None
                        inference_runtime_ms = inference_peak_memory = None
                        error = "CUDA out of memory"
                    rows.append(
                        {
                            **case,
                            "noise": float(noise),
                            "backend": backend,
                            "reservoir_seed": seed,
                            "top_k": args.top_k,
                            "nucleotide_scale": args.nucleotide_scale,
                            "loss": loss_value,
                            "dense_loss": dense_loss_value,
                            "loss_error": loss_error,
                            "gradient_relative_l2": grad_l2,
                            "gradient_cosine": grad_cosine,
                            "gradient_max_abs": grad_max,
                            "runtime_ms": runtime_ms,
                            "peak_memory_bytes": peak_memory,
                            "inference_runtime_ms": inference_runtime_ms,
                            "inference_peak_memory_bytes": inference_peak_memory,
                            "wall_seconds": time.time() - started,
                            "error": error,
                        }
                    )
                    print(json.dumps(rows[-1], sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
