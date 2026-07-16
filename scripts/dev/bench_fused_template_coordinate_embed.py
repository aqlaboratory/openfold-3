#!/usr/bin/env python
"""Benchmark precomputed versus coordinate-derived template embedding."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from openfold3.core.data.primitives.featurization.template import (
    create_template_distogram,
    create_template_unit_vector,
)


def _coordinates(n_token: int) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(n_token, dtype=torch.float32)
    ca = torch.stack(
        (index * 3.8, torch.sin(index * 0.03), torch.cos(index * 0.02)), dim=-1
    )
    pseudo_beta = ca + torch.tensor([0.2, -0.3, 0.7])
    frame = torch.stack(
        (
            ca + torch.tensor([-0.7, 1.1, 0.2]),
            ca,
            ca + torch.tensor([1.3, 0.2, -0.1]),
        ),
        dim=-2,
    )
    return pseudo_beta, frame


def _make_cpu_batches(n_token: int) -> tuple[dict, dict]:
    pseudo_beta, frame = _coordinates(n_token)
    mask = torch.ones(n_token)
    asym = torch.ones(1, n_token, dtype=torch.int32)
    restype = F.one_hot(torch.arange(n_token) % 32, num_classes=32).to(torch.int32)

    template_mask = mask[None]
    pair_mask = torch.ones(1, n_token, n_token, 1)
    dgram = create_template_distogram(
        pseudo_beta[None].numpy(), template_mask, pair_mask
    )
    unit = create_template_unit_vector(frame[None].numpy(), template_mask, pair_mask)

    common = {
        "template_restype": restype[None, None],
        "template_pseudo_beta_mask": mask[None, None],
        "template_backbone_frame_mask": mask[None, None],
        "asym_id": asym,
    }
    precomputed = {
        **common,
        "template_distogram": dgram[None],
        "template_unit_vector": unit[None],
    }
    coordinate = {
        **common,
        "template_pseudo_beta_coords": pseudo_beta[None, None],
        "template_frame_atom_coords": frame[None, None],
    }
    return precomputed, coordinate


def _to_cuda(batch: dict) -> dict:
    return {
        key: value.to("cuda", non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _payload_bytes(batch: dict) -> int:
    return sum(
        value.numel() * value.element_size()
        for key, value in batch.items()
        if key.startswith("template_")
    )


def _measure(fn, warmups: int, reps: int) -> tuple[float, int]:
    for _ in range(warmups):
        out = fn()
        del out
    torch.cuda.synchronize()

    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    transient = torch.cuda.max_memory_allocated() - baseline
    del out

    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        del out
    samples.sort()
    return samples[len(samples) // 2], transient


def _bench_length(n_token: int, warmups: int, reps: int) -> dict:
    from openfold3.core.model.feature_embedders.template_embedders import (
        TemplatePairEmbedderAllAtom,
    )
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_coordinate_pair_embedder_inference,
        fused_template_pair_embedder_inference,
    )

    torch.manual_seed(0)
    module = TemplatePairEmbedderAllAtom(128, 39, 32, 64).cuda().eval()
    precomputed_cpu, coordinate_cpu = _make_cpu_batches(n_token)
    precomputed_gpu = _to_cuda(precomputed_cpu)
    coordinate_gpu = _to_cuda(coordinate_cpu)
    z = torch.randn(1, n_token, n_token, 128, device="cuda")

    def legacy_compute():
        return fused_template_pair_embedder_inference(
            module, precomputed_gpu, z, template_index=0
        )

    def coordinate_compute():
        return fused_template_coordinate_pair_embedder_inference(
            module, coordinate_gpu, z, template_index=0
        )

    rows = {}
    with torch.inference_mode():
        os.environ["OPENFOLD3_FUSED_TEMPLATE_EMBED"] = "1"
        for name, fn in (
            ("legacy_compute", legacy_compute),
            ("coordinate_compute", coordinate_compute),
        ):
            median_ms, transient = _measure(fn, warmups, reps)
            rows[name] = {
                "median_ms": median_ms,
                "peak_transient_bytes": transient,
            }

    u_bytes = n_token * n_token * 128 * 4
    result = {
        "n_token": n_token,
        "U_bytes": u_bytes,
        "legacy_host_bytes": _payload_bytes(precomputed_cpu),
        "coordinate_host_bytes": _payload_bytes(coordinate_cpu),
        "host_reduction": (
            _payload_bytes(precomputed_cpu) / _payload_bytes(coordinate_cpu)
        ),
        **rows,
    }
    for value in rows.values():
        value["peak_transient_U"] = value["peak_transient_bytes"] / u_bytes
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[384, 590, 1264])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    rows = []
    for n_token in args.n:
        row = _bench_length(n_token, args.warmups, args.reps)
        rows.append(row)
        print(json.dumps(row, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
