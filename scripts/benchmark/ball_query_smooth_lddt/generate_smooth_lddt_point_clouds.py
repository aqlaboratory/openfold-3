"""Generate deterministic point clouds for smooth-lDDT backend benchmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PROTEIN_RADIUS = 15.0
NUCLEOTIDE_RADIUS = 30.0
PROTEIN_NEIGHBORS = 512
NUCLEOTIDE_NEIGHBORS = 2048


def _uniform_cloud(
    rng: np.random.Generator,
    n_atom: int,
    radius: float,
    target_neighbors: int,
) -> np.ndarray:
    target = max(1, min(target_neighbors, n_atom - 1))
    density = target / ((4.0 / 3.0) * math.pi * radius**3)
    side = (n_atom / density) ** (1.0 / 3.0)
    return rng.uniform(0.0, side, size=(n_atom, 3)).astype(np.float32)


def generate_case(
    profile: str,
    n_atom: int,
    seed: int,
    noise_levels: tuple[float, ...],
    resolved_fraction: float,
) -> dict[str, np.ndarray]:
    """Generate one controlled cloud and noisy predictions."""
    rng = np.random.default_rng(seed)
    if profile == "protein_only":
        x_gt = _uniform_cloud(rng, n_atom, PROTEIN_RADIUS, PROTEIN_NEIGHBORS)
        is_nucleotide = np.zeros(n_atom, dtype=np.bool_)
    elif profile == "nucleotide_only":
        x_gt = _uniform_cloud(rng, n_atom, NUCLEOTIDE_RADIUS, NUCLEOTIDE_NEIGHBORS)
        is_nucleotide = np.ones(n_atom, dtype=np.bool_)
    elif profile in {"mixed_separated", "overflow"}:
        n_protein = n_atom // 2
        n_nucleotide = n_atom - n_protein
        multiplier = 1.5 if profile == "overflow" else 1.0
        protein = _uniform_cloud(
            rng,
            n_protein,
            PROTEIN_RADIUS,
            int(PROTEIN_NEIGHBORS * multiplier),
        )
        nucleotide = _uniform_cloud(
            rng,
            n_nucleotide,
            NUCLEOTIDE_RADIUS,
            int(NUCLEOTIDE_NEIGHBORS * multiplier),
        )
        nucleotide[:, 0] += protein[:, 0].max(initial=0.0) + 65.0
        x_gt = np.concatenate((protein, nucleotide))
        is_nucleotide = np.zeros(n_atom, dtype=np.bool_)
        is_nucleotide[n_protein:] = True
    elif profile == "mixed_overlap":
        x_gt = _uniform_cloud(rng, n_atom, PROTEIN_RADIUS, PROTEIN_NEIGHBORS)
        is_nucleotide = np.zeros(n_atom, dtype=np.bool_)
        is_nucleotide[rng.choice(n_atom, n_atom // 2, replace=False)] = True
    else:
        raise ValueError(f"Unknown profile {profile!r}")

    atom_mask = rng.random(n_atom) < resolved_fraction
    predictions = np.stack(
        [
            x_gt + rng.normal(0.0, noise, x_gt.shape).astype(np.float32)
            for noise in noise_levels
        ]
    )
    return {
        "x_gt": x_gt,
        "x_pred": predictions,
        "is_nucleotide": is_nucleotide,
        "atom_mask": atom_mask,
        "loss_atom_mask": atom_mask.copy(),
        "noise_levels": np.asarray(noise_levels, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-atoms", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--noise", type=float, nargs="+", default=[0.1, 1.0, 3.0])
    parser.add_argument("--resolved-fraction", type=float, default=0.9)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=[
            "protein_only",
            "nucleotide_only",
            "mixed_separated",
            "mixed_overlap",
            "overflow",
        ],
    )
    args = parser.parse_args()
    if not 0.0 < args.resolved_fraction <= 1.0:
        parser.error("--resolved-fraction must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for profile in args.profiles:
        for n_atom in args.n_atoms:
            for seed in args.seeds:
                name = f"{profile}_n{n_atom}_seed{seed}.npz"
                data = generate_case(
                    profile=profile,
                    n_atom=n_atom,
                    seed=seed,
                    noise_levels=tuple(args.noise),
                    resolved_fraction=args.resolved_fraction,
                )
                np.savez_compressed(args.output_dir / name, **data)
                manifest.append(
                    {
                        "file": name,
                        "profile": profile,
                        "n_atom": n_atom,
                        "seed": seed,
                        "resolved_fraction": args.resolved_fraction,
                    }
                )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
