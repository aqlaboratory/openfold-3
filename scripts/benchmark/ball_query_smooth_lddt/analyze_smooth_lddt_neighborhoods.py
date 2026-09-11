"""Measure 15/30 Angstrom atom-neighborhood sizes in structure NPZ files."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RADIUS_NON_NUCLEOTIDE = 15.0
RADIUS_NUCLEOTIDE = 30.0
NUCLEOTIDE_MOLECULE_TYPE_IDS = (1, 2)


def _histogram_quantile(histogram: Counter[int], quantile: float) -> int | None:
    total = histogram.total()
    if total == 0:
        return None
    rank = max(1, math.ceil(quantile * total))
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= rank:
            return value
    raise AssertionError("unreachable")


@dataclass
class NeighborhoodStats:
    centers: int = 0
    total_15: int = 0
    total_30: int = 0
    histogram_15: Counter[int] = field(default_factory=Counter)
    histogram_30: Counter[int] = field(default_factory=Counter)

    def update(self, counts_15: np.ndarray, counts_30: np.ndarray) -> None:
        self.centers += counts_15.size
        self.total_15 += int(counts_15.sum(dtype=np.int64))
        self.total_30 += int(counts_30.sum(dtype=np.int64))
        values_15, frequencies_15 = np.unique(counts_15, return_counts=True)
        values_30, frequencies_30 = np.unique(counts_30, return_counts=True)
        self.histogram_15.update(
            dict(zip(values_15.tolist(), frequencies_15.tolist(), strict=True))
        )
        self.histogram_30.update(
            dict(zip(values_30.tolist(), frequencies_30.tolist(), strict=True))
        )

    def mean(self, radius: int) -> float | None:
        if self.centers == 0:
            return None
        total = self.total_15 if radius == 15 else self.total_30
        return total / self.centers

    def _radius_summary(
        self, total: int, histogram: Counter[int]
    ) -> dict[str, float | int | None]:
        return {
            "total_ordered_pairs": total,
            "mean_neighbors_per_center": total / self.centers if self.centers else None,
            "p50_neighbors": _histogram_quantile(histogram, 0.50),
            "p90_neighbors": _histogram_quantile(histogram, 0.90),
            "p95_neighbors": _histogram_quantile(histogram, 0.95),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "center_atoms": self.centers,
            "neighbors_within_15_angstrom": self._radius_summary(
                self.total_15, self.histogram_15
            ),
            "neighbors_within_30_angstrom": self._radius_summary(
                self.total_30, self.histogram_30
            ),
            "same_center_30_to_15_pair_ratio": (
                self.total_30 / self.total_15 if self.total_15 else None
            ),
        }


def _neighbor_counts(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(coordinates)
    radius_15 = np.nextafter(RADIUS_NON_NUCLEOTIDE, 0.0)
    radius_30 = np.nextafter(RADIUS_NUCLEOTIDE, 0.0)
    counts_15 = tree.query_ball_point(
        coordinates, radius_15, return_length=True, workers=-1
    )
    counts_30 = tree.query_ball_point(
        coordinates, radius_30, return_length=True, workers=-1
    )
    return counts_15 - 1, counts_30 - 1


def analyze_structure_files(structure_files: list[Path]) -> dict[str, object]:
    groups = {
        "all": NeighborhoodStats(),
        "nucleotide": NeighborhoodStats(),
        "non_nucleotide": NeighborhoodStats(),
    }
    mixed_nucleotide = NeighborhoodStats()
    mixed_non_nucleotide = NeighborhoodStats()
    mixed_structure_count = 0
    atom_count = 0
    skipped_files: list[str] = []
    for structure_file in structure_files:
        with np.load(structure_file, allow_pickle=False) as structure:
            if "coord" not in structure or "molecule_type_id" not in structure:
                skipped_files.append(str(structure_file))
                continue
            coordinates = structure["coord"]
            molecule_types = structure["molecule_type_id"]

        resolved = np.isfinite(coordinates).all(axis=-1)
        coordinates = coordinates[resolved]
        molecule_types = molecule_types[resolved]
        if coordinates.size == 0:
            skipped_files.append(str(structure_file))
            continue

        counts_15, counts_30 = _neighbor_counts(coordinates)
        is_nucleotide = np.isin(molecule_types, NUCLEOTIDE_MOLECULE_TYPE_IDS)
        groups["all"].update(counts_15, counts_30)
        groups["nucleotide"].update(counts_15[is_nucleotide], counts_30[is_nucleotide])
        groups["non_nucleotide"].update(
            counts_15[~is_nucleotide], counts_30[~is_nucleotide]
        )
        if is_nucleotide.any() and (~is_nucleotide).any():
            mixed_structure_count += 1
            mixed_nucleotide.update(counts_15[is_nucleotide], counts_30[is_nucleotide])
            mixed_non_nucleotide.update(
                counts_15[~is_nucleotide], counts_30[~is_nucleotide]
            )
        atom_count += coordinates.shape[0]

    nucleotide_mean_30 = mixed_nucleotide.mean(30)
    non_nucleotide_mean_15 = mixed_non_nucleotide.mean(15)
    suggested_scale = (
        nucleotide_mean_30 / non_nucleotide_mean_15
        if nucleotide_mean_30 is not None and non_nucleotide_mean_15 not in (None, 0.0)
        else None
    )
    return {
        "structure_files_analyzed": len(structure_files) - len(skipped_files),
        "resolved_atoms_analyzed": atom_count,
        "skipped_files": skipped_files,
        "groups": {name: stats.as_dict() for name, stats in groups.items()},
        "mixed_structure_scale_estimate": {
            "mixed_structure_count": mixed_structure_count,
            "nucleotide": mixed_nucleotide.as_dict(),
            "non_nucleotide": mixed_non_nucleotide.as_dict(),
            "nucleotide_30_to_non_nucleotide_15_mean_neighbor_ratio": suggested_scale,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "structure_directory",
        type=Path,
        help="Directory recursively containing preprocessed structure NPZ files.",
    )
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    structure_files = sorted(args.structure_directory.rglob("*.npz"))
    if args.max_structures is not None:
        structure_files = structure_files[: args.max_structures]
    if not structure_files:
        parser.error(f"No NPZ files found under {args.structure_directory}")

    result = analyze_structure_files(structure_files)
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
