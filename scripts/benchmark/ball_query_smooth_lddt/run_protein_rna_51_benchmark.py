"""Generate and benchmark the exact 51 protein-RNA structure cases.

Pull the public OpenFold3 assets first with::

    python -m scripts.benchmark.ball_query_smooth_lddt.pull_protein_rna_51_cases

Then run this module from the repository root. Existing non-empty case and
result paths are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.benchmark.ball_query_smooth_lddt.pull_protein_rna_51_cases import (
    PDB_IDS,
)


def _run(command: list[str], repository_root: Path, env: dict[str, str]) -> None:
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=repository_root, env=env, check=True)


def _validate_cases(case_dir: Path) -> None:
    manifest_path = case_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Case manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    structure_ids = [case["structure_id"] for case in manifest]
    expected = set(PDB_IDS)
    actual = set(structure_ids)
    if len(structure_ids) != 51 or len(actual) != 51 or actual != expected:
        raise ValueError(
            "Case membership mismatch: "
            f"rows={len(structure_ids)}, unique={len(actual)}, "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    cropped = [
        case["structure_id"]
        for case in manifest
        if case["n_atom"] != case["original_n_atom"]
    ]
    if cropped:
        raise ValueError(
            f"Expected full structures, but these cases were cropped: {cropped}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        default=Path("datasets/validation_cache_with_templates_protein_rna_51.json"),
    )
    parser.add_argument(
        "--structure-dir",
        type=Path,
        default=Path(
            "datasets/pdb_training_set/preprocessed_pdb_data/standard/structure_files"
        ),
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("outputs/smooth_lddt_benchmark/protein_rna_51_cases_full"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/smooth_lddt_benchmark"),
    )
    parser.add_argument("--output-prefix", default="protein_rna_51_strict_dynamic")
    parser.add_argument("--top-k", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--reservoir-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--case-seed", type=int, default=20260804)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them."
    )
    args = parser.parse_args()

    if any(top_k <= 0 for top_k in args.top_k):
        parser.error("--top-k values must be positive")
    if len(set(args.top_k)) != len(args.top_k):
        parser.error("--top-k values must be unique")

    repository_root = Path(__file__).resolve().parents[3]
    module_root = "scripts.benchmark.ball_query_smooth_lddt"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    manifest_path = args.case_dir / "manifest.json"
    if not manifest_path.exists():
        command = [
            sys.executable,
            "-u",
            "-m",
            f"{module_root}.generate_smooth_lddt_structure_cases",
            "--structure-dir",
            str(args.structure_dir),
            "--output-dir",
            str(args.case_dir),
            "--dataset-cache",
            str(args.dataset_cache),
            "--no-crop",
            "--noise",
            str(args.noise),
            "--seed",
            str(args.case_seed),
        ]
        if args.dry_run:
            print(shlex.join(command))
        else:
            _run(command, repository_root, env)

    if not args.dry_run:
        _validate_cases(args.case_dir)

    for top_k in args.top_k:
        output = args.output_dir / f"{args.output_prefix}_k{top_k}.json"
        command = [
            sys.executable,
            "-u",
            "-m",
            f"{module_root}.benchmark_smooth_lddt",
            "--dataset-dir",
            str(args.case_dir),
            "--output",
            str(output),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--reservoir-seeds",
            *(str(seed) for seed in args.reservoir_seeds),
            "--top-k",
            str(top_k),
        ]
        if args.dry_run:
            print(shlex.join(command))
        else:
            _run(command, repository_root, env)


if __name__ == "__main__":
    main()
