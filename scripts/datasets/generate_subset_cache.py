#!/usr/bin/env python3
"""Sample a small PDB training subset cache + fetch the pinned validation set.

Downloads the full (un-sampled) caches from S3 first if not already present
locally. Writes training_cache_with_templates_subset_{train_size}.json
(randomly sampled) and validation_cache_with_templates_subset_{N}.json (the
fixed set in pdb_subset_helpers.SMOKE_VALIDATION_PDB_IDS -- see
select_smoke_validation_set.py for how that set was chosen and why it's
pinned rather than randomly sampled) under --target-dir (default: ./datasets
next to wherever this is invoked from), plus a run_openfold training runner
yaml pointed at them.

Does not download any of the structure/alignment/template/reference-mol
files the subset references -- see download_subset.py for that.

Usage:
    python generate_subset_cache.py
    python generate_subset_cache.py --target-dir /data/of3-subset
    python generate_subset_cache.py --train-size 32 --seed 1234
    python generate_subset_cache.py --force  # regenerate even if a cache already exists
"""

import argparse
from pathlib import Path

from pdb_subset_helpers import (
    SMOKE_VALIDATION_PDB_IDS,
    build_pinned_subset_cache,
    default_target_dir,
    download_full_cache,
    sample_subset_cache,
    subset_cache_path,
    write_runner_yaml,
)


def get_or_create_subset_cache(
    full_cache: Path, size: int, seed: int, target_dir: Path, force: bool = False
) -> Path:
    """Return the path to a size-N random subset cache, generating it if missing.

    If the full cache itself isn't present locally either, it's downloaded
    from S3 first. If `force`, the subset is resampled even if a cache for
    this size already exists.
    """
    subset_path = subset_cache_path(full_cache, size, target_dir)
    if subset_path.exists() and not force:
        return subset_path

    download_full_cache(full_cache)
    print(f"Sampling {size} structures from {full_cache.name} with seed={seed}...")
    sample_subset_cache(full_cache, [size], target_dir, seed=seed)
    return subset_path


def get_or_create_pinned_validation_cache(
    full_cache: Path, target_dir: Path, force: bool = False
) -> Path:
    """Return the path to the pinned validation subset cache, building it if missing.

    Always the fixed set in `SMOKE_VALIDATION_PDB_IDS`, not a random sample --
    see that constant's docstring and select_smoke_validation_set.py for why.
    """
    subset_path = subset_cache_path(
        full_cache, len(SMOKE_VALIDATION_PDB_IDS), target_dir
    )
    if subset_path.exists() and not force:
        return subset_path

    download_full_cache(full_cache)
    print(
        f"Fetching pinned validation set from {full_cache.name}:"
        f"{SMOKE_VALIDATION_PDB_IDS}"
    )
    return build_pinned_subset_cache(full_cache, SMOKE_VALIDATION_PDB_IDS, target_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for the subset cache(s) and runner yaml, and "
            "(via download_subset.py) the downloaded files -- default: "
            "./datasets next to wherever this script is invoked from."
        ),
    )
    parser.add_argument(
        "--train-size", type=int, default=8, help="Train subset size (default: 8)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for train subset sampling"
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=None,
        help=(
            "Full training cache to sample from (default: "
            "<target-dir>/training_cache_with_templates.json)"
        ),
    )
    parser.add_argument(
        "--val-cache",
        type=Path,
        default=None,
        help=(
            "Full validation cache the pinned validation set is fetched from "
            "(default: <target-dir>/validation_cache_with_templates.json)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if a cache for the requested size already exists",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory download_subset.py will download files into -- baked "
            "into the generated runner yaml (default: <target-dir>/pdb_training_set)"
        ),
    )
    parser.add_argument(
        "--runner-yaml",
        type=str,
        default=None,
        help=(
            "Where to write the run_openfold training runner yaml (default: "
            "<target-dir>/train_pdb_subset.yaml). Pass an empty string to "
            "skip writing it."
        ),
    )
    args = parser.parse_args()

    target_dir = args.target_dir or default_target_dir()
    train_cache = args.train_cache or (
        target_dir / "training_cache_with_templates.json"
    )
    val_cache = args.val_cache or (target_dir / "validation_cache_with_templates.json")
    output_dir = args.output_dir or (target_dir / "pdb_training_set")
    runner_yaml = (
        str(target_dir / "train_pdb_subset.yaml")
        if args.runner_yaml is None
        else args.runner_yaml
    )

    cache_files = {
        "train": get_or_create_subset_cache(
            train_cache, args.train_size, args.seed, target_dir, force=args.force
        ),
        "val": get_or_create_pinned_validation_cache(
            val_cache, target_dir, force=args.force
        ),
    }

    if runner_yaml:
        write_runner_yaml(Path(runner_yaml), cache_files, output_dir)


if __name__ == "__main__":
    main()
