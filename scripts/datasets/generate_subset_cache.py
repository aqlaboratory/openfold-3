#!/usr/bin/env python3
"""Sample a small PDB training/validation subset cache from the full caches.

Downloads the full (un-sampled) caches from S3 first if not already present
locally. Writes training_cache_with_templates_subset_{train_size}.json /
validation_cache_with_templates_subset_{val_size}.json next to this script,
plus a run_openfold training runner yaml pointed at them.

Does not download any of the structure/alignment/template/reference-mol
files the subset references -- see download_subset.py for that.

Usage:
    python generate_subset_cache.py
    python generate_subset_cache.py --train-size 32 --val-size 16 --seed 1234
    python generate_subset_cache.py --force  # resample even if a cache already exists
"""

import argparse
from pathlib import Path

from pdb_subset_helpers import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUNNER_YAML,
    DEFAULT_TRAIN_CACHE,
    DEFAULT_VAL_CACHE,
    ROOT_DIR,
    download_full_cache,
    sample_subset_cache,
    subset_cache_path,
    write_runner_yaml,
)


def get_or_create_subset_cache(
    full_cache: Path, size: int, seed: int, force: bool = False
) -> Path:
    """Return the path to a size-N subset cache, generating it if missing.

    If the full cache itself isn't present locally either, it's downloaded
    from S3 first. If `force`, the subset is resampled even if a cache for
    this size already exists.
    """
    subset_path = subset_cache_path(full_cache, size)
    if subset_path.exists() and not force:
        return subset_path

    download_full_cache(full_cache)
    print(f"Sampling {size} structures from {full_cache.name} with seed={seed}...")
    sample_subset_cache(full_cache, [size], ROOT_DIR, seed=seed)
    return subset_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-size", type=int, default=8, help="Train subset size (default: 8)"
    )
    parser.add_argument(
        "--val-size", type=int, default=4, help="Validation subset size (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subset sampling"
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=DEFAULT_TRAIN_CACHE,
        help=f"Full training cache to sample from (default: {DEFAULT_TRAIN_CACHE})",
    )
    parser.add_argument(
        "--val-cache",
        type=Path,
        default=DEFAULT_VAL_CACHE,
        help=f"Full validation cache to sample from (default: {DEFAULT_VAL_CACHE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Resample even if a cache for the requested size already exists",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory download_subset.py will download files into -- baked "
            f"into the generated runner yaml (default: {DEFAULT_OUTPUT_DIR})"
        ),
    )
    parser.add_argument(
        "--runner-yaml",
        type=str,
        default=str(DEFAULT_RUNNER_YAML),
        help=(
            "Where to write the run_openfold training runner yaml (default: "
            f"{DEFAULT_RUNNER_YAML}). Pass an empty string to skip writing it."
        ),
    )
    args = parser.parse_args()

    cache_files = {
        "train": get_or_create_subset_cache(
            args.train_cache, args.train_size, args.seed, force=args.force
        ),
        "val": get_or_create_subset_cache(
            args.val_cache, args.val_size, args.seed, force=args.force
        ),
    }

    if args.runner_yaml:
        write_runner_yaml(Path(args.runner_yaml), cache_files, args.output_dir)


if __name__ == "__main__":
    main()
