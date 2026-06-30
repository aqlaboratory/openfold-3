#!/usr/bin/env python3
"""Download the minimal PDB benchmark dataset for smooth lDDT ball-query tests.

Downloads 6 preprocessed NPZ structure files from the public OpenFold3 S3
bucket, covering a range of atom counts from ~450 to ~7500. If you already have
the full PDB training set locally, pass --training-set-dir to symlink instead
of downloading.

Usage:
    # Download from S3
    python scripts/download_smooth_lddt_benchmark_data.py

    # Symlink from existing training set
    python scripts/download_smooth_lddt_benchmark_data.py \
        --training-set-dir /shared/openfold3/pdb_training_set
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openfold3.core.utils.s3 import download_s3_file, s3_file_matches_local

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

S3_BUCKET = "openfold3-data"
S3_PREFIX = "pdb_training_set/preprocessed_pdb_data/standard/structure_files"

SAMPLE_IDS = ["102m", "12e8", "134d", "17ra", "1tii", "1xfy"]

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "pdb_training_minimal"
    / "preprocessed_pdb_data"
    / "standard"
    / "structure_files"
)


def download_from_s3(output_dir: Path) -> None:
    for sample_id in SAMPLE_IDS:
        s3_key = f"{S3_PREFIX}/{sample_id}/{sample_id}.npz"
        local_path = output_dir / sample_id / f"{sample_id}.npz"

        if local_path.exists() and s3_file_matches_local(local_path, S3_BUCKET, s3_key):
            logger.info(f"  {sample_id}.npz — already up to date, skipping")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        download_s3_file(S3_BUCKET, s3_key, local_path)
        logger.info(f"  {sample_id}.npz — downloaded")


def symlink_from_training_set(training_set_dir: Path, output_dir: Path) -> None:
    structure_dir = (
        training_set_dir / "preprocessed_pdb_data" / "standard" / "structure_files"
    )
    if not structure_dir.is_dir():
        logger.error(f"Structure files directory not found: {structure_dir}")
        sys.exit(1)

    for sample_id in SAMPLE_IDS:
        src = structure_dir / sample_id / f"{sample_id}.npz"
        dst = output_dir / sample_id / f"{sample_id}.npz"

        if not src.exists():
            logger.error(f"  {sample_id}.npz — not found in training set at {src}")
            sys.exit(1)

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src.resolve(), dst)
        logger.info(f"  {sample_id}.npz — symlinked from {src}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download minimal PDB dataset for smooth lDDT benchmark."
    )
    parser.add_argument(
        "--training-set-dir",
        type=Path,
        default=None,
        help=(
            "Path to existing full PDB training set;"
            " symlinks samples instead of downloading."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target directory for benchmark data (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    logger.info(f"Output directory: {args.output_dir}")

    if args.training_set_dir is not None:
        logger.info(f"Symlinking from training set: {args.training_set_dir}")
        symlink_from_training_set(args.training_set_dir, args.output_dir)
    else:
        logger.info(f"Downloading from s3://{S3_BUCKET}/{S3_PREFIX}/")
        download_from_s3(args.output_dir)

    logger.info("Done. Run the benchmark with:")
    logger.info(
        "  pytest --benchmark-only openfold3/tests/test_smooth_lddt_benchmark.py"
    )


if __name__ == "__main__":
    main()
