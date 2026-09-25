#!/usr/bin/env python3
"""Build and pull the pinned 51-structure protein-RNA benchmark dataset.

Run from the repository root with::

    python -m scripts.benchmark.ball_query_smooth_lddt.pull_protein_rna_51_cases
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.datasets.pdb_subset_helpers import (
    download_full_cache,
    stream_subset,
    write_subset,
)

BOUNDED_PDB_IDS = (
    "7t1n",
    "7qdd",
    "7zex",
    "7sos",
    "7sop",
    "7sow",
    "7f3l",
    "7vrl",
    "7sov",
    "7f3j",
    "7zew",
    "7p0v",
    "7rgu",
    "7ozq",
    "7eiu",
    "7vsj",
    "7vkl",
    "7qde",
    "7x34",
    "7q4l",
    "7s3b",
    "7s3c",
    "7zhh",
    "7s3a",
    "7wl0",
    "7bah",
    "8af0",
    "7qr4",
    "7qr3",
    "7xs4",
    "7f3i",
    "8dzk",
    "8edj",
    "7uu4",
    "7z55",
    "7s02",
    "7rzz",
    "7v2z",
    "8d29",
    "7uo5",
    "7uo2",
    "6ww6",
    "6wxq",
    "8b0r",
    "7e8o",
)
LARGE_PDB_TOKEN_COUNTS = {
    "7vtn": 863,
    "7r9f": 1158,
    "7v93": 1344,
    "8h2h": 1513,
    "7ozs": 1824,
    "7r7c": 2012,
}
PDB_IDS = (*BOUNDED_PDB_IDS, *LARGE_PDB_TOKEN_COUNTS)


def _eligible_protein_rna_interfaces(entry: dict) -> list[str]:
    eligible = []
    for interface_id, interface in entry["interfaces"].items():
        chain_ids = interface_id.split("_")
        if len(chain_ids) != 2:
            continue
        molecule_types = {
            entry["chains"][chain_id]["molecule_type"] for chain_id in chain_ids
        }
        if molecule_types == {"PROTEIN", "RNA"} and interface.get(
            "metric_eligible", interface.get("use_metrics", True)
        ):
            eligible.append(interface_id)
    return eligible


def validate_selection(structure_data: dict) -> None:
    """Validate exact membership and benchmark-specific metadata invariants."""
    expected = set(PDB_IDS)
    actual = set(structure_data)
    if len(PDB_IDS) != 51 or len(expected) != 51:
        raise ValueError("The pinned selection must contain 51 unique PDB IDs")
    if actual != expected:
        raise ValueError(
            f"Cache membership mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    for pdb_id, entry in structure_data.items():
        if not _eligible_protein_rna_interfaces(entry):
            raise ValueError(f"{pdb_id} has no metric-eligible protein-RNA interface")

    for pdb_id, expected_tokens in LARGE_PDB_TOKEN_COUNTS.items():
        entry = structure_data[pdb_id]
        if entry["token_count"] != expected_tokens:
            raise ValueError(
                f"{pdb_id} token count is {entry['token_count']}, "
                f"expected {expected_tokens}"
            )
        molecule_types = {chain["molecule_type"] for chain in entry["chains"].values()}
        if not molecule_types <= {"PROTEIN", "RNA"}:
            raise ValueError(
                f"{pdb_id} contains unexpected molecule types: {molecule_types}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-cache",
        type=Path,
        default=Path("datasets/validation_cache_with_templates.json"),
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=Path("datasets/validation_cache_with_templates_protein_rna_51.json"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output cache after validating the new selection.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        help="Download assets here instead of <output-cache parent>/pdb_training_set.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Build and validate the 51-entry cache without downloading its assets.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify previously downloaded assets instead of downloading them.",
    )
    args = parser.parse_args()

    if args.output_cache.exists() and not args.force:
        cache = json.loads(args.output_cache.read_text())
        validate_selection(cache["structure_data"])
        print(f"Validated existing cache: {args.output_cache}")
    else:
        download_full_cache(args.full_cache)
        metadata, structure_data = stream_subset(args.full_cache, set(PDB_IDS))
        missing = set(PDB_IDS) - set(structure_data)
        if missing:
            raise ValueError(f"PDB IDs missing from full cache: {sorted(missing)}")
        validate_selection(structure_data)

        output_metadata = {
            **metadata,
            "name": "protein-rna-validation-51",
        }
        args.output_cache.parent.mkdir(parents=True, exist_ok=True)
        write_subset(
            args.output_cache,
            output_metadata,
            {pdb_id: structure_data[pdb_id] for pdb_id in sorted(PDB_IDS)},
        )

    if args.cache_only:
        return

    repository_root = Path(__file__).resolve().parents[3]
    command = [
        sys.executable,
        "-u",
        str(repository_root / "scripts/datasets/download_subset.py"),
        "--skip-train",
        "--val-subset-cache",
        str(args.output_cache),
        "--target-dir",
        str(args.output_cache.parent),
        "--workers",
        str(args.workers),
    ]
    if args.asset_dir is not None:
        command.extend(("--output-dir", str(args.asset_dir)))
    if args.verify:
        command.append("--verify")
    subprocess.run(command, cwd=repository_root, check=True)


if __name__ == "__main__":
    main()
