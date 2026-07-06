#!/usr/bin/env python3
"""Download only the S3 files needed for a subset of PDB training data.

Usage:
    python download_subset.py                  # download missing files
    python download_subset.py --verify         # check all files exist (no download)
    python download_subset.py --sync-ref-mols  # also sync full reference_mols dir
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "openfold3-data"
S3_PREFIX = "pdb_training_set"
ROOT_DIR = Path(__file__).parent
LOCAL_ROOT = ROOT_DIR / "pdb_training_set"


CACHE_FILES = {
    "train": ROOT_DIR / "training_cache_with_templates_subset_512.json",
    "val": ROOT_DIR / "validation_cache_with_templates_subset_32.json",
}


def extract_ids_from_cache(cache_path: Path) -> dict:
    """Extract all file IDs needed from a dataset cache JSON."""
    with open(cache_path) as f:
        cache = json.load(f)

    pdb_ids = set()
    alignment_rep_ids = set()
    template_ids = set()
    reference_mol_ids = set()

    for pdb_id, entry in cache["structure_data"].items():
        pdb_ids.add(pdb_id)
        for chain_data in entry["chains"].values():
            if rep_id := chain_data.get("alignment_representative_id"):
                alignment_rep_ids.add(rep_id)
            for tmpl_id in chain_data.get("template_ids") or []:
                template_ids.add(tmpl_id)
            if ref_mol := chain_data.get("reference_mol_id"):
                reference_mol_ids.add(ref_mol)

    return {
        "pdb_ids": pdb_ids,
        "alignment_rep_ids": alignment_rep_ids,
        "template_ids": template_ids,
        "reference_mol_ids": reference_mol_ids,
    }


def build_manifest(ids: dict, template_cache_subdir: str) -> list[tuple[str, Path]]:
    """Build list of (s3_key, local_path) tuples."""
    manifest = []

    # Target structures: {pdb_id}/{pdb_id}.npz
    for pdb_id in ids["pdb_ids"]:
        s3_key = f"{S3_PREFIX}/preprocessed_pdb_data/standard/structure_files/{pdb_id}/{pdb_id}.npz"  # noqa: E501
        local = (
            LOCAL_ROOT
            / "preprocessed_pdb_data"
            / "standard"
            / "structure_files"
            / pdb_id
            / f"{pdb_id}.npz"
        )
        manifest.append((s3_key, local))

    # Alignment arrays: {rep_id}.npz
    for rep_id in ids["alignment_rep_ids"]:
        s3_key = f"{S3_PREFIX}/alignment_arrays/{rep_id}.npz"
        local = LOCAL_ROOT / "alignment_arrays" / f"{rep_id}.npz"
        manifest.append((s3_key, local))

    # Template cache: {rep_id}.npz
    for rep_id in ids["alignment_rep_ids"]:
        s3_key = f"{S3_PREFIX}/templates/{template_cache_subdir}/{rep_id}.npz"
        local = LOCAL_ROOT / "templates" / template_cache_subdir / f"{rep_id}.npz"
        manifest.append((s3_key, local))

    # Template structure arrays: {tmpl_pdb}/{tmpl_id}.npz
    for tmpl_id in ids["template_ids"]:
        tmpl_pdb = tmpl_id.split("_")[0]
        s3_key = (
            f"{S3_PREFIX}/templates/template_structure_arrays/{tmpl_pdb}/{tmpl_id}.npz"
        )
        local = (
            LOCAL_ROOT
            / "templates"
            / "template_structure_arrays"
            / tmpl_pdb
            / f"{tmpl_id}.npz"
        )
        manifest.append((s3_key, local))

    # Also grab chain_id_to_moltype.npz for each template PDB
    seen_tmpl_pdbs = {tmpl_id.split("_")[0] for tmpl_id in ids["template_ids"]}
    for tmpl_pdb in seen_tmpl_pdbs:
        s3_key = f"{S3_PREFIX}/templates/template_structure_arrays/{tmpl_pdb}/chain_id_to_moltype.npz"  # noqa: E501
        local = (
            LOCAL_ROOT
            / "templates"
            / "template_structure_arrays"
            / tmpl_pdb
            / "chain_id_to_moltype.npz"
        )
        manifest.append((s3_key, local))

    return manifest


def download_file(s3_client, s3_key: str, local_path: Path) -> tuple[str, str]:
    """Download a single file. Returns (s3_key, status)."""
    if local_path.exists():
        return (s3_key, "skipped")
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(BUCKET, s3_key, str(local_path))
        return (s3_key, "downloaded")
    except Exception as e:
        return (s3_key, f"FAILED: {e}")


def verify_manifest(manifest: list[tuple[str, Path]]) -> list[str]:
    """Check which files are missing."""
    missing = []
    for s3_key, local_path in manifest:
        if not local_path.exists():
            missing.append(s3_key)
    return missing


def sync_reference_mols():
    """Sync full reference_mols directory from S3."""
    local_dir = LOCAL_ROOT / "preprocessed_pdb_data" / "standard" / "reference_mols"
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSyncing reference_mols to {local_dir} ...")
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            f"s3://{BUCKET}/{S3_PREFIX}/preprocessed_pdb_data/standard/reference_mols/",
            str(local_dir) + "/",
            "--no-sign-request",
        ],
        check=True,
    )
    n_files = sum(1 for _ in local_dir.iterdir())
    print(f"reference_mols synced: {n_files} files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="Only verify, don't download"
    )
    parser.add_argument(
        "--sync-ref-mols", action="store_true", help="Also sync full reference_mols dir"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Parallel download threads"
    )
    args = parser.parse_args()

    # Build combined manifest from train + val caches
    all_manifest = []

    for split, cache_path in CACHE_FILES.items():
        print("cache_path:", cache_path)
        if not cache_path.exists():
            print(f"WARNING: {cache_path} not found, skipping {split}")
            continue

        template_cache_subdir = (
            "train_template_cache" if split == "train" else "val_template_cache"
        )
        ids = extract_ids_from_cache(cache_path)
        manifest = build_manifest(ids, template_cache_subdir)

        print(f"\n{split} ({cache_path.name}):")
        print(f"  PDB IDs: {len(ids['pdb_ids'])}")
        print(f"  Alignment rep IDs: {len(ids['alignment_rep_ids'])}")
        print(f"  Template IDs: {len(ids['template_ids'])}")
        print(f"  Reference mol IDs: {len(ids['reference_mol_ids'])}")
        print(f"  Files to download: {len(manifest)}")

        all_manifest.extend(manifest)

    # Deduplicate (alignment arrays shared between train/val)
    seen = set()
    deduped = []
    for s3_key, local_path in all_manifest:
        if s3_key not in seen:
            seen.add(s3_key)
            deduped.append((s3_key, local_path))
    all_manifest = deduped
    print(f"\nTotal unique files: {len(all_manifest)}")

    if args.verify:
        missing = verify_manifest(all_manifest)
        if missing:
            print(f"\nMISSING {len(missing)} files:")
            for m in missing[:20]:
                print(f"  {m}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
            sys.exit(1)
        else:
            print("\nAll files present!")
            sys.exit(0)

    # Download
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_file, s3, s3_key, local_path): s3_key
            for s3_key, local_path in all_manifest
        }
        for i, future in enumerate(as_completed(futures), 1):
            s3_key, status = future.result()
            if status == "downloaded":
                counts["downloaded"] += 1
            elif status == "skipped":
                counts["skipped"] += 1
            else:
                counts["failed"] += 1
                print(f"  {status}: {s3_key}")

            if i % 100 == 0 or i == len(futures):
                print(
                    f"  Progress: {i}/{len(futures)} "
                    f"(downloaded={counts['downloaded']}, "
                    f"skipped={counts['skipped']}, "
                    f"failed={counts['failed']})"
                )

    print(
        f"\nDone: {counts['downloaded']} downloaded, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )

    if args.sync_ref_mols:
        sync_reference_mols()


if __name__ == "__main__":
    main()
